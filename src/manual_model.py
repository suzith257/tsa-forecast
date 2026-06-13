"""
src/manual_model.py
-------------------
A hand-rolled multiplicative decomposition forecaster.

No forecasting library is used here. The purpose is to estimate every
component of daily travel demand transparently, from first principles,
and show that the result is competitive with library models:

    passengers_t = level_t * annual_t * dow_t * holiday_t * bridge_t

COMPONENTS

  LEVEL      A 28-day centred rolling median of the training series
             (28 = 4 weekly cycles, so the weekly pattern averages out).
             For forecasting, the long-run level is projected with a
             log-linear regression fitted on the last 730 days of this
             trend - exactly two annual cycles, so the seasonal phase
             cancels out of the slope - and the slope is damped toward
             zero (multiplied by 0.5) because recent growth rates should
             not be extrapolated at full strength a year ahead.

  ANNUAL     Smooth within-year seasonality, estimated by regressing
             log(level_t) on year fixed-effects plus K=6 Fourier
             harmonics of day-of-year. The year dummies absorb growth
             between years; the harmonics capture the annual shape
             (deep January trough, early-summer peak) as a smooth curve
             rather than 12 step-function buckets - important because
             months like January are not internally flat: the first
             week carries post-holiday return traffic while the middle
             of the month is the quietest stretch of the year.

  DOW        Day-of-week factors: the geometric mean of
             passengers / level on each weekday, normalised so the
             seven factors multiply to one.

  HOLIDAY    One multiplicative factor per US federal holiday (the
             `holidays` library calendar - the same calendar the Prophet
             benchmark uses, so the model comparison is like-for-like),
             estimated as the geometric mean of the day's
             level-and-DOW-adjusted residual across training years.

  BRIDGE     Travel peaks rarely sit on the holiday itself; they sit on
             the surrounding days, and their placement depends on which
             weekday the holiday occupies (a Monday holiday creates a
             different long-weekend shape than a Saturday one). Two
             estimation strategies, chosen by holiday type:

             * Fixed-DATE holidays (New Year's Day, Juneteenth,
               Independence Day, Veterans Day, Christmas) fall on a
               different weekday each year, and each occurs only ~3
               times in training - too few to learn per-holiday weekday
               interactions. So their surrounding-day effects are POOLED
               into one table keyed by (weekday-of-holiday, offset).

             * Fixed-WEEKDAY holidays (MLK Day, Washington's Birthday,
               Memorial Day, Labor Day, Columbus Day, Thanksgiving)
               always fall on the same weekday, so each gets its own
               per-holiday offset profile over -3..+3 days. This is
               where the biggest travel days of the year live: the
               Sunday after Thanksgiving is Thanksgiving at offset +3.

All factor estimation uses geometric means (the model is multiplicative)
and requires at least two observations per cell; cells with fewer
observations default to a factor of 1.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

OUT_DIR = Path("outputs")

TREND_WINDOW = 28          # days; 4 weekly cycles
TREND_FIT_DAYS = 730       # 2 annual cycles -> season-neutral slope
TREND_DAMPING = 0.5        # shrink extrapolated slope toward zero
FOURIER_K = 6              # annual harmonics (finest wavelength ~2 months)
BRIDGE_OFFSETS = (-3, -2, -1, 1, 2, 3)
MIN_OBS = 2                # minimum observations per estimated factor

# Holidays that always fall on the same weekday: learn a per-holiday
# offset profile. All other federal holidays are fixed-date: pool their
# surrounding-day effects by (weekday-of-holiday, offset).
FIXED_WEEKDAY_HOLIDAYS = {
    "Martin Luther King Jr. Day",
    "Washington's Birthday",
    "Memorial Day",
    "Labor Day",
    "Columbus Day",
    "Thanksgiving Day",
}


@dataclass
class ManualModel:
    """Fitted components of the decomposition."""
    trend: pd.Series = field(default_factory=pd.Series)
    dow_factors: dict[int, float] = field(default_factory=dict)
    fourier_coefs: np.ndarray = field(default_factory=lambda: np.zeros(0))
    holiday_factors: dict[str, float] = field(default_factory=dict)
    # pooled bridges for fixed-date holidays: (weekday, offset) -> factor
    bridge_factors: dict[tuple[int, int], float] = field(default_factory=dict)
    # per-holiday bridges for fixed-weekday holidays: (name, offset) -> factor
    weekday_holiday_offsets: dict[tuple[str, int], float] = field(
        default_factory=dict)
    log_level_slope: float = 0.0
    log_level_anchor_date: pd.Timestamp = field(
        default_factory=lambda: pd.Timestamp("1970-01-01"))
    log_level_anchor_value: float = 0.0


# ---------------------------------------------------------------------------
# Calendar helpers
# ---------------------------------------------------------------------------
def federal_holidays(years: range) -> pd.DataFrame:
    """US federal holidays as [date, name]; '(observed)' entries dropped."""
    us = holidays.UnitedStates(years=list(years))
    rows = [(pd.Timestamp(d), n) for d, n in sorted(us.items())
            if "(observed)" not in n]
    return pd.DataFrame(rows, columns=["date", "name"])


def _fourier_design(dates: pd.DatetimeIndex, k: int = FOURIER_K) -> np.ndarray:
    """Sin/cos design matrix on day-of-year (period 365.25)."""
    doy = dates.dayofyear.values.astype(float)
    cols = []
    for h in range(1, k + 1):
        ang = 2.0 * np.pi * h * doy / 365.25
        cols.append(np.sin(ang))
        cols.append(np.cos(ang))
    return np.column_stack(cols)


def _geo_mean(values: list[float] | pd.Series) -> float:
    return float(np.exp(np.mean(np.log(values))))


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------
def fit(train: pd.Series) -> ManualModel:
    m = ManualModel()
    s = train.astype(float)

    # ---- LEVEL: rolling median trend
    trend = s.rolling(TREND_WINDOW, center=True, min_periods=14).median()
    trend = trend.bfill().ffill()
    m.trend = trend

    # ---- ANNUAL: log(trend) ~ year dummies + Fourier(doy)
    X_f = _fourier_design(trend.index)
    years = sorted(trend.index.year.unique())
    X_y = np.column_stack([(trend.index.year == y).astype(float)
                           for y in years])
    X = np.column_stack([X_y, X_f])
    beta, *_ = np.linalg.lstsq(X, np.log(trend.values), rcond=None)
    m.fourier_coefs = beta[len(years):]          # keep harmonics only

    # ---- DOW: geometric mean of detrended values per weekday
    detrended = s / trend
    dow_raw = {w: _geo_mean(detrended[detrended.index.dayofweek == w])
               for w in range(7)}
    norm = _geo_mean(list(dow_raw.values()))
    m.dow_factors = {w: v / norm for w, v in dow_raw.items()}

    # ---- Residual after level + DOW (annual is inside `trend` in-sample)
    dow_series = pd.Series([m.dow_factors[w] for w in detrended.index.dayofweek],
                           index=detrended.index)
    residual = detrended / dow_series

    hol = federal_holidays(range(s.index.year.min(), s.index.year.max() + 1))
    hol = hol[hol["date"].isin(residual.index)]

    # ---- HOLIDAY: day-of effect per holiday name
    holiday_obs: dict[str, list[float]] = {}
    for _, row in hol.iterrows():
        holiday_obs.setdefault(row["name"], []).append(
            float(residual.loc[row["date"]]))
    m.holiday_factors = {n: _geo_mean(v) for n, v in holiday_obs.items()
                         if len(v) >= MIN_OBS}

    # ---- BRIDGE: surrounding-day effects
    pooled: dict[tuple[int, int], list[float]] = {}
    per_holiday: dict[tuple[str, int], list[float]] = {}
    holiday_dates = set(hol["date"])
    for _, row in hol.iterrows():
        d, name = row["date"], row["name"]
        for off in BRIDGE_OFFSETS:
            target = d + pd.Timedelta(days=off)
            if target not in residual.index:
                continue
            if target in holiday_dates:
                continue   # that day's effect belongs to its own holiday
            val = float(residual.loc[target])
            if name in FIXED_WEEKDAY_HOLIDAYS:
                per_holiday.setdefault((name, off), []).append(val)
            else:
                pooled.setdefault((d.dayofweek, off), []).append(val)
    m.bridge_factors = {k: _geo_mean(v) for k, v in pooled.items()
                        if len(v) >= MIN_OBS}
    m.weekday_holiday_offsets = {k: _geo_mean(v)
                                 for k, v in per_holiday.items()
                                 if len(v) >= MIN_OBS}

    # ---- LEVEL projection: log-linear over the last two annual cycles
    recent = trend.iloc[-TREND_FIT_DAYS:]
    x = (recent.index - recent.index[0]).days.values.astype(float)
    slope, intercept = np.polyfit(x, np.log(recent.values), 1)
    m.log_level_slope = float(slope)
    m.log_level_anchor_date = recent.index[0]
    m.log_level_anchor_value = float(intercept)
    return m


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------
def _calendar_multiplier(model: ManualModel, date: pd.Timestamp,
                         hol_table: pd.DataFrame,
                         holiday_dates: set) -> float:
    """Holiday + bridge factor for one date."""
    factor = 1.0
    # Day-of holiday effect
    for _, row in hol_table[hol_table["date"] == date].iterrows():
        factor *= model.holiday_factors.get(row["name"], 1.0)
    if date in holiday_dates:
        return factor          # a holiday is never also a bridge day
    # Bridge effects from any holiday within +-3 days
    for off in BRIDGE_OFFSETS:
        source = date - pd.Timedelta(days=off)
        for _, row in hol_table[hol_table["date"] == source].iterrows():
            if row["name"] in FIXED_WEEKDAY_HOLIDAYS:
                factor *= model.weekday_holiday_offsets.get(
                    (row["name"], off), 1.0)
            else:
                factor *= model.bridge_factors.get(
                    (source.dayofweek, off), 1.0)
    return factor


def forecast(model: ManualModel, dates: pd.DatetimeIndex,
             damping: float = TREND_DAMPING) -> pd.Series:
    # Long-run level, damped
    x = (dates - model.log_level_anchor_date).days.values.astype(float)
    log_level = model.log_level_anchor_value + damping * model.log_level_slope * x
    # Annual shape
    annual = _fourier_design(dates) @ model.fourier_coefs
    base = np.exp(log_level + annual)

    hol_table = federal_holidays(range(dates.year.min(), dates.year.max() + 1))
    holiday_dates = set(hol_table["date"])

    out = [b
           * model.dow_factors[d.dayofweek]
           * _calendar_multiplier(model, d, hol_table, holiday_dates)
           for d, b in zip(dates, base)]
    return pd.Series(out, index=dates)


def fit_and_forecast(train: pd.Series,
                     forecast_dates: pd.DatetimeIndex
                     ) -> tuple[pd.Series, ManualModel]:
    model = fit(train)
    return forecast(model, forecast_dates), model


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------
def annual_curve(model: ManualModel) -> pd.Series:
    """The fitted smooth annual shape, evaluated on a non-leap year."""
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
    return pd.Series(np.exp(_fourier_design(dates) @ model.fourier_coefs),
                     index=dates.dayofyear)


def month_factors(model: ManualModel) -> dict[int, float]:
    """Monthly geometric means of the annual curve (for display)."""
    curve = annual_curve(model)
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
    out = {}
    for mo in range(1, 13):
        vals = curve.values[dates.month == mo]
        out[mo] = _geo_mean(list(vals))
    return out


def factors_summary(model: ManualModel) -> dict:
    wd = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    return {
        "dow_factors": {wd[w]: round(f, 3)
                        for w, f in model.dow_factors.items()},
        "month_factors": {f"{mo:02d}": round(f, 3)
                          for mo, f in month_factors(model).items()},
        "holiday_factors": {k: round(v, 3)
                            for k, v in sorted(model.holiday_factors.items())},
        "fixed_date_bridges": {
            f"holiday_on_{wd[w]}_offset_{off:+d}": round(v, 3)
            for (w, off), v in sorted(model.bridge_factors.items())},
        "fixed_weekday_bridges": {
            f"{name}_offset_{off:+d}": round(v, 3)
            for (name, off), v in sorted(model.weekday_holiday_offsets.items())},
        "annualised_growth_pct": round(
            (np.exp(model.log_level_slope * 365) - 1) * 100, 2),
    }


if __name__ == "__main__":
    import json
    from src.clean import clean
    df, _ = clean()
    train = df.loc["2022-01-01":"2024-12-31", "passengers"].astype(float)
    yhat, model = fit_and_forecast(
        train, pd.date_range("2025-01-01", "2025-12-31"))
    print(json.dumps(factors_summary(model), indent=2))
    print(f"\n2025 forecast: mean {yhat.mean():,.0f}, "
          f"peak {yhat.max():,.0f} on {yhat.idxmax().date()}")
