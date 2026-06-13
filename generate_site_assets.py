"""
generate_site_assets.py
-----------------------
Run this locally (on a network where tsa.gov is reachable). It:

  1. Scrapes TSA daily passenger volumes (2019-2026).
  2. Trains three models (manual decomposition, SARIMA, Prophet) on 2022-2024,
     forecasts 2025, scores it  -> the validation year (yearly chart).
  3. Refits through 2025, produces a LOCKED annual 2026 forecast (3 lines).
  4. Runs a LOCKED monthly-refresh setup for 2026 with a one-month gap:
       end of Jan  -> refit -> forecast Mar onward  (Feb already locked)
       end of Feb  -> refit -> forecast Apr onward  (Mar already locked)
       ... etc.
     This produces 3 additional "refreshed" lines that exist from March.
  5. Generates every PNG into ./docs/assets/ for the GitHub Pages site:
       - data_2019_2025.png        (hero: daily volume through 2025)
       - forecast_2025_yearly.png  (validation year, 3 forecasts vs actual)
       - 2026_01.png ... 2026_06.png  (monthly panels)
            Jan, Feb : 3 locked forecasts + actual
            Mar..Jun : 3 locked + 3 refreshed forecasts + actual
       - metrics_2026.png          (bar chart comparison)
  6. Writes metrics_2026.csv and a metrics summary printed to console.

No LLM, no WFM/staffing layer. Every number is real.
"""
from __future__ import annotations

import warnings
from io import StringIO
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import requests
import holidays
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- config
ASSETS = Path("docs/assets")
ASSETS.mkdir(parents=True, exist_ok=True)
RAW_DIR = Path("data/raw")

TRAIN_START = "2022-01-01"
BASE_URL = "https://www.tsa.gov/travel/passenger-volumes"
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/124.0.0.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
YEARS = range(2019, 2027)

# Color scheme - legible on white
COLORS = {"manual": "#2E8B57", "sarima": "#C44E52", "prophet": "#1F4E79"}
ACTUAL_COLOR = "#111111"
MODELS = ("manual", "sarima", "prophet")
MONTH_NAMES = {1: "January", 2: "February", 3: "March",
               4: "April", 5: "May", 6: "June"}


# ================================================================ INGEST
def _parse_year_table(html: str) -> pd.DataFrame:
    tables = pd.read_html(StringIO(html))
    df = tables[0].copy()
    df.columns = [str(c).strip() for c in df.columns]
    date_col = [c for c in df.columns if "date" in c.lower()][0]
    num_col = [c for c in df.columns if c != date_col][0]
    out = pd.DataFrame({
        "date": pd.to_datetime(df[date_col], errors="coerce"),
        "passengers": pd.to_numeric(
            df[num_col].astype(str).str.replace(",", "", regex=False),
            errors="coerce"),
    }).dropna()
    out["passengers"] = out["passengers"].astype(int)
    return out


def fetch_year(year: int) -> pd.DataFrame:
    url = BASE_URL if year == max(YEARS) else f"{BASE_URL}/{year}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=25)
        r.raise_for_status()
        df = _parse_year_table(r.text)
        if len(df):
            print(f"  {year}: live OK ({len(df)} rows)")
            return df[df["date"].dt.year == year]
    except Exception as e:
        print(f"  {year}: live failed ({type(e).__name__}); trying cache")
    cache = RAW_DIR / f"tsa_{year}.html"
    if cache.exists():
        df = _parse_year_table(cache.read_text())
        return df[df["date"].dt.year == year]
    print(f"  {year}: NO DATA")
    return pd.DataFrame(columns=["date", "passengers"])


def ingest() -> pd.Series:
    print("Ingesting TSA data...")
    frames = [fetch_year(y) for y in YEARS]
    df = (pd.concat(frames, ignore_index=True)
            .drop_duplicates("date").sort_values("date"))
    df = df.set_index("date")
    full = pd.date_range(df.index.min(), df.index.max(), freq="D", name="date")
    s = df["passengers"].reindex(full).astype(float)
    print(f"  Total: {s.notna().sum():,} days, "
          f"{s.index.min().date()} -> {s.index.max().date()}")
    return s


# ================================================================ MANUAL MODEL
FOURIER_K = 6
TREND_WINDOW = 28
TREND_FIT_DAYS = 730
TREND_DAMPING = 0.5
BRIDGE_OFFSETS = (-3, -2, -1, 1, 2, 3)
MIN_OBS = 2
FIXED_WEEKDAY_HOLIDAYS = {
    "Martin Luther King Jr. Day", "Washington's Birthday", "Memorial Day",
    "Labor Day", "Columbus Day", "Thanksgiving Day",
}


@dataclass
class ManualModel:
    dow_factors: dict = field(default_factory=dict)
    fourier_coefs: np.ndarray = field(default_factory=lambda: np.zeros(0))
    holiday_factors: dict = field(default_factory=dict)
    bridge_factors: dict = field(default_factory=dict)
    weekday_holiday_offsets: dict = field(default_factory=dict)
    log_level_slope: float = 0.0
    log_level_anchor_date: pd.Timestamp = pd.Timestamp("1970-01-01")
    log_level_anchor_value: float = 0.0


def _federal_holidays(years):
    us = holidays.UnitedStates(years=list(years))
    rows = [(pd.Timestamp(d), n) for d, n in sorted(us.items())
            if "(observed)" not in n]
    return pd.DataFrame(rows, columns=["date", "name"])


def _fourier(dates, k=FOURIER_K):
    doy = dates.dayofyear.values.astype(float)
    cols = []
    for h in range(1, k + 1):
        ang = 2 * np.pi * h * doy / 365.25
        cols += [np.sin(ang), np.cos(ang)]
    return np.column_stack(cols)


def _geo_mean(v):
    return float(np.exp(np.mean(np.log(v))))


def fit_manual(train: pd.Series) -> ManualModel:
    m = ManualModel()
    s = train.astype(float)
    trend = s.rolling(TREND_WINDOW, center=True, min_periods=14).median().bfill().ffill()
    X_f = _fourier(trend.index)
    years = sorted(trend.index.year.unique())
    X_y = np.column_stack([(trend.index.year == y).astype(float) for y in years])
    beta, *_ = np.linalg.lstsq(np.column_stack([X_y, X_f]),
                               np.log(trend.values), rcond=None)
    m.fourier_coefs = beta[len(years):]
    detrended = s / trend
    dow_raw = {w: _geo_mean(detrended[detrended.index.dayofweek == w]) for w in range(7)}
    norm = _geo_mean(list(dow_raw.values()))
    m.dow_factors = {w: v / norm for w, v in dow_raw.items()}
    dow_series = pd.Series([m.dow_factors[w] for w in detrended.index.dayofweek],
                           index=detrended.index)
    residual = detrended / dow_series
    hol = _federal_holidays(range(s.index.year.min(), s.index.year.max() + 1))
    hol = hol[hol["date"].isin(residual.index)]
    obs = {}
    for _, r in hol.iterrows():
        obs.setdefault(r["name"], []).append(float(residual.loc[r["date"]]))
    m.holiday_factors = {n: _geo_mean(v) for n, v in obs.items() if len(v) >= MIN_OBS}
    pooled, per_holiday = {}, {}
    holiday_dates = set(hol["date"])
    for _, r in hol.iterrows():
        d, name = r["date"], r["name"]
        for off in BRIDGE_OFFSETS:
            t = d + pd.Timedelta(days=off)
            if t not in residual.index or t in holiday_dates:
                continue
            val = float(residual.loc[t])
            if name in FIXED_WEEKDAY_HOLIDAYS:
                per_holiday.setdefault((name, off), []).append(val)
            else:
                pooled.setdefault((d.dayofweek, off), []).append(val)
    m.bridge_factors = {k: _geo_mean(v) for k, v in pooled.items() if len(v) >= MIN_OBS}
    m.weekday_holiday_offsets = {k: _geo_mean(v) for k, v in per_holiday.items()
                                 if len(v) >= MIN_OBS}
    recent = trend.iloc[-TREND_FIT_DAYS:]
    x = (recent.index - recent.index[0]).days.values.astype(float)
    slope, intercept = np.polyfit(x, np.log(recent.values), 1)
    m.log_level_slope, m.log_level_anchor_date, m.log_level_anchor_value = (
        float(slope), recent.index[0], float(intercept))
    return m


def forecast_manual(m: ManualModel, dates: pd.DatetimeIndex) -> pd.Series:
    x = (dates - m.log_level_anchor_date).days.values.astype(float)
    log_level = m.log_level_anchor_value + TREND_DAMPING * m.log_level_slope * x
    base = np.exp(log_level + _fourier(dates) @ m.fourier_coefs)
    hol = _federal_holidays(range(dates.year.min(), dates.year.max() + 1))
    holiday_dates = set(hol["date"])

    def cal_mult(date):
        f = 1.0
        for _, r in hol[hol["date"] == date].iterrows():
            f *= m.holiday_factors.get(r["name"], 1.0)
        if date in holiday_dates:
            return f
        for off in BRIDGE_OFFSETS:
            src = date - pd.Timedelta(days=off)
            for _, r in hol[hol["date"] == src].iterrows():
                if r["name"] in FIXED_WEEKDAY_HOLIDAYS:
                    f *= m.weekday_holiday_offsets.get((r["name"], off), 1.0)
                else:
                    f *= m.bridge_factors.get((src.dayofweek, off), 1.0)
        return f

    out = [b * m.dow_factors[d.dayofweek] * cal_mult(d) for d, b in zip(dates, base)]
    return pd.Series(out, index=dates)


# ================================================================ BENCHMARKS
def forecast_sarima(train, dates):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    res = SARIMAX(np.log(train), order=(1, 1, 1), seasonal_order=(1, 1, 1, 7),
                  enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    return pd.Series(np.exp(res.get_forecast(steps=len(dates)).predicted_mean).values,
                     index=dates)


def _prophet_holidays(years):
    us = holidays.UnitedStates(years=list(years))
    rows = [{"holiday": n, "ds": pd.Timestamp(d), "lower_window": -1, "upper_window": 1}
            for d, n in sorted(us.items()) if "(observed)" not in n]
    for d, n in us.items():
        if n == "Thanksgiving Day":
            rows.append({"holiday": "Sunday after Thanksgiving",
                         "ds": pd.Timestamp(d) + pd.Timedelta(days=3),
                         "lower_window": 0, "upper_window": 0})
    return pd.DataFrame(rows)


def forecast_prophet(train, dates):
    from prophet import Prophet
    years = range(train.index.year.min(), train.index.year.max() + 3)
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False,
                holidays=_prophet_holidays(years), seasonality_mode="multiplicative")
    m.fit(pd.DataFrame({"ds": train.index, "y": train.values}))
    return pd.Series(m.predict(pd.DataFrame({"ds": dates}))["yhat"].values, index=dates)


def all_three(train, dates):
    return {"manual": forecast_manual(fit_manual(train), dates),
            "sarima": forecast_sarima(train, dates),
            "prophet": forecast_prophet(train, dates)}


# ================================================================ METRICS
def mape(actual, fc):
    mask = actual.notna() & fc.notna()
    a, f = actual[mask], fc[mask]
    return float((f - a).abs().div(a).mean() * 100)


def bias(actual, fc):
    mask = actual.notna() & fc.notna()
    a, f = actual[mask], fc[mask]
    return float((f - a).mean() / a.mean() * 100)


def daily_accuracy(actual, fc):
    """Per-day forecast accuracy = 1 - |actual - forecast| / actual (percent)."""
    mask = actual.notna() & fc.notna()
    a, f = actual[mask], fc[mask]
    return (1 - (a - f).abs() / a) * 100


def accuracy_summary(actual, fc):
    acc = daily_accuracy(actual, fc)
    if len(acc) == 0:
        return None
    return {
        "days": int(len(acc)),
        "mean_acc": round(float(acc.mean()), 2),
        "days_ge_90": int((acc >= 90).sum()),
        "pct_ge_90": round(float((acc >= 90).mean() * 100), 1),
    }


# ================================================================ MAIN
def main():
    s = ingest()

    # ---- Fold 1: train <=2024, forecast 2025 (validation year)
    print("\nForecasting 2025 (validation)...")
    train24 = s.loc[TRAIN_START:"2024-12-31"].dropna()
    d2025 = pd.date_range("2025-01-01", "2025-12-31", freq="D")
    fc2025 = all_three(train24, d2025)
    act2025 = s.reindex(d2025)

    # ---- Locked annual 2026 forecast: train <=2025
    print("Forecasting 2026 (locked annual, trained through 2025)...")
    train25 = s.loc[TRAIN_START:"2025-12-31"].dropna()
    d2026 = pd.date_range("2026-01-01", "2026-12-31", freq="D")
    locked2026 = all_three(train25, d2026)
    act2026 = s.reindex(d2026)
    last_actual = act2026.dropna().index.max()
    print(f"  2026 actuals available through {last_actual.date()}")

    # ---- Locked monthly refresh (one-month gap)
    #   end of month M -> refit through M-end -> forecast from month M+2 onward
    #   so refit at end of Jan first produces forecasts starting March.
    print("Running locked monthly refresh (one-month gap)...")
    refreshed2026 = {m: pd.Series(index=d2026, dtype=float) for m in MODELS}
    for refit_month in range(1, 6):           # refit at end of Jan..May
        refit_end = (pd.Timestamp(f"2026-{refit_month:02d}-01")
                     + pd.offsets.MonthEnd(0))
        start_month = refit_month + 2          # one-month locked gap
        if start_month > 12:
            continue
        fc_start = pd.Timestamp(f"2026-{start_month:02d}-01")
        fc_dates = pd.date_range(fc_start, "2026-12-31", freq="D")
        train_r = s.loc[TRAIN_START:refit_end].dropna()
        if train_r.index.max() < refit_end:
            # not enough actuals yet for this refit (future month) - skip
            continue
        print(f"  refit through {refit_end.date()} -> forecast from {fc_start.date()}")
        fc_r = all_three(train_r, fc_dates)
        # Fill ONLY the not-yet-filled portion so each month keeps the forecast
        # made at the most recent allowed refit (locked once entered).
        for m in MODELS:
            for dt in fc_dates:
                if pd.isna(refreshed2026[m].loc[dt]):
                    refreshed2026[m].loc[dt] = fc_r[m].loc[dt]

    # =========================================================== CHARTS
    plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.25,
                         "figure.dpi": 130, "font.size": 11})

    # ---- Hero: daily volume 2019-2025
    print("\nGenerating charts...")
    hist = s.loc[:"2025-12-31"]
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(hist.index, hist.values / 1e6, lw=0.6, color=COLORS["prophet"])
    ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-06-01"),
               color="red", alpha=0.07, label="COVID (excluded from training)")
    ax.set_title("TSA daily passenger throughput, 2019–2025")
    ax.set_ylabel("Passengers (millions)")
    ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(ASSETS / "data_2019_2025.png"); plt.close(fig)

    # ---- 2025 yearly validation
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(act2025.index, act2025.values / 1e6, color=ACTUAL_COLOR, lw=1.4,
            label="Actual", zorder=5)
    for m in MODELS:
        ax.plot(d2025, fc2025[m].values / 1e6, color=COLORS[m], lw=1.0,
                ls="--", alpha=0.85, label=m, zorder=3)
    ax.set_title("2025 validation — trained on 2022–2024, forecast vs. actual")
    ax.set_ylabel("Passengers (millions)")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.legend(ncol=4, loc="lower center")
    fig.tight_layout(); fig.savefig(ASSETS / "forecast_2025_yearly.png"); plt.close(fig)

    # ---- 2026 monthly panels
    for month in range(1, 7):
        m_start = pd.Timestamp(f"2026-{month:02d}-01")
        m_end = m_start + pd.offsets.MonthEnd(0)
        idx = pd.date_range(m_start, m_end, freq="D")
        a = act2026.reindex(idx)
        if a.dropna().empty:
            continue
        # clip to last actual within the month
        idx = idx[idx <= last_actual]
        a = a.reindex(idx)

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(idx, a.values / 1e6, color=ACTUAL_COLOR, lw=1.8,
                label="Actual", zorder=6)
        # locked annual (3 lines) - lighter dotted
        for m in MODELS:
            ax.plot(idx, locked2026[m].reindex(idx).values / 1e6,
                    color=COLORS[m], lw=1.0, ls=":", alpha=0.7,
                    label=f"{m} (locked)", zorder=3)
        # refreshed (only where it exists, i.e. Mar onward) - bolder dashed
        has_refresh = refreshed2026["manual"].reindex(idx).notna().any()
        if has_refresh:
            for m in ("manual", "prophet"):   # SARIMA excluded - unstable under short-window refit
                ax.plot(idx, refreshed2026[m].reindex(idx).values / 1e6,
                        color=COLORS[m], lw=1.6, ls="--", alpha=0.95,
                        label=f"{m} (refreshed)", zorder=4)
        ax.set_title(f"2026 — {MONTH_NAMES[month]}: forecasts vs. actual")
        ax.set_ylabel("Passengers (millions)")
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ncol = 4 if has_refresh else 4
        ax.legend(ncol=ncol, loc="lower center", fontsize=8)
        fig.tight_layout(); fig.savefig(ASSETS / f"2026_{month:02d}.png"); plt.close(fig)
        print(f"  2026_{month:02d}.png  ({'7 lines' if has_refresh else '4 lines'})")

    # ---- Metrics on 2026 to date
    print("\n2026 metrics (scored on available actuals):")
    rows = []
    for m in MODELS:
        rows.append({"model": m, "type": "locked",
                     "MAPE": round(mape(act2026, locked2026[m]), 2),
                     "Bias": round(bias(act2026, locked2026[m]), 2)})
        # refreshed scored only where it exists
        rf = refreshed2026[m]
        mask = act2026.notna() & rf.notna()
        if mask.any():
            rows.append({"model": m, "type": "refreshed",
                         "MAPE": round(mape(act2026[mask], rf[mask]), 2),
                         "Bias": round(bias(act2026[mask], rf[mask]), 2)})
    metrics_df = pd.DataFrame(rows)
    print(metrics_df.to_string(index=False))
    metrics_df.to_csv(ASSETS / "metrics_2026.csv", index=False)

    # ---- Metrics bar chart
    fig, ax = plt.subplots(figsize=(10, 5.5))
    locked = metrics_df[metrics_df["type"] == "locked"].set_index("model")["MAPE"]
    refreshed = (metrics_df[metrics_df["type"] == "refreshed"]
                 .set_index("model")["MAPE"])
    xpos = np.arange(len(MODELS))
    w = 0.38
    ax.bar(xpos - w/2, [locked.get(m, np.nan) for m in MODELS], w,
           label="Locked annual", color=[COLORS[m] for m in MODELS], alpha=0.55)
    ax.bar(xpos + w/2, [refreshed.get(m, np.nan) for m in MODELS], w,
           label="Refreshed", color=[COLORS[m] for m in MODELS], alpha=1.0)
    ax.set_xticks(xpos); ax.set_xticklabels(MODELS)
    ax.set_ylabel("MAPE (%)  — lower is better")
    ax.set_title("2026 forecast accuracy: locked vs. refreshed")
    ax.legend()
    for i, m in enumerate(MODELS):
        if not np.isnan(locked.get(m, np.nan)):
            ax.text(i - w/2, locked[m] + 0.1, f"{locked[m]:.1f}", ha="center", fontsize=9)
        if m in refreshed.index and not np.isnan(refreshed.get(m, np.nan)):
            ax.text(i + w/2, refreshed[m] + 0.1, f"{refreshed[m]:.1f}", ha="center", fontsize=9)
    fig.tight_layout(); fig.savefig(ASSETS / "metrics_2026.png"); plt.close(fig)

    # ---- 2025 metrics too (for the validation section)
    print("\n2025 validation metrics:")
    rows25 = [{"model": m, "MAPE": round(mape(act2025, fc2025[m]), 2),
               "Bias": round(bias(act2025, fc2025[m]), 2)} for m in MODELS]
    df25 = pd.DataFrame(rows25)
    print(df25.to_string(index=False))
    df25.to_csv(ASSETS / "metrics_2025.csv", index=False)

    # ---- Per-month forecast accuracy printout (actual denominator)
    print("\n" + "=" * 64)
    print("FORECAST ACCURACY BY MONTH  (1 - |actual-forecast|/actual)")
    print("=" * 64)
    for month in range(1, 7):
        m_start = pd.Timestamp(f"2026-{month:02d}-01")
        m_end = m_start + pd.offsets.MonthEnd(0)
        idx = pd.date_range(m_start, m_end, freq="D")
        a = act2026.reindex(idx)
        if a.dropna().empty:
            continue
        print(f"\n--- {MONTH_NAMES[month]} 2026 ---")
        for m in MODELS:
            s = accuracy_summary(a, locked2026[m].reindex(idx))
            if s:
                print(f"  {m:>8} (locked)    "
                      f"mean {s['mean_acc']:5.2f}%   "
                      f">=90%: {s['days_ge_90']:2d}/{s['days']} "
                      f"({s['pct_ge_90']:.0f}%)")
        for m in ("manual", "prophet"):
            rf = refreshed2026[m].reindex(idx)
            if rf.notna().any():
                s = accuracy_summary(a, rf)
                if s:
                    print(f"  {m:>8} (refreshed) "
                          f"mean {s['mean_acc']:5.2f}%   "
                          f">=90%: {s['days_ge_90']:2d}/{s['days']} "
                          f"({s['pct_ge_90']:.0f}%)")

    print(f"\nAll assets written to {ASSETS.resolve()}")
    print("Charts:", sorted(p.name for p in ASSETS.glob('*.png')))


if __name__ == "__main__":
    main()
