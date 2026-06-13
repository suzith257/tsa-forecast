"""
src/models.py
-------------
Fit three forecasting models and run walk-forward rolling-refit evaluation.

Models:
  manual   - Multiplicative decomposition built by hand (src.manual_model)
  sarima   - SARIMA(1,1,1)(1,1,1,7) on log(passengers)
  prophet  - Prophet with US-federal-holiday regressors

Walk-forward design (what a real WFM team does):
  Fold 1: train 2022-01-01 .. 2024-12-31 -> forecast 2025-01-01 .. 2025-12-31
  Fold 2: train 2022-01-01 .. 2025-12-31 -> forecast 2026-01-01 .. 2026-12-31
          (scored only on the 2026 days for which actuals exist)

Why this matters:
  - In production we refit periodically as new actuals arrive. Walk-forward
    evaluation mimics that and is the honest way to estimate future-year
    performance, especially when the 2026 model gets to "see" all of 2025.
  - It is also how to defend the model in an interview: we don't train
    once and hope - we refit on a rolling origin.

Output:
  outputs/forecasts_<year>.csv   one per fold, columns: date, actual,
                                 manual, sarima, prophet
"""
from __future__ import annotations

import warnings
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src import manual_model
from src.clean import clean

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_START = "2022-01-01"


# ---------------------------------------------------------------------------
# SARIMA
# ---------------------------------------------------------------------------
def _fit_sarima(train: pd.Series):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    model = SARIMAX(
        np.log(train),
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)


def _forecast_sarima(res, periods: int) -> np.ndarray:
    fc = res.get_forecast(steps=periods)
    return np.exp(fc.predicted_mean).values


# ---------------------------------------------------------------------------
# Prophet
# ---------------------------------------------------------------------------
def _prophet_holidays(years: range) -> pd.DataFrame:
    us = holidays.UnitedStates(years=list(years))
    rows = []
    for d, name in sorted(us.items()):
        if "(observed)" in name:
            continue
        rows.append({"holiday": name, "ds": pd.Timestamp(d),
                     "lower_window": -1, "upper_window": 1})
    for d, name in us.items():
        if name == "Thanksgiving Day":
            rows.append({"holiday": "Sunday after Thanksgiving",
                         "ds": pd.Timestamp(d) + pd.Timedelta(days=3),
                         "lower_window": 0, "upper_window": 0})
    return pd.DataFrame(rows)


def _fit_prophet(train: pd.Series):
    from prophet import Prophet
    df = pd.DataFrame({"ds": train.index, "y": train.values})
    years = range(train.index.year.min(), train.index.year.max() + 3)
    m = Prophet(
        yearly_seasonality=True, weekly_seasonality=True,
        daily_seasonality=False,
        holidays=_prophet_holidays(years),
        seasonality_mode="multiplicative",
    )
    m.fit(df)
    return m


def _forecast_prophet(model, dates: pd.DatetimeIndex) -> np.ndarray:
    future = pd.DataFrame({"ds": dates})
    return model.predict(future)["yhat"].values


# ---------------------------------------------------------------------------
# One walk-forward fold
# ---------------------------------------------------------------------------
def run_fold(full_series: pd.Series, train_end: str,
             forecast_year: int) -> pd.DataFrame:
    """Train through `train_end`, forecast all of `forecast_year`."""
    train = full_series.loc[TRAIN_START:train_end].dropna()
    fc_dates = pd.date_range(f"{forecast_year}-01-01",
                             f"{forecast_year}-12-31", freq="D")
    print(f"  Fold: train {train.index.min().date()} -> "
          f"{train.index.max().date()} ({len(train)} d); "
          f"forecast {fc_dates[0].date()} -> {fc_dates[-1].date()} "
          f"({len(fc_dates)} d)")

    print("    manual decomposition...")
    manual_fc, _ = manual_model.fit_and_forecast(train, fc_dates)

    print("    SARIMA(1,1,1)(1,1,1,7) on log...")
    sarima_res = _fit_sarima(train)
    sarima_fc = _forecast_sarima(sarima_res, len(fc_dates))

    print("    Prophet w/ holidays...")
    prophet_m = _fit_prophet(train)
    prophet_fc = _forecast_prophet(prophet_m, fc_dates)

    actuals = full_series.reindex(fc_dates)

    out = pd.DataFrame({
        "actual": actuals.values,
        "manual": manual_fc.values,
        "sarima": sarima_fc,
        "prophet": prophet_fc,
    }, index=fc_dates)
    out.index.name = "date"
    return out


def run_walk_forward() -> dict[int, pd.DataFrame]:
    """Two folds: forecast 2025 (trained through 2024) and 2026 (through 2025)."""
    df, _ = clean()
    series = df["passengers"].astype(float)

    folds = {}
    print("Fold 1 (test year = 2025):")
    folds[2025] = run_fold(series, "2024-12-31", 2025)
    folds[2025].to_csv(OUT_DIR / "forecasts_2025.csv")

    print("Fold 2 (test year = 2026):")
    folds[2026] = run_fold(series, "2025-12-31", 2026)
    folds[2026].to_csv(OUT_DIR / "forecasts_2026.csv")

    return folds


if __name__ == "__main__":
    folds = run_walk_forward()
    for year, frame in folds.items():
        n_actuals = frame["actual"].notna().sum()
        print(f"\n{year}: {len(frame)} forecast days, "
              f"{n_actuals} actuals available")
