"""
src/refresh.py
--------------
Monthly refresh study: how much accuracy does retraining buy as each
month of actuals closes?

THE QUESTION THIS ANSWERS
  The frozen annual forecast (src/models.py) is produced once, from data
  available before the forecast year begins, and never updated. Real
  planning teams refresh: when January closes, they retrain and reissue
  February onward; when February closes, March onward; and so on. This
  module simulates that cadence and measures, month by month, whether
  the refreshed forecast beats the frozen one.

DESIGN
  For each month m of the study year (February onward):
    1. Train every model on all data through the last day of month m-1.
    2. Forecast the days of month m.
    3. Score month m against its actuals (which were never part of that
       training window).
  January has no preceding in-year actuals, so its refreshed forecast is
  by construction identical to the frozen one and serves as the shared
  baseline row.

  Each monthly forecast is a genuine out-of-sample forecast at one-month
  horizon: the rolling origin moves forward, but no model ever sees the
  month it is scored on. The frozen forecast remains the headline result;
  this study isolates the incremental value of refresh cadence, which is
  an operating-policy question, not a model question.

OUTPUT
  outputs/refresh_study_<year>.csv     per (month, model): frozen vs
                                       refreshed MAPE and bias
  outputs/10_monthly_refresh.png       the comparison chart
  outputs/refresh_report_<year>.json   machine-readable summary
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src import manual_model
from src.clean import clean
from src.models import TRAIN_START, _fit_prophet, _fit_sarima, \
    _forecast_prophet, _forecast_sarima

OUT_DIR = Path("outputs")
MODELS = ("manual", "sarima", "prophet")
MODEL_COLORS = {"manual": "#2e8b57", "sarima": "#c0504d", "prophet": "#1f4e79"}


def _mape(actual: pd.Series, forecast: pd.Series) -> float:
    mask = actual.notna()
    return float(((forecast[mask] - actual[mask]).abs()
                  / actual[mask]).mean() * 100)


def _bias(actual: pd.Series, forecast: pd.Series) -> float:
    mask = actual.notna()
    return float((forecast[mask] - actual[mask]).mean()
                 / actual[mask].mean() * 100)


def _forecast_month(train: pd.Series, dates: pd.DatetimeIndex) -> dict:
    """Fit all three models on `train` and forecast `dates`."""
    out = {}
    fc, _ = manual_model.fit_and_forecast(train, dates)
    out["manual"] = fc.values
    out["sarima"] = _forecast_sarima(_fit_sarima(train), len(dates))
    out["prophet"] = _forecast_prophet(_fit_prophet(train), dates)
    return out


def run_refresh_study(year: int = 2026) -> pd.DataFrame:
    """Compare frozen vs monthly-refreshed accuracy for each month of `year`."""
    df, _ = clean()
    series = df["passengers"].astype(float)
    frozen = pd.read_csv(OUT_DIR / f"forecasts_{year}.csv",
                         parse_dates=["date"]).set_index("date")

    rows = []
    for month in range(1, 13):
        m_start = pd.Timestamp(f"{year}-{month:02d}-01")
        m_end = m_start + pd.offsets.MonthEnd(0)
        month_actuals = series.loc[m_start:m_end]
        if month_actuals.dropna().empty:
            break                              # no scored months beyond data
        dates = pd.date_range(m_start, m_end, freq="D")
        partial = month_actuals.dropna().index.max() < m_end

        if month == 1:
            refreshed = {m: frozen.loc[dates, m].values for m in MODELS}
        else:
            cutoff = m_start - pd.Timedelta(days=1)
            train = series.loc[TRAIN_START:cutoff].dropna()
            print(f"  refit through {cutoff.date()} -> forecast "
                  f"{m_start:%Y-%m}{' (partial actuals)' if partial else ''}")
            refreshed = _forecast_month(train, dates)

        actual = series.reindex(dates)
        for m in MODELS:
            frozen_fc = frozen.loc[dates, m]
            refreshed_fc = pd.Series(refreshed[m], index=dates)
            rows.append({
                "month": f"{year}-{month:02d}",
                "model": m,
                "n_days": int(actual.notna().sum()),
                "partial": bool(partial),
                "frozen_MAPE": round(_mape(actual, frozen_fc), 2),
                "refreshed_MAPE": round(_mape(actual, refreshed_fc), 2),
                "frozen_bias": round(_bias(actual, frozen_fc), 2),
                "refreshed_bias": round(_bias(actual, refreshed_fc), 2),
            })

    study = pd.DataFrame(rows)
    study.to_csv(OUT_DIR / f"refresh_study_{year}.csv", index=False)
    _plot(study, year)

    summary = {}
    scored = study[study["month"] != f"{year}-01"]   # months with refresh
    for m in MODELS:
        g = scored[scored["model"] == m]
        summary[m] = {
            "months_compared": int(len(g)),
            "frozen_MAPE_mean": round(float(g["frozen_MAPE"].mean()), 2),
            "refreshed_MAPE_mean": round(float(g["refreshed_MAPE"].mean()), 2),
            "months_improved": int((g["refreshed_MAPE"]
                                    < g["frozen_MAPE"]).sum()),
        }
    (OUT_DIR / f"refresh_report_{year}.json").write_text(
        json.dumps({"year": year, "models": summary}, indent=2))
    return study


def _plot(study: pd.DataFrame, year: int) -> Path:
    months = study["month"].unique().tolist()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    x = np.arange(len(months))
    width = 0.38
    for ax, m in zip(axes, MODELS):
        g = study[study["model"] == m].set_index("month").loc[months]
        ax.bar(x - width / 2, g["frozen_MAPE"], width,
               color="#b0b0b0", label="Frozen (issued before Jan 1)")
        ax.bar(x + width / 2, g["refreshed_MAPE"], width,
               color=MODEL_COLORS[m], label="Refreshed monthly")
        for i, (f, r) in enumerate(zip(g["frozen_MAPE"], g["refreshed_MAPE"])):
            ax.text(i - width / 2, f + 0.1, f"{f:.1f}", ha="center", fontsize=7)
            ax.text(i + width / 2, r + 0.1, f"{r:.1f}", ha="center", fontsize=7)
        labels = [mo[-2:] + ("*" if g.loc[mo, "partial"] else "")
                  for mo in months]
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(m)
        ax.set_xlabel(f"Month of {year} (* partial actuals)")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("MAPE (%)")
    fig.suptitle(f"Value of monthly refresh, {year}: frozen annual forecast "
                 f"vs. retrain-as-months-close", y=1.02)
    fig.tight_layout()
    out = OUT_DIR / "10_monthly_refresh.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    study = run_refresh_study(2026)
    print("\nPer-month MAPE, frozen -> refreshed:")
    for m in MODELS:
        g = study[study["model"] == m]
        parts = [f"{r['month'][-2:]}: {r['frozen_MAPE']:.1f}->"
                 f"{r['refreshed_MAPE']:.1f}" for _, r in g.iterrows()]
        print(f"  {m:<8} " + " | ".join(parts))
