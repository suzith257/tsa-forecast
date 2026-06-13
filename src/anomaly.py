"""
src/anomaly.py
--------------
Flag days where actual TSA throughput diverged materially from the
Prophet forecast. Runs on every fold we have actuals for.

Method:
  residual = actual - forecast
  robust_z = (residual - median) / (1.4826 * MAD)
  Flag if |robust_z| >= 3.0  OR  |residual / actual| >= 15%

This mimics a production WFM monitor: each morning, compare yesterday's
actual to what we predicted, and surface days that broke the pattern.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = Path("outputs")
Z_THRESHOLD = 3.0
PCT_THRESHOLD = 0.15


def _robust_z(x: pd.Series) -> pd.Series:
    med = x.median()
    mad = (x - med).abs().median()
    if mad == 0:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - med) / (1.4826 * mad)


def detect(df: pd.DataFrame, model: str = "prophet") -> pd.DataFrame:
    """Return the days flagged as anomalous against `model`'s forecast."""
    scored = df.dropna(subset=["actual"]).copy()
    scored["residual"] = scored["actual"] - scored[model]
    scored["residual_pct"] = scored["residual"] / scored["actual"] * 100
    scored["robust_z"] = _robust_z(scored["residual"])
    flagged = scored[
        (scored["robust_z"].abs() >= Z_THRESHOLD)
        | (scored["residual_pct"].abs() >= PCT_THRESHOLD * 100)
    ].copy()
    flagged["direction"] = np.where(
        flagged["residual"] > 0,
        "actual > forecast (surge)",
        "actual < forecast (shortfall)",
    )
    return flagged.sort_values("robust_z", key=lambda s: s.abs(),
                                ascending=False)


def plot(df: pd.DataFrame, flagged: pd.DataFrame, year: int) -> Path:
    fig, ax = plt.subplots(figsize=(13, 5))
    scored = df.dropna(subset=["actual"])
    ax.plot(scored.index, scored["actual"] / 1e6,
            color="black", lw=0.9, label="Actual")
    ax.plot(scored.index, scored["prophet"] / 1e6,
            color="#1f4e79", lw=0.7, alpha=0.7, label="Prophet forecast")
    surges = flagged[flagged["residual"] > 0]
    short = flagged[flagged["residual"] < 0]
    ax.scatter(surges.index, surges["actual"] / 1e6,
               color="#1f4e79", s=40, zorder=5, label="Surge")
    ax.scatter(short.index, short["actual"] / 1e6,
               color="#c0504d", s=40, zorder=5, label="Shortfall")
    ax.set_title(f"{year} anomalies vs. Prophet forecast")
    ax.set_ylabel("Passengers (millions)")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = OUT_DIR / f"09_anomalies_{year}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def run_anomaly() -> dict[int, pd.DataFrame]:
    out = {}
    for path in sorted(OUT_DIR.glob("forecasts_*.csv")):
        year = int(path.stem.split("_")[1])
        df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
        flagged = detect(df, "prophet")
        if flagged.empty:
            print(f"{year}: no anomalies flagged.")
        else:
            (flagged[["actual", "prophet", "residual", "residual_pct",
                      "robust_z", "direction"]]
             .to_csv(OUT_DIR / f"anomalies_{year}.csv"))
            plot(df, flagged, year)
        out[year] = flagged
    return out


if __name__ == "__main__":
    out = run_anomaly()
    for year, flagged in out.items():
        print(f"\n{year}: {len(flagged)} anomalous day(s).")
        if not flagged.empty:
            cols = ["actual", "prophet", "residual_pct", "robust_z", "direction"]
            preview = flagged[cols].head(10).copy()
            preview["actual"] = preview["actual"].astype(int)
            preview["prophet"] = preview["prophet"].astype(int)
            preview["residual_pct"] = preview["residual_pct"].round(1)
            preview["robust_z"] = preview["robust_z"].round(1)
            print(preview.to_string())
