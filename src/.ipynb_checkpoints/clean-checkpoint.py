"""
src/clean.py
------------
Light cleaning and a data-quality report on the ingested TSA series.

Design notes:
  - The raw TSA per-year tables already arrive as one row per calendar day
    with integer passenger counts and no comma artifacts after parsing, so
    we don't 'clean' values - we VERIFY them and surface anything unusual.
  - We force a complete daily index across the observed range so any future
    gap (e.g. a day TSA failed to post) shows up immediately as a NaN.
  - We compute a robust seasonal residual (passengers vs. its own
    same-day-of-week 28-day median) and emit a QA report rather than
    silently editing the series. The forecasting models see the raw data.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED_DIR = Path("data/processed")


def load_daily(path: Path = PROCESSED_DIR / "tsa_daily.csv") -> pd.DataFrame:
    """Load ingested daily series with a DatetimeIndex named 'date'."""
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
    df = df.set_index("date")
    return df


def reindex_to_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex to a continuous daily DatetimeIndex; missing days become NaN."""
    full = pd.date_range(df.index.min(), df.index.max(), freq="D", name="date")
    return df.reindex(full)


def quality_report(df: pd.DataFrame) -> dict:
    """Compute a data-quality summary; pure function, prints nothing."""
    # 1. Missing days after calendar reindex.
    missing = df.index[df["passengers"].isna()].tolist()

    # 2. Duplicate dates (should be impossible after ingest but verify).
    dups = df.index[df.index.duplicated()].tolist()

    # 3. Implausible values - non-positive, or so far from neighbours that
    #    they could be data-entry errors. Use a same-DOW rolling median
    #    (28 days = 4 same-weekday observations) for a robust baseline,
    #    flag |dev| > 40%.
    s = df["passengers"].astype("float")
    dow = s.index.dayofweek
    baselines = pd.Series(index=s.index, dtype="float")
    for d in range(7):
        mask = dow == d
        baselines.loc[mask] = (
            s[mask].rolling(window=4, min_periods=2).median()
        )
    rel_dev = (s - baselines) / baselines
    suspicious = rel_dev[rel_dev.abs() > 0.40].dropna()

    return {
        "rows": int(df.shape[0]),
        "date_min": str(df.index.min().date()),
        "date_max": str(df.index.max().date()),
        "missing_days": [str(d.date()) for d in missing],
        "duplicate_days": [str(d.date()) for d in dups],
        "non_positive_days": [
            str(d.date()) for d in df.index[(df["passengers"] <= 0).fillna(False)]
        ],
        # Note: 'suspicious' days are INFORMATIONAL - 2020 COVID collapse will
        # dominate this list, which is exactly correct. We don't drop anything.
        "suspicious_days_top": (
            suspicious.abs().sort_values(ascending=False)
            .head(15)
            .index.strftime("%Y-%m-%d")
            .tolist()
        ),
    }


def clean(write_report: bool = True) -> tuple[pd.DataFrame, dict]:
    """Full cleaning pipeline; returns the calendar-complete frame and QA dict."""
    df = load_daily()
    df = reindex_to_calendar(df)
    report = quality_report(df)
    if write_report:
        (PROCESSED_DIR / "qa_report.json").write_text(json.dumps(report, indent=2))
    return df, report


if __name__ == "__main__":
    df, report = clean()
    print(f"Rows:       {report['rows']:,}")
    print(f"Range:      {report['date_min']} -> {report['date_max']}")
    print(f"Missing:    {len(report['missing_days'])}")
    print(f"Duplicates: {len(report['duplicate_days'])}")
    print(f"Non-positive: {len(report['non_positive_days'])}")
    print("\nTop 'unusual vs. same-weekday baseline' days (informational, not removed):")
    for d in report["suspicious_days_top"]:
        v = df.loc[d, "passengers"]
        print(f"  {d}: {int(v):,}")
