"""
src/registry.py
---------------
Append-only run log: every pipeline execution records what data it saw,
how the models were configured, and every metric produced.

Why this matters at scale:
  In production you will re-run this weekly/monthly. Six months later
  someone asks "did accuracy degrade after the March refit?" - the answer
  must come from a log, not from memory. This registry is the audit trail:
  one JSON entry per run in outputs/run_log.json.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
OUT_DIR = Path("outputs")
LOG_PATH = OUT_DIR / "run_log.json"


def record_run(extra: dict | None = None) -> dict:
    """Collect everything the pipeline produced into one log entry."""
    entry: dict = {
        "run_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Data provenance + QA
    src = PROCESSED_DIR / "SOURCE.txt"
    if src.exists():
        entry["source_note"] = src.read_text().strip().splitlines()
    qa = PROCESSED_DIR / "qa_report.json"
    if qa.exists():
        q = json.loads(qa.read_text())
        entry["data"] = {
            "rows": q["rows"], "date_min": q["date_min"],
            "date_max": q["date_max"],
            "missing_days": len(q["missing_days"]),
            "duplicate_days": len(q["duplicate_days"]),
        }

    # Model configuration (kept in sync with src/models.py & manual_model.py)
    entry["config"] = {
        "train_start": "2022-01-01",
        "folds": {"2025": "train->2024-12-31", "2026": "train->2025-12-31"},
        "models": {
            "manual": "multiplicative decomposition: 28d-median trend, "
                      "DOW/month geo-mean factors, holiday + pooled "
                      "bridge factors, 730d damped log-linear trend (0.5)",
            "sarima": "SARIMA(1,1,1)(1,1,1,7) on log(passengers)",
            "prophet": "multiplicative, yearly+weekly, US federal holidays "
                       "+-1d, Sunday-after-Thanksgiving event",
        },
    }

    # Accuracy metrics per fold
    metrics = OUT_DIR / "metrics.json"
    if metrics.exists():
        entry["metrics"] = json.loads(metrics.read_text())

    # Monthly refresh study
    refresh = {}
    for path in sorted(OUT_DIR.glob("refresh_report_*.json")):
        year = path.stem.split("_")[-1]
        refresh[year] = json.loads(path.read_text())
    if refresh:
        entry["refresh_study"] = refresh

    # Anomaly counts
    anomalies = {}
    for path in sorted(OUT_DIR.glob("anomalies_*.csv")):
        year = path.stem.split("_")[-1]
        anomalies[year] = sum(1 for _ in open(path)) - 1
    if anomalies:
        entry["anomaly_days_flagged"] = anomalies

    if extra:
        entry.update(extra)

    # Append to the log
    log = json.loads(LOG_PATH.read_text()) if LOG_PATH.exists() else []
    log.append(entry)
    LOG_PATH.write_text(json.dumps(log, indent=2))
    return entry


if __name__ == "__main__":
    e = record_run()
    print(f"Recorded run at {e['run_at_utc']} -> {LOG_PATH}")
    print(f"Log now contains {len(json.loads(LOG_PATH.read_text()))} run(s).")
