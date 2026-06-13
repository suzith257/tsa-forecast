"""
run_pipeline.py
---------------
Single entrypoint that runs the full pipeline end-to-end.

    python run_pipeline.py             # full pipeline (live-first ingest)
    python run_pipeline.py --no-live   # skip live scrape; use cached HTML
    python run_pipeline.py --no-llm    # skip LLM summary

Stages:
  1. Ingest (live first, cached fallback)  -> data/processed/tsa_daily.csv
  2. Clean + QA                            -> data/processed/qa_report.json
  3. EDA + fitted-factors viz              -> outputs/01..04, 10_*.png
  4. Walk-forward training:
       Fold 1 train 2022-2024 -> forecast 2025
       Fold 2 train 2022-2025 -> forecast 2026
     Three models per fold (manual / SARIMA / Prophet)
                                           -> outputs/forecasts_<year>.csv
  5. Evaluate both folds                   -> outputs/metrics.json, 05/07/09
  6. Anomaly detection per fold            -> outputs/anomalies_<year>.csv, 08
  7. Illustrative WFM scenario             -> outputs/wfm_scenario_2025.csv
  8. Executive summary                     -> outputs/executive_summary.md
"""
from __future__ import annotations

import argparse
import sys
import time

from src import (anomaly, clean, eda, evaluate, ingest, llm_summary,
                 models, refresh, registry, wfm_scenario)


def stage(name: str):
    print(f"\n=========== {name} ===========")
    return time.time()


def main(use_live: bool = True, skip_llm: bool = False) -> int:
    t = stage("1/10 INGEST")
    df = ingest.ingest(prefer_live=use_live)
    print(f"  {len(df):,} rows ({time.time()-t:.1f}s)")

    t = stage("2/10 CLEAN + QA")
    _, report = clean.clean()
    print(f"  missing: {len(report['missing_days'])}, "
          f"duplicates: {len(report['duplicate_days'])} "
          f"({time.time()-t:.1f}s)")

    t = stage("3/10 EDA + FITTED FACTORS")
    info = eda.run_eda()
    for k, v in info["figures"].items():
        print(f"  {k}: {v}")
    print(f"  ({time.time()-t:.1f}s)")

    t = stage("4/10 WALK-FORWARD TRAINING (3 models x 2 folds)")
    models.run_walk_forward()
    print(f"  ({time.time()-t:.1f}s)")

    t = stage("5/10 EVALUATE")
    scores = evaluate.run_evaluation()
    evaluate.print_scoreboard(scores)
    print(f"  ({time.time()-t:.1f}s)")

    t = stage("6/10 ANOMALY DETECTION (per fold)")
    a = anomaly.run_anomaly()
    for year, flagged in a.items():
        print(f"  {year}: {len(flagged)} anomalous day(s)")
    print(f"  ({time.time()-t:.1f}s)")

    t = stage("7/10 MONTHLY REFRESH STUDY (2026)")
    refresh.run_refresh_study(2026)
    print(f"  ({time.time()-t:.1f}s)")

    t = stage("8/10 WFM SCENARIO (illustrative, 2025)")
    wfm_scenario.run_wfm_scenario()
    print(f"  ({time.time()-t:.1f}s)")

    if skip_llm:
        print("\nSkipping LLM summary (--no-llm).")
    else:
        t = stage("9/10 EXECUTIVE SUMMARY")
        out = llm_summary.run_summary()
        print(f"  {out} ({time.time()-t:.1f}s)")

    t = stage("10/10 RECORD RUN IN REGISTRY")
    entry = registry.record_run()
    print(f"  logged run {entry['run_at_utc']} -> outputs/run_log.json "
          f"({time.time()-t:.1f}s)")

    print("\nDone. See outputs/ for results.")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--no-live", action="store_true",
                   help="Skip live scrape; use cached HTML only")
    p.add_argument("--no-llm", action="store_true",
                   help="Skip executive-summary LLM call")
    args = p.parse_args()
    sys.exit(main(use_live=not args.no_live, skip_llm=args.no_llm))
