"""
src/wfm_scenario.py
-------------------
OPTIONAL illustrative module: translate a passenger forecast into a
notional customer-support contact volume and required headcount.

IMPORTANT - READ THIS BEFORE USING:
  Every parameter below is an ILLUSTRATIVE assumption, not an empirical
  figure from any travel-tech company. They are user-set, transparent,
  and clearly labelled here, in the output, and in the README. This module
  exists to demonstrate the workflow a WFM analyst would build on top of
  a passenger forecast - NOT to claim any specific staffing number.

  The passenger forecast itself uses ONLY real public TSA data; nothing
  in this module changes that. If you don't want the illustrative layer,
  simply don't import this module - the rest of the pipeline runs without it.

The conversion uses the standard Erlang-derived workload arithmetic:
    contacts          = passengers * contact_rate
    raw_agent_hours   = contacts * AHT_seconds / 3600
    productive_hours  = raw_agent_hours / occupancy
    scheduled_hours   = productive_hours / (1 - shrinkage)
    FTE_required      = scheduled_hours / hours_per_FTE_per_day
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

OUT_DIR = Path("outputs")


@dataclass
class WFMAssumptions:
    """All ILLUSTRATIVE - tweak freely. None of these are empirical truth."""
    contact_rate: float = 0.005       # 0.5% of passengers contact support
    aht_seconds: int = 360            # 6-min average handle time
    occupancy_target: float = 0.85    # agents productive 85% of logged time
    shrinkage: float = 0.30           # 30% break/training/PTO etc
    hours_per_fte_per_day: float = 8.0


def passengers_to_fte(passengers: pd.Series,
                      assumptions: WFMAssumptions) -> pd.DataFrame:
    """Compute the daily required-FTE series from a daily passenger forecast."""
    a = assumptions
    contacts = passengers * a.contact_rate
    raw_hours = contacts * a.aht_seconds / 3600
    productive_hours = raw_hours / a.occupancy_target
    scheduled_hours = productive_hours / (1.0 - a.shrinkage)
    fte = scheduled_hours / a.hours_per_fte_per_day
    return pd.DataFrame({
        "passengers_forecast": passengers,
        "contacts_forecast": contacts.round().astype(int),
        "scheduled_hours": scheduled_hours.round(1),
        "fte_required": fte.round(2),
    })


def run_wfm_scenario() -> pd.DataFrame:
    """Run the scenario on the Prophet forecast for 2025; write a CSV."""
    fc = pd.read_csv(OUT_DIR / "forecasts_2025.csv",
                     parse_dates=["date"]).set_index("date")
    assumptions = WFMAssumptions()
    out = passengers_to_fte(fc["prophet"], assumptions)
    out.attrs["assumptions"] = assumptions.__dict__
    out.to_csv(OUT_DIR / "wfm_scenario_2025.csv")
    return out


if __name__ == "__main__":
    df = run_wfm_scenario()
    print("\n===== ILLUSTRATIVE WFM SCENARIO =====")
    print("Assumptions (all USER-SET, not empirical):")
    a = WFMAssumptions()
    for k, v in a.__dict__.items():
        print(f"  {k}: {v}")
    print("\nDaily FTE requirement summary (2025 forecast):")
    print(df["fte_required"].describe().round(1))
    print("\nTop 5 staffing-demand days (Sun-after-Thanksgiving etc.):")
    print(df.sort_values("fte_required", ascending=False).head(5).round(1))
