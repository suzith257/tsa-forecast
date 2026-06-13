"""
src/eda.py
----------
Exploratory analysis on the TSA daily series.

Outputs (PNG, into outputs/):
  - 01_full_series.png            Full 2019-now series with COVID and holidays annotated
  - 02_dow_seasonality.png        Day-of-week pattern (post-recovery years)
  - 03_annual_seasonality.png     Day-of-year overlay by year
  - 04_holiday_impact.png         Effect size of major US holidays vs. surrounding baseline

We intentionally compute DOW and holiday effects on POST-COVID years only
(2022 onwards), because pandemic-era data distorts the structural patterns
this section is meant to illustrate. The full series plot still shows
2019-onwards so the reader sees the structural break.
"""
from __future__ import annotations

from pathlib import Path

import holidays
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.clean import clean

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# US federal holidays we annotate. We use the `holidays` library so the
# dates are authoritative (e.g. observed-day rules, Juneteenth from 2021).
MAJOR_HOLIDAY_NAMES = {
    "New Year's Day",
    "Memorial Day",
    "Juneteenth National Independence Day",
    "Independence Day",
    "Labor Day",
    "Thanksgiving Day",
    "Christmas Day",
}


def federal_holiday_frame(years: range) -> pd.DataFrame:
    """All US federal holidays in `years` as a tidy DataFrame [date, name]."""
    us = holidays.UnitedStates(years=list(years))
    rows = [(pd.Timestamp(d), n) for d, n in sorted(us.items())]
    return pd.DataFrame(rows, columns=["date", "name"])


def plot_full_series(df: pd.DataFrame) -> Path:
    """Full daily series with COVID period shaded and major holidays marked."""
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(df.index, df["passengers"] / 1e6, lw=0.6, color="#1f4e79")

    # COVID structural-break shading (purely visual context, not a data edit).
    ax.axvspan(
        pd.Timestamp("2020-03-15"), pd.Timestamp("2021-12-31"),
        color="grey", alpha=0.10, label="COVID period (excluded from training)",
    )

    # Mark Thanksgiving Wednesday + Sunday across years (the biggest spikes).
    years = range(df.index.year.min(), df.index.year.max() + 1)
    hol = federal_holiday_frame(years)
    thanksgiving = hol[hol["name"] == "Thanksgiving"]
    for d in thanksgiving["date"]:
        sunday_after = d + pd.Timedelta(days=3)
        if sunday_after in df.index:
            ax.axvline(sunday_after, color="#c0504d", alpha=0.25, lw=0.7)

    ax.set_title(
        "TSA Daily Checkpoint Passenger Throughput, Jan 2019 - present\n"
        "Vertical lines mark Sunday after Thanksgiving (peak rebooking day)",
        fontsize=12,
    )
    ax.set_ylabel("Passengers (millions)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = OUT_DIR / "01_full_series.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_dow_seasonality(df: pd.DataFrame) -> Path:
    """Average passengers by day-of-week, post-COVID years only."""
    post = df.loc["2022":"2024"].copy()
    post["dow"] = post.index.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
             "Saturday", "Sunday"]
    means = post.groupby("dow")["passengers"].mean().reindex(order) / 1e6

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(order, means.values, color="#1f4e79")
    ax.set_title("Average daily throughput by day of week (2022-2024)")
    ax.set_ylabel("Passengers (millions)")
    overall = means.mean()
    ax.axhline(overall, color="grey", linestyle="--", lw=0.8,
               label=f"7-day mean = {overall:.2f}M")
    for b, v in zip(bars, means.values):
        pct = (v / overall - 1) * 100
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02,
                f"{pct:+.0f}%", ha="center", fontsize=9)
    ax.legend(loc="lower center")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = OUT_DIR / "02_dow_seasonality.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_annual_seasonality(df: pd.DataFrame) -> Path:
    """Year-over-year overlay on day-of-year, post-COVID years."""
    fig, ax = plt.subplots(figsize=(13, 5))
    for year, color in zip([2022, 2023, 2024, 2025], ["#9bbcd8", "#5b8db7",
                                                      "#1f4e79", "#c0504d"]):
        s = df.loc[str(year), "passengers"]
        if s.empty:
            continue
        doy = s.index.dayofyear
        ax.plot(doy, s.values / 1e6, lw=0.9, color=color, label=str(year),
                alpha=0.85)

    # Holiday markers along the x-axis (use 2024 as the reference DOY mapping).
    ref_hol = federal_holiday_frame(range(2024, 2025))
    ref_hol = ref_hol[ref_hol["name"].isin(MAJOR_HOLIDAY_NAMES)]
    for _, row in ref_hol.iterrows():
        doy = row["date"].dayofyear
        ax.axvline(doy, color="grey", alpha=0.25, lw=0.6)
        ax.text(doy, ax.get_ylim()[1] * 0.98, row["name"].split()[0],
                rotation=90, va="top", fontsize=7, alpha=0.7)

    ax.set_title("Annual seasonality: TSA throughput by day-of-year (2022-2025)")
    ax.set_xlabel("Day of year")
    ax.set_ylabel("Passengers (millions)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = OUT_DIR / "03_annual_seasonality.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def holiday_effect_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Quantify each major holiday's effect by comparing throughput on the
    holiday itself (and key adjacent travel days) vs. a 14-day surrounding
    baseline of the same day-of-week.
    Returns a small DataFrame with %-deviation per holiday.
    """
    post = df.loc["2022":"2024", "passengers"].astype(float)
    years = range(2022, 2025)
    hol = federal_holiday_frame(years)
    hol = hol[hol["name"].isin(MAJOR_HOLIDAY_NAMES)].copy()

    # The travel-industry peak is rarely the holiday itself - it's the
    # adjacent travel days. Add Sunday-after-Thanksgiving explicitly because
    # it's consistently the single highest-volume day of the year.
    tg = hol[hol["name"] == "Thanksgiving Day"].copy()
    tg["date"] = tg["date"] + pd.Timedelta(days=3)
    tg["name"] = "Sunday after Thanksgiving"
    hol = pd.concat([hol, tg], ignore_index=True)

    rows = []
    for _, h in hol.iterrows():
        d = h["date"]
        # 14-day window on each side, same day-of-week only
        window = post.loc[d - pd.Timedelta(days=21): d + pd.Timedelta(days=21)]
        same_dow = window[window.index.dayofweek == d.dayofweek]
        same_dow = same_dow.drop(d, errors="ignore")
        if same_dow.empty or d not in post.index:
            continue
        baseline = same_dow.mean()
        pct = (post.loc[d] - baseline) / baseline * 100
        rows.append({"holiday": h["name"], "date": d.date(),
                     "actual": int(post.loc[d]),
                     "baseline": int(baseline),
                     "pct_vs_baseline": round(pct, 1)})
    return pd.DataFrame(rows)


def plot_holiday_impact(df: pd.DataFrame) -> Path:
    tbl = holiday_effect_table(df)
    summary = (tbl.groupby("holiday")["pct_vs_baseline"]
               .mean()
               .sort_values())
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#c0504d" if v < 0 else "#1f4e79" for v in summary.values]
    ax.barh(summary.index, summary.values, color=colors)
    for y, v in enumerate(summary.values):
        ax.text(v + (0.5 if v > 0 else -0.5), y, f"{v:+.1f}%",
                va="center", ha="left" if v > 0 else "right", fontsize=9)
    ax.axvline(0, color="black", lw=0.7)
    ax.set_title("Holiday effect on day-of vs. same-weekday baseline (2022-2024 avg)")
    ax.set_xlabel("% deviation from baseline")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out = OUT_DIR / "04_holiday_impact.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_manual_factors() -> Path:
    """Visualise the manual model's fitted DOW, monthly, holiday, bridge factors."""
    from src.manual_model import fit
    df, _ = clean()
    train = df.loc["2022-01-01":"2024-12-31", "passengers"].astype(float)
    m = fit(train)
    wd_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # (a) DOW factors
    ax = axes[0, 0]
    dow_vals = [m.dow_factors[w] for w in range(7)]
    colors = ["#1f4e79" if v >= 1 else "#c0504d" for v in dow_vals]
    ax.bar(wd_names, dow_vals, color=colors)
    ax.axhline(1.0, color="grey", lw=0.8, ls="--")
    for i, v in enumerate(dow_vals):
        ax.text(i, v + 0.005, f"{(v-1)*100:+.1f}%", ha="center", fontsize=9)
    ax.set_title("Weekly factors (multiplicative, geometric mean = 1)")
    ax.set_ylim(0.8, 1.15)
    ax.grid(axis="y", alpha=0.3)

    # (b) Annual seasonality: smooth fitted curve with monthly means
    from src.manual_model import annual_curve, month_factors
    ax = axes[0, 1]
    curve = annual_curve(m)
    ax.plot(curve.index, curve.values, color="#1f4e79", lw=1.6,
            label="Fitted annual curve (Fourier, K=6)")
    mf = month_factors(m)
    month_mid_doy = [15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349]
    ax.scatter(month_mid_doy, [mf[mo] for mo in range(1, 13)],
               color="#c0504d", zorder=5, s=28, label="Monthly geo-mean")
    for doy, mo in zip(month_mid_doy, range(1, 13)):
        ax.text(doy, mf[mo] + 0.012, f"{(mf[mo]-1)*100:+.0f}%",
                ha="center", fontsize=8)
    ax.axhline(1.0, color="grey", lw=0.8, ls="--")
    ax.set_xticks(month_mid_doy)
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_title("Annual seasonality (smooth curve; monthly means as markers)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)

    # (c) Holiday factors
    ax = axes[1, 0]
    hol_items = sorted(m.holiday_factors.items(), key=lambda kv: kv[1])
    names = [k.replace(" National Independence Day", "") for k, _ in hol_items]
    vals = [v for _, v in hol_items]
    colors = ["#1f4e79" if v >= 1 else "#c0504d" for v in vals]
    ax.barh(names, vals, color=colors)
    ax.axvline(1.0, color="grey", lw=0.8, ls="--")
    for i, v in enumerate(vals):
        ax.text(v, i, f" {(v-1)*100:+.1f}%", va="center", fontsize=9)
    ax.set_title("Holiday factors (level- and DOW-adjusted)")
    ax.grid(axis="x", alpha=0.3)

    # (d) Bridge factors heatmap (weekday-of-holiday x offset)
    ax = axes[1, 1]
    offsets = sorted({off for (_, off) in m.bridge_factors.keys()})
    weekdays = sorted({wd for (wd, _) in m.bridge_factors.keys()})
    grid = np.full((len(weekdays), len(offsets)), np.nan)
    for (wd, off), v in m.bridge_factors.items():
        i = weekdays.index(wd)
        j = offsets.index(off)
        grid[i, j] = v
    im = ax.imshow(grid, cmap="RdBu_r", vmin=0.85, vmax=1.15, aspect="auto")
    ax.set_xticks(range(len(offsets)))
    ax.set_xticklabels([f"{o:+d}d" for o in offsets])
    ax.set_yticks(range(len(weekdays)))
    ax.set_yticklabels([wd_names[w] for w in weekdays])
    ax.set_xlabel("Offset from holiday")
    ax.set_ylabel("Weekday holiday falls on")
    ax.set_title("Bridge-day multipliers (pooled fixed-date holidays)")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if not np.isnan(grid[i, j]):
                v = grid[i, j]
                txt = f"{(v-1)*100:+.0f}%"
                ax.text(j, i, txt, ha="center", va="center",
                        color="white" if abs(v - 1) > 0.08 else "black",
                        fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.04)

    fig.suptitle("Manual model: fitted factors from 2022-2024 training data",
                 y=1.00)
    fig.tight_layout()
    out = OUT_DIR / "05_manual_factors.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def run_eda() -> dict:
    df, _ = clean()
    paths = {
        "full_series": plot_full_series(df),
        "dow_seasonality": plot_dow_seasonality(df),
        "annual_seasonality": plot_annual_seasonality(df),
        "holiday_impact": plot_holiday_impact(df),
        "manual_factors": plot_manual_factors(),
    }
    tbl = holiday_effect_table(df)
    tbl.to_csv(OUT_DIR / "holiday_effects.csv", index=False)
    return {"figures": {k: str(v) for k, v in paths.items()},
            "holiday_table_rows": len(tbl)}


if __name__ == "__main__":
    info = run_eda()
    for k, v in info["figures"].items():
        print(f"  {k}: {v}")
    print(f"  holiday_effects.csv rows: {info['holiday_table_rows']}")
