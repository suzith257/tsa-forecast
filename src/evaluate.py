"""
src/evaluate.py
---------------
Score the walk-forward forecasts honestly.

For each fold (2025 full year, 2026 year-to-date) we report:
  - MAPE, wMAPE, Bias, RMSE
  - Same metrics restricted to the busiest-decile days (peak accuracy)
  - Monthly MAPE breakdown
  - Forecast-vs-actual and residual diagnostic plots

The 2026 fold is scored only on days where actuals exist (currently
Jan 1 - Jun 10, 2026 = 161 days). The score is explicitly labelled
'partial' because it excludes Q4, where holiday effects dominate.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = Path("outputs")
MODELS = ("manual", "sarima", "prophet")
MODEL_COLORS = {"manual": "#2e8b57", "sarima": "#c0504d", "prophet": "#1f4e79"}


def metrics(actual: pd.Series, forecast: pd.Series) -> dict:
    mask = actual.notna() & forecast.notna()
    a, f = actual[mask], forecast[mask]
    if len(a) == 0:
        return {"MAPE_pct": np.nan, "wMAPE_pct": np.nan,
                "Bias_pct": np.nan, "RMSE": np.nan, "n": 0}
    err = f - a
    return {
        "MAPE_pct": float((err.abs() / a).mean() * 100),
        "wMAPE_pct": float(err.abs().sum() / a.sum() * 100),
        "Bias_pct": float(err.mean() / a.mean() * 100),
        "RMSE": float(np.sqrt((err ** 2).mean())),
        "n": int(len(a)),
    }


def score_fold(df: pd.DataFrame) -> dict:
    scored = df.dropna(subset=["actual"]).copy()
    overall = {m: metrics(scored["actual"], scored[m]) for m in MODELS}

    if len(scored) >= 50:
        thr = scored["actual"].quantile(0.90)
        peaks = scored[scored["actual"] >= thr]
        peak = {m: metrics(peaks["actual"], peaks[m]) for m in MODELS}
        peak_meta = {"threshold": int(thr), "n": int(len(peaks))}
    else:
        peak, peak_meta = {}, {}

    monthly = {}
    for m in MODELS:
        rows = []
        for month, g in scored.groupby(scored.index.to_period("M")):
            rows.append({"month": str(month), **metrics(g["actual"], g[m])})
        monthly[m] = pd.DataFrame(rows)
    return {"overall": overall, "peak": peak, "peak_meta": peak_meta,
            "monthly": monthly, "n_scored": int(len(scored))}


# ---------- plots ----------
def plot_forecast_vs_actual(df: pd.DataFrame, year: int) -> Path:
    fig, ax = plt.subplots(figsize=(13, 5.5))
    scored = df.dropna(subset=["actual"])
    ax.plot(scored.index, scored["actual"] / 1e6,
            color="black", lw=1.0, label="Actual", zorder=4)
    for m in MODELS:
        ax.plot(df.index, df[m] / 1e6, color=MODEL_COLORS[m],
                lw=0.9, alpha=0.85, label=m)
    ax.set_title(f"Forecast vs. actual, {year} "
                 f"(scored on {len(scored)} actual day(s))")
    ax.set_ylabel("Passengers (millions)")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = OUT_DIR / f"06_forecast_vs_actual_{year}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_monthly_mape(scores: dict, year: int) -> Path:
    months = scores["monthly"]["manual"]["month"].tolist()
    if not months:
        return None
    x = np.arange(len(months))
    width = 0.27
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for i, m in enumerate(MODELS):
        ax.bar(x + (i - 1) * width,
               scores["monthly"][m]["MAPE_pct"],
               width, color=MODEL_COLORS[m], label=m)
    ax.set_xticks(x)
    ax.set_xticklabels([mo[-2:] for mo in months])
    ax.set_xlabel(f"Month of {year}")
    ax.set_ylabel("MAPE (%)")
    ax.set_title(f"Monthly MAPE on {year}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = OUT_DIR / f"07_monthly_mape_{year}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_walk_forward_summary(scores_by_year: dict[int, dict]) -> Path:
    """Side-by-side overall MAPE/wMAPE/Bias for both folds."""
    years = sorted(scores_by_year.keys())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    metric_names = ["MAPE_pct", "wMAPE_pct", "Bias_pct"]
    titles = ["MAPE (%)", "wMAPE (%)", "Bias (%)"]
    width = 0.27
    x = np.arange(len(years))
    for ax, mname, t in zip(axes, metric_names, titles):
        for i, m in enumerate(MODELS):
            vals = [scores_by_year[y]["overall"][m][mname] for y in years]
            bars = ax.bar(x + (i - 1) * width, vals, width,
                          color=MODEL_COLORS[m], label=m)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2,
                        v + (0.4 if v >= 0 else -0.4),
                        f"{v:+.1f}" if mname == "Bias_pct" else f"{v:.1f}",
                        ha="center",
                        va="bottom" if v >= 0 else "top", fontsize=8)
        ax.set_xticks(x)
        labels = []
        for y in years:
            n = scores_by_year[y]["n_scored"]
            labels.append(f"{y}\n(n={n})")
        ax.set_xticklabels(labels)
        ax.set_title(t)
        ax.grid(axis="y", alpha=0.3)
        if mname == "Bias_pct":
            ax.axhline(0, color="black", lw=0.6)
    axes[0].legend(loc="upper left")
    fig.suptitle("Walk-forward accuracy comparison", y=1.02)
    fig.tight_layout()
    out = OUT_DIR / "08_walkforward_summary.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def run_evaluation() -> dict[int, dict]:
    """Score every forecast file in outputs/forecasts_<year>.csv."""
    scores_by_year = {}
    for path in sorted(OUT_DIR.glob("forecasts_*.csv")):
        year = int(path.stem.split("_")[1])
        df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
        s = score_fold(df)
        scores_by_year[year] = s
        plot_forecast_vs_actual(df, year)
        plot_monthly_mape(s, year)
        for m, frame in s["monthly"].items():
            frame.to_csv(OUT_DIR / f"monthly_metrics_{year}_{m}.csv",
                         index=False)
    plot_walk_forward_summary(scores_by_year)

    # JSON-safe dump
    dump = {}
    for y, s in scores_by_year.items():
        dump[str(y)] = {
            "n_scored": s["n_scored"],
            "overall": s["overall"],
            "peak": s["peak"], "peak_meta": s["peak_meta"],
        }
    (OUT_DIR / "metrics.json").write_text(json.dumps(dump, indent=2))
    return scores_by_year


def print_scoreboard(scores_by_year: dict[int, dict]) -> None:
    for year in sorted(scores_by_year):
        s = scores_by_year[year]
        partial = " (PARTIAL - scored on days with actuals)" if s["n_scored"] < 300 else ""
        print(f"\n========== {year} HOLD-OUT (n={s['n_scored']}){partial} ==========")
        print(f"{'':<10} {'MAPE':>8} {'wMAPE':>8} {'Bias':>8} {'RMSE':>12}")
        print("-" * 50)
        for m in MODELS:
            r = s["overall"][m]
            print(f"{m.upper():<10} {r['MAPE_pct']:>7.2f}% "
                  f"{r['wMAPE_pct']:>7.2f}% {r['Bias_pct']:>+7.2f}% "
                  f"{r['RMSE']:>12,.0f}")
        if s["peak"]:
            print(f"Peak days only (top 10%, >= "
                  f"{s['peak_meta']['threshold']:,} pax, "
                  f"n={s['peak_meta']['n']}):")
            for m in MODELS:
                r = s["peak"][m]
                print(f"{m.upper():<10} {r['MAPE_pct']:>7.2f}% "
                      f"{r['wMAPE_pct']:>7.2f}% {r['Bias_pct']:>+7.2f}%")


if __name__ == "__main__":
    s = run_evaluation()
    print_scoreboard(s)
