"""
src/llm_summary.py
------------------
Optional plain-language executive summary built from the real metrics.

Provider via env var LLM_PROVIDER in {anthropic, openai, gemini} and the
corresponding API key. Without a key it writes a deterministic template
summary so the pipeline always produces executive_summary.md.

We hand the LLM only a tightly-structured fact sheet derived from the
real metrics, never raw data or code. That prevents number hallucination.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

OUT_DIR = Path("outputs")
MODELS = ("manual", "sarima", "prophet")


def build_fact_sheet() -> str:
    metrics = json.loads((OUT_DIR / "metrics.json").read_text())
    lines = ["FACT SHEET (all numbers computed on real TSA actuals):", ""]
    for year_str in sorted(metrics):
        s = metrics[year_str]
        partial = " (partial-year)" if s["n_scored"] < 300 else ""
        lines.append(f"=== {year_str} hold-out, n={s['n_scored']}{partial} ===")
        for m in MODELS:
            r = s["overall"][m]
            lines.append(f"  {m.upper():<8} MAPE {r['MAPE_pct']:.2f}% | "
                         f"wMAPE {r['wMAPE_pct']:.2f}% | "
                         f"bias {r['Bias_pct']:+.2f}%")
        if s["peak"]:
            lines.append(f"  Peak days (top 10%, n={s['peak_meta']['n']}):")
            for m in MODELS:
                r = s["peak"][m]
                lines.append(f"    {m.upper():<8} MAPE {r['MAPE_pct']:.2f}% | "
                             f"bias {r['Bias_pct']:+.2f}%")
        lines.append("")

    # Sample anomalies
    for path in sorted(OUT_DIR.glob("anomalies_*.csv")):
        year = path.stem.split("_")[1]
        df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
        if df.empty:
            continue
        lines.append(f"Top anomalies, {year}:")
        for d, row in df.head(5).iterrows():
            lines.append(f"  {d:%Y-%m-%d}: actual {int(row['actual']):,}, "
                         f"forecast {int(row['prophet']):,} "
                         f"({row['residual_pct']:+.1f}%) - {row['direction']}")
        lines.append("")

    return "\n".join(lines)


PROMPT = """\
You are a senior workforce-management analyst writing a one-page exec summary
for travel-industry leadership. Use ONLY the facts in the brief below; do not
invent numbers. ~300 words. Structure:

1. Headline (one sentence on best model + 2025/2026 accuracy).
2. Why three models, and what we learned from comparing them.
3. Where forecasts were reliable and where they broke (cite specific dates).
4. Two concrete WFM recommendations from the anomaly pattern.

Brief:
{facts}
"""


def _call_anthropic(p: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    msg = client.messages.create(
        model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
        max_tokens=800,
        messages=[{"role": "user", "content": p}],
    )
    return msg.content[0].text


def _call_openai(p: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    r = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[{"role": "user", "content": p}],
        max_tokens=800,
    )
    return r.choices[0].message.content


def _call_gemini(p: str) -> str:
    import google.generativeai as genai
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    return genai.GenerativeModel(
        os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
    ).generate_content(p).text


PROVIDERS = {"anthropic": _call_anthropic, "openai": _call_openai,
             "gemini": _call_gemini}


def run_summary() -> Path:
    facts = build_fact_sheet()
    provider = os.environ.get("LLM_PROVIDER", "").lower().strip()
    if provider in PROVIDERS:
        try:
            text = PROVIDERS[provider](PROMPT.format(facts=facts))
            body = (f"# Executive Summary (via {provider})\n\n{text}\n\n"
                    f"---\n\n## Source facts\n\n```\n{facts}\n```\n")
        except Exception as exc:
            body = (f"# Executive Summary (template - {provider} failed: {exc})\n\n"
                    f"```\n{facts}\n```\n")
    else:
        body = ("# Executive Summary (template; set LLM_PROVIDER for natural language)\n\n"
                f"```\n{facts}\n```\n")
    out = OUT_DIR / "executive_summary.md"
    out.write_text(body)
    return out


if __name__ == "__main__":
    out = run_summary()
    print(f"Wrote {out}")
    print(out.read_text()[:1500])
