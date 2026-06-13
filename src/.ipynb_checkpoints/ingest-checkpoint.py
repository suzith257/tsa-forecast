"""
src/ingest.py
-------------
Ingest TSA Daily Checkpoint Passenger Throughput.

Live-first ingestion strategy:
  1. Scrape each year page from tsa.gov directly into pandas via
     requests + pd.read_html. No HTML stored on disk by default.
  2. If a year fails to fetch (network down, IP blocked, table moved),
     fall back to a local cached copy under data/raw/tsa_<YEAR>.html.
  3. Optionally accept a tab-separated partial-year file for the current
     year if you already have it (e.g. pasted from the landing page).

Why hybrid (not pure live, not pure cache):
  - Pure-live breaks the moment tsa.gov changes its layout or blocks the
    caller; a repo must be runnable by anyone, months from now.
  - Pure-cache is reproducible but goes stale and demonstrates nothing
    about working against a live source.

Output:
  data/processed/tsa_daily.csv   columns [date, passengers]
  data/processed/SOURCE.txt      retrieval-time provenance note
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://www.tsa.gov/travel/passenger-volumes"
RAW_DIR = Path("data/raw")          # used only as a fallback cache
PROCESSED_DIR = Path("data/processed")

# A normal desktop-browser User-Agent. tsa.gov blocks data-center IPs
# entirely (cloud sandboxes, CI runners), but accepts requests from home
# / office networks with this header set.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def _parse_html_text(html: str, year: int) -> pd.DataFrame:
    """Parse the single Date | Numbers table from a TSA year page."""
    tables = pd.read_html(StringIO(html))
    if not tables:
        raise ValueError(f"{year}: no HTML tables on page")
    df = tables[0].copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Identify the date and value columns defensively. The landing page
    # may carry weekday-aligned prior-year columns; we only keep the
    # current-year calendar column.
    date_col = next((c for c in df.columns if "date" in c.lower()),
                    df.columns[0])
    value_cols = [c for c in df.columns if c != date_col]
    if len(value_cols) > 1:
        match = [c for c in value_cols if str(year) in c]
        value_col = match[0] if match else value_cols[0]
    else:
        value_col = value_cols[0]

    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    df["passengers"] = pd.to_numeric(
        df[value_col].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )
    df = df.dropna(subset=["date", "passengers"])
    df = df[df["date"].dt.year == year]
    df["passengers"] = df["passengers"].astype(int)
    return df[["date", "passengers"]].sort_values("date").reset_index(drop=True)


def _fetch_live(year: int, current_year: int,
                session: requests.Session) -> pd.DataFrame | None:
    """Fetch a single year page from tsa.gov; return parsed frame or None."""
    # The current year sits at the base URL; archive years at /<year>.
    url = BASE_URL if year == current_year else f"{BASE_URL}/{year}"
    try:
        resp = session.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        return _parse_html_text(resp.text, year)
    except (requests.RequestException, ValueError) as exc:
        print(f"  [WARN] live fetch failed for {year}: {exc}")
        return None


def _fallback_cache(year: int) -> pd.DataFrame | None:
    """Read a locally cached HTML page if one exists; otherwise return None."""
    path = RAW_DIR / f"tsa_{year}.html"
    if not path.exists():
        return None
    print(f"  [INFO] using cached {path}")
    try:
        return _parse_html_text(path.read_text(encoding="utf-8",
                                               errors="ignore"), year)
    except ValueError as exc:
        print(f"  [WARN] cache parse failed for {year}: {exc}")
        return None


def _parse_partial_tsv(path: Path) -> pd.DataFrame:
    """Parse an optional Date<TAB>Numbers TSV (e.g. pasted from the page)."""
    df = pd.read_csv(path, sep="\t")
    df.columns = [c.strip() for c in df.columns]
    df["date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["passengers"] = pd.to_numeric(
        df["Numbers"].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )
    return (df.dropna(subset=["date", "passengers"])
            [["date", "passengers"]]
            .assign(passengers=lambda d: d["passengers"].astype(int))
            .sort_values("date").reset_index(drop=True))


def ingest(years: range | None = None,
           prefer_live: bool = True,
           processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """
    Build the combined daily series.

    Parameters
    ----------
    years : range, optional
        Which years to ingest. Defaults to 2019 through current year.
    prefer_live : bool
        If True (default), try scraping tsa.gov first and use cached HTML
        only as a fallback. If False, use the cache directly.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    current_year = datetime.now().year
    if years is None:
        years = range(2019, current_year + 1)

    session = requests.Session()
    frames: list[pd.DataFrame] = []

    for year in years:
        print(f"Year {year}:")
        frame = None
        if prefer_live:
            frame = _fetch_live(year, current_year, session)
            time.sleep(1)  # be polite
        if frame is None:
            frame = _fallback_cache(year)
        if frame is None:
            print(f"  [WARN] no data for {year} - skipping.")
            continue
        print(f"  [OK] {len(frame)} rows")
        frames.append(frame)

    # Optional partial-year TSV (e.g. pasted from landing page during testing).
    for path in sorted(RAW_DIR.glob("tsa_*_partial.tsv")):
        print(f"Partial: {path.name}")
        frames.append(_parse_partial_tsv(path))

    if not frames:
        raise RuntimeError("No data ingested - check network or cache.")

    combined = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset="date", keep="last")
        .sort_values("date")
        .reset_index(drop=True)
        .astype({"passengers": int})
    )
    out_path = processed_dir / "tsa_daily.csv"
    combined.to_csv(out_path, index=False)
    (processed_dir / "SOURCE.txt").write_text(
        "TSA Daily Checkpoint Passenger Throughput\n"
        f"Source: {BASE_URL} (per-year pages)\n"
        f"Ingested: {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}\n"
        "Note: TSA revises archived figures (Known Crewmember and late-flight "
        "additions). Figures reflect the source pages as of the retrieval date.\n"
    )
    return combined


if __name__ == "__main__":
    df = ingest()
    print(f"\nRows:       {len(df):,}")
    print(f"Date range: {df['date'].min().date()} -> {df['date'].max().date()}")
    print("Rows per year:")
    print(df.groupby(df["date"].dt.year).size().to_string())
