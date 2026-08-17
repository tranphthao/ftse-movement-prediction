"""Merge per-ticker headline sentiment, technical, and SUE feature files.

For every ticker subfolder inside ``ROOT_DIR`` (e.g. w8_model/SBRY), this
script looks for three files:

    1. <ticker>_headline_sentiments_gdelt.csv   (news headline sentiment)
    2. <ticker>_tech_features.csv                (technical indicators)
    3. <ticker>_sue_feature.csv                  (EPS surprise / SUE proxy)

and combines them into a single ``<ticker>_combined_features.csv`` written
back into the same ticker folder. Original input files are never modified.

Merge logic
-----------
- Headline sentiment: multiple articles per day are aggregated to one row
  per day using the mean of the ``compound`` score
  (-> ``daily_compound_sentiment``).
- The three datasets are first restricted to their common overlapping date
  range: max(start dates) to min(end dates), inclusive.
- The technical-features file is used as the "base calendar" (it has a true
  daily trading-day frequency), and the sentiment + SUE data are LEFT-joined
  onto it. Sentiment/SUE values are therefore NaN on days without a news
  article or an earnings event -- this is expected, since those are
  event-driven series, not continuously observed ones. If you want an inner
  join instead (only dates present in ALL three files), change
  ``HOW_TO_MERGE`` below to "inner".
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT_DIR = Path(r"C:\Users\X1 Gen 9\Documents\Study_2\MA7908 Final project\Coding\w8_model")

HEADLINE_SUFFIX = "_headline_sentiments_gdelt.csv"
TECH_SUFFIX = "_tech_features.csv"
SUE_SUFFIX = "_sue_feature.csv"

# "left"  -> keep every trading date from tech_features, attach sentiment/SUE
#            where available, NaN elsewhere (recommended default)
# "inner" -> keep only dates where sentiment, tech, AND sue all have a value
HOW_TO_MERGE = "left"

OUTPUT_SUFFIX = "_combined_features.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_one(folder: Path, suffix: str) -> Path | None:
    """Return the single file in ``folder`` ending with ``suffix``, or None."""
    matches = [p for p in folder.glob(f"*{suffix}") if p.is_file()]
    if len(matches) == 0:
        return None
    if len(matches) > 1:
        print(
            f"  WARNING: multiple files match *{suffix} in {folder.name}: "
            f"{[m.name for m in matches]}. Using {matches[0].name}."
        )
    return matches[0]


def normalize_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """Rename whichever of 'date'/'Date' exists to lowercase 'date'."""
    if "date" in df.columns:
        return df
    if "Date" in df.columns:
        return df.rename(columns={"Date": "date"})
    raise ValueError(f"No 'date' or 'Date' column found. Columns: {list(df.columns)}")


def load_headline_sentiment(path: Path) -> pd.DataFrame:
    """Load headline sentiment file and aggregate to one row per day.

    Raw date format example: '20230201T093000Z' -> calendar date 2023-02-01.
    Only the 'compound' column is used; multiple articles on the same day
    are averaged into 'daily_compound_sentiment'.
    """
    df = pd.read_csv(path)
    df = normalize_date_column(df)

    if "compound" not in df.columns:
        raise ValueError(f"'compound' column not found in {path.name}")

    # First 8 characters of the raw string are YYYYMMDD.
    raw_date_str = df["date"].astype(str)
    df["date"] = pd.to_datetime(raw_date_str.str[:8], format="%Y%m%d", errors="raise")

    daily = (
        df.groupby("date")["compound"]
        .mean()
        .rename("daily_compound_sentiment")
        .reset_index()
    )
    return daily.sort_values("date").reset_index(drop=True)


def load_tech_features(path: Path) -> pd.DataFrame:
    """Load the technical-features file with a parsed datetime 'date' column."""
    df = pd.read_csv(path)
    df = normalize_date_column(df)
    df["date"] = pd.to_datetime(df["date"])

    before = len(df)
    df = df.drop_duplicates(subset="date", keep="last")
    if len(df) != before:
        print(f"  NOTE: dropped {before - len(df)} duplicate date rows in {path.name}")

    return df.sort_values("date").reset_index(drop=True)


def load_sue_feature(path: Path) -> pd.DataFrame:
    """Load the SUE/EPS-surprise file, dropping the redundant 'ticker' column."""
    df = pd.read_csv(path)
    df = normalize_date_column(df)
    df["date"] = pd.to_datetime(df["date"])

    if "ticker" in df.columns:
        df = df.drop(columns=["ticker"])

    before = len(df)
    df = df.drop_duplicates(subset="date", keep="last")
    if len(df) != before:
        print(f"  NOTE: dropped {before - len(df)} duplicate date rows in {path.name}")

    return df.sort_values("date").reset_index(drop=True)


def common_date_range(*frames: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    starts = [f["date"].min() for f in frames]
    ends = [f["date"].max() for f in frames]
    return max(starts), min(ends)


def restrict_to_range(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    mask = (df["date"] >= start) & (df["date"] <= end)
    return df.loc[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per-ticker processing
# ---------------------------------------------------------------------------

def process_ticker_folder(folder: Path) -> str:
    """Process one ticker folder. Returns a short status string."""
    ticker = folder.name.upper()
    print(f"\n=== {ticker} ({folder}) ===")

    headline_path = find_one(folder, HEADLINE_SUFFIX)
    tech_path = find_one(folder, TECH_SUFFIX)
    sue_path = find_one(folder, SUE_SUFFIX)

    missing = [
        name
        for name, p in [
            (HEADLINE_SUFFIX, headline_path),
            (TECH_SUFFIX, tech_path),
            (SUE_SUFFIX, sue_path),
        ]
        if p is None
    ]
    if missing:
        print(f"  SKIPPED: missing file(s) matching {missing}")
        return "skipped (missing files)"

    sentiment = load_headline_sentiment(headline_path)
    tech = load_tech_features(tech_path)
    sue = load_sue_feature(sue_path)

    print(f"  headline sentiment: {sentiment['date'].min().date()} -> {sentiment['date'].max().date()} ({len(sentiment)} days with news)")
    print(f"  tech features:      {tech['date'].min().date()} -> {tech['date'].max().date()} ({len(tech)} rows)")
    print(f"  sue feature:        {sue['date'].min().date()} -> {sue['date'].max().date()} ({len(sue)} rows)")

    start, end = common_date_range(sentiment, tech, sue)
    if start > end:
        print(f"  SKIPPED: no overlapping date range (common_start={start.date()} > common_end={end.date()})")
        return "skipped (no overlap)"

    print(f"  common overlapping range: {start.date()} -> {end.date()}")

    sentiment = restrict_to_range(sentiment, start, end)
    tech = restrict_to_range(tech, start, end)
    sue = restrict_to_range(sue, start, end)

    if HOW_TO_MERGE == "left":
        base = tech
    else:
        # inner join: build the base as the intersection of dates present
        # in all three, then merge everything onto it.
        common_dates = (
            set(sentiment["date"]) & set(tech["date"]) & set(sue["date"])
        )
        base = tech[tech["date"].isin(common_dates)].reset_index(drop=True)

    merged = base.merge(sentiment, on="date", how="left")
    merged = merged.merge(sue, on="date", how="left")

    merged = merged.sort_values("date").reset_index(drop=True)
    merged = merged.drop_duplicates(subset="date", keep="last")

    merged.insert(1, "ticker", ticker)

    n_rows = len(merged)
    pct_sentiment = merged["daily_compound_sentiment"].notna().mean() * 100
    pct_sue = merged["sue"].notna().mean() * 100 if "sue" in merged.columns else float("nan")
    print(f"  merged rows: {n_rows} | days with sentiment: {pct_sentiment:.1f}% | days with sue value: {pct_sue:.1f}%")

    output_path = folder / f"{ticker.lower()}{OUTPUT_SUFFIX}"
    merged.to_csv(output_path, index=False)
    print(f"  wrote {output_path}")

    return f"ok ({n_rows} rows)"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    if not ROOT_DIR.exists():
        raise FileNotFoundError(f"ROOT_DIR does not exist: {ROOT_DIR}")

    results: dict[str, str] = {}
    for folder in sorted(ROOT_DIR.iterdir()):
        if not folder.is_dir():
            continue
        # Only treat it as a ticker folder if it actually has at least one
        # of the three expected file types -- this naturally skips
        # __pycache__, .venv, .venv-1, archived, etc.
        has_any = any(
            find_one(folder, suf) is not None
            for suf in (HEADLINE_SUFFIX, TECH_SUFFIX, SUE_SUFFIX)
        )
        if not has_any:
            continue
        try:
            results[folder.name] = process_ticker_folder(folder)
        except Exception as exc:  # noqa: BLE001 - log and keep going
            print(f"  ERROR processing {folder.name}: {exc}")
            results[folder.name] = f"error: {exc}"

    print("\n--- Summary ---")
    for ticker, status in results.items():
        print(f"  {ticker}: {status}")


if __name__ == "__main__":
    main()