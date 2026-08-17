import requests
import time
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import nltk
nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# --- Step 1: Get headlines directly from GDELT ---
def get_headlines_gdelt(query, start_date, end_date, max_records=250, retries=3):
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "startdatetime": f"{start_date}000000",
        "enddatetime": f"{end_date}000000",
        "maxrecords": max_records,
        # Return the oldest matches first so the 250-record API limit does not
        # hide the first available headline in a busy month.
        "sort": "dateasc"
    }
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) research-script"}

    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=20)
            if response.status_code == 429:
                wait = 15 * (attempt + 1)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            response.raise_for_status()

            try:
                data = response.json()
            except ValueError as exc:
                print(
                    f"  Non-JSON response for {query}: {response.status_code} "
                    f"{response.text[:200]!r}"
                )
                return []

            if not isinstance(data, dict):
                print(f"  Unexpected response structure for {query}: {type(data)}")
                return []

            articles = data.get("articles", [])
            rows = []
            for a in articles:
                if "title" in a and "url" in a:
                    rows.append({
                        "date": a.get("seendate", ""),
                        "title": a["title"],
                        "url": a["url"],
                        "domain": a.get("domain", ""),
                        "language": a.get("language", "")
                    })
            return rows
        except requests.RequestException as exc:
            print(f"  Request failed for {query}: {exc}")
            if attempt < retries - 1:
                time.sleep(10 * (attempt + 1))
                continue
            return []

    return []

# --- Step 2: Loop month-by-month ---
def collect_headlines_over_range(query, start_year, start_month, end_year, end_month):
    all_rows = []
    for year in range(start_year, end_year + 1):
        first_month = start_month if year == start_year else 1
        last_month = end_month if year == end_year else 12
        for month in range(first_month, last_month + 1):
            start = f"{year}{month:02d}01"
            if month == 12:
                end = f"{year + 1}0101"
            else:
                end = f"{year}{month + 1:02d}01"

            rows = get_headlines_gdelt(query, start, end)
            print(f"{start} to {end}: {len(rows)} headlines")
            all_rows.extend(rows)
            time.sleep(5)  # longer delay to avoid rate limits

    return all_rows

# --- Step 3: Test LLOY headline availability during 2020 ---
TICKER_QUERIES = {
    "CRST": "Crest Nicholson"
}
START_YEAR, START_MONTH = 2020, 1
END_YEAR, END_MONTH = 2023, 1

# --- Step 4: Score sentiment on headlines and save one dataset per ticker ---
sid = SentimentIntensityAnalyzer()

for ticker, query in TICKER_QUERIES.items():
    try:
        print(f"\nExtracting headlines for {ticker}: {query}")
        all_headlines = collect_headlines_over_range(
            query, START_YEAR, START_MONTH, END_YEAR, END_MONTH
        )
        print(f"Total headlines for {ticker}: {len(all_headlines)}")

        for row in all_headlines:
            polarity = sid.polarity_scores(row["title"])
            row.update(polarity)
            row["ticker"] = ticker

        df = pd.DataFrame(all_headlines)

        # Filter English only.
        if "language" in df.columns:
            df = df[df["language"].str.contains("English", case=False, na=False)]

        print(f"English headlines scored for {ticker}: {len(df)}")
        if not df.empty:
            print(df[["date", "title", "neg", "neu", "pos", "compound"]].head(10))
            dates = pd.to_datetime(df["date"], errors="coerce", utc=True)
            earliest_date = dates.min()
            print(
                f"Earliest LLOY headline returned in the test period: "
                f"{earliest_date.strftime('%Y-%m-%d %H:%M:%S UTC')}"
            )
        else:
            print("No English LLOY headlines were returned for 2020.")

        output_stem = f"{ticker.lower()}_headline_sentiments_gdelt"
        df.to_csv(f"{output_stem}.csv", index=False, encoding="utf-8")
        df.to_pickle(f"{output_stem}.pkl")
        print(f"Saved {output_stem}.")
    except Exception as exc:
        print(f"Failed for {ticker}: {exc}")
        continue
