"""Extract SUE, Turnover, and Volatility features from Refinitiv (LSEG) Workspace.

One CSV is written for each selected UK ticker, for example
``asos_sue_feature.csv``.

Features
--------
sue          : SUE proxy, forward-filled event-driven predicted EPS surprise (%).
turnover     : Volume_t / SharesOutstanding_t.
volatility   : Rolling (20-trading-day) standard deviation of daily close-to-close
               returns, annualized by sqrt(252).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import refinitiv.data as rd


TICKERS = {
    "LLOY": "LLOY.L",
    "RR": "RR.L",
    "SBRY": "SBRY.L"
}
START_DATE = "2020-01-01"
END_DATE = "2026-07-30"

# Fields pulled from Workspace, in this exact order. Columns on the returned
# dataframe are renamed *positionally* to these canonical names (see
# `ticker_frame`) rather than matched against Workspace's own display titles,
# since those titles can vary slightly by entitlement/product version.
EPS_SURPRISE_FIELD = "TR.EpsPreSurprisePct"
VOLUME_FIELD = "TR.Volume"
# If your entitlement doesn't return this field, try "TR.SharesOutstandingCommon"
# or "TR.SharesOutstandingTotal" (check the Data Item Browser in Workspace).
SHARES_OUTSTANDING_FIELD = "TR.SharesOutstanding"
CLOSE_PRICE_FIELD = "TR.PriceClose"

FIELDS = [EPS_SURPRISE_FIELD, VOLUME_FIELD, SHARES_OUTSTANDING_FIELD, CLOSE_PRICE_FIELD]
FIELD_NAMES = ["eps_surprise_pct", "volume", "shares_outstanding", "close"]

VOLATILITY_WINDOW = 20  # trading days (~1 calendar month)
TRADING_DAYS_PER_YEAR = 252

OUTPUT_DIRECTORY = Path(__file__).resolve().parent


def pull_history(ric: str) -> pd.DataFrame:
    """Retrieve daily EPS surprise, volume, shares outstanding, and close price."""
    return rd.get_history(
        universe=[ric],
        fields=FIELDS,
        start=START_DATE,
        end=END_DATE,
        interval="1D",
    )


def ticker_frame(raw: pd.DataFrame, ric: str) -> pd.DataFrame:
    """Extract one RIC from the wide response and rename to canonical field names."""
    if isinstance(raw.columns, pd.MultiIndex):
        try:
            data = raw[ric].copy()
        except KeyError as exc:
            raise RuntimeError(f"No data returned for {ric}.") from exc
    else:
        data = raw.copy()

    if len(data.columns) != len(FIELD_NAMES):
        raise RuntimeError(
            f"Expected {len(FIELD_NAMES)} fields for {ric}, got {len(data.columns)} "
            f"({list(data.columns)}). Check field availability/entitlement, or "
            "that no field was silently dropped/duplicated by Workspace."
        )
    data.columns = FIELD_NAMES
    data.index = pd.to_datetime(data.index)
    data.index.name = "date"
    return data


def build_features(raw: pd.DataFrame, ticker: str, ric: str) -> pd.DataFrame:
    """Return SUE, Turnover, and realized Volatility features."""
    data = ticker_frame(raw, ric)

    for col in FIELD_NAMES:
        if data[col].isna().all():
            raise RuntimeError(
                f"No values returned for '{col}' on {ric}. Verify the field is "
                "available for this Workspace entitlement."
            )

    # SUE proxy: event-driven predicted EPS surprise, forward-filled so the
    # latest known value is available on every trading day.
    data["sue"] = data["eps_surprise_pct"].ffill()

    # Turnover_t = Volume_t / SharesOutstanding_t.
    # Shares outstanding updates infrequently (quarterly filings, buybacks,
    # issuance), so forward-fill it to have a value on every trading day.
    shares_outstanding_ffilled = data["shares_outstanding"].ffill()
    data["turnover"] = data["volume"] / shares_outstanding_ffilled

    # Volatility_t: rolling standard deviation of daily close-to-close returns,
    # annualized. Uses a 20-trading-day (~1 month) window.
    daily_return = data["close"].pct_change()
    data["volatility"] = daily_return.rolling(VOLATILITY_WINDOW).std() * np.sqrt(
        TRADING_DAYS_PER_YEAR
    )

    data["ticker"] = ticker
    return data.reset_index().sort_values("date")[
        [
            "date",
            "ticker",
            "eps_surprise_pct",
            "sue",
            "volume",
            "shares_outstanding",
            "turnover",
            "close",
            "volatility",
        ]
    ]


def main() -> None:
    app_key = os.environ.get("REFINITIV_APP_KEY")
    if not app_key:
        raise RuntimeError(
            "Set REFINITIV_APP_KEY before running. Generate it from the "
            "App Key Generator in Refinitiv Workspace."
        )

    rd.open_session(app_key=app_key)
    try:
        for ticker, ric in TICKERS.items():
            print(f"Extracting {ticker} ({ric}) from {START_DATE} to {END_DATE}...")
            features = build_features(pull_history(ric), ticker, ric)
            output_file = OUTPUT_DIRECTORY / f"{ticker.lower()}_sue_feature.csv"
            features.to_csv(output_file, index=False)
            print(f"Wrote {len(features):,} rows to {output_file}")
    finally:
        rd.close_session()


if __name__ == "__main__":
    main()