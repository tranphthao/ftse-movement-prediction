"""Download daily OHLCV data and calculate manual technical indicators.

All calculations are asset-specific and chronologically ordered.  Rolling
windows include the current trading day and prior trading days only.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path


START_DATE = "2016-06-01"
END_DATE = "2026-07-01"  # yfinance's end date is exclusive; this includes 30 June 2026.
OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def download_and_clean(ticker: str, label: str) -> pd.DataFrame:
    """Download one ticker's daily OHLCV data and forward-fill missing values."""
    dataset = yf.download(
        tickers=ticker,
        start=START_DATE,
        end=END_DATE,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    # Keep only requested fields and remove the single-ticker MultiIndex if present.
    if isinstance(dataset.columns, pd.MultiIndex):
        dataset = dataset.loc[:, dataset.columns.get_level_values("Price").isin(OHLCV_COLUMNS)]
        dataset.columns = dataset.columns.get_level_values("Price")
    else:
        dataset = dataset.loc[:, dataset.columns.isin(OHLCV_COLUMNS)]

    print(f"\n{label} — initial rows:")
    print(dataset.head())

    # Check missing/empty values before cleaning and report their counts.
    empty_by_column = dataset.isna().sum()
    total_empty = int(empty_by_column.sum())
    print(f"\n{label} — empty values by column (before forward fill):")
    print(empty_by_column[empty_by_column.gt(0)] if total_empty else empty_by_column)
    print(f"{label} — total empty values: {total_empty}")

    if total_empty > 0:
        dataset = dataset.ffill()
        print(f"{label} — empty values remaining after forward fill: {int(dataset.isna().sum().sum())}")
    else:
        print(f"{label} — no forward fill was needed.")

    return dataset


def validate_ohlcv(dataset: pd.DataFrame, label: str) -> None:
    """Report missing and invalid OHLCV values without changing the source data."""
    missing_columns = set(OHLCV_COLUMNS).difference(dataset.columns)
    if missing_columns:
        raise ValueError(f"{label} is missing required columns: {sorted(missing_columns)}")

    price_columns = ["Open", "High", "Low", "Close"]
    missing_prices = int(dataset[price_columns].isna().sum().sum())
    invalid_high_low = int((dataset["High"] < dataset["Low"]).sum())
    non_positive_prices = int((dataset[price_columns] <= 0).any(axis=1).sum())
    negative_volume = int((dataset["Volume"] < 0).sum())

    print(
        f"{label} - OHLCV validation: {missing_prices} missing price values, "
        f"{invalid_high_low} High < Low rows, {non_positive_prices} non-positive price rows, "
        f"and {negative_volume} negative-volume rows."
    )

    if invalid_high_low:
        raise ValueError(f"{label} contains rows where High is below Low.")


def recursive_ema(values: pd.Series, span: int, min_periods: int) -> pd.Series:
    """Calculate an inspectable recursive EMA and withhold early values as NaN."""
    alpha = 2 / (span + 1)
    ema = pd.Series(np.nan, index=values.index, dtype="float64")
    previous_ema = np.nan
    consecutive_values = 0

    for date, value in values.items():
        if pd.isna(value):
            # A gap is not filled from the future; restart once valid data resumes.
            previous_ema = np.nan
            consecutive_values = 0
            continue

        previous_ema = value if pd.isna(previous_ema) else previous_ema + alpha * (value - previous_ema)
        consecutive_values += 1
        if consecutive_values >= min_periods:
            ema.loc[date] = previous_ema

    return ema


def calculate_technical_features(dataset: pd.DataFrame, label: str) -> pd.DataFrame:
    """Return OHLCV data plus SMA, Stochastic %K, MACD signal, RSI, and CCI.

    Formula and convention summary:
    - SMA_10: 10-day arithmetic mean of Close, with min_periods=10.
    - STOCH_K_14: 100 * (Close - 14-day rolling Low) / (14-day rolling High
      - 14-day rolling Low); zero ranges are NaN.
    - MACD_signal_9: a 9-period EMA of EMA(12, Close) - EMA(26, Close).
      Each EMA uses the recursive standard EMA = prior EMA + alpha *
      (current value - prior EMA), alpha = 2 / (span + 1), initialized from
      its first available input. Values are withheld until 12, 26, and then
      9 respective observations are available.
    - RSI_14: 14-day simple averages of positive and negative Close changes,
      as specified (not Wilder smoothing). Zero loss with positive gain is
      100; zero gain with positive loss is 0; both zero is NaN.
    - CCI_20: Typical Price = (High + Low + Close) / 3, using a 20-day simple
      mean and mean absolute deviation around that same current-window mean.
      A zero deviation produces NaN.

    Missing source values are not backfilled here. The import stage may
    forward-fill internal gaps; any remaining missing values cause indicators
    to remain NaN until a full valid lookback window exists.
    """
    dataset = dataset.sort_index().copy()
    validate_ohlcv(dataset, label)

    close = dataset["Close"]
    high = dataset["High"]
    low = dataset["Low"]

    # Every rolling window ends at t, so it contains only t and prior dates.
    dataset["SMA_10"] = close.rolling(window=10, min_periods=10).mean()

    lowest_low_14 = low.rolling(window=14, min_periods=14).min()
    highest_high_14 = high.rolling(window=14, min_periods=14).max()
    stoch_range = (highest_high_14 - lowest_low_14).replace(0, np.nan)
    dataset["STOCH_K_14"] = 100 * (close - lowest_low_14) / stoch_range

    fast_ema_12 = recursive_ema(close, span=12, min_periods=12)
    slow_ema_26 = recursive_ema(close, span=26, min_periods=26)
    macd_diff = fast_ema_12 - slow_ema_26
    dataset["MACD_signal_9"] = recursive_ema(macd_diff, span=9, min_periods=9)

    close_change = close.diff()
    upward_move = close_change.clip(lower=0)
    downward_move = (-close_change).clip(lower=0)
    average_gain_14 = upward_move.rolling(window=14, min_periods=14).mean()
    average_loss_14 = downward_move.rolling(window=14, min_periods=14).mean()
    relative_strength = average_gain_14 / average_loss_14.replace(0, np.nan)
    rsi_14 = 100 - (100 / (1 + relative_strength))
    dataset["RSI_14"] = rsi_14.mask((average_loss_14 == 0) & (average_gain_14 > 0), 100)
    dataset["RSI_14"] = dataset["RSI_14"].mask((average_gain_14 == 0) & (average_loss_14 > 0), 0)
    dataset["RSI_14"] = dataset["RSI_14"].mask((average_gain_14 == 0) & (average_loss_14 == 0), np.nan)

    typical_price = (high + low + close) / 3
    typical_price_mean_20 = typical_price.rolling(window=20, min_periods=20).mean()
    mean_absolute_deviation_20 = typical_price.rolling(window=20, min_periods=20).apply(
        lambda values: np.mean(np.abs(values - np.mean(values))), raw=True
    )
    cci_denominator = (0.015 * mean_absolute_deviation_20).replace(0, np.nan)
    dataset["CCI_20"] = (typical_price - typical_price_mean_20) / cci_denominator

    # Make the trading date explicit while preserving all original OHLCV columns.
    dataset.index.name = "Date"
    return dataset.reset_index()


# Each ticker is stored in its own DataFrame.
data_lloy = download_and_clean("LLOY.L", "LLOY")
data_rr = download_and_clean("RR.L", "RR")
data_sbry = download_and_clean("SBRY.L", "SBRY")
data_asc = download_and_clean("ASC.L", "ASC")
data_cpi = download_and_clean("CPI.L", "CPI")
data_crst = download_and_clean("CRST.L", "CRST")

# Technical features are calculated independently, never across tickers.
lloy_tech_features = calculate_technical_features(data_lloy, "LLOY")
rr_tech_features = calculate_technical_features(data_rr, "RR")
sbry_tech_features = calculate_technical_features(data_sbry, "SBRY")
asc_tech_features = calculate_technical_features(data_asc, "ASC")
cpi_tech_features = calculate_technical_features(data_cpi, "CPI")
crst_tech_features = calculate_technical_features(data_crst, "CRST")

# Small validation table requested for one ticker.
validation_columns = [
    "Date", "Open", "High", "Low", "Close", "Volume", "SMA_10",
    "STOCH_K_14", "MACD_signal_9", "RSI_14", "CCI_20",
]
print("\nLLOY technical-feature validation table:")
print(lloy_tech_features.loc[:, validation_columns].tail())

# Export each ticker's complete OHLCV and technical-feature dataset beside this script.
OUTPUT_DIRECTORY = Path(__file__).resolve().parent
feature_datasets = {
    "lloy_tech_features": lloy_tech_features,
    "rr_tech_features": rr_tech_features,
    "sbry_tech_features": sbry_tech_features,
    "asc_tech_features": asc_tech_features,
    "cpi_tech_features": cpi_tech_features,
    "crst_tech_features": crst_tech_features,
}

for dataset_name, feature_dataset in feature_datasets.items():
    output_path = OUTPUT_DIRECTORY / f"{dataset_name}.csv"
    feature_dataset.to_csv(output_path, index=False)
    print(f"Exported {dataset_name} to {output_path}")
