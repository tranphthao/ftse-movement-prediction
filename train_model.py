"""Leakage-safe multi-horizon stock-direction classification experiment.

Each ticker subfolder inside ``--data-dir`` (e.g. w8_model/SBRY, w8_model/ASC,
...) is expected to hold one ``<ticker>_combined_features.csv`` file, written
by the earlier merge script: one row per trading date, with technical
indicators, ``daily_compound_sentiment``, and ``sue`` already aligned to a
common date range. Those files are auto-discovered and combined into one
panel so each horizon has one common, chronological train/test sample.
Rows from different stocks are never joined into an LSTM sequence. A
sequence ending on date t contains only feature observations at or before t;
``Close`` is used only when constructing its corresponding future-direction
target.

Feature engineering:
    - SMA_10 and MACD_signal_9 are raw price-scale indicators, so their
      absolute level drifts along with the stock's price level over a
      multi-year window. Because the train/test split is chronological (not
      random), a stock that has trended up or down leaves the test period at
      a structurally different price level than the training period --
      standardizing with a scaler fit on training data does not fix this,
      since it only rescales, it doesn't remove the drift. Both indicators
      are therefore converted to price-relative versions before modelling:
        SMA_10_pct        = (Close - SMA_10) / SMA_10
        MACD_signal_9_pct = MACD_signal_9 / Close
      RSI_14, STOCH_K_14, and CCI_20 are already relative-to-recent-range by
      construction and are used unchanged.

Missing-value handling (see load_and_validate_data for details):
    - daily_compound_sentiment: NaN means no news that day -> filled with 0
      (neutral), not "unknown".
    - sue: forward-filled per ticker (the most recently known EPS-surprise
      estimate carries forward until the next update). Rows before a
      ticker's first SUE observation are dropped, since there is no
      leakage-safe way to fill them.
    - Technical indicators: any stray NaN (rare zero-division edge cases)
      and missing turnover/volatility values fall through to the existing
      dropna() safety net in create_horizon_dataset. These market measures
      are not imputed, since doing so would fabricate observed trading data.

Behavioral-feature ablation (default mode):
    By default, running this script trains everything TWICE on the exact
    same loaded panel -- once with every feature, once with
    daily_compound_sentiment and sue removed -- so any accuracy difference
    is attributable to those two features alone, not to a different sample
    or date range. Each run's full output goes to its own subfolder
    (model_results/with_behavioral, model_results/without_behavioral), and
    a combined before/after comparison (model_comparison_results.csv-style,
    per-ticker, and per-group) is written at the top of --output-dir. Use
    --mode full or --mode technical-only to run just one feature set.

Pooled vs. per-group modelling:
    By default every ticker is pooled into one panel and trained as a single
    shared model per horizon (model_comparison_by_ticker.csv / by_group.csv
    then re-slice that ONE model's predictions by ticker/group afterward --
    no retraining involved). Pass --by-group to instead train fully
    independent models per market-cap group (Large Cap / Small Cap, see
    MARKET_CAP_GROUPS) -- each group gets its own chronological split, its
    own scaler, and its own SVM/LR/RF/LSTM fits, written to
    model_results/large_cap/... and model_results/small_cap/.... This is a
    genuinely different model per group, not a re-sliced view of one pooled
    model. --by-group combines with --mode (ablation/full/technical-only).

Run from any directory, for example::

    python train_model.py --epochs 30
    python train_model.py --mode full        # every feature, single run
    python train_model.py --mode technical-only
    python train_model.py --by-group         # separate models per market-cap group

Outputs (CSV tables and PNG figures) are written to ``model_results`` beside
this script. TensorFlow is required only for the LSTM experiments.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # headless: this script only saves PNGs, never shows a window.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# Raw technical-indicator columns as they appear in the combined-features CSV.
RAW_TECH_COLUMNS = ["SMA_10", "STOCH_K_14", "MACD_signal_9", "RSI_14", "CCI_20"]

# Final model inputs. SMA_10 and MACD_signal_9 are replaced by price-relative
# versions (see module docstring); the rest are used as-is.
FEATURES = [
    "SMA_10_pct",
    "STOCH_K_14",
    "MACD_signal_9_pct",
    "RSI_14",
    "CCI_20",
    "turnover",
    "volatility",
    "daily_compound_sentiment",
    "sue",
]

# The two "behavioral" features, used to build the ablation feature set
# (FEATURES with these removed) for the before/after comparison.
BEHAVIORAL_FEATURES = ["daily_compound_sentiment", "sue"]

HORIZONS = (1, 7, 30)
LOOKBACK = 30  # 30 prior-and-current usable trading observations per ticker.
RANDOM_STATE = 42
COMBINED_SUFFIX = "_combined_features.csv"  # produced by the merge script.

# Market-cap grouping for the per-group accuracy breakdown. This is a manual
# classification -- it can't be inferred from the data -- so any ticker not
# listed here (e.g. one added later) falls into "Ungrouped" with a printed
# warning rather than being silently dropped or miscategorized.
MARKET_CAP_GROUPS = {
    "LLOY": "Large Cap",
    "SBRY": "Large Cap",
    "RR": "Large Cap",
    "ASC": "Small Cap",
    "CPI": "Small Cap",
    "CRST": "Small Cap",
}


def add_relative_price_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive SMA_10_pct and MACD_signal_9_pct from raw price-scale columns.

    Both Close and SMA_10 are always strictly positive for a real equity
    price series, so no zero-division guard is needed here.
    """
    frame["SMA_10_pct"] = (frame["Close"] - frame["SMA_10"]) / frame["SMA_10"]
    frame["MACD_signal_9_pct"] = frame["MACD_signal_9"] / frame["Close"]
    return frame


def prepare_uploaded_data(raw_data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Validate and prepare one uploaded price/feature CSV for experimentation.

    Files exported by this project can be uploaded directly. Plain OHLCV
    files are also supported: the same technical indicators used by the
    project are calculated before modelling. Sentiment and SUE cannot be
    derived from OHLCV alone, so if either is absent it defaults to 0
    (neutral / no-signal), matching the same convention used for the main
    per-ticker panel. No uploaded data is written to disk.
    """
    frame = raw_data.copy()
    required_base = {"Date", "Close"}
    missing_base = required_base.difference(frame.columns)
    if missing_base:
        raise ValueError(f"Uploaded CSV is missing required columns: {sorted(missing_base)}")

    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame["Close"] = pd.to_numeric(frame["Close"], errors="coerce")
    if frame["Date"].isna().any() or frame["Close"].isna().any():
        raise ValueError("Date and Close must contain valid, non-empty values.")
    if frame["Date"].duplicated().any():
        raise ValueError("Uploaded CSV contains duplicate Date values.")
    frame = frame.sort_values("Date").reset_index(drop=True)

    # Sentiment/SUE can't be computed from price data; default to neutral
    # rather than failing an otherwise-valid OHLCV-only upload.
    for optional_feature in ("daily_compound_sentiment", "sue"):
        if optional_feature not in frame.columns:
            frame[optional_feature] = 0.0

    absent_raw = set(RAW_TECH_COLUMNS).difference(frame.columns)
    if absent_raw:
        ohlcv = {"Open", "High", "Low", "Volume"}
        missing_ohlcv = ohlcv.difference(frame.columns)
        if missing_ohlcv:
            raise ValueError(
                "Upload either the project technical-feature CSV or an OHLCV CSV. "
                f"Missing columns needed to calculate features: {sorted(missing_ohlcv)}"
            )
        for column in ["Open", "High", "Low", "Volume"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame[["Open", "High", "Low", "Close"]].isna().any().any():
            raise ValueError("OHLC price columns must contain valid numeric values.")
        frame["SMA_10"] = frame["Close"].rolling(10, min_periods=10).mean()
        low_14, high_14 = frame["Low"].rolling(14, min_periods=14).min(), frame["High"].rolling(14, min_periods=14).max()
        frame["STOCH_K_14"] = 100 * (frame["Close"] - low_14) / (high_14 - low_14).replace(0, np.nan)
        macd = frame["Close"].ewm(span=12, adjust=False, min_periods=12).mean() - frame["Close"].ewm(span=26, adjust=False, min_periods=26).mean()
        frame["MACD_signal_9"] = macd.ewm(span=9, adjust=False, min_periods=9).mean()
        change = frame["Close"].diff()
        gains, losses = change.clip(lower=0), -change.clip(upper=0)
        rs = gains.rolling(14, min_periods=14).mean() / losses.rolling(14, min_periods=14).mean().replace(0, np.nan)
        frame["RSI_14"] = (100 - 100 / (1 + rs)).mask((losses.rolling(14, min_periods=14).mean() == 0) & (gains.rolling(14, min_periods=14).mean() > 0), 100)
        typical = (frame["High"] + frame["Low"] + frame["Close"]) / 3
        average = typical.rolling(20, min_periods=20).mean()
        deviation = typical.rolling(20, min_periods=20).apply(lambda values: np.mean(np.abs(values - values.mean())), raw=True)
        frame["CCI_20"] = (typical - average) / (0.015 * deviation).replace(0, np.nan)
    else:
        for column in RAW_TECH_COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    # Derive the price-relative versions the model actually uses, regardless
    # of whether the raw indicators were uploaded directly or just computed.
    frame = add_relative_price_features(frame)

    frame["Ticker"] = ticker.upper().strip() or "UPLOADED"
    return frame


def set_seeds() -> None:
    """Set all applicable seeds; TensorFlow is imported lazily."""
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(RANDOM_STATE)
        try:
            tf.config.experimental.enable_op_determinism()
        except (AttributeError, RuntimeError):
            pass
    except ImportError:
        pass


def find_combined_file(folder: Path) -> Path | None:
    """Return the single '<ticker>_combined_features.csv' file in ``folder``."""
    matches = sorted(p for p in folder.glob(f"*{COMBINED_SUFFIX}") if p.is_file())
    if not matches:
        return None
    if len(matches) > 1:
        print(
            f"  WARNING: multiple files match *{COMBINED_SUFFIX} in {folder.name}: "
            f"{[m.name for m in matches]}. Skipping this folder; each ticker folder "
            "must contain exactly one combined-features file."
        )
        return None
    return matches[0]


def discover_combined_files(data_directory: Path) -> dict[str, Path]:
    """Map TICKER -> path to its combined-features CSV.

    Checks two layouts so it doesn't matter which one you used:
      - directly inside data_directory, e.g. w8_model/asc_combined_features.csv
        (ticker taken from the filename)
      - inside a per-ticker subfolder, e.g. w8_model/ASC/asc_combined_features.csv
        (ticker taken from the folder name)
    If both exist for the same ticker, the subfolder version wins and a
    warning is printed.
    """
    found: dict[str, Path] = {}

    for path in sorted(data_directory.glob(f"*{COMBINED_SUFFIX}")):
        if path.is_file():
            found[path.name[: -len(COMBINED_SUFFIX)].upper()] = path

    for folder in sorted(data_directory.iterdir()):
        if not folder.is_dir():
            continue
        path = find_combined_file(folder)
        if path is None:
            continue
        ticker = folder.name.upper()
        if ticker in found and found[ticker] != path:
            print(
                f"  WARNING: found combined files for {ticker} both directly in "
                f"{data_directory} and in {folder}. Using the one inside {folder}."
            )
        found[ticker] = path

    return found


def load_and_validate_data(data_directory: Path) -> pd.DataFrame:
    """Auto-discover every ticker's ``<ticker>_combined_features.csv`` under
    ``data_directory`` (see discover_combined_files for the two supported
    layouts) and load it, applying feature-specific missing-value rules and
    the price-relative transforms before returning one concatenated,
    chronologically sorted panel.
    """
    required = {
        "date", "Close", *RAW_TECH_COLUMNS, "turnover", "volatility",
        "daily_compound_sentiment", "sue",
    }
    combined_files = discover_combined_files(data_directory)
    frames = []
    for ticker, path in sorted(combined_files.items()):
        frame = pd.read_csv(path)
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

        frame = frame.rename(columns={"date": "Date"})
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
        for column in [
            "Close", *RAW_TECH_COLUMNS, "turnover", "volatility",
            "daily_compound_sentiment", "sue",
        ]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame["Date"].isna().any():
            raise ValueError(f"{path} contains invalid Date values.")
        if frame["Date"].duplicated().any():
            raise ValueError(f"{path} contains duplicate trading dates.")
        frame = frame.sort_values("Date").reset_index(drop=True)

        # --- Missing-value handling, one rule per feature -------------------
        # No news that day -> neutral sentiment, not "unknown".
        frame["daily_compound_sentiment"] = frame["daily_compound_sentiment"].fillna(0.0)

        # SUE only updates when a new EPS-surprise estimate is published.
        # Forward-fill carries the most recently *known* value forward to
        # every day until the next update -- leakage-safe because ffill only
        # ever looks backward in time. Backfilling or averaging over the
        # whole series would both leak future information into earlier rows.
        frame["sue"] = frame["sue"].ffill()
        # Rows before a ticker's very first SUE observation have nothing to
        # carry forward from; there is no leakage-safe way to fill them, so
        # they are dropped rather than fabricated.
        frame = frame.dropna(subset=["sue"]).reset_index(drop=True)

        # --- Price-relative transforms (see module docstring) --------------
        frame = add_relative_price_features(frame)

        frame["Ticker"] = ticker
        frames.append(frame.loc[:, ["Date", "Ticker", "Close", *FEATURES]])

    if not frames:
        raise FileNotFoundError(
            f"No '*{COMBINED_SUFFIX}' files found directly in or in a subfolder of {data_directory}."
        )
    return pd.concat(frames, ignore_index=True).sort_values(["Date", "Ticker"]).reset_index(drop=True)


def create_horizon_dataset(panel: pd.DataFrame, horizon: int, features: list[str] = FEATURES) -> tuple[pd.DataFrame, int]:
    """Align Target_h(t) = Close(t+h) > Close(t), then remove unusable rows.

    ``features`` controls which columns the dropna() safety net checks --
    pass a subset (e.g. FEATURES minus BEHAVIORAL_FEATURES) to run an
    ablation without touching the underlying panel.
    """
    work = panel.sort_values(["Ticker", "Date"]).copy()
    # shift(-h) is h *trading observations* ahead, separately for every stock.
    future_close = work.groupby("Ticker", sort=False)["Close"].shift(-horizon)
    work["Target"] = np.where(future_close.notna(), (future_close > work["Close"]).astype(int), np.nan)
    before = len(work)
    work = work.dropna(subset=[*features, "Target"]).copy()
    work["Target"] = work["Target"].astype(int)
    work = work.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    return work, before - len(work)


def chronological_split(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the first 80% and final 20% in chronological panel order."""
    split_index = int(len(dataset) * 0.80)
    if split_index == 0 or split_index == len(dataset):
        raise ValueError("Dataset is too small for an 80/20 chronological split.")
    return dataset.iloc[:split_index].copy(), dataset.iloc[split_index:].copy()


def scale_without_leakage(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[StandardScaler, np.ndarray, np.ndarray]:
    """Fit StandardScaler on training predictors only and reuse it for test data."""
    scaler = StandardScaler()
    return scaler, scaler.fit_transform(X_train), scaler.transform(X_test)


def train_logistic_regression(X_train: np.ndarray, y_train: pd.Series) -> LogisticRegression:
    return LogisticRegression(max_iter=2_000, random_state=RANDOM_STATE)


def train_svm(X_train: np.ndarray, y_train: pd.Series) -> SVC:
    return SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1, class_weight=None)


def make_lstm_sequences(dataset: pd.DataFrame, scaled_features: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Make per-ticker, historical sequences and return endpoints in panel row order.

    ``endpoint`` is the original horizon dataset row position, allowing an
    exact chronological split of sequence targets. No observation after the
    endpoint is ever placed in a sequence.
    """
    sequences, targets, endpoints = [], [], []
    work = dataset.copy()
    work["_row"] = np.arange(len(work))
    for _, group in work.sort_values(["Ticker", "Date"]).groupby("Ticker", sort=False):
        rows = group["_row"].to_numpy()
        for stop in range(lookback - 1, len(rows)):
            sequence_rows = rows[stop - lookback + 1 : stop + 1]
            sequences.append(scaled_features[sequence_rows])
            targets.append(int(group.iloc[stop]["Target"]))
            endpoints.append(int(rows[stop]))
    if not sequences:
        raise ValueError(f"No LSTM sequences available with LOOKBACK={lookback}.")
    return np.asarray(sequences, dtype=np.float32), np.asarray(targets), np.asarray(endpoints)


def train_lstm(X_train: np.ndarray, y_train: np.ndarray, epochs: int):
    """Train a small binary LSTM; validation is the later 10% of training only."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError("TensorFlow is required for LSTM. Install it with `pip install tensorflow`.") from exc
    set_seeds()
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2])),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
    # shuffle=False preserves chronological ordering in the train/validation split.
    model.fit(X_train, y_train, validation_split=0.10, epochs=epochs, batch_size=32,
              shuffle=False, verbose=0, callbacks=callbacks)
    return model


def evaluate_predictions(y_true: np.ndarray, probabilities: np.ndarray, baseline_class: int) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Calculate requested test metrics using a 0.5 probability threshold."""
    predictions = (probabilities >= 0.5).astype(int)
    roc_auc = roc_auc_score(y_true, probabilities) if len(np.unique(y_true)) == 2 else np.nan
    metrics = {
        "Accuracy": accuracy_score(y_true, predictions),
        "ROC_AUC": roc_auc,
        "Precision": precision_score(y_true, predictions, zero_division=0),
        "Recall": recall_score(y_true, predictions, zero_division=0),
        "F1": f1_score(y_true, predictions, zero_division=0),
        "Baseline_Accuracy": accuracy_score(y_true, np.full(len(y_true), baseline_class)),
    }
    return metrics, predictions, confusion_matrix(y_true, predictions, labels=[0, 1])


def evaluate_by_ticker(prediction_rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Break the same test predictions down by ticker instead of pooling them.

    The pooled ``comparison`` table (one row per Model x Horizon) tells you
    how a model does on the panel as a whole; this tells you whether that
    number is actually representative of every ticker or being carried by
    one or two of them. Computed straight from ``prediction_rows`` -- no
    retraining, just re-slicing the test predictions already collected.
    """
    if not prediction_rows:
        return pd.DataFrame(columns=["Model", "Horizon", "Ticker", "Observations",
                                     "Accuracy", "ROC_AUC", "Precision", "Recall", "F1",
                                     "Up_Proportion_Actual"])
    frame = pd.DataFrame(prediction_rows)
    rows = []
    for (model, horizon, ticker), group in frame.groupby(["Model", "Horizon", "Ticker"], sort=False):
        y_true = group["Actual"].to_numpy()
        y_pred = group["Predicted"].to_numpy()
        probabilities = group["Up_Probability"].to_numpy()
        roc_auc = roc_auc_score(y_true, probabilities) if len(np.unique(y_true)) == 2 else np.nan
        rows.append({
            "Model": model, "Horizon": horizon, "Ticker": ticker,
            "Observations": len(group),
            "Accuracy": accuracy_score(y_true, y_pred),
            "ROC_AUC": roc_auc,
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1": f1_score(y_true, y_pred, zero_division=0),
            "Up_Proportion_Actual": y_true.mean(),
        })
    order = {horizon: index for index, horizon in enumerate(dict.fromkeys(frame["Horizon"]))}
    return (
        pd.DataFrame(rows)
        .assign(_horizon_order=lambda d: d["Horizon"].map(order))
        .sort_values(["_horizon_order", "Model", "Ticker"])
        .drop(columns="_horizon_order")
        .reset_index(drop=True)
    )


def plot_performance_by_ticker(by_ticker: pd.DataFrame, output_dir: Path) -> None:
    """One Accuracy bar chart per horizon, tickers on the x-axis, grouped by model."""
    if by_ticker.empty:
        return
    horizons = list(dict.fromkeys(by_ticker["Horizon"]))
    fig, axes = plt.subplots(1, len(horizons), figsize=(6 * len(horizons), 4.5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, horizon in zip(axes, horizons):
        subset = by_ticker.loc[by_ticker["Horizon"].eq(horizon)]
        sns.barplot(data=subset, x="Ticker", y="Accuracy", hue="Model", ax=ax)
        ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="Chance")
        ax.set_title(horizon)
        ax.set_ylim(0, 1)
        if ax is not axes[0]:
            ax.legend_.remove()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels))
    fig.suptitle("Accuracy by ticker")
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    fig.savefig(output_dir / "accuracy_by_ticker.png", dpi=160)
    plt.close(fig)


def evaluate_by_group(by_ticker: pd.DataFrame, group_map: dict[str, str]) -> pd.DataFrame:
    """Average the per-ticker metrics within named ticker groups (e.g. Large
    Cap vs Small Cap).

    This averages the already-computed per-ticker Accuracy/ROC_AUC/etc.
    across the tickers in each group -- every ticker counts equally,
    regardless of how many test rows it happens to have -- rather than
    pooling all of a group's rows and computing one metric over the
    combined set. Tickers with no entry in ``group_map`` are bucketed into
    "Ungrouped" with a printed warning, instead of being silently dropped.
    """
    if by_ticker.empty:
        return pd.DataFrame(columns=["Model", "Horizon", "Group", "Tickers", "N_Tickers",
                                     "Total_Observations", "Accuracy", "ROC_AUC",
                                     "Precision", "Recall", "F1"])
    frame = by_ticker.copy()
    unmapped = sorted(set(frame["Ticker"]) - set(group_map))
    if unmapped:
        print(f"  WARNING: no market-cap group assigned for {unmapped}; grouping them as 'Ungrouped'.")
    frame["Group"] = frame["Ticker"].map(group_map).fillna("Ungrouped")

    rows = []
    for (model, horizon, group), subset in frame.groupby(["Model", "Horizon", "Group"], sort=False):
        rows.append({
            "Model": model, "Horizon": horizon, "Group": group,
            "Tickers": ", ".join(sorted(subset["Ticker"])),
            "N_Tickers": subset["Ticker"].nunique(),
            "Total_Observations": int(subset["Observations"].sum()),
            "Accuracy": subset["Accuracy"].mean(),
            "ROC_AUC": subset["ROC_AUC"].mean(),
            "Precision": subset["Precision"].mean(),
            "Recall": subset["Recall"].mean(),
            "F1": subset["F1"].mean(),
        })
    order = {horizon: index for index, horizon in enumerate(dict.fromkeys(frame["Horizon"]))}
    return (
        pd.DataFrame(rows)
        .assign(_horizon_order=lambda d: d["Horizon"].map(order))
        .sort_values(["_horizon_order", "Model", "Group"])
        .drop(columns="_horizon_order")
        .reset_index(drop=True)
    )


def split_panel_by_group(panel: pd.DataFrame, group_map: dict[str, str]) -> dict[str, pd.DataFrame]:
    """Split one loaded panel into one sub-panel per market-cap group.

    Unlike evaluate_by_group (which re-slices one pooled model's already-
    computed predictions), this splits the raw panel *before* any model is
    trained -- the sub-panels returned here go on to get their own
    chronological split, scaler, and model fits in run_by_group. Tickers
    absent from ``group_map`` are bucketed into "Ungrouped" (same
    convention as evaluate_by_group) with a printed warning, rather than
    being silently dropped from every group.
    """
    frame = panel.copy()
    unmapped = sorted(set(frame["Ticker"]) - set(group_map))
    if unmapped:
        print(f"  WARNING: no market-cap group assigned for {unmapped}; grouping them as 'Ungrouped'.")
    frame["_Group"] = frame["Ticker"].map(group_map).fillna("Ungrouped")
    return {
        group_name: group_frame.drop(columns="_Group").reset_index(drop=True)
        for group_name, group_frame in frame.groupby("_Group", sort=False)
    }


def plot_performance_by_group(by_group: pd.DataFrame, output_dir: Path) -> None:
    """One Accuracy bar chart per horizon, Large Cap vs Small Cap, grouped by model."""
    if by_group.empty:
        return
    horizons = list(dict.fromkeys(by_group["Horizon"]))
    fig, axes = plt.subplots(1, len(horizons), figsize=(5 * len(horizons), 4.5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, horizon in zip(axes, horizons):
        subset = by_group.loc[by_group["Horizon"].eq(horizon)]
        sns.barplot(data=subset, x="Group", y="Accuracy", hue="Model", ax=ax)
        ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="Chance")
        ax.set_title(horizon)
        ax.set_ylim(0, 1)
        if ax is not axes[0]:
            ax.legend_.remove()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels))
    fig.suptitle("Average accuracy by market-cap group")
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    fig.savefig(output_dir / "accuracy_by_group.png", dpi=160)
    plt.close(fig)


def print_horizon_summary(horizon: int, dataset: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame, removed: int) -> None:
    def distribution(frame: pd.DataFrame) -> str:
        values = frame["Target"].value_counts(normalize=True).reindex([0, 1], fill_value=0)
        return f"Down={values[0]:.2%}, Up={values[1]:.2%}"
    print(f"\n{'=' * 72}\nHorizon: {horizon}-day | usable observations: {len(dataset)} | removed: {removed}")
    print(f"First/last usable date: {dataset.Date.min().date()} to {dataset.Date.max().date()}")
    print(f"Train ({len(train)}): {train.Date.min().date()} to {train.Date.max().date()} | {distribution(train)}")
    print(f"Test  ({len(test)}): {test.Date.min().date()} to {test.Date.max().date()} | {distribution(test)}")


def plot_confusion(matrix: np.ndarray, model: str, horizon: int, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Predicted Down", "Predicted Up"], yticklabels=["Actual Down", "Actual Up"])
    ax.set_title(f"{model} — {horizon}-day confusion matrix")
    fig.tight_layout()
    fig.savefig(output_dir / f"confusion_{model.lower().replace(' ', '_')}_{horizon}d.png", dpi=160)
    plt.close(fig)


def plot_roc_curves(roc_data: dict[int, list[tuple[str, np.ndarray, np.ndarray]]], output_dir: Path) -> None:
    for horizon, entries in roc_data.items():
        fig, ax = plt.subplots(figsize=(6, 5))
        for model, y_true, probabilities in entries:
            if len(np.unique(y_true)) == 2:
                fpr, tpr, _ = roc_curve(y_true, probabilities)
                ax.plot(fpr, tpr, label=f"{model} (AUC={roc_auc_score(y_true, probabilities):.3f})")
        ax.plot([0, 1], [0, 1], "k--", label="Chance")
        ax.set(xlabel="False positive rate", ylabel="True positive rate", title=f"ROC curves — {horizon}-day horizon")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / f"roc_{horizon}d.png", dpi=160)
        plt.close(fig)


def plot_performance(results: pd.DataFrame, output_dir: Path) -> None:
    metrics = ["Accuracy", "ROC_AUC", "Precision", "Recall", "F1"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(21, 4.5), sharey=True)
    for ax, metric in zip(axes, metrics):
        sns.barplot(data=results, x="Horizon", y=metric, hue="Model", ax=ax)
        ax.set_title(metric.replace("_", " "))
        ax.set_ylim(0, 1)
        ax.legend_.remove()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(output_dir / "model_performance_comparison.png", dpi=160)
    plt.close(fig)


def plot_actual_vs_predicted(records: list[dict[str, Any]], output_dir: Path) -> None:
    for record in records:
        frame = record["frame"].sort_values(["Ticker", "Date"])
        tickers = list(frame["Ticker"].unique())
        fig, axes = plt.subplots(len(tickers), 1, figsize=(11, 2.2 * len(tickers)), sharex=True)
        axes = np.atleast_1d(axes)
        for ax, ticker in zip(axes, tickers):
            ticker_frame = frame.loc[frame["Ticker"].eq(ticker)]
            ax.step(ticker_frame["Date"], ticker_frame["Target"], where="mid", label="Actual", linewidth=1.3)
            ax.step(ticker_frame["Date"], ticker_frame["Predicted"], where="mid", label="Predicted", linewidth=1.0, alpha=0.8)
            ax.set(ylim=(-0.1, 1.1), yticks=[0, 1], yticklabels=["Down (0)", "Up (1)"], ylabel=ticker)
        axes[0].set_title(f"Actual vs predicted direction — {record['Model']}, {record['Horizon']}-day")
        axes[0].legend(loc="upper right")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(output_dir / f"direction_{record['Model'].lower().replace(' ', '_')}_{record['Horizon']}d.png", dpi=160)
        plt.close(fig)


def save_dashboard_data(prediction_rows: list[dict[str, Any]], roc_rows: list[dict[str, Any]], output_dir: Path) -> None:
    """Persist the app's tabular inputs as soon as an experiment completes.

    The dashboard requires row-level test predictions and probability scores.
    Writing them incrementally means that a later optional LSTM error cannot
    prevent the completed Logistic Regression, SVM, and Random Forest outputs
    from being available to Streamlit.
    """
    prediction_columns = ["Model", "Horizon", "Ticker", "Date", "Actual", "Predicted", "Up_Probability"]
    roc_columns = ["Model", "Horizon", "False_Positive_Rate", "True_Positive_Rate", "Threshold"]
    pd.DataFrame(prediction_rows, columns=prediction_columns).to_csv(
        output_dir / "test_predictions.csv", index=False
    )
    pd.DataFrame(roc_rows, columns=roc_columns).to_csv(
        output_dir / "roc_curve_points.csv", index=False
    )


def run_experiments(data_directory: Path | None, output_dir: Path, epochs: int,
                    panel: pd.DataFrame | None = None,
                    features: list[str] | None = None) -> pd.DataFrame:
    """Run every horizon x model combination and write evaluation artefacts.

    ``features`` selects which of the loaded columns to actually model with
    (default: the full FEATURES list). The panel loaded by
    load_and_validate_data always contains every feature -- passing a subset
    here (e.g. FEATURES minus BEHAVIORAL_FEATURES) runs an ablation without
    reloading or re-cleaning the data, and without touching which rows get
    dropped for reasons unrelated to the feature subset (missing Date,
    duplicate trading dates, etc.).
    """
    set_seeds()
    features = list(features) if features is not None else list(FEATURES)
    output_dir.mkdir(parents=True, exist_ok=True)
    if panel is None:
        if data_directory is None:
            raise ValueError("A data directory or prepared panel is required.")
        panel = load_and_validate_data(data_directory)
    else:
        panel = panel.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    print(f"Features used this run: {features}")
    results, roc_data, direction_records = [], {}, []
    summary_rows, prediction_rows, confusion_rows, roc_rows, importance_rows = [], [], [], [], []

    for horizon in HORIZONS:
        dataset, removed = create_horizon_dataset(panel, horizon, features)
        train, test = chronological_split(dataset)
        print_horizon_summary(horizon, dataset, train, test, removed)
        X_train, X_test = train[features], test[features]
        y_train, y_test = train["Target"], test["Target"]
        baseline_class = int(y_train.mode().iloc[0])
        for split_name, split_frame in (("Train", train), ("Test", test)):
            target_counts = split_frame["Target"].value_counts().reindex([0, 1], fill_value=0)
            summary_rows.append({
                "Horizon": f"{horizon}-day", "Split": split_name,
                "Observations": len(split_frame), "First_Date": split_frame["Date"].min().date(),
                "Last_Date": split_frame["Date"].max().date(), "Down_Count": target_counts[0],
                "Up_Count": target_counts[1], "Down_Proportion": target_counts[0] / len(split_frame),
                "Up_Proportion": target_counts[1] / len(split_frame), "Majority_Training_Class": baseline_class,
            })
        _, X_train_scaled, X_test_scaled = scale_without_leakage(X_train, X_test)

        models = {
            "Logistic Regression": (train_logistic_regression(X_train_scaled, y_train), X_train_scaled, X_test_scaled),
            "SVM": (train_svm(X_train_scaled, y_train), X_train_scaled, X_test_scaled),
            "Random Forest": (train_random_forest(X_train, y_train), X_train, X_test),
        }
        for name, (model, model_X_train, model_X_test) in models.items():
            model.fit(model_X_train, y_train)
            probabilities = model.predict_proba(model_X_test)[:, 1]
            metrics, predictions, matrix = evaluate_predictions(y_test.to_numpy(), probabilities, baseline_class)
            results.append({"Model": name, "Horizon": f"{horizon}-day", **metrics})
            if name == "Random Forest":
                importance_rows.extend({"Horizon": f"{horizon}-day", "Feature": feature, "Importance": importance}
                                       for feature, importance in zip(features, model.feature_importances_))
            roc_data.setdefault(horizon, []).append((name, y_test.to_numpy(), probabilities))
            plot_confusion(matrix, name, horizon, output_dir)
            for actual, predicted, probability, (_, test_row) in zip(y_test, predictions, probabilities, test.iterrows()):
                prediction_rows.append({"Model": name, "Horizon": f"{horizon}-day", "Ticker": test_row["Ticker"],
                                        "Date": test_row["Date"].date(), "Actual": actual,
                                        "Predicted": predicted, "Up_Probability": probability})
            for actual_label, predicted_label, count in (
                ("Down", "Down", matrix[0, 0]), ("Down", "Up", matrix[0, 1]),
                ("Up", "Down", matrix[1, 0]), ("Up", "Up", matrix[1, 1]),
            ):
                confusion_rows.append({"Model": name, "Horizon": f"{horizon}-day", "Actual": actual_label,
                                       "Predicted": predicted_label, "Count": count})
            if len(np.unique(y_test)) == 2:
                fpr, tpr, thresholds = roc_curve(y_test, probabilities)
                roc_rows.extend({"Model": name, "Horizon": f"{horizon}-day", "False_Positive_Rate": x,
                                 "True_Positive_Rate": y, "Threshold": threshold}
                                for x, y, threshold in zip(fpr, tpr, thresholds))
            direction_records.append({"Model": name, "Horizon": horizon,
                                      "frame": test[["Date", "Ticker", "Target"]].assign(Predicted=predictions)})
            save_dashboard_data(prediction_rows, roc_rows, output_dir)

        # One scaler fitted only on the tabular training rows also scales LSTM
        # sequences. Test sequences may include earlier training history, never
        # future history; their target endpoint remains in the test partition.
        #
        # The whole block is wrapped so a missing/broken TensorFlow install
        # (or any other LSTM-specific failure) skips just this horizon's LSTM
        # row instead of losing the already-computed Logistic Regression,
        # SVM, and Random Forest results for every horizon.
        try:
            scaler = StandardScaler().fit(X_train)
            all_scaled = scaler.transform(dataset[features])
            sequences, sequence_targets, endpoints = make_lstm_sequences(dataset, all_scaled, LOOKBACK)
            split_index = len(train)
            train_mask, test_mask = endpoints < split_index, endpoints >= split_index
            if not train_mask.any() or not test_mask.any():
                raise ValueError("Chronological split leaves no LSTM train or test sequences.")
            lstm = train_lstm(sequences[train_mask], sequence_targets[train_mask], epochs)
            probabilities = lstm.predict(sequences[test_mask], verbose=0).ravel()
            y_lstm_test = sequence_targets[test_mask]
            metrics, predictions, matrix = evaluate_predictions(y_lstm_test, probabilities, baseline_class)
            results.append({"Model": "LSTM", "Horizon": f"{horizon}-day", **metrics})
            roc_data.setdefault(horizon, []).append(("LSTM", y_lstm_test, probabilities))
            plot_confusion(matrix, "LSTM", horizon, output_dir)
            lstm_test_rows = dataset.iloc[endpoints[test_mask]]
            for actual, predicted, probability, (_, test_row) in zip(y_lstm_test, predictions, probabilities, lstm_test_rows.iterrows()):
                prediction_rows.append({"Model": "LSTM", "Horizon": f"{horizon}-day", "Ticker": test_row["Ticker"],
                                        "Date": test_row["Date"].date(), "Actual": actual,
                                        "Predicted": predicted, "Up_Probability": probability})
            for actual_label, predicted_label, count in (
                ("Down", "Down", matrix[0, 0]), ("Down", "Up", matrix[0, 1]),
                ("Up", "Down", matrix[1, 0]), ("Up", "Up", matrix[1, 1]),
            ):
                confusion_rows.append({"Model": "LSTM", "Horizon": f"{horizon}-day", "Actual": actual_label,
                                       "Predicted": predicted_label, "Count": count})
            if len(np.unique(y_lstm_test)) == 2:
                fpr, tpr, thresholds = roc_curve(y_lstm_test, probabilities)
                roc_rows.extend({"Model": "LSTM", "Horizon": f"{horizon}-day", "False_Positive_Rate": x,
                                 "True_Positive_Rate": y, "Threshold": threshold}
                                for x, y, threshold in zip(fpr, tpr, thresholds))
            lstm_test_frame = dataset.iloc[endpoints[test_mask]][["Date", "Ticker", "Target"]].copy()
            direction_records.append({"Model": "LSTM", "Horizon": horizon,
                                      "frame": lstm_test_frame.assign(Predicted=predictions)})
            save_dashboard_data(prediction_rows, roc_rows, output_dir)
        except ImportError as exc:
            print(f"  SKIPPED LSTM for {horizon}-day horizon: {exc}")
        except Exception as exc:  # noqa: BLE001 - log and keep going
            print(f"  SKIPPED LSTM for {horizon}-day horizon due to an error: {exc}")

    comparison = pd.DataFrame(results, columns=["Model", "Horizon", "Accuracy", "ROC_AUC", "Precision", "Recall", "F1", "Baseline_Accuracy"])
    comparison.to_csv(output_dir / "model_comparison_results.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(output_dir / "horizon_split_summaries.csv", index=False)
    save_dashboard_data(prediction_rows, roc_rows, output_dir)
    pd.DataFrame(confusion_rows).to_csv(output_dir / "confusion_matrices.csv", index=False)
    pd.DataFrame(importance_rows).to_csv(output_dir / "feature_importance.csv", index=False)

    by_ticker = evaluate_by_ticker(prediction_rows)
    by_ticker.to_csv(output_dir / "model_comparison_by_ticker.csv", index=False)
    plot_performance_by_ticker(by_ticker, output_dir)

    by_group = evaluate_by_group(by_ticker, MARKET_CAP_GROUPS)
    by_group.to_csv(output_dir / "model_comparison_by_group.csv", index=False)
    plot_performance_by_group(by_group, output_dir)

    plot_performance(comparison, output_dir)
    plot_roc_curves(roc_data, output_dir)
    plot_actual_vs_predicted(direction_records, output_dir)
    print(f"\nSaved {len(comparison)}-row comparison table and figures to: {output_dir}")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nPer-ticker breakdown ({len(by_ticker)} rows) saved to model_comparison_by_ticker.csv")
    print(by_ticker.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nLarge Cap (LLOY/SBRY/RR) vs Small Cap (ASC/CPI/CRST) saved to model_comparison_by_group.csv")
    print(by_group.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    return comparison


def plot_before_after(combined_comparison: pd.DataFrame, output_dir: Path) -> None:
    """Accuracy before vs after adding the behavioral features, one panel per model."""
    if combined_comparison.empty:
        return
    models = list(dict.fromkeys(combined_comparison["Model"]))
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4.5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, model in zip(axes, models):
        subset = combined_comparison.loc[combined_comparison["Model"].eq(model)]
        sns.barplot(data=subset, x="Horizon", y="Accuracy", hue="Feature_Set", ax=ax)
        ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
        ax.set_title(model)
        ax.set_ylim(0, 1)
        if ax is not axes[0]:
            ax.legend_.remove()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)
    fig.suptitle("Accuracy before vs after adding behavioral features (sentiment + SUE)")
    fig.tight_layout(rect=(0, 0.09, 1, 0.93))
    fig.savefig(output_dir / "accuracy_before_after.png", dpi=160)
    plt.close(fig)


def run_ablation(data_directory: Path | None, output_dir: Path, epochs: int,
                  panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Run the full experiment twice on the identical loaded panel -- once
    with every feature, once with the two behavioral features (sentiment,
    SUE) removed -- so any accuracy difference is attributable to those two
    features alone, not to a different sample, date range, or train/test
    split. Both runs use the exact same rows: load_and_validate_data (and
    its missing-value handling) only runs once, up front.

    ``panel`` lets a caller (e.g. run_by_group) pass an already-loaded,
    already-cleaned sub-panel directly -- e.g. just the Large Cap tickers --
    instead of re-reading and re-cleaning every ticker's CSV from disk for
    each group. When omitted, ``data_directory`` is loaded as before.

    Each run's full output (confusion matrices, ROC curves, per-ticker and
    per-group breakdowns, etc.) is written to its own subfolder --
    'with_behavioral' and 'without_behavioral' -- and a combined
    before/after comparison is written at the top level of ``output_dir``.
    """
    if panel is None:
        if data_directory is None:
            raise ValueError("A data directory or prepared panel is required.")
        panel = load_and_validate_data(data_directory)
    no_behavioral_features = [f for f in FEATURES if f not in BEHAVIORAL_FEATURES]
    feature_sets = {
        "With Behavioral": FEATURES,
        "Without Behavioral": no_behavioral_features,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    combined = {"comparison": [], "by_ticker": [], "by_group": []}
    for label, features in feature_sets.items():
        run_dir = output_dir / label.lower().replace(" ", "_")
        print(f"\n{'#' * 72}\nFeature set: {label} -> {features}\n{'#' * 72}")
        run_experiments(data_directory=None, output_dir=run_dir, epochs=epochs, panel=panel, features=features)
        combined["comparison"].append(pd.read_csv(run_dir / "model_comparison_results.csv").assign(Feature_Set=label))
        combined["by_ticker"].append(pd.read_csv(run_dir / "model_comparison_by_ticker.csv").assign(Feature_Set=label))
        combined["by_group"].append(pd.read_csv(run_dir / "model_comparison_by_group.csv").assign(Feature_Set=label))

    combined_comparison = pd.concat(combined["comparison"], ignore_index=True)
    combined_by_ticker = pd.concat(combined["by_ticker"], ignore_index=True)
    combined_by_group = pd.concat(combined["by_group"], ignore_index=True)
    combined_comparison.to_csv(output_dir / "before_after_comparison.csv", index=False)
    combined_by_ticker.to_csv(output_dir / "before_after_by_ticker.csv", index=False)
    combined_by_group.to_csv(output_dir / "before_after_by_group.csv", index=False)
    plot_before_after(combined_comparison, output_dir)

    print(f"\n{'=' * 72}\nBefore/after comparison ({len(combined_comparison)} rows) saved to "
          f"{output_dir / 'before_after_comparison.csv'}")
    print(combined_comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    return combined_comparison


def run_by_group(data_directory: Path, output_dir: Path, epochs: int,
                  group_map: dict[str, str] = MARKET_CAP_GROUPS,
                  mode: str = "ablation") -> dict[str, pd.DataFrame]:
    """Train fully separate models per market-cap group instead of one pooled
    model across every ticker.

    This is genuinely different from model_comparison_by_group.csv (produced
    by evaluate_by_group in a normal pooled run): that re-slices ONE shared
    model's predictions by group after the fact, to check whether a single
    model performs consistently across groups. This function instead gives
    each group its own chronological train/test split, its own scaler, and
    its own independently-fit SVM / Logistic Regression / Random Forest /
    LSTM -- Large Cap stocks never influence the Small Cap model's training
    at all, and vice versa.

    The panel is loaded and cleaned once (load_and_validate_data), then
    split into one sub-panel per group (split_panel_by_group) before any
    model is trained. Each group's full output goes to its own subfolder,
    e.g. model_results/large_cap/... and model_results/small_cap/... --
    with the internal layout matching whichever ``mode`` was requested
    (ablation writes with_behavioral/without_behavioral subfolders under
    each group; full/technical-only write flat into the group's folder).

    Returns a dict mapping group name -> that group's comparison DataFrame
    (the same return type run_ablation/run_experiments already produce).
    """
    if mode not in ("ablation", "full", "technical-only"):
        raise ValueError(f"Unknown mode: {mode!r}. Expected 'ablation', 'full', or 'technical-only'.")

    panel = load_and_validate_data(data_directory)
    groups = split_panel_by_group(panel, group_map)
    no_behavioral_features = [f for f in FEATURES if f not in BEHAVIORAL_FEATURES]

    results: dict[str, pd.DataFrame] = {}
    for group_name, group_panel in groups.items():
        group_dir = output_dir / group_name.lower().replace(" ", "_")
        tickers = sorted(group_panel["Ticker"].unique())
        print(f"\n{'#' * 72}\nGroup: {group_name} ({', '.join(tickers)})\n{'#' * 72}")
        if mode == "ablation":
            results[group_name] = run_ablation(data_directory=None, output_dir=group_dir, epochs=epochs, panel=group_panel)
        elif mode == "full":
            results[group_name] = run_experiments(data_directory=None, output_dir=group_dir, epochs=epochs, panel=group_panel, features=FEATURES)
        else:  # technical-only
            results[group_name] = run_experiments(data_directory=None, output_dir=group_dir, epochs=epochs, panel=group_panel, features=no_behavioral_features)

    print(f"\n{'=' * 72}\nTrained {len(results)} separate group model(s): {sorted(results)}. "
          f"See subfolders under {output_dir}.")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run leakage-safe stock direction experiments.")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent,
                        help="Directory containing one subfolder per ticker (e.g. w8_model/SBRY), "
                             "each holding a '<ticker>_combined_features.csv' file.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "model_results",
                        help="Directory for result tables and PNG charts.")
    parser.add_argument("--epochs", type=int, default=30, help="Maximum LSTM epochs (early stopping is enabled).")
    parser.add_argument("--mode", choices=["ablation", "full", "technical-only"], default="ablation",
                        help="'ablation' (default): run both feature sets and write a before/after "
                             "comparison. 'full': run once with every feature. 'technical-only': run "
                             "once with the two behavioral features (daily_compound_sentiment, sue) removed.")
    parser.add_argument("--by-group", action="store_true",
                        help="Train separate models per market-cap group (Large Cap / Small Cap, see "
                             "MARKET_CAP_GROUPS) instead of one pooled model across every ticker. "
                             "Combines with --mode: e.g. --by-group --mode full trains one full-feature "
                             "model per group; the default --mode ablation trains both feature sets per group.")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.by_group:
        run_by_group(arguments.data_dir, arguments.output_dir, arguments.epochs, mode=arguments.mode)
    elif arguments.mode == "ablation":
        run_ablation(arguments.data_dir, arguments.output_dir, arguments.epochs)
    elif arguments.mode == "full":
        run_experiments(arguments.data_dir, arguments.output_dir, arguments.epochs, features=FEATURES)
    else:  # technical-only
        run_experiments(arguments.data_dir, arguments.output_dir, arguments.epochs,
                        features=[f for f in FEATURES if f not in BEHAVIORAL_FEATURES])