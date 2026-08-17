"""Streamlit dashboard for the FTSE stock-movement prediction project.

Run with: ``streamlit run app.py``.

Two ways to view results:

1. Default (no upload): browse whatever ``python train_model.py`` already
   computed for the project's local tickers. Works with either the default
   ablation-mode output (``model_results/with_behavioral`` /
   ``model_results/without_behavioral``) or a flat ``--mode full`` /
   ``--mode technical-only`` output.
2. Upload: drop in one or more ``*_combined_features.csv`` (or OHLCV) files
   -- one file per ticker -- pick a historical time window, then click
   "Run uploaded-data prediction". That single click trains every model at
   every horizon for BOTH feature sets (with and without the behavioral
   features) on the uploaded data. After that, the ticker / model / horizon
   / feature-set dropdowns just filter the already-computed results -- no
   retraining needed until you change the time period and click Run again.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve

from train_model import (
    BEHAVIORAL_FEATURES,
    COMBINED_SUFFIX,
    FEATURES,
    HORIZONS as MODEL_HORIZONS,
    prepare_uploaded_data,
    run_experiments,
)

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "model_results"
# Built from train_model.py's own HORIZONS tuple so the dropdown can never
# offer a horizon (e.g. a stray "90-day") that was never actually trained.
HORIZONS = [f"{h}-day" for h in MODEL_HORIZONS]
MODELS = ["Logistic Regression", "SVM", "Random Forest", "LSTM"]
FEATURE_SET_OPTIONS = {
    "With Behavioral": list(FEATURES),
    "Without Behavioral": [f for f in FEATURES if f not in BEHAVIORAL_FEATURES],
}
NAVY, RED, GREEN = "#102a43", "#b7193f", "#16803c"
PLOT_CONFIG = {"displayModeBar": False, "responsive": True}

st.set_page_config(page_title="FTSE Stock Movement Prediction", page_icon="📈", layout="wide", initial_sidebar_state="expanded")


@st.cache_data(show_spinner=False)
def discover_stocks() -> list[str]:
    return sorted(p.name.removesuffix(COMBINED_SUFFIX).upper() for p in BASE_DIR.glob(f"*{COMBINED_SUFFIX}"))


@st.cache_data(show_spinner=False)
def load_market_data(ticker: str) -> pd.DataFrame:
    data = pd.read_csv(BASE_DIR / f"{ticker.lower()}{COMBINED_SUFFIX}")
    return clean_market_data(data)


def normalize_date_column(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename a lowercase 'date' header to 'Date' if 'Date' isn't already present.

    The project's own combined-features CSVs use a lowercase 'date' header
    (train_model.py renames it internally); prepare_uploaded_data() and the
    dashboard's own charts both expect 'Date' (capital D).
    """
    if "Date" not in frame.columns and "date" in frame.columns:
        frame = frame.rename(columns={"date": "Date"})
    return frame


def infer_ticker(raw: pd.DataFrame, filename: str) -> str:
    """Best-effort ticker for one uploaded file: a Ticker/ticker column if
    present and single-valued, otherwise the filename with known suffixes
    stripped (matching the project's own '<ticker>_combined_features.csv'
    naming convention).
    """
    for column in ("Ticker", "ticker"):
        if column in raw.columns:
            values = raw[column].dropna().unique()
            if len(values) == 1:
                return str(values[0]).upper().strip()
    stem = Path(filename).stem
    for suffix in ("_combined_features", "_tech_features"):
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem.upper().strip() or "UPLOADED"


def resolve_results_folder(base_folder: Path, feature_set_label: str) -> Path:
    """Point at the with/without-behavioral subfolder produced by ablation
    mode when it exists; otherwise fall back to the folder itself (a flat
    '--mode full' / '--mode technical-only' run, or a single-feature-set
    upload run).
    """
    with_dir = base_folder / "with_behavioral"
    without_dir = base_folder / "without_behavioral"
    if with_dir.exists() or without_dir.exists():
        return without_dir if feature_set_label == "Without Behavioral" else with_dir
    return base_folder


def clean_market_data(data: pd.DataFrame) -> pd.DataFrame:
    data = normalize_date_column(data.copy())
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    for col in ("Close", "Volume"):
        if col not in data:
            data[col] = np.nan
        data[col] = pd.to_numeric(data[col], errors="coerce")
    return data.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def read_results(results_folder: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    folder = Path(results_folder)
    def read(name: str) -> pd.DataFrame:
        path = folder / name
        return pd.read_csv(path) if path.exists() else pd.DataFrame()
    predictions, comparison, importance = read("test_predictions.csv"), read("model_comparison_results.csv"), read("feature_importance.csv")
    if not predictions.empty:
        predictions["Date"] = pd.to_datetime(predictions["Date"], errors="coerce")
        for col in ("Actual", "Predicted", "Up_Probability"):
            predictions[col] = pd.to_numeric(predictions[col], errors="coerce")
        predictions = predictions.dropna(subset=["Date", "Actual", "Predicted", "Up_Probability"])
    return predictions, comparison, importance


def card(title: str, value: str, accent: str = NAVY, detail: str = "") -> None:
    st.markdown(f"""<div class='metric-card' style='border-top-color:{accent}'>
    <div class='metric-label'>{title}</div><div class='metric-value' style='color:{accent}'>{value}</div>
    <div class='metric-detail'>{detail}</div></div>""", unsafe_allow_html=True)

def section(title: str, subtitle: str = "") -> None:
    st.markdown(f"<div class='section-title'>{title}</div><div class='section-subtitle'>{subtitle}</div>", unsafe_allow_html=True)

def chart_layout(fig: go.Figure, title: str, height: int = 360) -> go.Figure:
    fig.update_layout(title=title, height=height, margin=dict(l=12, r=12, t=52, b=18), plot_bgcolor="white", paper_bgcolor="white", font=dict(family="Arial, sans-serif", size=13, color="#334155"), legend=dict(orientation="h", y=1.12, x=0), hovermode="x unified")
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="#edf1f5", zeroline=False)
    return fig

st.markdown("""<style>
.block-container {max-width: 1480px; padding: 1.65rem 2.4rem 3rem;}
[data-testid='stSidebar'] {background:#f6f7f9;} [data-testid='stSidebar'] > div:first-child {padding:0.95rem 1rem;}
.brand-title {font-size:1.45rem; line-height:1.2; font-weight:700; color:#102a43; margin:.8rem 0 .35rem; letter-spacing:.02em;}
.brand-meta {font-size:.83rem; color:#64748b; margin-bottom:1.1rem;}
.sidebar-logo {margin-bottom:.75rem;}
.section-title {font-size:1.42rem; font-weight:700; color:#102a43; margin:1.9rem 0 .05rem;}.section-subtitle {color:#64748b; font-size:.92rem; margin-bottom:.75rem;}
.metric-card {background:#fff; border:1px solid #e5eaf0; border-top:3px solid; border-radius:12px; box-shadow:0 3px 12px rgba(15,23,42,.06); padding:.75rem 1rem .68rem; min-height:94px;}
.metric-label {font-size:.78rem; color:#64748b; font-weight:650; text-transform:uppercase; letter-spacing:.045em;}.metric-value {font-size:1.72rem; font-weight:750; line-height:1.25; margin-top:.18rem;}.metric-detail {font-size:.78rem; color:#94a3b8; min-height:.8rem; margin-top:.08rem;}
.disclaimer {font-size:.82rem; color:#64748b; margin-top:.55rem;}.stPlotlyChart {border:1px solid #eef1f4; border-radius:12px; padding:.2rem;}
</style>""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Sidebar. Widget slots are created up front, in the exact visual order
# requested (Upload -> Run -> Ticker -> Time period -> Model -> Horizon ->
# Feature set), then filled in further down in whatever code order is
# convenient -- st.empty() lets a widget's *position* and its *definition*
# live at different points in the script, which is what makes it possible
# for the Run button (position #2) to depend on the time-period value
# (defined later, position #4) without the visual order being affected.
# --------------------------------------------------------------------------
with st.sidebar:
    logo = BASE_DIR / "unileicester_logo.png"
    if logo.exists(): st.image(str(logo), width=170, caption=None)
    st.markdown("<div class='brand-title'>FTSE STOCK<br>MOVEMENT PREDICTION</div><div class='brand-meta'>Phuong-Thao Tran · DABI 2025–26</div>", unsafe_allow_html=True)
    uploads = st.file_uploader(
        "Upload CSV(s)", type=["csv"], accept_multiple_files=True,
        help="Upload one or more project feature CSVs or OHLCV CSVs -- one file per ticker.",
    )
    run_slot = st.empty()
    ticker_slot = st.empty()
    date_slot = st.empty()
    model_slot = st.empty()
    horizon_slot = st.empty()
    featureset_slot = st.empty()
    st.caption(
        "Ticker, model, horizon, and feature set only change what's displayed. "
        "The time period changes the underlying training data - click Run again after adjusting it."
    )

# --------------------------------------------------------------------------
# Parse and validate every uploaded file into one multi-ticker panel.
# --------------------------------------------------------------------------
uploaded_panel = None
if uploads:
    prepared_frames, upload_errors = [], []
    for uploaded_file in uploads:
        try:
            raw = normalize_date_column(pd.read_csv(uploaded_file))
            ticker_name = infer_ticker(raw, uploaded_file.name)
            prepared_frames.append(prepare_uploaded_data(raw, ticker_name))
        except (ValueError, pd.errors.ParserError) as exc:
            upload_errors.append(f"{uploaded_file.name}: {exc}")
    if upload_errors:
        st.sidebar.error("Some files failed validation:\n" + "\n".join(f"- {msg}" for msg in upload_errors))
    if prepared_frames:
        combined = pd.concat(prepared_frames, ignore_index=True)
        duplicate_mask = combined.duplicated(subset=["Ticker", "Date"])
        if duplicate_mask.any():
            dupes = sorted(combined.loc[duplicate_mask, "Ticker"].unique())
            st.sidebar.error(
                f"Duplicate (Ticker, Date) rows for: {dupes}. Two uploaded files may map to the same ticker -- "
                "check filenames or add a Ticker column to disambiguate."
            )
        else:
            uploaded_panel = combined.sort_values(["Date", "Ticker"]).reset_index(drop=True)
            st.sidebar.success(
                f"Validated {len(uploaded_panel):,} rows across {uploaded_panel['Ticker'].nunique()} "
                f"ticker(s): {', '.join(sorted(uploaded_panel['Ticker'].unique()))}."
            )

using_uploaded = uploaded_panel is not None

# --- Ticker: only uploaded tickers once something's been uploaded --------
available_tickers = sorted(uploaded_panel["Ticker"].unique()) if using_uploaded else discover_stocks()
if not available_tickers:
    st.error(f"No *{COMBINED_SUFFIX} files were found beside app.py, and no CSV has been uploaded yet.")
    st.stop()
ticker = ticker_slot.selectbox("Select ticker", available_tickers)

# --- Time period: bounded by the *whole* uploaded panel (all tickers), so
# the window applied at training time is the same for every ticker -------
if using_uploaded:
    min_date, max_date = uploaded_panel["Date"].min().date(), uploaded_panel["Date"].max().date()
    if min_date == max_date:
        date_slot.info(f"Only one date available in the uploaded data: {min_date}.")
        date_range = (min_date, max_date)
    else:
        date_range = date_slot.slider("Select analysed time period", min_value=min_date, max_value=max_date, value=(min_date, max_date))
else:
    date_range = None
    date_slot.caption("Time-period filtering applies to uploaded data only.")

# --- Model / Horizon -------------------------------------------------------
model = model_slot.selectbox("Select prediction model", MODELS, index=1)
horizon = horizon_slot.selectbox("Select prediction horizon", HORIZONS, index=1)

# --- Feature set -------------------------------------------------------
feature_set_label = featureset_slot.selectbox("Select feature set", list(FEATURE_SET_OPTIONS.keys()))

# --- Run button (rendered into the slot reserved earlier, right after Upload) ---
results_folder = RESULTS_DIR
if using_uploaded:
    digest_source = b"".join(sorted(f.getvalue() for f in uploads))
    digest = sha256(digest_source).hexdigest()[:12]
    range_tag = f"{date_range[0].isoformat()}_{date_range[1].isoformat()}"
    batch_folder = RESULTS_DIR / f"uploaded_{digest}_{range_tag}"
    run_requested = run_slot.button("Run uploaded-data prediction", type="primary", use_container_width=True)
    if run_requested:
        filtered_panel = uploaded_panel.loc[
            uploaded_panel["Date"].between(pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]))
        ].copy()
        if filtered_panel.empty:
            st.sidebar.error("No uploaded rows fall inside the selected time period.")
        else:
            with st.spinner("Preprocessing and evaluating all models, for both feature sets…"):
                run_experiments(None, batch_folder / "with_behavioral", epochs=30, panel=filtered_panel, features=FEATURE_SET_OPTIONS["With Behavioral"])
                run_experiments(None, batch_folder / "without_behavioral", epochs=30, panel=filtered_panel, features=FEATURE_SET_OPTIONS["Without Behavioral"])
            read_results.clear()
            st.sidebar.success("Prediction results are ready - try the feature-set and ticker dropdowns above.")
    results_folder = resolve_results_folder(batch_folder, feature_set_label)
else:
    run_slot.button("Run uploaded-data prediction", disabled=True, use_container_width=True, help="Upload one or more CSVs above first.")
    results_folder = resolve_results_folder(RESULTS_DIR, feature_set_label)

# --------------------------------------------------------------------------
# From here down, `data`, `predictions`, `comparison`, `importance`,
# `ticker`, `model`, and `horizon` are populated the same way regardless of
# which branch above ran -- everything below is unchanged from the
# single-ticker version.
# --------------------------------------------------------------------------
if using_uploaded:
    ticker_rows = uploaded_panel.loc[uploaded_panel["Ticker"].eq(ticker)]
    ticker_rows = ticker_rows.loc[ticker_rows["Date"].between(pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]))]
    data = clean_market_data(ticker_rows)
else:
    data = load_market_data(ticker)
predictions, comparison, importance = read_results(str(results_folder))
returns = data["Close"].pct_change().dropna()

section("Initial Analysis", "Market price behaviour and daily return distribution")
left, right = st.columns(2, gap="large")
with left:
    price = make_subplots(specs=[[{"secondary_y": True}]])
    if data["Volume"].notna().any(): price.add_trace(go.Bar(x=data.Date, y=data.Volume, name="Volume", marker_color="rgba(183,25,63,.25)"), secondary_y=True)
    price.add_trace(go.Scatter(x=data.Date, y=data.Close, name="Close", mode="lines", line=dict(color=NAVY, width=2.5)), secondary_y=False)
    chart_layout(price, "Closing price and trading volume")
    price.update_yaxes(title_text="Close", secondary_y=False); price.update_yaxes(title_text="Volume", tickformat="~s", secondary_y=True)
    st.plotly_chart(price, use_container_width=True, config=PLOT_CONFIG)
with right:
    hist = go.Figure(go.Histogram(x=returns * 100, nbinsx=42, marker_color=RED, opacity=.82))
    chart_layout(hist, "Distribution of daily returns")
    hist.update_xaxes(title="Daily return (%)"); hist.update_yaxes(title="Frequency")
    st.plotly_chart(hist, use_container_width=True, config=PLOT_CONFIG)

section("Summary Statistics", "Descriptive measures from the currently selected dataset")
growth = ((data.Close.iloc[-1] / data.Close.iloc[0]) - 1) * 100 if len(data) > 1 else np.nan
summary = st.columns(3, gap="large")
with summary[0]: card("Price growth", f"{growth:,.2f}%", RED)
with summary[1]: card("Average volume", f"{data.Volume.mean():,.0f}" if data.Volume.notna().any() else "N/A", NAVY)
with summary[2]: card("Standard deviation", f"{returns.std() * 100:,.2f}%", NAVY)

selection = predictions.loc[(predictions.get("Ticker", pd.Series(dtype=str)).eq(ticker)) & (predictions.get("Horizon", pd.Series(dtype=str)).eq(horizon)) & (predictions.get("Model", pd.Series(dtype=str)).eq(model))].sort_values("Date") if not predictions.empty else pd.DataFrame()
latest = "—" if selection.empty else ("UP" if int(selection.iloc[-1].Predicted) else "DOWN")
recommendation = "BUY" if latest == "UP" else "SELL" if latest == "DOWN" else "—"
accuracy = "—" if selection.empty else f"{accuracy_score(selection.Actual, selection.Predicted) * 100:.1f}%"
auc = "—" if selection.empty else (f"{roc_auc_score(selection.Actual, selection.Up_Probability) * 100:.1f}%" if selection.Actual.nunique() == 2 else "N/A")
section("Prediction", f"{model} · {horizon} horizon · {feature_set_label}")
prediction_cards = st.columns(4, gap="large")
with prediction_cards[0]: card("Direction", latest, GREEN if latest == "UP" else RED if latest == "DOWN" else NAVY)
with prediction_cards[1]: card("Recommendation", recommendation, GREEN if recommendation == "BUY" else RED if recommendation == "SELL" else NAVY)
with prediction_cards[2]: card("Accuracy", accuracy)
with prediction_cards[3]: card("ROC-AUC", auc)
st.markdown("<div class='disclaimer'>This recommendation is generated from the selected machine learning model and should not be considered financial advice.</div>", unsafe_allow_html=True)

if selection.empty:
    st.info("No results are available for this selection. Run `python train_model.py` for the default data, or use the uploaded-data prediction button.")
else:
    confusion_col, roc_col = st.columns(2, gap="large")
    with confusion_col:
        matrix = confusion_matrix(selection.Actual, selection.Predicted, labels=[0, 1])
        fig = go.Figure(go.Heatmap(z=matrix, x=["Down", "Up"], y=["Down", "Up"], text=matrix, texttemplate="%{text}", colorscale=[[0, "#f1f5f9"], [1, RED]], showscale=False))
        chart_layout(fig, "Confusion matrix (test period)"); fig.update_xaxes(title="Predicted direction"); fig.update_yaxes(title="Actual direction", autorange="reversed")
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)
    with roc_col:
        roc = go.Figure()
        if selection.Actual.nunique() == 2:
            fpr, tpr, _ = roc_curve(selection.Actual, selection.Up_Probability)
            roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{model} ({auc})", line=dict(color=RED, width=2.5)))
        roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random classifier", line=dict(color="#94a3b8", dash="dash")))
        chart_layout(roc, "ROC curve (test period)"); roc.update_xaxes(title="False positive rate", range=[0, 1]); roc.update_yaxes(title="True positive rate", range=[0, 1])
        st.plotly_chart(roc, use_container_width=True, config=PLOT_CONFIG)

section("Model Comparison", f"All models for {ticker} at the {horizon} horizon")
# Calculate comparison metrics from the selected ticker's saved test rows.  The
# training summary is panel-level, whereas this view must respect the ticker.
comparison_rows = []
all_model_rows = predictions.loc[(predictions.get("Ticker", pd.Series(dtype=str)).eq(ticker)) & (predictions.get("Horizon", pd.Series(dtype=str)).eq(horizon))] if not predictions.empty else pd.DataFrame()
for name in MODELS:
    model_rows = all_model_rows.loc[all_model_rows.Model.eq(name)] if not all_model_rows.empty else pd.DataFrame()
    if model_rows.empty:
        continue
    comparison_rows.append({
        "Model": name,
        "Accuracy": accuracy_score(model_rows.Actual, model_rows.Predicted),
        "Precision": precision_score(model_rows.Actual, model_rows.Predicted, zero_division=0),
        "Recall": recall_score(model_rows.Actual, model_rows.Predicted, zero_division=0),
        "F1": f1_score(model_rows.Actual, model_rows.Predicted, zero_division=0),
        "ROC_AUC": roc_auc_score(model_rows.Actual, model_rows.Up_Probability) if model_rows.Actual.nunique() == 2 else np.nan,
    })
comparison_view = pd.DataFrame(comparison_rows)
if comparison_view.empty:
    st.info("Model comparison will appear once prediction results are available.")
else:
    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
    best_model = comparison_view.loc[comparison_view["Accuracy"].idxmax(), "Model"]
    long = comparison_view.melt(id_vars="Model", value_vars=metrics, var_name="Metric", value_name="Score")
    long["Metric"] = long["Metric"].replace({"ROC_AUC": "ROC-AUC", "F1": "F1-score"})
    bars = go.Figure()
    for metric in long.Metric.unique():
        subset = long[long.Metric.eq(metric)]
        bars.add_trace(go.Bar(name=metric, x=subset.Model, y=subset.Score, text=[f"{x:.1%}" for x in subset.Score], textposition="outside"))
    chart_layout(bars, f"Model performance — best accuracy: {best_model}", 410)
    bars.update_layout(barmode="group", title=dict(text=f"Model performance — best accuracy: {best_model}", x=0, xanchor="left", y=0.96, yanchor="top"), margin=dict(l=12, r=12, t=100, b=18), legend=dict(orientation="h", y=1.12, x=0, xanchor="left"))
    bars.update_yaxes(title="Score", range=[0, 1.12], tickformat=".0%")
    st.plotly_chart(bars, use_container_width=True, config=PLOT_CONFIG)
    table = comparison_view[["Model", *metrics]].copy().sort_values("Accuracy", ascending=False)
    table = table.rename(columns={"ROC_AUC": "ROC-AUC", "F1": "F1-score"})
    st.dataframe(table.style.format({column: "{:.2%}" for column in table.columns if column != "Model"}).highlight_max(subset=["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"], color="#dff4e5"), use_container_width=True, hide_index=True)

    # Ticker comparison section
    ticker_section_label = f"{model} across all available tickers at the {horizon} horizon"
    section("Ticker Comparison", ticker_section_label)
    ticker_rows_pred = predictions.loc[(predictions.get("Model", pd.Series(dtype=str)).eq(model)) & (predictions.get("Horizon", pd.Series(dtype=str)).eq(horizon))]
    all_tickers = sorted(predictions["Ticker"].dropna().unique()) if not predictions.empty else []
    if ticker_rows_pred.empty:
        st.info("Ticker comparison is unavailable because no saved results exist for the selected model and horizon.")
    else:
        ticker_metrics = []
        for ticker_name in all_tickers:
            ticker_subset = ticker_rows_pred.loc[ticker_rows_pred["Ticker"].eq(ticker_name)]
            if ticker_subset.empty:
                ticker_metrics.append({"Ticker": ticker_name, "Predicted movement": "N/A", "Accuracy": np.nan})
                continue
            ticker_metrics.append({
                "Ticker": ticker_name,
                "Predicted movement": "UP" if int(ticker_subset.iloc[-1].Predicted) == 1 else "DOWN",
                "Accuracy": accuracy_score(ticker_subset.Actual, ticker_subset.Predicted),
            })
        ticker_comparison_view = pd.DataFrame(ticker_metrics)
        valid_tickers = ticker_comparison_view.dropna(subset=["Accuracy"])
        total_tickers = len(all_tickers)
        valid_count = len(valid_tickers)
        if total_tickers and valid_count < total_tickers:
            st.markdown(f"<div style='color:#64748b; margin-bottom:0.75rem;'>Comparison based on {valid_count} of {total_tickers} available tickers with valid evaluation results.</div>", unsafe_allow_html=True)
        if valid_tickers.empty:
            st.info("No ticker-level results are available for this model and horizon.")
        else:
            best_ticker = valid_tickers.loc[valid_tickers["Accuracy"].idxmax(), "Ticker"]
            ticker_bars = go.Figure(go.Bar(
                x=valid_tickers.Ticker,
                y=valid_tickers.Accuracy,
                marker_color=[GREEN if name == best_ticker else NAVY for name in valid_tickers.Ticker],
                text=[f"{value:.1%}" for value in valid_tickers.Accuracy],
                textposition="outside",
                hovertemplate="Ticker: %{x}<br>Accuracy: %{y:.1%}<extra></extra>",
            ))
            chart_layout(ticker_bars, f"Ticker performance — best accuracy: {best_ticker}", 430)
            ticker_bars.update_yaxes(title="Accuracy", range=[0, 1.12], tickformat=".0%")
            st.plotly_chart(ticker_bars, use_container_width=True, config=PLOT_CONFIG)
            ticker_table = ticker_comparison_view[["Ticker", "Predicted movement", "Accuracy"]].copy()
            st.dataframe(ticker_table.style.format({"Accuracy": "{:.2%}"}).highlight_max(subset=["Accuracy"], color="#dff4e5"), use_container_width=True, hide_index=True)

if not importance.empty:
    feature_view = importance.loc[importance.Horizon.eq(horizon)].sort_values("Importance")
    if not feature_view.empty:
        section("Feature Importance", "Random Forest feature contribution for the selected horizon")
        feature_chart = go.Figure(go.Bar(x=feature_view.Importance, y=feature_view.Feature, orientation="h", marker_color=NAVY))
        chart_layout(feature_chart, "Random Forest feature importance", 300); feature_chart.update_xaxes(title="Importance")
        st.plotly_chart(feature_chart, use_container_width=True, config=PLOT_CONFIG)