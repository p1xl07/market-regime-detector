"""Streamlit interactive dashboard for Market Regime Detector."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.data import DEFAULT_END, DEFAULT_START, DEFAULT_TICKER, fetch_nifty_data
from src.evaluation import evaluate_model
from src.features import build_features
from src.model import RegimeModel, split_chronological
from src.visualization import plot_price_regimes, plot_scatter

st.set_page_config(
    page_title="Market Regime Detector — NIFTY 50",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Market Regime Detector: Indian Equity Market (NIFTY 50)")
st.markdown(
    """
> **Educational & Research Disclaimer:** This application is strictly an educational tool for analyzing historical market states using unsupervised machine learning (KMeans clustering). It is **not** a stock-price prediction model, trading system, or financial advisory tool. Past cluster characteristics do not guarantee future market behavior.
"""
)

# Sidebar controls
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Ticker Symbol", value=DEFAULT_TICKER)
start_date = st.sidebar.date_input("Start Date", pd.to_datetime(DEFAULT_START))
end_date = st.sidebar.date_input("End Date", pd.to_datetime(DEFAULT_END))
train_frac = st.sidebar.slider("Training Set Proportion", 0.5, 0.9, 0.7, 0.05)
n_clusters = st.sidebar.slider("Number of Clusters (K)", 2, 5, 3)

refresh_btn = st.sidebar.button("Fetch / Refresh Data")


@st.cache_data(show_spinner=True)
def load_and_process(tk: str, start: str, end: str, n_c: int, t_frac: float, force_refresh: bool = False):
    df_raw = fetch_nifty_data(ticker=tk, start=start, end=end, use_cache=not force_refresh)
    df_feat = build_features(df_raw)

    train_raw, test_raw = split_chronological(df_feat, train_frac=t_frac)

    model = RegimeModel(n_clusters=n_c)
    model.fit(train_raw)

    train_preds = model.train_df
    test_preds = model.predict(test_raw)

    eval_res = evaluate_model(train_preds, test_preds, model.scaler, n_clusters=n_c)
    return df_raw, train_raw, test_raw, train_preds, test_preds, eval_res, model.scaler


try:
    with st.spinner("Downloading data and running clustering pipeline..."):
        df_raw, train_raw, test_raw, train_df, test_df, eval_res, scaler = load_and_process(
            ticker, str(start_date), str(end_date), n_clusters, train_frac, force_refresh=refresh_btn
        )
except Exception as exc:
    st.error(f"Error loading or processing data: {exc}")
    st.info(
        "If yfinance is blocked or rate-limited, please ensure internet connectivity or check the date range."
    )
    st.stop()

# Expandable Diagnostic Section
with st.expander("🔍 Pipeline Diagnostics & Data Inspector", expanded=False):
    st.markdown("### Step-by-Step Pipeline Diagnostics")
    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        st.metric("Requested Ticker", ticker)
        st.metric("Raw Downloaded Rows", len(df_raw))
    with col_d2:
        st.metric("Train Period Rows", len(train_raw))
        st.metric("Test Period Rows", len(test_raw))
    with col_d3:
        st.metric("Training Feature Rows (Valid)", eval_res["n_train"])
        st.metric("Test Feature Rows (Valid)", eval_res["n_test"])

    st.markdown("#### Missing Values per Feature (Raw Features DataFrame)")
    st.dataframe(train_raw[['close', 'daily_return', 'rolling_20_return', 'rolling_20_vol', 'dist_from_ma50', 'momentum_5']].isnull().sum())

# Tabs for structured navigation
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📊 Overview & Methodology",
        "🔍 Feature Engineering",
        "🤖 ML Model & Evaluation",
        "📈 Visualizations",
        "📝 Summary & Limitations",
    ]
)

with tab1:
    st.header("Project Overview")
    st.markdown(
        f"""
    This project applies **unsupervised machine learning** to historical **NIFTY 50** index data (`{ticker}`) to identify distinct market regimes (e.g., high volatility vs. trending vs. range-bound).

    ### Key Pillars:
    1. **Strict Temporal Isolation:** Chronological train/test split. Preprocessing (`StandardScaler`) and clustering (`KMeans`) are fitted **only** on the training period and applied unmodified to the test period to prevent data leakage.
    2. **Unsupervised Discovery:** KMeans clusters observations based on return, volatility, and distance from moving averages without knowing future outcomes.
    3. **Transparency:** Full evaluation including silhouette scores and baseline comparisons.
    """
    )
    st.metric("Total Trading Days", len(df_raw))
    st.metric("Training Period Rows", eval_res["n_train"])
    st.metric("Test Period Rows", eval_res["n_test"])

with tab2:
    st.header("Feature Engineering")
    st.markdown(
        """
    Features are computed using **past-only data** at each observation date to avoid look-ahead bias:
    - **Daily Return (`daily_return`)**: Percentage change from previous close.
    - **20-Day Return (`rolling_20_return`)**: Cumulative return over 20 trading days.
    - **20-Day Annualized Volatility (`rolling_20_vol`)**: Rolling standard deviation of daily returns multiplied by $\\sqrt{252}$.
    - **Distance from 50-Day MA (`dist_from_ma50`)**: Proportional distance $\\frac{\\text{Close} - \\text{MA}_{50}}{\\text{MA}_{50}}$.
    - **5-Day Momentum (`momentum_5`)**: Short-term directional change.
    """
    )
    st.subheader("Sample Feature Values (Recent)")
    st.dataframe(train_df[["close", "daily_return", "rolling_20_vol", "dist_from_ma50"]].tail(10))

with tab3:
    st.header("Model & Evaluation")
    st.markdown(
        f"""
    - **Algorithm:** KMeans Clustering ($K={n_clusters}$) with `StandardScaler`.
    - **Training Silhouette Score:** `{eval_res['sil_train']}`
    - **Test Silhouette Score:** `{eval_res['sil_test']}`
    - **Random Baseline Silhouette (`random_state=42`):** `{eval_res['sil_baseline']}`

    ### Methodological Note on Cluster Populations Across Splits:
    - **Training vs. Test Profiles:** The training profile displays all $K$ clusters (e.g., Clusters 0, 1, and 2). However, the test profile may display a subset (e.g., only Clusters 0 and 1) with **zero count for Cluster 2**.
    - **Why?** KMeans centroids are fixed from training. If the test period (e.g., 2023–2024) lacked extreme pandemic-style dislocation (such as March–April 2020 which populated Cluster 2), test observations will not fall into Cluster 2. **We never refit on test data or force cluster assignments.** Showing zero-count clusters transparently reflects actual out-of-sample regime distribution.
    - **Rare Extreme Cluster (Cluster 2):** Captures severe outlier market events (e.g., March 2020 COVID crash), characterized by extreme negative 20-day returns ($\\sim -20\\%$), massive annualized volatility ($\\sim 70\\%+$), and deep distance from the 50-day moving average ($\\sim -20\\%$). It represents rare high-stress market anomalies rather than everyday trading regimes.
    """
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Training Cluster Profile (All K Clusters)")
        st.dataframe(eval_res["train_profile"])
    with col2:
        st.subheader("Test Cluster Profile (Out-of-Sample Assignment)")
        st.dataframe(eval_res["test_profile"])

with tab4:
    st.header("Visualizations")
    st.subheader("Historical NIFTY 50 Price with Test Regimes")
    fig_price = plot_price_regimes(df_raw, train_df, test_df)
    st.pyplot(fig_price)

    st.subheader("Return vs Volatility Scatter Plot (Test Period)")
    fig_scatter = plot_scatter(test_df)
    st.pyplot(fig_scatter)

with tab5:
    st.header("Summary, Limitations & Educational Takeaways")
    st.markdown(
        """
    ### What We Learned
    - **Unsupervised grouping** highlights that markets transition between distinct volatility and return states.
    - **Cluster IDs are arbitrary**: Cluster 0 in one run may correspond to Cluster 2 in another. Always interpret clusters via their statistical profile (mean return and volatility) rather than assuming fixed labels.
    - **No predictive power**: Knowing the current regime does not predict whether tomorrow will go up or down.

    ### Limitations & Risks
    - **Lookback sensitivity**: Results change depending on the chosen historical window and hyperparameters.
    - **Stationarity assumption**: Market dynamics shift over structural timeframes; historical patterns may not recur.
    """
    )
