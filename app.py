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
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "📊 Overview & Methodology",
        "🔍 Feature Engineering",
        "🤖 ML Model & Evaluation",
        "📈 Visualizations",
        "📝 Summary & Limitations",
        "⚖️ Evaluation & Stability",
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

with tab6:
    st.header("⚖️ Evaluation & Stability")
    st.markdown(
        """
    ### Silhouette Scores — What They Measure (and Don't)
    - **Training Silhouette:** Evaluates how well-separated and compact the training-period clusters are, using the fitted `StandardScaler` and assigned cluster labels.
    - **Test Silhouette:** Evaluates the same on held-out test-period observations assigned to the fixed training centroids (no refitting).
    - **Baseline:** Random uniform labels (`random_state=42`) provide a reproducible reference only — not a definitive benchmark of market predictability.
    - **Important:** Silhouette measures cluster cohesion/separation in feature space. It does **not** establish predictive accuracy, profitability, or future market behavior.
    """
    )

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        st.metric("Train Silhouette", f"{eval_res['sil_train']:.3f}" if eval_res.get('sil_train') is not None else "N/A")
    with col_s2:
        st.metric("Test Silhouette", f"{eval_res['sil_test']:.3f}" if eval_res.get('sil_test') is not None else "N/A")
    with col_s3:
        st.metric("Random Baseline", f"{eval_res['sil_baseline']:.3f}" if eval_res.get('sil_baseline') is not None else "N/A")

    st.markdown("---")
    st.subheader("Chronological Walk-Forward Stability")
    st.markdown(
        """
    Each split fits KMeans **only on past data** and evaluates on the immediately following window. No future data leaks into model fitting.
    Small samples and missing clusters in any window are shown explicitly — never fabricated.
    """
    )
    try:
        from src.evaluation import chronological_stability
        from src.features import build_features
        # Build feature DataFrame if not already in session (uses past-only data)
        df_for_stab = df_raw
        try:
            df_for_stab = build_features(df_raw)
        except Exception:
            pass
        stab = chronological_stability(df_for_stab, n_clusters=n_clusters, n_splits=2)
        if stab:
            stab_df = pd.DataFrame([{k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items() if k in ("split", "train_n", "test_n", "status", "silhouette")} for s in stab])
            st.dataframe(stab_df, use_container_width=True)
        else:
            st.info("Stability results not available (insufficient history or empty download).")
    except Exception as e:
        # Expose actual exception to diagnostics while keeping user-facing message concise
        st.info(f"Stability analysis skipped: {e}")
        with st.expander("Diagnostic details"):
            st.code(str(e), language="text")

    st.markdown("---")
    st.subheader("Forward-Return Evaluation (Strictly Outcome-Based)")
    st.markdown(
        """
    - **Future prices used only for evaluation**, never as clustering inputs.
    - **Overlapping windows:** 5-day and 20-day horizons overlap; observations are not independent.
    - **Rare clusters:** Cluster 2 has very small counts; estimates have high uncertainty.
    - **No causality/profit claim:** These show historical association, not predictive power.
    """
    )
    try:
        from src.evaluation import forward_return_analysis, align_cluster_labels
        # Build full evaluation dataset: close prices + assigned clusters
        full_eval = df_raw.copy()
        full_eval["close"] = df_raw["close"]
        train_assigned = train_df if 'train_df' in locals() else None
        test_assigned = test_df if 'test_df' in locals() else None
        # Assign clusters to matching indices; use only complete feature-row indices
        if train_assigned is not None and "cluster" in train_assigned.columns:
            common_train = full_eval.index.intersection(train_assigned.index)
            full_eval.loc[common_train, "cluster"] = train_assigned.loc[common_train, "cluster"].values
        if test_assigned is not None and "cluster" in test_assigned.columns:
            common_test = full_eval.index.intersection(test_assigned.index)
            full_eval.loc[common_test, "cluster"] = test_assigned.loc[common_test, "cluster"].values
        # Forward return uses close prices + cluster assignments; never uses future prices as features
        fwd = forward_return_analysis(full_eval.dropna(subset=["close"]), horizon_5=5, horizon_20=20)
        if not fwd.empty:
            display_cols = [c for c in fwd.columns if "mean" in c or "count" in c or c == "cluster"]
            if display_cols:
                fwd_display = fwd[display_cols].copy()
                fwd_display.columns = [c.replace("_mean", " (mean)").replace("_count", " (count)") for c in fwd_display.columns]
                st.dataframe(fwd_display.round(4), use_container_width=True)
                st.caption("Overlapping 5-day / 20-day windows; sample counts reflect non-independent observations. No causality/profitability claimed.")
            else:
                st.info("Forward-return results available but no display columns selected.")
        else:
            st.info("Forward-return analysis produced empty results (insufficient cluster assignments or missing future prices).")
    except Exception as e:
        st.info(f"Forward-return analysis skipped: {e}")

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
