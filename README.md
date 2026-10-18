# Market Regime Detector: Indian Equity Market (NIFTY 50)

An educational and research-oriented machine learning project built for Indian equity market analysis using historical NIFTY 50 index data.

## Problem Statement & Motivation

Financial markets undergo distinct behavioral phases—such as calm trending bull markets, high-volatility corrections, and range-bound consolidations. Traditional forecasting models often fail because they assume stationary market dynamics. 

This project explores **unsupervised machine learning (KMeans clustering)** to group historical market states based on price action, volatility, and momentum without requiring future labels or predictive claims. It serves as a pedagogical template for students and practitioners learning rigorous ML pipelines and time-series feature engineering.

---

## Features & Technology Stack

- **Python** (3.10+)
- **pandas & NumPy** for data manipulation and vectorized calculations
- **yfinance** for historical NIFTY 50 (`^NSEI`) data retrieval
- **scikit-learn** for `StandardScaler` and `KMeans` clustering
- **matplotlib** for static visualizations
- **Streamlit** for the interactive dashboard
- **pytest** for automated testing

---

## Repository Structure

```text
market-regime-detector/
├── README.md
├── requirements.txt
├── .gitignore
├── app.py
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── features.py
│   ├── model.py
│   ├── evaluation.py
│   └── visualization.py
└── tests/
    ├── test_data.py
    ├── test_features.py
    ├── test_model.py
    └── test_evaluation.py
```

---

## Installation & Setup (Windows PowerShell)

1. **Clone or open the repository folder:**
   ```powershell
   cd C:\Users\PARAM\Desktop\market-regime-detector
   ```

2. **Create and activate a Python virtual environment:**
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

---

## Running Tests & Launching the Dashboard

### 1. Run Automated Tests
```powershell
.venv\Scripts\python.exe -m pytest tests/ -v
```

### 2. Launch Streamlit Dashboard
```powershell
.venv\Scripts\streamlit.exe run app.py
```

---

## Feature Definitions & Formulas

All features use **past-only data** at each observation $t$:
1. **Daily Return**: $R_t = \frac{P_t - P_{t-1}}{P_{t-1}}$
2. **20-Day Cumulative Return**: $R_{t,20} = \frac{P_t - P_{t-20}}{P_{t-20}}$
3. **20-Day Annualized Volatility**: $\sigma_{t,20} = \text{std}(R_{t-19:t}) \times \sqrt{252}$
4. **Distance from 50-Day Moving Average**: $D_{t,50} = \frac{P_t - \text{MA}_{50,t}}{\text{MA}_{50,t}}$

---

## Train/Test Methodology & Leakage Prevention

- **Chronological Split**: Data is split chronologically (e.g., first 70% for training, remaining 30% for testing) to prevent time-series data leakage.
- **Preprocessing Isolation**: `StandardScaler` is fitted **strictly on the training set**. The test set is transformed using training parameters.
- **KMeans Unsupervised Clustering**: The model learns cluster centroids from training features only and assigns test points to the nearest existing centroid without refitting.

---

## Methodological Insights & Audit Findings

### 1. Missing Test Clusters (Zero-Count Display)
- **Observation**: Training profile displays all $K$ clusters (e.g., 0, 1, 2), while the test profile may show only clusters 0 and 1 (with 0 count for cluster 2).
- **Explanation**: KMeans test assignment maps observations to fixed training centroids. If out-of-sample periods lack severe crisis events (e.g., March–April 2020 COVID crash), test observations do not map to Cluster 2. **We never refit on test data or force cluster assignments.** Showing zero-count clusters transparently reflects actual out-of-sample regime distributions.

### 2. Rare Extreme Cluster (Cluster 2)
- **Characteristics**: ~2.2% of training observations (concentrated in March–April 2020), mean 20-day return $\sim -20.4\%$, annualized volatility $\sim 70.5\%$, and distance from 50-day MA $\sim -20.5\%$.
- **Interpretation**: This captures extreme pandemic-style market dislocating shocks. It is an outlier/stress regime rather than a regular market phase.

### 3. Evaluation & Silhouette Scores
- **Metric**: Evaluated on scaled feature matrices (`StandardScaler.transform(X)`) using assigned cluster labels.
- **What it establishes**: Relative cohesion and separation of clusters in feature space compared to random baselines.
- **What it does not establish**: It does not prove predictive accuracy or financial profitability.

---

## Disclaimer

This is a research and educational project, **not** financial advice or a trading system.
