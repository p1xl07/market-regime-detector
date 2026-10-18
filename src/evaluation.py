"""Regime interpretation and evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score


def profile_clusters(df: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
    """Mean statistics per cluster, ensuring all cluster IDs (0 to n_clusters-1) are represented."""
    # Ensure cluster column exists and is integer where possible
    if "cluster" not in df.columns:
        df = df.copy()
        df["cluster"] = 0

    # Group by cluster
    groups = df.groupby("cluster")
    profile = groups[["rolling_20_return", "rolling_20_vol", "dist_from_ma50"]].mean()
    profile["count"] = groups.size()
    profile["pct"] = profile["count"] / profile["count"].sum() * 100

    # Reindex to include all expected cluster IDs (0, 1, ..., n_clusters-1)
    all_clusters = list(range(n_clusters))
    profile = profile.reindex(all_clusters)

    # Fill NaN counts/pct with 0 for unassigned/missing clusters in test period
    profile["count"] = profile["count"].fillna(0).astype(int)
    profile["pct"] = profile["pct"].fillna(0.0)

    return profile.round(4)


def safe_silhouette(X: np.ndarray, labels: np.ndarray) -> float | None:
    """Silhouette only when >=2 distinct labels and valid."""
    unique = np.unique(labels)
    if len(unique) < 2 or len(unique) >= len(labels) - 1:
        return None
    try:
        return float(silhouette_score(X, labels))
    except Exception:
        return None


def evaluate_model(train_df: pd.DataFrame, test_df: pd.DataFrame, scaler, n_clusters: int = 3) -> dict:
    """Return metrics and interpretations."""
    # Profile ensuring full cluster spectrum
    train_prof = profile_clusters(train_df.dropna(subset=["cluster"]), n_clusters=n_clusters)
    test_prof = profile_clusters(test_df.dropna(subset=["cluster"]), n_clusters=n_clusters)

    # Silhouette on training scaled data
    feature_cols = [
        "daily_return",
        "rolling_20_return",
        "rolling_20_vol",
        "dist_from_ma50",
        "momentum_5",
    ]
    X_train = train_df[feature_cols].dropna()
    y_train = train_df.loc[X_train.index, "cluster"]
    sil_train = safe_silhouette(scaler.transform(X_train.values), y_train.values)

    X_test = test_df[feature_cols].dropna()
    y_test = test_df.loc[X_test.index, "cluster"]
    sil_test = safe_silhouette(scaler.transform(X_test.values), y_test.values) if len(y_test) >= 2 else None

    # Baseline comparison: random stratified/uniform clustering on same train data (reproducible with random_state=42)
    rng = np.random.default_rng(42)
    baseline_labels = rng.integers(0, n_clusters, size=len(y_train))
    sil_baseline = safe_silhouette(scaler.transform(X_train.values), baseline_labels)

    return {
        "train_profile": train_prof,
        "test_profile": test_prof,
        "sil_train": sil_train,
        "sil_test": sil_test,
        "sil_baseline": sil_baseline,
        "n_train": len(train_df),
        "n_test": len(test_df),
    }
