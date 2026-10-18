"""Regime interpretation, stability, and forward-return evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def profile_clusters(df: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
    if "cluster" not in df.columns:
        df = df.copy()
        df["cluster"] = 0
    groups = df.groupby("cluster")
    profile = groups[["rolling_20_return", "rolling_20_vol", "dist_from_ma50"]].mean()
    profile["count"] = groups.size()
    profile["pct"] = profile["count"] / profile["count"].sum() * 100
    all_clusters = list(range(n_clusters))
    profile = profile.reindex(all_clusters)
    profile["count"] = profile["count"].fillna(0).astype(int)
    profile["pct"] = profile["pct"].fillna(0.0)
    return profile.round(4)


def safe_silhouette(X: np.ndarray, labels: np.ndarray) -> float | None:
    unique = np.unique(labels)
    if len(unique) < 2 or len(unique) >= len(labels) - 1:
        return None
    try:
        return float(silhouette_score(X, labels))
    except Exception:
        return None


def evaluate_model(train_df: pd.DataFrame, test_df: pd.DataFrame, scaler, n_clusters: int = 3) -> dict:
    train_prof = profile_clusters(train_df.dropna(subset=["cluster"]), n_clusters=n_clusters)
    test_prof = profile_clusters(test_df.dropna(subset=["cluster"]), n_clusters=n_clusters)
    feature_cols = [
        "daily_return", "rolling_20_return", "rolling_20_vol", "dist_from_ma50", "momentum_5"
    ]
    X_train = train_df[feature_cols].dropna()
    y_train = train_df.loc[X_train.index, "cluster"]
    sil_train = safe_silhouette(scaler.transform(X_train.values), y_train.values)
    X_test = test_df[feature_cols].dropna()
    y_test = test_df.loc[X_test.index, "cluster"]
    sil_test = safe_silhouette(scaler.transform(X_test.values), y_test.values) if len(y_test) >= 2 else None
    rng = np.random.default_rng(42)
    baseline_labels = rng.integers(0, n_clusters, size=len(y_train))
    sil_baseline = safe_silhouette(scaler.transform(X_train.values), baseline_labels)
    return {
        "train_profile": train_prof, "test_profile": test_prof,
        "sil_train": sil_train, "sil_test": sil_test, "sil_baseline": sil_baseline,
        "n_train": len(train_df), "n_test": len(test_df),
    }


def align_cluster_labels(ref_df: pd.DataFrame, target_df: pd.DataFrame, feature_cols=None, n_clusters: int = 3) -> pd.DataFrame:
    """Align target cluster IDs to reference using feature-centroid matching."""
    if feature_cols is None:
        feature_cols = ["daily_return", "rolling_20_return", "rolling_20_vol", "dist_from_ma50", "momentum_5"]
    # Compute centroids from reference clusters
    ref_centroids = ref_df.groupby("cluster")[feature_cols].mean()
    # For each target cluster, find nearest ref centroid by Euclidean distance
    mapping = {}
    used_ref = set()
    for c in sorted(target_df["cluster"].dropna().unique().astype(int)):
        c_df = target_df[target_df["cluster"] == c]
        centroid = c_df[feature_cols].mean()
        best_ref = None
        best_dist = float("inf")
        for r in sorted(ref_centroids.index):
            if r in used_ref:
                continue
            d = np.linalg.norm(centroid.values - ref_centroids.loc[r].values)
            if d < best_dist:
                best_dist = d
                best_ref = r
        if best_ref is not None:
            mapping[c] = best_ref
            used_ref.add(best_ref)
    # Apply mapping; leave unmapped as-is (should not happen with same K)
    aligned = target_df.copy()
    aligned["cluster_aligned"] = aligned["cluster"].replace(mapping)
    return aligned


def chronological_stability(df_feat: pd.DataFrame, n_clusters: int = 3, n_splits: int = 3) -> dict:
    """Walk-forward evaluation: model fitted only on past, tested on next window."""
    df_feat = df_feat.copy()
    # Use only complete feature rows for consistent windows
    df_clean = df_feat.dropna(subset=["daily_return","rolling_20_return","rolling_20_vol","dist_from_ma50","momentum_5"])
    total = len(df_clean)
    split_size = total // (n_splits + 1)
    results = []
    for i in range(n_splits):
        train_end = (i + 1) * split_size
        # Train: from start to train_end (past only)
        train_part = df_clean.iloc[:train_end].copy()
        # Test: next window after train_end, fixed size
        test_part = df_clean.iloc[train_end:train_end + split_size].copy()
        if len(train_part) < n_clusters * 2 or len(test_part) < 5:
            results.append({"split": i+1, "train_n": len(train_part), "test_n": len(test_part), "status": "skipped_insufficient"})
            continue
        scaler = StandardScaler()
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        feature_cols = ["daily_return","rolling_20_return","rolling_20_vol","dist_from_ma50","momentum_5"]
        X_train = scaler.fit_transform(train_part[feature_cols].values)
        model.fit(X_train)
        # Apply to test (past-only scaler, fixed model)
        X_test = scaler.transform(test_part[feature_cols].values)
        labels = model.predict(X_test)
        test_part = test_part.copy()
        test_part.loc[test_part.index[:len(X_test)], "cluster"] = labels
        # Profile and silhouette
        test_prof = profile_clusters(test_part.dropna(subset=["cluster"]), n_clusters=n_clusters)
        sil = safe_silhouette(X_test, labels)
        results.append({
            "split": i+1,
            "train_n": len(train_part),
            "test_n": len(test_part),
            "test_profile": test_prof,
            "silhouette": sil,
            "status": "ok",
        })
    return results


def forward_return_analysis(df_feat: pd.DataFrame, horizon_5: int = 5, horizon_20: int = 20) -> pd.DataFrame:
    """Compute forward returns strictly as evaluation outcomes (not inputs). Overlapping windows noted."""
    df = df_feat.copy()
    # Use cluster assignments already present in df (must be fitted before calling)
    # Compute future returns from close price
    df["fwd_5d"] = df["close"].shift(-horizon_5).pct_change(periods=horizon_5)  # incorrect; need proper forward return
    # Correct: future return = (close_t+h - close_t) / close_t
    df["fwd_return_5"] = (df["close"].shift(-horizon_5) - df["close"]) / df["close"]
    df["fwd_return_20"] = (df["close"].shift(-horizon_20) - df["close"]) / df["close"]

    # Group by cluster
    groups = df.dropna(subset=["cluster", "fwd_return_5", "fwd_return_20"]).groupby("cluster")
    summary = groups[["fwd_return_5", "fwd_return_20"]].agg(["mean", "std", "count"])
    summary.columns = [f"{c}_{m}" for c,m in summary.columns]
    summary = summary.reset_index()
    # Note on overlap: observations within rolling windows overlap; sample sizes reflect non-independent observations
    return summary
