"""Tests for evaluation."""
import numpy as np
import pandas as pd
from src.evaluation import safe_silhouette, profile_clusters


def test_degenerate_silhouette():
    # Only one unique label -> should return None
    X = np.random.randn(10, 3)
    labels = np.zeros(10, dtype=int)
    assert safe_silhouette(X, labels) is None


def test_profile_clusters():
    df = pd.DataFrame({
        "cluster": [0, 0, 1, 1],
        "rolling_20_return": [0.1, 0.2, -0.05, -0.01],
        "rolling_20_vol": [0.2, 0.3, 0.5, 0.6],
        "dist_from_ma50": [0.01, 0.02, -0.03, -0.04],
    })
    prof = profile_clusters(df, n_clusters=2)
    assert len(prof) == 2
    assert "pct" in prof.columns


def test_profile_clusters_default_reindex():
    df = pd.DataFrame({
        "cluster": [0, 0, 1, 1],
        "rolling_20_return": [0.1, 0.2, -0.05, -0.01],
        "rolling_20_vol": [0.2, 0.3, 0.5, 0.6],
        "dist_from_ma50": [0.01, 0.02, -0.03, -0.04],
    })
    prof = profile_clusters(df, n_clusters=3)
    assert len(prof) == 3
    # Cluster 2 should have count 0
    assert prof.loc[2, "count"] == 0
