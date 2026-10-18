"""Tests for stability, alignment, and forward-return evaluation."""
import numpy as np
import pandas as pd
import pytest
from src.evaluation import (
    chronological_stability, forward_return_analysis,
    align_cluster_labels, profile_clusters,
)
from src.features import build_features
from src.model import RegimeModel, split_chronological


@pytest.fixture
def price_df():
    dates = pd.date_range("2020-01-01", periods=80, freq="B")
    prices = 100 + np.cumsum(np.random.default_rng(9).normal(0, 0.5, 80))
    df = pd.DataFrame({"close": prices}, index=dates)
    return build_features(df)


def test_chronological_stability_runs(price_df):
    result = chronological_stability(price_df, n_clusters=3, n_splits=2)
    assert isinstance(result, list)
    for r in result:
        if r["status"] == "ok":
            assert "test_profile" in r
            assert "silhouette" in r


def test_forward_return_analysis(price_df):
    # Forward return uses future close prices as evaluation only; clusters assigned externally
    price_df = price_df.copy()
    price_df["cluster"] = 0  # dummy assignment for all rows
    fwd = forward_return_analysis(price_df, horizon_5=5, horizon_20=20)
    assert isinstance(fwd, pd.DataFrame)
    assert "fwd_return_5_mean" in fwd.columns
    assert isinstance(fwd, pd.DataFrame)
    assert len(fwd) > 0
    # Should contain forward columns
    assert "fwd_return_5_mean" in fwd.columns


def test_align_cluster_labels(price_df):
    train_df, test_df = split_chronological(price_df, 0.7)
    model = RegimeModel(n_clusters=3)
    model.fit(train_df)
    test_pred = model.predict(test_df)
    # Align test predictions to train reference
    aligned = align_cluster_labels(model.train_df, test_pred.dropna(subset=["cluster"]), n_clusters=3)
    assert "cluster_aligned" in aligned.columns
    # Aligned clusters should be within 0..2
    assert aligned["cluster_aligned"].dropna().min() >= 0
    assert aligned["cluster_aligned"].dropna().max() < 3
