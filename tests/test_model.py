"""Tests for clustering pipeline."""
import numpy as np
import pandas as pd
import pytest
from src.model import RegimeModel, split_chronological


@pytest.fixture
def dummy_df():
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    prices = np.cumsum(np.random.default_rng(7).normal(0, 0.5, 100)) + 100
    df = pd.DataFrame({"close": prices}, index=dates)
    from src.features import build_features
    df = build_features(df)
    return df


def test_chronological_split(dummy_df):
    train, test = split_chronological(dummy_df, train_frac=0.7)
    assert len(train) == int(len(dummy_df) * 0.7)
    assert train.index[-1] < test.index[0]


def test_no_preprocessing_leakage(dummy_df):
    train, test = split_chronological(dummy_df, 0.7)
    model = RegimeModel(n_clusters=3)
    model.fit(train)
    # Scaler fitted only on train; predict transforms test using fitted scaler
    result = model.predict(test)
    assert "cluster" in result.columns
    # All test predictions should be in valid range
    assert result["cluster"].dropna().min() >= 0
    assert result["cluster"].dropna().max() < 3


def test_output_shape(dummy_df):
    train, test = split_chronological(dummy_df, 0.7)
    model = RegimeModel(n_clusters=3)
    model.fit(train)
    result = model.predict(test)
    # Should not drop additional rows unexpectedly
    assert len(result) == len(test)


def test_insufficient_observations_raises():
    dates = pd.date_range("2020-01-01", periods=2, freq="B")
    df = pd.DataFrame({"close": [100, 101],
                       "daily_return": [0.0, 0.01],
                       "rolling_20_return": [0.0, 0.01],
                       "rolling_20_vol": [0.1, 0.12],
                       "dist_from_ma50": [0.0, 0.01],
                       "momentum_5": [0.0, 0.01]}, index=dates)
    model = RegimeModel(n_clusters=3)
    with pytest.raises(ValueError, match="Not enough non-null observations"):
        model.fit(df)


def test_insufficient_distinct_observations_raises():
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    # 5 rows but only 1 distinct row
    df = pd.DataFrame({"close": [100]*5,
                       "daily_return": [0.0]*5,
                       "rolling_20_return": [0.0]*5,
                       "rolling_20_vol": [0.1]*5,
                       "dist_from_ma50": [0.0]*5,
                       "momentum_5": [0.0]*5}, index=dates)
    model = RegimeModel(n_clusters=3)
    with pytest.raises(ValueError, match="Not enough distinct observations"):
        model.fit(df)
