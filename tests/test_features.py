"""Tests for feature engineering."""
import numpy as np
import pandas as pd
import pytest
from src.features import build_features


@pytest.fixture
def price_series():
    dates = pd.date_range("2020-01-01", periods=60, freq="B")
    prices = 100 + np.cumsum(np.random.default_rng(1).normal(0, 1, 60))
    return pd.DataFrame({"close": prices}, index=dates)


def test_features_shape(price_series):
    df = build_features(price_series)
    assert "daily_return" in df.columns
    assert "rolling_20_vol" in df.columns
    assert len(df) == len(price_series)


def test_warmup_rows_handle(price_series):
    df = build_features(price_series)
    # First day has NaN daily_return; first 20 have NaN rolling_20_return/vol; first 50 have NaN dist_from_ma50
    assert df["daily_return"].isna().sum() >= 1
    assert df["rolling_20_return"].isna().sum() >= 20
    assert df["dist_from_ma50"].isna().sum() >= 49
    # Remaining rows after warm-up are non-null
    complete = df.dropna()
    assert len(complete) > 0


def test_invalid_prices():
    bad = pd.DataFrame({"close": [100, 0, -5]}, index=pd.date_range("2020-01-01", periods=3))
    result = build_features(bad)
    assert result is not None
