"""Tests for data collection."""
import numpy as np
import pandas as pd
import pytest
from src.data import _validate_data, fetch_nifty_data, DEFAULT_TICKER, MIN_OBSERVATIONS


def test_validate_rejects_duplicates(tmp_path):
    dates = pd.date_range("2020-01-01", periods=MIN_OBSERVATIONS, freq="B").tolist()
    dates[1] = dates[0]  # duplicate
    df = pd.DataFrame({"close": np.ones(MIN_OBSERVATIONS) * 100}, index=dates)
    with pytest.raises(ValueError, match="Duplicate"):
        _validate_data(df)


def test_validate_rejects_nonpositive():
    dates = pd.date_range("2020-01-01", periods=MIN_OBSERVATIONS, freq="B")
    prices = np.ones(MIN_OBSERVATIONS) * 100
    prices[5] = 0
    df = pd.DataFrame({"close": prices}, index=dates)
    with pytest.raises(ValueError, match="Non-positive"):
        _validate_data(df)


def test_validate_rejects_insufficient_rows():
    dates = pd.date_range("2020-01-01", periods=20, freq="B")
    df = pd.DataFrame({"close": np.ones(20) * 100}, index=dates)
    with pytest.raises(ValueError, match="At least 60 trading days are required"):
        _validate_data(df)


def test_regression_multiindex_columns(monkeypatch, tmp_path):
    """Regression test for MultiIndex columns from yfinance."""
    dates = pd.date_range("2020-01-01", periods=MIN_OBSERVATIONS + 10, freq="B")
    columns = pd.MultiIndex.from_tuples(
        [
            ("Close", "^NSEI"),
            ("High", "^NSEI"),
            ("Low", "^NSEI"),
            ("Open", "^NSEI"),
            ("Volume", "^NSEI"),
        ],
        names=["Price", "Ticker"],
    )
    data = np.ones((len(dates), len(columns))) * 100.0
    mock_df = pd.DataFrame(data, index=dates, columns=columns)

    import yfinance as yf
    import src.data as data_mod

    # Redirect cache file to temporary directory so tests don't poison project cache
    monkeypatch.setattr(data_mod, "CACHE_FILE", tmp_path / "test_cache.csv")
    monkeypatch.setattr(yf, "download", lambda *args, **kwargs: mock_df)

    df = fetch_nifty_data(ticker="^NSEI", use_cache=False)

    assert isinstance(df, pd.DataFrame)
    assert df.columns.tolist() == ["close"]
    assert len(df) == MIN_OBSERVATIONS + 10
    assert (df["close"] == 100.0).all()


def test_blank_ticker_fallback(monkeypatch, tmp_path):
    """Ensure passing empty/blank ticker falls back to default without crashing."""
    dates = pd.date_range("2020-01-01", periods=MIN_OBSERVATIONS + 10, freq="B")
    mock_df = pd.DataFrame({"Close": np.ones(len(dates)) * 100.0}, index=dates)

    import yfinance as yf
    import src.data as data_mod

    monkeypatch.setattr(data_mod, "CACHE_FILE", tmp_path / "test_cache.csv")
    called_ticker = []

    def mock_download(ticker, *args, **kwargs):
        called_ticker.append(ticker)
        return mock_df

    monkeypatch.setattr(yf, "download", mock_download)

    df = fetch_nifty_data(ticker="   ", use_cache=False)
    assert called_ticker[0] == DEFAULT_TICKER
    assert len(df) == MIN_OBSERVATIONS + 10
