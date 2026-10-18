"""Data collection and validation for NIFTY 50 index."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import yfinance as yf

DEFAULT_TICKER = "^NSEI"
DEFAULT_START = "2019-01-01"
DEFAULT_END = "2024-12-31"
CACHE_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_FILE = CACHE_DIR / "nifty50_cache.csv"
MIN_OBSERVATIONS = 60  # Require at least 60 trading days so rolling 50-day MA and 20-day returns have valid rows


def fetch_nifty_data(
    ticker: str = DEFAULT_TICKER,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Download or load cached NIFTY 50 data with validation."""
    ticker = (ticker or DEFAULT_TICKER).strip()
    if not ticker:
        ticker = DEFAULT_TICKER

    if use_cache and CACHE_FILE.exists():
        df = pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)
        return _validate_data(df)

    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    except Exception as exc:
        raise RuntimeError(f"Failed to download data for {ticker}: {exc}") from exc

    if df is None or df.empty:
        raise ValueError(
            f"No data returned for ticker '{ticker}' between {start} and {end}. Check that the ticker symbol and date range are valid."
        )

    if len(df) < MIN_OBSERVATIONS:
        raise ValueError(
            f"Downloaded data for '{ticker}' contains only {len(df)} rows, which is insufficient. "
            f"At least {MIN_OBSERVATIONS} trading days are required to calculate 50-day moving averages and 20-day rolling metrics."
        )

    # Handle MultiIndex columns returned by modern yfinance versions
    if isinstance(df.columns, pd.MultiIndex):
        price_col = None
        for col in df.columns:
            if col[0] in ("Adj Close", "Close"):
                price_col = col
                break
        if price_col is None:
            raise ValueError(f"No price column ('Adj Close' or 'Close') found in downloaded data. Columns: {df.columns.tolist()}")
        df = df[[price_col]].copy()
        df.columns = ["close"]
    else:
        # Single-level columns
        if "Adj Close" in df.columns:
            price_col = "Adj Close"
        elif "Close" in df.columns:
            price_col = "Close"
        else:
            raise ValueError(f"No price column ('Adj Close' or 'Close') found in downloaded data. Columns: {df.columns.tolist()}")

        df = df[[price_col]].rename(columns={price_col: "close"})

    df.index.name = "date"
    # Ensure numeric conversion
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.sort_index()
    df = _validate_data(df)

    # Cache only valid market data
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE_FILE)
    return df


def _validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate price data and raise on issues."""
    if len(df) < MIN_OBSERVATIONS:
        raise ValueError(
            f"Data contains only {len(df)} rows. At least {MIN_OBSERVATIONS} trading days are required."
        )
    if df.index.duplicated().any():
        raise ValueError("Duplicate dates detected in data.")
    if df["close"].isnull().any():
        raise ValueError("Missing or invalid price values detected.")
    if (df["close"] <= 0).any():
        raise ValueError("Non-positive prices detected.")
    return df
