"""Feature engineering using only historical information."""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create regime-detection features from closing prices.

    Features (all using past-only data at each observation):
    - daily_return: pct change from previous close
    - rolling_20_return: 20-trading-day cumulative return
    - rolling_20_vol: 20-day annualized vol (factor 252)
    - dist_from_ma50: (close - 50-day MA) / 50-day MA
    - momentum_5: 5-day return (optional interpretation boost)
    """
    df = df.copy()
    df["daily_return"] = df["close"].pct_change()
    df["rolling_20_return"] = df["close"].pct_change(periods=20)

    # Annualized volatility: daily vol * sqrt(252)
    df["rolling_20_vol"] = df["daily_return"].rolling(window=20).std() * np.sqrt(252)

    ma50 = df["close"].rolling(window=50).mean()
    df["dist_from_ma50"] = (df["close"] - ma50) / ma50

    # Optional: 5-day momentum improves cluster separation interpretation
    df["momentum_5"] = df["close"].pct_change(periods=5)

    # Handle warm-up and infinities
    df = df.replace([np.inf, -np.inf], np.nan)
    return df
