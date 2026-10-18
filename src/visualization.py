"""Visualization helpers."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_price_regimes(df: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df["close"], color="black", label="NIFTY 50 (close)")
    # Test-period regimes
    test_plot = test_df.dropna(subset=["cluster"])
    for c, color in zip(sorted(test_plot["cluster"].unique()), ["blue", "red", "green"]):
        sub = test_plot[test_plot["cluster"] == c]
        ax.scatter(sub.index, sub["close"], color=color, label=f"Regime {int(c)} (test)", s=20)
    ax.axvline(train_df.index[-1], color="gray", linestyle="--", label="Train/Test split")
    ax.set_title("Historical Price with Detected Test Regimes")
    ax.set_ylabel("Index Level")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    return fig


def plot_scatter(test_df: pd.DataFrame):
    sub = test_df.dropna(subset=["cluster"])
    fig, ax = plt.subplots(figsize=(7, 5))
    for c, color in zip(sorted(sub["cluster"].unique()), ["blue", "red", "green"]):
        s = sub[sub["cluster"] == c]
        ax.scatter(s["rolling_20_vol"], s["rolling_20_return"], color=color, label=f"Regime {int(c)}", alpha=0.7)
    ax.set_title("Return vs Volatility (Test Period)")
    ax.set_xlabel("20-day Annualized Volatility")
    ax.set_ylabel("20-day Return")
    ax.legend()
    plt.tight_layout()
    return fig
