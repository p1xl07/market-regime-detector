"""Clustering pipeline with chronological split and no leakage."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def split_chronological(df: pd.DataFrame, train_frac: float = 0.7):
    """Chronological train/test split for time-series."""
    n = len(df)
    split_idx = int(n * train_frac)
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    return train, test


class RegimeModel:
    """KMeans with StandardScaler; fit only on train."""

    def __init__(self, n_clusters: int = 3, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = KMeans(
            n_clusters=n_clusters, random_state=random_state, n_init="auto"
        )
        self.feature_cols = [
            "daily_return",
            "rolling_20_return",
            "rolling_20_vol",
            "dist_from_ma50",
            "momentum_5",
        ]

    def fit(self, train_df: pd.DataFrame):
        train_df = train_df.copy()
        for col in self.feature_cols:
            if col not in train_df.columns:
                raise ValueError(f"Missing feature column: {col}")
        X_train = train_df[self.feature_cols].dropna()
        if len(X_train) < self.n_clusters:
            raise ValueError(
                f"Not enough non-null observations for clustering. Found {len(X_train)} complete rows, but need at least K={self.n_clusters}."
            )
        # Check distinct observations
        n_distinct = len(np.unique(X_train.values, axis=0))
        if n_distinct < self.n_clusters:
            raise ValueError(
                f"Not enough distinct observations for clustering. Found {n_distinct} distinct rows, but need at least K={self.n_clusters}."
            )

        X_scaled = self.scaler.fit_transform(X_train.values)
        self.kmeans.fit(X_scaled)
        self.train_df = train_df.loc[X_train.index].copy()
        self.train_df["cluster"] = self.kmeans.labels_
        return self

    def predict(self, test_df: pd.DataFrame) -> pd.DataFrame:
        test_df = test_df.copy()
        X_test = test_df[self.feature_cols].dropna()
        if len(X_test) > 0:
            X_scaled = self.scaler.transform(X_test.values)
            labels = self.kmeans.predict(X_scaled)
            test_df.loc[X_test.index, "cluster"] = labels
        else:
            test_df["cluster"] = np.nan
        return test_df
