"""
Customer Behavior Clustering using K-Means.
"""

import os
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.utils.constants import RANDOM_STATE


def perform_kmeans_clustering(
    df: pd.DataFrame,
    feature_cols: list = None,
    n_clusters: int = 4,
) -> Tuple[pd.DataFrame, KMeans, StandardScaler]:
    """
    Fit K-Means clustering on customer behavioral features.
    """
    if feature_cols is None:
        feature_cols = [
            "obs_total_events", "obs_active_days", "obs_decay_ratio",
            "obs_recency_days", "obs_respond_count", "obs_play_video_count"
        ]
        
    valid_cols = [c for c in feature_cols if c in df.columns]
    if not valid_cols:
        raise ValueError("No valid clustering feature columns were found.")
    if not 2 <= n_clusters <= len(df):
        raise ValueError(f"n_clusters must be between 2 and {len(df):,}.")

    X = df[valid_cols].copy().replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Some Windows/cloud containers report no physical cores to joblib. Apply a
    # local soft limit during K-Means only, then restore the process environment.
    previous_loky_limit = os.environ.get("LOKY_MAX_CPU_COUNT")
    os.environ["LOKY_MAX_CPU_COUNT"] = "1"
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
    finally:
        if previous_loky_limit is None:
            os.environ.pop("LOKY_MAX_CPU_COUNT", None)
        else:
            os.environ["LOKY_MAX_CPU_COUNT"] = previous_loky_limit
    
    df_result = df.copy()
    df_result["cluster"] = clusters
    
    return df_result, kmeans, scaler


def get_cluster_summary(df_clustered: pd.DataFrame, feature_cols: list = None) -> pd.DataFrame:
    """Compute mean statistics and churn rates per cluster."""
    if feature_cols is None:
        feature_cols = [
            "obs_total_events", "obs_active_days", "obs_decay_ratio",
            "obs_recency_days", "obs_respond_count", "is_churn"
        ]
    valid_cols = [c for c in feature_cols if c in df_clustered.columns]
    return df_clustered.groupby("cluster")[valid_cols].mean()
