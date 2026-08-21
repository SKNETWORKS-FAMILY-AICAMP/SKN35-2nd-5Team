"""
Customer Behavior Clustering using K-Means.
"""

from typing import Dict, Any, Tuple
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
    X = df[valid_cols].copy()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
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
