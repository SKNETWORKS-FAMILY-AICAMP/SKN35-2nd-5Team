"""
Exploratory Data Analysis (EDA) module for churn dataset.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np


def compute_dataset_overview(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute top-level summary metrics of the dataset."""
    total_users = len(df)
    churn_users = int(df["is_churn"].sum()) if "is_churn" in df.columns else 0
    retained_users = total_users - churn_users
    
    refund_churn = int(df["is_refund_churn"].sum()) if "is_refund_churn" in df.columns else 0
    non_renewal_churn = int(df["is_non_renewal_churn"].sum()) if "is_non_renewal_churn" in df.columns else 0
    
    return {
        "total_users": total_users,
        "churn_users": churn_users,
        "churn_rate": (churn_users / total_users * 100) if total_users > 0 else 0,
        "retained_users": retained_users,
        "retained_rate": (retained_users / total_users * 100) if total_users > 0 else 0,
        "refund_churn": refund_churn,
        "non_renewal_churn": non_renewal_churn,
        "num_features": df.shape[1],
    }


def compare_churn_groups(df: pd.DataFrame, feature_cols: List[str] = None) -> pd.DataFrame:
    """Compare feature statistics between retained (0) and churned (1) users."""
    if feature_cols is None:
        feature_cols = [
            "obs_total_events", "obs_active_days", "obs_events_per_active_day",
            "obs_w1_events", "obs_w2_events", "obs_decay_ratio",
            "obs_recency_days", "obs_last_active_day",
            "obs_respond_count", "obs_submit_count", "obs_play_video_count", "obs_play_audio_count",
            "pre_pay_events", "pre_pay_active_days", "total_lifetime_events"
        ]
    valid_cols = [c for c in feature_cols if c in df.columns]
    
    stats = df.groupby("is_churn")[valid_cols].agg(["mean", "median"]).T
    return stats


def compute_correlations(df: pd.DataFrame, target_col: str = "is_churn") -> pd.Series:
    """Calculate feature correlations with the target churn label."""
    numeric_df = df.select_dtypes(include=[np.number])
    if target_col not in numeric_df.columns:
        return pd.Series()
    corrs = numeric_df.corr()[target_col].drop(
        index=[target_col, "is_refund_churn", "is_non_renewal_churn"],
        errors="ignore"
    ).sort_values()
    return corrs
