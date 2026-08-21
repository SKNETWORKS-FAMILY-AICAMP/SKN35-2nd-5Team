"""
Data validation module to check schema, missing values, and data integrity.
"""

from typing import Dict, Any
import pandas as pd


def validate_feature_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate the churn modeling feature dataset.
    """
    total_rows = len(df)
    null_counts = df.isnull().sum().to_dict()
    has_nulls = sum(null_counts.values()) > 0
    
    required_cols = [
        "user_id", "obs_total_events", "obs_w1_events", "obs_w2_events",
        "obs_active_days", "obs_recency_days", "is_churn"
    ]
    missing_required = [col for col in required_cols if col not in df.columns]
    
    churn_counts = df["is_churn"].value_counts().to_dict() if "is_churn" in df.columns else {}
    
    is_valid = (not has_nulls) and (len(missing_required) == 0) and (total_rows > 0)
    
    return {
        "is_valid": is_valid,
        "total_rows": total_rows,
        "total_columns": df.shape[1],
        "has_nulls": has_nulls,
        "null_counts": null_counts,
        "missing_required_columns": missing_required,
        "target_distribution": churn_counts,
    }
