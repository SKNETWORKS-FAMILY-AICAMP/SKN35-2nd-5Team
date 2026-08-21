"""
Test fixtures and mock dataset generators.
"""

import pandas as pd
import numpy as np


def create_mock_feature_df(n_samples: int = 100) -> pd.DataFrame:
    """Generate synthetic customer features for unit testing."""
    np.random.seed(42)
    return pd.DataFrame({
        "user_id": [f"u{i}" for i in range(n_samples)],
        "first_pay_ts": [1565000000000 + i * 1000 for i in range(n_samples)],
        "first_pay_item": ["p124"] * n_samples,
        "obs_total_events": np.random.randint(10, 2000, n_samples),
        "obs_w1_events": np.random.randint(5, 1000, n_samples),
        "obs_w2_events": np.random.randint(0, 1000, n_samples),
        "obs_decay_ratio": np.random.uniform(0.0, 1.5, n_samples),
        "obs_active_days": np.random.randint(1, 15, n_samples),
        "obs_solve_count": np.random.randint(0, 300, n_samples),
        "obs_recency_days": np.random.uniform(0.0, 14.0, n_samples),
        "is_refund_churn": np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05]),
        "is_non_renewal_churn": np.random.choice([0, 1], size=n_samples, p=[0.15, 0.85]),
        "is_churn": np.random.choice([0, 1], size=n_samples, p=[0.1, 0.9]),
    })
