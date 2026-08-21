"""
LightGBM Model for Churn Prediction.
"""

import time
from typing import Tuple, Dict, Any
from lightgbm import LGBMClassifier
from src.ml.evaluate import evaluate_classifier
from src.utils.constants import RANDOM_STATE


def train_lightgbm(
    X_train, X_test, y_train, y_test, feature_names=None, n_estimators=200
) -> Tuple[LGBMClassifier, Dict[str, Any]]:
    """Train LightGBM with unbalance handling."""
    t0 = time.time()
    model = LGBMClassifier(
        n_estimators=n_estimators,
        is_unbalance=True,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    duration = time.time() - t0
    
    metrics = evaluate_classifier(model, X_train, X_test, y_train, y_test, feature_names)
    metrics["model_name"] = "LightGBM"
    metrics["train_time_sec"] = float(duration)
    return model, metrics
