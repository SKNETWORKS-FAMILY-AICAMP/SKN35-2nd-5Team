"""
XGBoost Model for Churn Prediction.
"""

import time
from typing import Tuple, Dict, Any
from xgboost import XGBClassifier
from src.ml.evaluate import evaluate_classifier
from src.utils.constants import RANDOM_STATE


def train_xgboost(
    X_train, X_test, y_train, y_test, feature_names=None, n_estimators=200
) -> Tuple[XGBClassifier, Dict[str, Any]]:
    """Train XGBoost with scale_pos_weight balancing."""
    t0 = time.time()
    num_0 = (y_train == 0).sum()
    num_1 = (y_train == 1).sum()
    scale_pos_weight = num_0 / max(1, num_1)
    
    model = XGBClassifier(
        n_estimators=n_estimators,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    duration = time.time() - t0
    
    metrics = evaluate_classifier(model, X_train, X_test, y_train, y_test, feature_names)
    metrics["model_name"] = "XGBoost"
    metrics["train_time_sec"] = float(duration)
    return model, metrics
