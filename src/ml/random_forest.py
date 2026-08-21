"""
Random Forest Model for Churn Prediction.
"""

import time
from typing import Tuple, Dict, Any
from sklearn.ensemble import RandomForestClassifier
from src.ml.evaluate import evaluate_classifier
from src.utils.constants import RANDOM_STATE


def train_random_forest(
    X_train, X_test, y_train, y_test, feature_names=None, n_estimators=200
) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    """Train Random Forest with balanced class weighting."""
    t0 = time.time()
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    duration = time.time() - t0
    
    metrics = evaluate_classifier(model, X_train, X_test, y_train, y_test, feature_names)
    metrics["model_name"] = "Random Forest"
    metrics["train_time_sec"] = float(duration)
    return model, metrics
