"""
Decision Tree Model for Churn Prediction.
"""

import time
from typing import Tuple, Dict, Any
from sklearn.tree import DecisionTreeClassifier
from src.ml.evaluate import evaluate_classifier
from src.utils.constants import RANDOM_STATE


def train_decision_tree(
    X_train, X_test, y_train, y_test, feature_names=None, max_depth=None
) -> Tuple[DecisionTreeClassifier, Dict[str, Any]]:
    """Train Decision Tree (unconstrained to observe overfitting)."""
    t0 = time.time()
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    duration = time.time() - t0
    
    metrics = evaluate_classifier(model, X_train, X_test, y_train, y_test, feature_names)
    metrics["model_name"] = "Decision Tree"
    metrics["train_time_sec"] = float(duration)
    return model, metrics
