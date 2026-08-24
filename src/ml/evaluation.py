"""Common binary-classification metrics and curves."""

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def positive_probability(model: Any, features: Any) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(features))[:, 1]
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(features), dtype=float)
        return 1.0 / (1.0 + np.exp(-scores))
    return np.asarray(model.predict(features), dtype=float)


def evaluate_classifier(model: Any, features: Any, target: Any) -> dict[str, Any]:
    predictions = np.asarray(model.predict(features), dtype=int)
    probabilities = positive_probability(model, features)
    tn, fp, fn, tp = confusion_matrix(target, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(target, predictions)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "average_precision": float(average_precision_score(target, probabilities)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def classification_curves(model: Any, features: Any, target: Any) -> dict[str, np.ndarray]:
    probabilities = positive_probability(model, features)
    fpr, tpr, roc_thresholds = roc_curve(target, probabilities)
    precision, recall, pr_thresholds = precision_recall_curve(target, probabilities)
    return {
        "fpr": fpr,
        "tpr": tpr,
        "roc_thresholds": roc_thresholds,
        "precision": precision,
        "recall": recall,
        "pr_thresholds": pr_thresholds,
    }

