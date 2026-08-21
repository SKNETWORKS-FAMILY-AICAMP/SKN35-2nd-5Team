"""
ML Model Evaluation Utility.
"""

from typing import Dict, Any, List
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)


def evaluate_classifier(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    feature_names: List[str] = None,
) -> Dict[str, Any]:
    """Evaluate a trained classifier and return comprehensive metrics."""
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    y_proba_test = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    metrics = {
        "train_accuracy": float(accuracy_score(y_train, y_pred_train)),
        "test_accuracy": float(accuracy_score(y_test, y_pred_test)),
        "precision": float(precision_score(y_test, y_pred_test, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred_test, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred_test, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba_test)) if y_proba_test is not None else None,
        "pr_auc": float(average_precision_score(y_test, y_proba_test)) if y_proba_test is not None else None,
        "confusion_matrix": confusion_matrix(y_test, y_pred_test).tolist(),
        "classification_report": classification_report(
            y_test, y_pred_test, target_names=["Retained (0)", "Churned (1)"]
        ),
    }
    
    if y_proba_test is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba_test)
        prec_c, rec_c, _ = precision_recall_curve(y_test, y_proba_test)
        metrics["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
        metrics["pr_curve"] = {"precision": prec_c.tolist(), "recall": rec_c.tolist()}
    else:
        metrics["roc_curve"] = None
        metrics["pr_curve"] = None
        
    if feature_names is not None and hasattr(model, "feature_importances_"):
        fi = sorted(
            zip(feature_names, model.feature_importances_.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        metrics["feature_importance"] = fi
    else:
        metrics["feature_importance"] = None
        
    return metrics
