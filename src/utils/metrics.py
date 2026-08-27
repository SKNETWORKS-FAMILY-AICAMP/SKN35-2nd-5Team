"""ML과 DL에서 공통으로 사용하는 이진 분류 평가 함수."""

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_binary_metrics(y_true, y_pred, y_prob) -> dict[str, float | int]:
    """예측 결과를 ML/DL 공통 리포트 스키마로 계산한다."""

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def evaluate_sklearn_model(model: Any, features, target) -> dict[str, float | int]:
    """학습된 scikit-learn 호환 모델의 지표를 계산한다."""

    predictions = model.predict(features)
    probabilities = np.asarray(model.predict_proba(features))[:, 1]
    return calculate_binary_metrics(target, predictions, probabilities)


def print_binary_metrics(y_true, y_pred, y_prob) -> dict[str, float | int]:
    """공통 지표를 출력하고 CSV 저장에 사용할 딕셔너리를 반환한다."""

    metrics = calculate_binary_metrics(y_true, y_pred, y_prob)
    print(f"Accuracy  : {metrics['accuracy']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"F1 Score  : {metrics['f1']:.4f}")
    print(f"ROC-AUC-Score : {metrics['roc_auc']:.4f}")
    print("\n[ Classification Report ]")
    print(classification_report(y_true, y_pred, zero_division=0))
    return metrics
