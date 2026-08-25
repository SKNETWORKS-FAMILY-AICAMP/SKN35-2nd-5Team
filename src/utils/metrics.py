# 모델 평가 유틸 함수
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(y_true, y_pred, y_prob):
    print(f"Accuracy  : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Recall    : {recall_score(y_true, y_pred):.4f}")
    print(f"Precision : {precision_score(y_true, y_pred):.4f}")
    print(f"F1 Score  : {f1_score(y_true, y_pred):.4f}")
    print(f"ROC-AUC-Score : {roc_auc_score(y_true, y_prob):.4f}")
    print("\n[ Classification Report ]")
    print(classification_report(y_true, y_pred))
