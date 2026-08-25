"""Shared utilities for machine learning models."""

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


# ============================================================
# 공통 설정값
# ============================================================

RANDOM_STATE = 42
TARGET_COLUMN = "Attrition"
TEST_SIZE = 0.2


# ============================================================
# 데이터 로드
# ============================================================

def load_data(
    csv_path: str | Path,
) -> tuple[pd.DataFrame, pd.Series]:
    """전처리된 이탈 예측 데이터셋을 불러온다."""

    df = pd.read_csv(csv_path)

    # CSV 저장 시 생성된 불필요한 인덱스 컬럼 제거
    df = df.drop(
        columns=["Unnamed: 0"],
        errors="ignore",
    )

    # 입력 변수와 Target 분리
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    return X, y


# ============================================================
# Train / Test 분리
# ============================================================

def split_data(
    X: pd.DataFrame,
    y: pd.Series,
):
    """데이터를 Train / Test 데이터로 분리한다."""

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


# ============================================================
# 모델 평가
# ============================================================

def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """학습된 이진 분류 모델의 성능을 평가한다."""

    # 최종 예측값 (0 / 1)
    y_pred = model.predict(X_test)

    # 이탈(1)일 확률
    y_prob = model.predict_proba(X_test)[:, 1]

    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred,
    ).ravel()

    return {
        "accuracy": accuracy_score(
            y_test,
            y_pred,
        ),
        "precision": precision_score(
            y_test,
            y_pred,
        ),
        "recall": recall_score(
            y_test,
            y_pred,
        ),
        "f1": f1_score(
            y_test,
            y_pred,
        ),
        "roc_auc": roc_auc_score(
            y_test,
            y_prob,
        ),
        "average_precision": average_precision_score(
            y_test,
            y_prob,
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }