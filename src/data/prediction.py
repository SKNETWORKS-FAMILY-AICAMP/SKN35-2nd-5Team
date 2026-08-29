"""직원 퇴사 예측 로직과 예측 테이블 INSERT SQL (최종 채택 모델: MLP)."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from src.data.loader import load_raw_test, load_raw_train
from src.data.preprocess import preprocess_pipeline
from src.models.dl.mlp_model import MLPClassifier
from src.utils.paths import (
    DL_ARTIFACTS_DIR,
    MLP_BEST_PARAMS_PATH,
    MLP_METADATA_PATH,
    MLP_MODEL_PATH,
    MLP_PREPROCESSOR_PATH,
    MLP_THRESHOLD_PATH,
)

# prediction_id는 DB에서 자동 생성하므로 INSERT 대상에서 제외한다.
INSERT_PREDICTION_SQL = """
    INSERT INTO employee_attrition_prediction (
        employee_id,
        prediction
    )
    VALUES (%s, %s)
    ON DUPLICATE KEY UPDATE
        prediction = VALUES(prediction)
"""


def load_prediction_model() -> tuple[MLPClassifier, Any, float, dict[str, Any]]:
    """최종 채택된 Deep Learning (MLP) 모델과 전처리 스케일러를 로드합니다."""
    best_params = joblib.load(MLP_BEST_PARAMS_PATH)
    preprocessor = joblib.load(MLP_PREPROCESSOR_PATH)
    threshold = float(joblib.load(MLP_THRESHOLD_PATH))
    metadata = joblib.load(MLP_METADATA_PATH)
    in_features = metadata.get("in_features", 28)

    model = MLPClassifier(best_params, in_features=in_features)
    state_dict = torch.load(MLP_MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    return model, preprocessor, threshold, metadata


def prepare_model_input(
    raw_frame: pd.DataFrame,
    raw_train: pd.DataFrame,
) -> pd.DataFrame:
    """원본 직원 데이터를 MLP 학습 시 사용된 전처리 피처로 변환합니다."""
    typed_frame = raw_frame.copy()
    for column in typed_frame.columns.intersection(raw_train.columns):
        if pd.api.types.is_numeric_dtype(raw_train[column]):
            typed_frame[column] = pd.to_numeric(typed_frame[column], errors="raise")

    processed = preprocess_pipeline(typed_frame, reference=raw_train.copy())
    return processed.drop(columns=["Attrition", "Unnamed: 0"], errors="ignore")


def attrition_probabilities(
    model: Any,
    preprocessor: Any,
    frame: pd.DataFrame,
) -> np.ndarray:
    """각 직원의 퇴사 확률(0.0 ~ 1.0)을 MLP 신경망으로 계산하여 반환합니다."""
    if isinstance(model, torch.nn.Module):
        X_scaled = preprocessor.transform(frame)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            logits = model(X_tensor)
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
        return np.asarray(probs, dtype=float)

    # fallback for scikit-learn models if passed
    classes = list(model.classes_)
    return np.asarray(model.predict_proba(frame)[:, classes.index(1)], dtype=float)


def attrition_probability(
    model: Any,
    preprocessor: Any,
    frame: pd.DataFrame,
) -> float:
    """단일 직원 시나리오에서 사용할 퇴사 확률을 반환합니다."""
    if len(frame) != 1:
        raise ValueError("단일 직원 확률 계산에는 한 행만 전달해야 합니다.")
    return float(attrition_probabilities(model, preprocessor, frame)[0])


def create_employee_predictions(
    raw_frame: pd.DataFrame | None = None,
    raw_train: pd.DataFrame | None = None,
    model_bundle: tuple | None = None,
) -> pd.DataFrame:
    """DB 저장용 ``employee_id, prediction`` 데이터프레임을 생성한다.

    최종 채택된 MLP 모델을 사용하여 퇴사 확률을 0.0~1.0 범위의 실수로 계산합니다.
    """
    source = load_raw_test() if raw_frame is None else raw_frame.copy()
    reference = load_raw_train() if raw_train is None else raw_train.copy()

    if model_bundle is None:
        model, preprocessor, threshold, metadata = load_prediction_model()
    else:
        model, preprocessor, threshold, metadata = model_bundle

    if "Employee ID" not in source.columns:
        raise ValueError("예측 데이터에 'Employee ID' 컬럼이 없습니다.")
    if source["Employee ID"].isna().any():
        raise ValueError("Employee ID에 빈 값이 있습니다.")
    if source["Employee ID"].duplicated().any():
        raise ValueError("Employee ID에 중복 값이 있습니다.")

    processed_features = prepare_model_input(source, reference)
    if processed_features.empty:
        raise ValueError("전처리 후 예측할 직원이 없습니다.")

    probabilities = attrition_probabilities(model, preprocessor, processed_features)
    employee_ids = source.loc[processed_features.index, "Employee ID"]

    return pd.DataFrame(
        {
            "employee_id": employee_ids.to_numpy(),
            "prediction": probabilities,
        }
    ).reset_index(drop=True)


def prediction_rows(predictions: pd.DataFrame) -> list[tuple[object, float]]:
    """예측 데이터프레임을 DB executemany용 튜플 목록으로 변환한다."""
    required_columns = {"employee_id", "prediction"}
    missing_columns = required_columns - set(predictions.columns)
    if missing_columns:
        raise ValueError("예측 결과 컬럼이 없습니다: " + ", ".join(sorted(missing_columns)))

    return [
        (
            employee_id.item() if hasattr(employee_id, "item") else employee_id,
            float(prediction),
        )
        for employee_id, prediction in predictions[
            ["employee_id", "prediction"]
        ].itertuples(index=False, name=None)
    ]


__all__ = [
    "INSERT_PREDICTION_SQL",
    "attrition_probabilities",
    "attrition_probability",
    "create_employee_predictions",
    "load_prediction_model",
    "prediction_rows",
    "prepare_model_input",
]

