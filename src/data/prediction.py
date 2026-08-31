"""직원 퇴사 예측 로직과 예측 테이블 INSERT SQL."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data.loader import load_raw_test, load_raw_train
from src.data.preprocess import preprocess_pipeline
from src.utils.paths import BEST_ML_MODEL_PATH

# prediction_id는 DB에서 자동 생성하므로 INSERT 대상에서 제외한다.
INSERT_PREDICTION_SQL = """
    INSERT INTO employee_attrition_prediction (
        employee_id,
        prediction
    )
    VALUES (%s, %s)
"""


def load_prediction_model(model_path: str | Path = BEST_ML_MODEL_PATH):
    """DB 적재와 page5에서 공통으로 사용할 최종 ML 모델을 불러온다."""

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"예측 모델을 찾을 수 없습니다: {path}")
    return joblib.load(path)


def prepare_model_input(
    raw_frame: pd.DataFrame,
    raw_train: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    """원본 직원 데이터를 학습 때와 같은 모델 입력 피처로 변환한다."""

    typed_frame = raw_frame.copy()
    for column in typed_frame.columns.intersection(raw_train.columns):
        if pd.api.types.is_numeric_dtype(raw_train[column]):
            typed_frame[column] = pd.to_numeric(typed_frame[column], errors="raise")

    processed = preprocess_pipeline(typed_frame, reference=raw_train.copy())
    processed = processed.drop(columns=["Attrition", "Unnamed: 0"], errors="ignore")

    # pd.get_dummies(drop_first=True)는 직원 한 명만 전달되면 해당 행의 범주를
    # 기준 범주로 오인할 수 있다. 모델 피처명에 맞춰 원본 값에서 더미 값을
    # 다시 계산하면 page5 단일 예측과 전체 배치 예측이 항상 동일해진다.
    raw_columns = sorted(raw_frame.columns, key=len, reverse=True)
    for feature in feature_names:
        source_column = next(
            (
                column
                for column in raw_columns
                if feature.startswith(f"{column}_")
            ),
            None,
        )
        if source_column is None:
            continue

        category = feature.removeprefix(f"{source_column}_")
        normalized = (
            raw_frame.loc[processed.index, source_column]
            .astype("string")
            .str.strip()
            .str.replace("'", "’", regex=False)
        )
        processed[feature] = normalized.eq(category).astype(int).to_numpy()

    missing_features = [
        feature for feature in feature_names if feature not in processed.columns
    ]
    if missing_features:
        raise ValueError(
            "전처리 후 누락된 모델 입력 항목: " + ", ".join(missing_features)
        )

    return processed.reindex(columns=feature_names)


def attrition_probabilities(model, frame: pd.DataFrame) -> np.ndarray:
    """각 직원이 퇴사 라벨 1일 확률을 반환한다."""

    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("모델 클래스에 퇴사 라벨 1이 없습니다.")
    return np.asarray(model.predict_proba(frame)[:, classes.index(1)], dtype=float)


def attrition_probability(model, frame: pd.DataFrame) -> float:
    """page5의 단일 직원 시나리오에서 사용할 퇴사 확률을 반환한다."""

    if len(frame) != 1:
        raise ValueError("단일 직원 확률 계산에는 한 행만 전달해야 합니다.")
    return float(attrition_probabilities(model, frame)[0])


def create_employee_predictions(
    raw_frame: pd.DataFrame | None = None,
    raw_train: pd.DataFrame | None = None,
    model=None,
) -> pd.DataFrame:
    """DB 저장용 ``employee_id, prediction`` 데이터프레임을 생성한다.

    raw_frame을 생략하면 test.csv의 모든 직원을 예측한다. prediction에는
    모델이 계산한 퇴사 확률을 0.0~1.0 범위의 실수로 그대로 저장한다.
    """

    source = load_raw_test() if raw_frame is None else raw_frame.copy()
    reference = load_raw_train() if raw_train is None else raw_train.copy()
    prediction_model = load_prediction_model() if model is None else model

    if "Employee ID" not in source.columns:
        raise ValueError("예측 데이터에 'Employee ID' 컬럼이 없습니다.")
    if source["Employee ID"].isna().any():
        raise ValueError("Employee ID에 빈 값이 있습니다.")
    if source["Employee ID"].duplicated().any():
        raise ValueError("Employee ID에 중복 값이 있습니다.")

    feature_names = list(getattr(prediction_model, "feature_names_in_", []))
    if not feature_names:
        raise ValueError("저장된 모델에서 입력 피처 정보를 찾을 수 없습니다.")

    model_input = prepare_model_input(source, reference, feature_names)
    if model_input.empty:
        raise ValueError("전처리 후 예측할 직원이 없습니다.")

    probabilities = attrition_probabilities(prediction_model, model_input)
    employee_ids = source.loc[model_input.index, "Employee ID"]

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
