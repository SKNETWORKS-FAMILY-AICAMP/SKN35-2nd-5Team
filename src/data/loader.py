"""MySQL 직원 데이터 로더와 학습 입력 스키마 검증."""

import pandas as pd

from src.database.load_db import (
    COLUMN_MAPPING,
    get_prediction_data,
    get_processed_data,
    get_raw_data,
)
from src.utils.constants import TARGET_COLUMN


def _from_db(data_type: str, processed: bool = False) -> pd.DataFrame:
    rows = get_processed_data(data_type) if processed else get_raw_data(data_type)
    if not rows:
        raise ValueError(f"DB에서 {data_type} 데이터를 찾을 수 없습니다.")
    frame = pd.DataFrame(rows).drop(columns=["id", "type"], errors="ignore")
    reverse_mapping = {database: source for source, database in COLUMN_MAPPING.items()}
    return frame.rename(columns=reverse_mapping)


def load_raw_train() -> pd.DataFrame:
    return _from_db("train")


def load_raw_test() -> pd.DataFrame:
    return _from_db("test")


def load_processed_train() -> pd.DataFrame:
    return _from_db("train", processed=True)


def load_processed_test() -> pd.DataFrame:
    return _from_db("test", processed=True)


def load_predictions() -> pd.DataFrame:
    """DB에 저장된 직원별 이탈 예측 결과를 반환합니다."""
    rows = get_prediction_data()
    if not rows:
        raise ValueError("DB에서 이탈 예측 데이터를 찾을 수 없습니다.")
    frame = pd.DataFrame(rows)
    required = {"employee_id", "prediction"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("예측 DB 컬럼이 없습니다: " + ", ".join(sorted(missing)))
    if frame["employee_id"].duplicated().any():
        raise ValueError("예측 DB에 중복된 employee_id가 있습니다.")
    frame["prediction"] = pd.to_numeric(frame["prediction"], errors="raise")
    return frame[["employee_id", "prediction"]]


def split_processed_features_target(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"타깃 컬럼이 없습니다: {TARGET_COLUMN}")
    target = pd.to_numeric(data[TARGET_COLUMN], errors="coerce")
    if target.isna().any() or set(target.unique()) - {0, 1}:
        raise ValueError("전처리 타깃은 0과 1이어야 합니다.")
    saved = [column for column in data.columns if column.startswith("Unnamed:")]
    features = data.drop(columns=[TARGET_COLUMN, *saved]).copy()
    bool_columns = features.select_dtypes(include="bool").columns
    features[bool_columns] = features[bool_columns].astype("int8")
    non_numeric = features.select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        raise ValueError("전처리되지 않은 피처가 있습니다: " + ", ".join(non_numeric))
    return features, target.astype("int8").rename(TARGET_COLUMN)


def load_processed_train_test_features() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    train_x, train_y = split_processed_features_target(load_processed_train())
    test_x, test_y = split_processed_features_target(load_processed_test())
    if list(train_x.columns) != list(test_x.columns):
        raise ValueError("학습/테스트 피처 구성이 다릅니다.")
    return train_x, train_y, test_x, test_y
