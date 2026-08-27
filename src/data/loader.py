"""원본/전처리 데이터 로드와 학습 입력 스키마 검증."""

import pandas as pd

from src.config import PROCESSED_DIR, RAW_DIR
from src.utils.constants import TARGET_COLUMN


def load_raw_train():
    return pd.read_csv(RAW_DIR / "train.csv")


def load_raw_test():
    return pd.read_csv(RAW_DIR / "test.csv")


def load_processed_train():
    return pd.read_csv(PROCESSED_DIR / "train_processed.csv")


def load_processed_test():
    return pd.read_csv(PROCESSED_DIR / "test_processed.csv")


def split_processed_features_target(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """전처리 데이터를 수치형 피처와 이진 타깃으로 분리하고 검증한다."""

    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"타깃 컬럼이 없습니다: {TARGET_COLUMN}")

    target = pd.to_numeric(data[TARGET_COLUMN], errors="coerce")
    if target.isna().any():
        raise ValueError("전처리 타깃에 숫자로 변환할 수 없는 값이 있습니다.")

    unexpected_labels = set(target.unique()) - {0, 1}
    if unexpected_labels:
        labels = ", ".join(map(str, sorted(unexpected_labels)))
        raise ValueError(f"예상하지 못한 타깃 값입니다: {labels}")

    saved_index_columns = [column for column in data.columns if column.startswith("Unnamed:")]
    features = data.drop(columns=[TARGET_COLUMN, *saved_index_columns]).copy()
    bool_columns = features.select_dtypes(include="bool").columns
    features[bool_columns] = features[bool_columns].astype("int8")

    non_numeric_columns = features.select_dtypes(exclude="number").columns.tolist()
    if non_numeric_columns:
        raise ValueError("전처리되지 않은 피처가 있습니다: " + ", ".join(non_numeric_columns))

    return features, target.astype("int8").rename(TARGET_COLUMN)


def load_processed_train_test_features(
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """전처리 학습·테스트 데이터를 피처/타깃으로 로드하고 스키마를 검증한다."""

    train_features, train_target = split_processed_features_target(load_processed_train())
    test_features, test_target = split_processed_features_target(load_processed_test())

    if list(train_features.columns) != list(test_features.columns):
        missing_from_test = sorted(set(train_features.columns) - set(test_features.columns))
        extra_in_test = sorted(set(test_features.columns) - set(train_features.columns))
        raise ValueError(
            "학습/테스트 피처 구성이 다릅니다. "
            f"테스트에 없음={missing_from_test}, 테스트에만 있음={extra_in_test}"
        )

    return train_features, train_target, test_features, test_target
