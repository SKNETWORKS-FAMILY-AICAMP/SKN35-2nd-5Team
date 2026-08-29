<<<<<<< Updated upstream
# csv 파일 또는 모델 불러오는 함수 정의(config.py에 경로 설정 후 import로 추가형 함수 정의)
import pandas as pd

from src.config import PROCESSED_DIR, RAW_DIR
=======
"""MySQL 원본/전처리 데이터 로드와 학습 입력 스키마 검증."""

import pandas as pd

from src.database.load_db import COLUMN_MAPPING, get_processed_data, get_raw_data
from src.utils.constants import TARGET_COLUMN
>>>>>>> Stashed changes

RAW_COLUMNS = [
    column
    for column in COLUMN_MAPPING
    if column not in {"Industry Experience Gap", "Promotion Rate"}
]
PROCESSED_COLUMNS = [
    "Age",
    "Gender",
    "Years at Company",
    "Monthly Income",
    "Work-Life Balance",
    "Job Satisfaction",
    "Performance Rating",
    "Number of Promotions",
    "Overtime",
    "Distance from Home",
    "Education Level",
    "Number of Dependents",
    "Job Level",
    "Company Size",
    "Company Tenure",
    "Remote Work",
    "Leadership Opportunities",
    "Innovation Opportunities",
    "Company Reputation",
    "Employee Recognition",
    "Attrition",
    "Industry Experience Gap",
    "Promotion Rate",
    "Job Role_Finance",
    "Job Role_Healthcare",
    "Job Role_Media",
    "Job Role_Technology",
    "Marital Status_Married",
    "Marital Status_Single",
]
PROCESSED_DB_MAPPING = {
    column.lower().replace("-", "_").replace(" ", "_"): column
    for column in PROCESSED_COLUMNS
}


def _rows_to_dataframe(
    rows: list[dict],
    column_mapping: dict[str, str],
    expected_columns: list[str],
) -> pd.DataFrame:
    """DB 조회 결과에서 메타 컬럼을 제거하고 앱의 표준 스키마로 복원한다."""

    if not rows:
        return pd.DataFrame(columns=expected_columns)

    frame = pd.DataFrame(rows).rename(columns=column_mapping)
    missing = [column for column in expected_columns if column not in frame.columns]
    if missing:
        raise ValueError("DB 테이블에 필요한 컬럼이 없습니다: " + ", ".join(missing))

    return frame.loc[:, expected_columns].copy()


<<<<<<< Updated upstream
def load_processed_train():
    return pd.read_csv(PROCESSED_DIR / "train_processed.csv")
=======
def load_raw_train() -> pd.DataFrame:
    return _rows_to_dataframe(
        get_raw_data("train"),
        {value: key for key, value in COLUMN_MAPPING.items()},
        RAW_COLUMNS,
    )


def load_raw_test() -> pd.DataFrame:
    return _rows_to_dataframe(
        get_raw_data("test"),
        {value: key for key, value in COLUMN_MAPPING.items()},
        RAW_COLUMNS,
    )


def load_processed_train() -> pd.DataFrame:
    frame = _rows_to_dataframe(
        get_processed_data("train"),
        PROCESSED_DB_MAPPING,
        PROCESSED_COLUMNS,
    )
    return frame.apply(pd.to_numeric, errors="raise")


def load_processed_test() -> pd.DataFrame:
    frame = _rows_to_dataframe(
        get_processed_data("test"),
        PROCESSED_DB_MAPPING,
        PROCESSED_COLUMNS,
    )
    return frame.apply(pd.to_numeric, errors="raise")


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
>>>>>>> Stashed changes
