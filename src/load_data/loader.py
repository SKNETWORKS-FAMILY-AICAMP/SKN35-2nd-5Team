"""Load and validate employee attrition CSV datasets."""

from pathlib import Path

import pandas as pd

from src.utils.constants import ID_COLUMN, NEGATIVE_LABEL, POSITIVE_LABEL, TARGET_COLUMN
from src.utils.paths import TEST_DATA_PATH, TRAIN_DATA_PATH


def load_dataset(path: str | Path, *, require_target: bool = True) -> pd.DataFrame:
    """Read a CSV and perform lightweight schema validation."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {csv_path}")
    if csv_path.suffix.lower() != ".csv":
        raise ValueError(f"CSV 파일만 지원합니다: {csv_path}")

    frame = pd.read_csv(csv_path)
    if frame.empty:
        raise ValueError(f"데이터 파일이 비어 있습니다: {csv_path}")
    if require_target and TARGET_COLUMN not in frame.columns:
        raise ValueError(f"필수 타깃 컬럼 '{TARGET_COLUMN}'이 없습니다.")
    if ID_COLUMN not in frame.columns:
        raise ValueError(f"필수 식별자 컬럼 '{ID_COLUMN}'이 없습니다.")
    if frame[ID_COLUMN].duplicated().any():
        raise ValueError(f"'{ID_COLUMN}' 컬럼에 중복값이 있습니다.")

    if require_target:
        labels = set(frame[TARGET_COLUMN].dropna().unique())
        expected = {NEGATIVE_LABEL, POSITIVE_LABEL}
        if not labels.issubset(expected):
            raise ValueError(
                f"'{TARGET_COLUMN}'은 {sorted(expected)}만 허용합니다. 현재값: {sorted(labels)}"
            )
    return frame


def load_train_data(path: str | Path = TRAIN_DATA_PATH) -> pd.DataFrame:
    return load_dataset(path, require_target=True)


def load_test_data(path: str | Path = TEST_DATA_PATH) -> pd.DataFrame:
    return load_dataset(path, require_target=False)


def split_features_target(
    frame: pd.DataFrame,
    *,
    drop_id: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a validated frame and encode Left=1, Stayed=0."""
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"'{TARGET_COLUMN}' 컬럼이 없어 학습 데이터를 분리할 수 없습니다.")
    features = frame.drop(columns=[TARGET_COLUMN])
    if drop_id:
        features = features.drop(columns=[ID_COLUMN], errors="ignore")
    target = frame[TARGET_COLUMN].map({NEGATIVE_LABEL: 0, POSITIVE_LABEL: 1})
    if target.isna().any():
        raise ValueError("타깃에 결측치 또는 알 수 없는 레이블이 있습니다.")
    return features, target.astype("int8")

