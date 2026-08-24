from typing import Any

import pandas as pd

from src.utils.constants import ID_COLUMN, POSITIVE_LABEL, TARGET_COLUMN


def numeric_summary(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.select_dtypes(include="number").drop(columns=[ID_COLUMN], errors="ignore")
    if numeric.empty:
        return pd.DataFrame()
    return numeric.describe().T.rename_axis("feature").reset_index()


def categorical_summary(frame: pd.DataFrame) -> pd.DataFrame:
    categorical = frame.select_dtypes(exclude="number")
    rows: list[dict[str, Any]] = []
    for column in categorical.columns:
        mode = categorical[column].mode(dropna=True)
        rows.append(
            {
                "feature": column,
                "unique": int(categorical[column].nunique(dropna=True)),
                "most_frequent": None if mode.empty else mode.iloc[0],
                "missing": int(categorical[column].isna().sum()),
            }
        )
    return pd.DataFrame(rows)


def attrition_rate_by(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """선택한 피처의 그룹별 직원 수와 이탈률을 계산한다."""
    if column not in frame.columns:
        raise KeyError(f"존재하지 않는 컬럼입니다: {column}")
    grouped = (
        frame.assign(_left=frame[TARGET_COLUMN].eq(POSITIVE_LABEL))
        .groupby(column, dropna=False, observed=True)
        .agg(employee_count=(TARGET_COLUMN, "size"), attrition_rate=("_left", "mean"))
        .reset_index()
        .sort_values("attrition_rate", ascending=False)
    )
    grouped["attrition_rate"] = grouped["attrition_rate"] * 100
    return grouped


def build_eda_report(frame: pd.DataFrame) -> dict[str, Any]:
    """화면과 CLI에서 사용할 데이터 요약 정보를 계산한다."""
    target_counts = frame[TARGET_COLUMN].value_counts(dropna=False)
    return {
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "duplicate_rows": int(frame.duplicated().sum()),
        "missing_values": int(frame.isna().sum().sum()),
        "attrition_rate": float(frame[TARGET_COLUMN].eq(POSITIVE_LABEL).mean()),
        "target_counts": target_counts.to_dict(),
    }
