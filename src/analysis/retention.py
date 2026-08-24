"""Turn attrition risk scores into reviewable retention suggestions."""

from typing import Any

import pandas as pd

from src.ml.evaluation import positive_probability
from src.utils.constants import ID_COLUMN, TARGET_COLUMN


def _recommendations(row: pd.Series) -> str:
    actions: list[str] = []
    if row.get("Overtime") == "Yes":
        actions.append("업무량·초과근무 조정 면담")
    if row.get("Job Satisfaction") in {"Low", "Medium"}:
        actions.append("직무 만족도 및 역할 재설계 점검")
    if row.get("Work-Life Balance") in {"Poor", "Fair"}:
        actions.append("유연근무·휴가 사용 계획 검토")
    if row.get("Employee Recognition") in {"Low", "Medium"}:
        actions.append("성과 인정 및 피드백 강화")
    if row.get("Leadership Opportunities") == "No":
        actions.append("성장·리더십 기회 논의")
    return " / ".join(actions[:3]) or "정기 체크인 및 경력 개발 면담"


def score_retention_risk(
    model: Any,
    frame: pd.DataFrame,
    *,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Score employees and attach non-causal, human-review suggestions."""
    features = frame.drop(columns=[TARGET_COLUMN, ID_COLUMN], errors="ignore")
    scores = positive_probability(model, features)
    output = pd.DataFrame(
        {
            ID_COLUMN: frame[ID_COLUMN].to_numpy(),
            "attrition_probability": scores,
        }
    )
    output["risk_level"] = pd.cut(
        output["attrition_probability"],
        bins=[-0.01, threshold * 0.6, threshold, 1.0],
        labels=["Low", "Medium", "High"],
    )
    output["recommended_action"] = frame.apply(_recommendations, axis=1).to_numpy()
    if TARGET_COLUMN in frame.columns:
        output["actual"] = frame[TARGET_COLUMN].to_numpy()
    return output.sort_values("attrition_probability", ascending=False, ignore_index=True)
