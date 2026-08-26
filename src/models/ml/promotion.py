"""튜닝 모델을 공통 ML 리더보드와 최고 모델로 승격하는 로직."""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from .train import (
    BEST_MODEL_PATH,
    FINAL_METRICS_PATH,
    LEADERBOARD_PATH,
    RESULT_COLUMNS,
)


def _score(metrics: dict[str, Any] | pd.Series) -> tuple[float, float]:
    """공통 우선순위인 ROC-AUC, F1 순으로 비교값을 만든다."""

    return float(metrics["roc_auc"]), float(metrics["f1"])


def promote_tuned_model(
    model_name: str,
    model: Pipeline,
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    artifact_path: Path,
) -> tuple[bool, str]:
    """기본 모델보다 좋은 튜닝 모델을 기록하고 전체 1위이면 최고 모델로 승격한다."""

    if not LEADERBOARD_PATH.exists():
        return False, "기본 리더보드가 없어 승격을 건너뜁니다. 먼저 train.py를 실행하세요."

    leaderboard = pd.read_csv(LEADERBOARD_PATH)
    current_rows = leaderboard[leaderboard["model"] == model_name]
    if not current_rows.empty and _score(validation_metrics) <= _score(current_rows.iloc[0]):
        return False, f"튜닝 {model_name}이 기존 {model_name} validation 성능보다 높지 않습니다."

    tuned_row = {
        "model": model_name,
        **validation_metrics,
        "train_seconds": float("nan"),
        "artifact_path": str(artifact_path),
    }
    leaderboard = leaderboard[leaderboard["model"] != model_name]
    leaderboard = pd.concat([leaderboard, pd.DataFrame([tuned_row])], ignore_index=True)
    leaderboard = (
        leaderboard[RESULT_COLUMNS]
        .sort_values(["roc_auc", "f1"], ascending=False)
        .reset_index(drop=True)
    )
    leaderboard.to_csv(LEADERBOARD_PATH, index=False)

    if str(leaderboard.iloc[0]["model"]) != model_name:
        return False, f"튜닝 {model_name}은 기본 버전보다 좋지만 전체 모델 1위는 아닙니다."

    BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    FINAL_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, BEST_MODEL_PATH)
    pd.DataFrame([{"model": model_name, **test_metrics}]).to_csv(
        FINAL_METRICS_PATH,
        index=False,
    )
    return True, f"튜닝 {model_name}을 전체 최고 모델로 승격했습니다."
