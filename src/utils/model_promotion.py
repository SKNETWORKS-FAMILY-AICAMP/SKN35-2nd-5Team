"""튜닝된 ML 모델을 공통 리더보드와 최고 모델로 승격한다."""

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline

from src.utils.artifact_io import save_metrics_csv, save_promoted_ml_artifacts
from src.utils.constants import ML_RESULT_COLUMNS
from src.utils.paths import ML_LEADERBOARD_PATH, project_relative_path


def _score(metrics: dict[str, Any] | pd.Series) -> tuple[float, float]:
    return float(metrics["roc_auc"]), float(metrics["f1"])


def promote_tuned_model(
    model_name: str,
    model: Pipeline,
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    artifact_path: Path,
) -> tuple[bool, str]:
    """기본 모델보다 좋은 튜닝 모델을 기록하고 전체 1위이면 승격한다."""

    if not ML_LEADERBOARD_PATH.exists():
        return False, "기본 리더보드가 없어 승격을 건너뜁니다. 먼저 train.py를 실행하세요."

    leaderboard = pd.read_csv(ML_LEADERBOARD_PATH)
    current_rows = leaderboard[leaderboard["model"] == model_name]
    if not current_rows.empty and _score(validation_metrics) <= _score(current_rows.iloc[0]):
        return False, f"튜닝 {model_name}이 기존 {model_name} validation 성능보다 높지 않습니다."

    tuned_row = {
        "model": model_name,
        **validation_metrics,
        "train_seconds": float("nan"),
        "artifact_path": project_relative_path(artifact_path),
    }
    leaderboard = leaderboard[leaderboard["model"] != model_name]
    leaderboard = pd.concat([leaderboard, pd.DataFrame([tuned_row])], ignore_index=True)
    leaderboard = (
        leaderboard[ML_RESULT_COLUMNS]
        .sort_values(["roc_auc", "f1"], ascending=False)
        .reset_index(drop=True)
    )
    save_metrics_csv(leaderboard, ML_LEADERBOARD_PATH)

    if str(leaderboard.iloc[0]["model"]) != model_name:
        return False, f"튜닝 {model_name}은 기본 버전보다 좋지만 전체 모델 1위는 아닙니다."

    save_promoted_ml_artifacts(model, {"model": model_name, **test_metrics})
    return True, f"튜닝 {model_name}을 전체 최고 모델로 승격했습니다."
