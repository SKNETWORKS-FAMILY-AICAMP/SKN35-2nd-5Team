"""Train, evaluate, rank, and persist classical ML baselines."""

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.load_data.loader import split_features_target
from src.ml.evaluation import evaluate_classifier
from src.ml.models import get_model_candidates
from src.ml.preprocessing import build_preprocessor
from src.utils.constants import RANDOM_STATE, TEST_SIZE
from src.utils.paths import (
    BEST_ML_MODEL_PATH,
    ML_LEADERBOARD_PATH,
    MODELS_DIR,
    ensure_artifact_dirs,
)


@dataclass
class TrainingResult:
    name: str
    pipeline: Pipeline
    metrics: dict[str, Any]
    train_seconds: float
    artifact_path: str | None = None


def make_train_valid_split(
    frame: pd.DataFrame,
    *,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    features, target = split_features_target(frame)
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )


def train_ml_models(
    frame: pd.DataFrame,
    *,
    selected: list[str] | tuple[str, ...] | None = None,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    save_artifacts: bool = True,
) -> tuple[list[TrainingResult], pd.DataFrame, dict[str, str]]:
    """Train requested baselines against the same holdout split."""
    x_train, x_valid, y_train, y_valid = make_train_valid_split(
        frame,
        test_size=test_size,
        random_state=random_state,
    )
    models, unavailable = get_model_candidates(
        random_state=random_state,
        selected=selected,
    )
    if not models:
        raise RuntimeError("학습 가능한 모델이 없습니다. 프로젝트 의존성을 확인하세요.")

    results: list[TrainingResult] = []
    if save_artifacts:
        ensure_artifact_dirs()

    for name, estimator in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(x_train)),
                ("model", estimator),
            ]
        )
        started = perf_counter()
        pipeline.fit(x_train, y_train)
        elapsed = perf_counter() - started
        metrics = evaluate_classifier(pipeline, x_valid, y_valid)
        artifact_path: str | None = None
        if save_artifacts:
            path = MODELS_DIR / f"{name}.joblib"
            joblib.dump(pipeline, path, compress=3)
            artifact_path = str(path)
        results.append(
            TrainingResult(
                name=name,
                pipeline=pipeline,
                metrics=metrics,
                train_seconds=elapsed,
                artifact_path=artifact_path,
            )
        )

    leaderboard = pd.DataFrame(
        [
            {
                "model": result.name,
                **result.metrics,
                "train_seconds": result.train_seconds,
                "artifact_path": result.artifact_path,
            }
            for result in results
        ]
    ).sort_values(["roc_auc", "f1"], ascending=False, ignore_index=True)

    if save_artifacts:
        leaderboard.to_csv(ML_LEADERBOARD_PATH, index=False)
        best_name = str(leaderboard.iloc[0]["model"])
        best_pipeline = next(result.pipeline for result in results if result.name == best_name)
        joblib.dump(best_pipeline, BEST_ML_MODEL_PATH, compress=3)
    return results, leaderboard, unavailable
