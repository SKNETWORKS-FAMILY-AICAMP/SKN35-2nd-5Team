"""ML과 DL 학습 산출물을 일관된 경로와 형식으로 저장한다."""

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.utils.paths import (
    BEST_ML_MODEL_PATH,
    BEST_ML_TEST_METRICS_PATH,
    DL_METRICS_PATH,
    ML_ARTIFACTS_DIR,
    ML_LEADERBOARD_PATH,
    MLP_BEST_PARAMS_PATH,
    MLP_METADATA_PATH,
    MLP_MODEL_PATH,
    MLP_PREPROCESSOR_PATH,
    MLP_THRESHOLD_PATH,
    ensure_artifact_dirs,
)

MetricRow = Mapping[str, Any]
MetricData = pd.DataFrame | MetricRow | Sequence[MetricRow]


def save_metrics_csv(metrics: MetricData, output_path: Path) -> Path:
    """단일/복수 성능 행 또는 DataFrame을 인덱스 없는 CSV로 저장한다."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(metrics, pd.DataFrame):
        report = metrics.copy()
    elif isinstance(metrics, Mapping):
        report = pd.DataFrame([metrics])
    else:
        report = pd.DataFrame(metrics)

    # 팀원마다 프로젝트 절대경로가 달라도 리포트가 깨지지 않도록
    # artifact_path에는 artifacts/부터 시작하는 프로젝트 상대경로만 저장한다.
    if "artifact_path" in report.columns:
        report["artifact_path"] = report["artifact_path"].apply(
            _normalize_artifact_reference
        )
    report.to_csv(output_path, index=False)
    return output_path


def _normalize_artifact_reference(value: Any) -> Any:
    """Windows/Unix 절대경로에서 프로젝트의 artifacts 이하 경로만 남긴다."""

    if pd.isna(value) or not str(value).strip():
        return value

    normalized = str(value).replace("\\", "/")
    marker = "artifacts/"
    marker_index = normalized.lower().find(marker)
    if marker_index >= 0:
        artifact_reference = normalized[marker_index:]
        legacy_ml_prefix = "artifacts/models/"
        if artifact_reference.lower().startswith(legacy_ml_prefix):
            filename = artifact_reference[len(legacy_ml_prefix) :]
            return f"artifacts/ml/{filename}"
        return artifact_reference
    return normalized


def save_ml_artifacts(
    leaderboard: pd.DataFrame,
    candidate_models: Mapping[str, Any],
    best_model: Any,
    final_metrics: MetricRow,
) -> None:
    """ML 후보/최고 모델과 검증·테스트 리포트를 저장한다."""

    ensure_artifact_dirs()
    for model_name, model in candidate_models.items():
        joblib.dump(model, ML_ARTIFACTS_DIR / f"{model_name}.joblib")

    joblib.dump(best_model, BEST_ML_MODEL_PATH)
    save_metrics_csv(leaderboard, ML_LEADERBOARD_PATH)
    save_metrics_csv(final_metrics, BEST_ML_TEST_METRICS_PATH)


def save_dl_artifacts(
    model: Any,
    preprocessor: Any,
    best_params: Mapping[str, Any],
    best_threshold: float,
    metadata: Mapping[str, Any],
    test_metrics: MetricRow,
) -> None:
    """DL 모델 구성요소와 테스트 성능 리포트를 저장한다."""

    import torch

    ensure_artifact_dirs()
    model_cpu = deepcopy(model).to("cpu")
    torch.save(model_cpu.state_dict(), MLP_MODEL_PATH)
    joblib.dump(preprocessor, MLP_PREPROCESSOR_PATH)
    joblib.dump(dict(best_params), MLP_BEST_PARAMS_PATH)
    joblib.dump(float(best_threshold), MLP_THRESHOLD_PATH)
    joblib.dump(dict(metadata), MLP_METADATA_PATH)
    save_metrics_csv(test_metrics, DL_METRICS_PATH)


def save_tuned_ml_artifacts(
    model: Any,
    model_path: Path,
    model_name: str,
    metrics: MetricRow,
    metrics_path: Path,
    params: Mapping[str, Any],
    params_path: Path,
) -> None:
    """튜닝된 ML 모델, 성능 CSV, 하이퍼파라미터 JSON을 저장한다."""

    ensure_artifact_dirs()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    save_metrics_csv({"model": model_name, **metrics}, metrics_path)
    params_path.parent.mkdir(parents=True, exist_ok=True)
    params_path.write_text(
        json.dumps(dict(params), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_promoted_ml_artifacts(model: Any, test_metrics: MetricRow) -> None:
    """튜닝 결과가 전체 1위일 때 최고 ML 모델과 테스트 지표를 갱신한다."""

    ensure_artifact_dirs()
    joblib.dump(model, BEST_ML_MODEL_PATH)
    save_metrics_csv(test_metrics, BEST_ML_TEST_METRICS_PATH)
