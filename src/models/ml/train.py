"""
머신러닝 모델 공통 학습·비교 관리자
=====================================

(모델 짜기전에 먼저 읽어야 함!!!!!)

이 파일은 Logistic Regression, Random Forest, XGBoost, LightGBM을 직접
구현하는 파일이 아니다. 팀원들이 각자 구현한 모델을 동일한 데이터와 평가
조건으로 학습시켜 공정하게 비교하고, 가장 좋은 모델을 저장하는 공통 실행 파일이다.

전체 처리 순서
--------------
1. ``train_processed.csv``와 ``test_processed.csv``를 불러온다.
2. 모든 피처가 숫자이고 타깃이 퇴사 ``1``, 재직 ``0``인지 검증한다.
3. ``train_processed.csv``만 학습 80%, 검증 20%로 계층 분할한다.
4. 각 모델 파일의 생성 함수를 불러와 동일한 학습셋으로 학습한다.
5. 검증셋의 ROC-AUC와 F1 등을 계산해 후보 모델의 순위를 정한다.
6. 검증 ROC-AUC 1위 모델을 전체 ``train_processed.csv``로 다시 학습한다.
7. 모델 선택에 사용하지 않은 ``test_processed.csv``로 최종 성능을 한 번 평가한다.
8. 공통 저장 유틸을 호출해 ``artifacts/ml``, ``artifacts/reports``에 저장한다.

팀원 작업 규칙
---------------
- 개별 모델 담당자는 원칙적으로 이 ``train.py``를 수정하지 않는다.
- 담당 모델 파일에 아래 이름의 생성 함수만 구현하면 공통 학습에 자동 포함된다.

  - ``logistic_regression.py`` → ``create_logistic_regression()``
  - ``random_forest.py`` → ``create_random_forest()``
  - ``xgboost.py`` → ``create_xgboost()``
  - ``lightgbm.py`` → ``create_lightgbm()``

- 생성 함수는 아직 학습되지 않은 scikit-learn 호환 분류기 또는 Pipeline을 반환해야
  한다. 데이터 로드·분할·평가·저장은 ``src/data``와 ``src/utils``의 공통 함수를 쓴다.
- 모델별 하이퍼파라미터 튜닝은 ``xgboost_tuning.py``처럼 별도 파일에서 진행한다.
- 아직 구현되지 않았거나 필요한 라이브러리가 없는 모델은 자동으로 제외되므로 다른
  팀원의 모델 학습을 막지 않는다.
- 딥러닝 모델은 ``src/models/dl``의 별도 학습 흐름을 사용하며 이 파일의 대상이 아니다.

실행 방법
---------
전체 구현 모델을 학습·비교하고 산출물을 저장하려면 프로젝트 루트에서 실행한다.

``uv run python -m src.models.ml.train``

개발 중 특정 모델만 시험하고 기존 산출물을 덮어쓰지 않으려면 다음처럼 호출한다.

``run_training(["xgboost"], save_artifacts=False)``

공통 학습 조건을 바꿔야 할 때만 이 파일의 담당자와 협의하여 수정한다. 이렇게 역할을
분리하면 여러 팀원이 각자 다른 모델 파일을 개발하고 병합해도 ``train.py``에서 충돌할
가능성을 줄일 수 있다.
"""

from time import perf_counter
from typing import Any, Callable

import pandas as pd
from sklearn.pipeline import Pipeline

from src.data.loader import load_processed_train_test_features
from src.utils.artifact_io import save_ml_artifacts
from src.utils.constants import ML_RESULT_COLUMNS
from src.utils.metrics import evaluate_sklearn_model
from src.utils.ml_training import (
    create_training_pipeline,
    load_model_factories,
    make_train_validation_split,
)
from src.utils.paths import (
    BEST_ML_MODEL_PATH,
    BEST_ML_TEST_METRICS_PATH,
    ML_ARTIFACTS_DIR,
    ML_LEADERBOARD_PATH,
)


def train_candidates(
    factories: dict[str, Callable[[], Any]],
    train_features: pd.DataFrame,
    train_target: pd.Series,
    validation_features: pd.DataFrame,
    validation_target: pd.Series,
) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    """모든 후보를 같은 학습셋으로 학습하고 검증셋 성능으로 순위를 정한다."""

    results: list[dict[str, Any]] = []
    fitted_models: dict[str, Pipeline] = {}

    for model_name, factory in factories.items():
        # 모델마다 새로운 전처리기와 모델 객체를 만들어 조건을 동일하게 맞춘다.
        model = create_training_pipeline(factory(), train_features)
        started_at = perf_counter()
        model.fit(train_features, train_target)
        elapsed = perf_counter() - started_at

        # 모델 선택에는 최종 테스트셋이 아닌 검증셋 성능만 사용한다.
        metrics = evaluate_sklearn_model(
            model,
            validation_features,
            validation_target,
        )
        metrics.update(
            {
                "model": model_name,
                "train_seconds": elapsed,
                "artifact_path": str(ML_ARTIFACTS_DIR / f"{model_name}.joblib"),
            }
        )
        results.append(metrics)
        fitted_models[model_name] = model

    if not results:
        raise RuntimeError("학습할 수 있는 머신러닝 모델이 없습니다.")

    # ROC-AUC를 우선 기준으로, 동률이면 F1이 높은 모델을 위에 배치한다.
    leaderboard = (
        pd.DataFrame(results)[ML_RESULT_COLUMNS]
        .sort_values(["roc_auc", "f1"], ascending=False)
        .reset_index(drop=True)
    )
    return leaderboard, fitted_models


def refit_best_model(
    best_model_name: str,
    factory: Callable[[], Any],
    training_features: pd.DataFrame,
    training_target: pd.Series,
    test_features: pd.DataFrame,
    test_target: pd.Series,
) -> tuple[Pipeline, dict[str, Any]]:
    """선정된 모델을 전체 학습 데이터로 재학습하고 외부 테스트를 한 번 수행한다."""

    # 검증 과정에서 선택된 모델을 train_processed.csv 전체로 처음부터 다시 학습한다.
    model = create_training_pipeline(factory(), training_features)
    started_at = perf_counter()
    model.fit(training_features, training_target)
    elapsed = perf_counter() - started_at

    metrics = evaluate_sklearn_model(model, test_features, test_target)
    metrics.update(
        {
            "model": best_model_name,
            "train_seconds": elapsed,
        }
    )
    return model, metrics


def run_training(
    selected: list[str] | tuple[str, ...] | None = None,
    *,
    save_artifacts: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, str]]:
    """공통 ML 학습 흐름을 실행하고 검증표와 최종 테스트 결과를 반환한다."""

    features, target, test_features, test_target = load_processed_train_test_features()

    (
        train_features,
        validation_features,
        train_target,
        validation_target,
    ) = make_train_validation_split(features, target)

    factories, unavailable = load_model_factories(selected)

    leaderboard, candidate_models = train_candidates(
        factories,
        train_features,
        train_target,
        validation_features,
        validation_target,
    )

    best_model_name = str(leaderboard.iloc[0]["model"])

    # 모델 선택이 끝났으므로 최종 모델은 train_processed.csv 전체로 재학습한다.
    best_model, final_metrics = refit_best_model(
        best_model_name,
        factories[best_model_name],
        features,
        target,
        test_features,
        test_target,
    )

    if save_artifacts:
        save_ml_artifacts(
            leaderboard,
            candidate_models,
            best_model,
            final_metrics,
        )

    return leaderboard, final_metrics, unavailable


def main() -> None:
    """터미널에서 공통 머신러닝 학습을 실행하는 진입점."""

    leaderboard, final_metrics, unavailable = run_training()

    print("\n검증 데이터 기준 모델 순위")
    print("=" * 100)
    print(leaderboard.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    print("\n선정 모델의 최종 테스트 성능")
    print("=" * 100)
    print(
        pd.DataFrame([final_metrics]).to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    if unavailable:
        print("\n현재 사용할 수 없는 모델")
        for model_name, reason in unavailable.items():
            print(f"- {model_name}: {reason}")

    print(f"\n검증 결과표: {ML_LEADERBOARD_PATH}")
    print(f"최고 모델: {BEST_ML_MODEL_PATH}")
    print(f"최종 테스트 결과: {BEST_ML_TEST_METRICS_PATH}")


if __name__ == "__main__":
    main()
