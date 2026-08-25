"""
머신러닝 모델 공통 학습·비교 관리자  
=====================================

(모델 짜기전에 먼저 읽어야 함!!!!!)

이 파일은 Logistic Regression, Random Forest, XGBoost, LightGBM을 직접
구현하는 파일이 아니다. 팀원들이 각자 구현한 모델을 동일한 데이터와 평가
조건으로 학습시켜 공정하게 비교하고, 가장 좋은 모델을 저장하는 공통 실행 파일이다.

전체 처리 순서
--------------
1. ``data/preprocessing/train_processed.csv``를 불러온다.
2. CSV 저장 인덱스를 제거하고 모든 피처가 숫자인지 검증한다.
3. 저장된 타깃 방향을 퇴사 ``Left=1``, 재직 ``Stayed=0``으로 통일한다.
4. 모든 모델이 함께 사용할 데이터를 학습 60%, 검증 20%, 최종 테스트 20%로
   한 번만 계층 분할한다.
5. 각 모델 파일의 생성 함수를 불러와 동일한 학습셋으로 학습한다.
6. 검증셋의 ROC-AUC와 F1 등을 계산해 후보 모델의 순위를 정한다.
7. 검증 ROC-AUC 1위 모델을 학습+검증 데이터로 다시 학습한다.
8. 마지막까지 사용하지 않은 최종 테스트셋으로 일반화 성능을 한 번 평가한다.
9. 모델 파일과 성능표를 ``artifacts/models``, ``artifacts/reports``에 저장한다.

팀원 작업 규칙
---------------
- 개별 모델 담당자는 원칙적으로 이 ``train.py``를 수정하지 않는다.
- 담당 모델 파일에 아래 이름의 생성 함수만 구현하면 공통 학습에 자동 포함된다.

  - ``logistic_regression.py`` → ``create_logistic_regression()``
  - ``random_forest.py`` → ``create_random_forest()``
  - ``xgboost.py`` → ``create_xgboost()``
  - ``lightgbm.py`` → ``create_lightgbm()``

- 생성 함수는 아직 학습되지 않은 scikit-learn 호환 분류기 또는 Pipeline을 반환해야
  한다. 데이터 로드, 데이터 분할, 성능 평가, 모델 저장은 각 모델 파일에서 중복해서
  작성하지 않고 이 파일에 맡긴다.
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

from importlib import import_module
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .utils import RANDOM_STATE, TARGET_COLUMN, evaluate_model


# 현재 파일 위치를 기준으로 프로젝트 최상위 경로를 계산한다.
# 실행 위치가 달라져도 데이터와 산출물 경로가 바뀌지 않도록 절대 경로를 사용한다.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DATA_PATH = (
    PROJECT_ROOT / "data" / "preprocessing" / "train_processed.csv"
)
MODELS_DIR = PROJECT_ROOT / "artifacts" / "models"
REPORTS_DIR = PROJECT_ROOT / "artifacts" / "reports"
LEADERBOARD_PATH = REPORTS_DIR / "ml_leaderboard.csv"
FINAL_METRICS_PATH = REPORTS_DIR / "best_ml_test_metrics.csv"
BEST_MODEL_PATH = MODELS_DIR / "best_ml_model.joblib"

# 전체 데이터의 20%는 최종 테스트용으로 끝까지 분리해 둔다.
# 나머지 80% 중 25%를 검증용으로 사용하면 전체 비율은 60:20:20이 된다.
FINAL_TEST_SIZE = 0.20
VALIDATION_SIZE_WITHIN_DEVELOPMENT = 0.25

# train_processed.csv는 LabelEncoder 결과로 Left=0, Stayed=1이 저장돼 있다.
# 모델과 평가에서는 퇴사를 양성 클래스로 사용하므로 0과 1을 반대로 변환한다.
PROCESSED_TARGET_MAPPING = {0: 1, 1: 0}

MODEL_SPECS = {
    "logistic_regression": (
        "logistic_regression",
        "create_logistic_regression",
    ),
    "random_forest": (
        "random_forest",
        "create_random_forest",
    ),
    "xgboost": (
        "xgboost",
        "create_xgboost",
    ),
    "lightgbm": (
        "lightgbm",
        "create_lightgbm",
    ),
}

RESULT_COLUMNS = [
    "model",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "average_precision",
    "tn",
    "fp",
    "fn",
    "tp",
    "train_seconds",
    "artifact_path",
]


def load_training_data(
    data_path: Path = PROCESSED_DATA_PATH,
) -> tuple[pd.DataFrame, pd.Series]:
    """전처리 데이터를 불러오고 퇴사 타깃을 Left=1로 변환한다."""

    if not data_path.exists():
        raise FileNotFoundError(f"학습 데이터가 없습니다: {data_path}")

    data = pd.read_csv(data_path)
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"타깃 컬럼이 없습니다: {TARGET_COLUMN}")

    # CSV 타깃이 숫자 0/1인지 확인하고, 변환 불가능한 값은 오류로 처리한다.
    encoded_target = pd.to_numeric(data[TARGET_COLUMN], errors="coerce")
    if encoded_target.isna().any():
        raise ValueError("전처리 타깃에 숫자로 변환할 수 없는 값이 있습니다.")

    unexpected_labels = set(encoded_target.unique()) - set(PROCESSED_TARGET_MAPPING)
    if unexpected_labels:
        labels = ", ".join(map(str, sorted(unexpected_labels)))
        raise ValueError(f"예상하지 못한 타깃 값입니다: {labels}")

    # 저장된 Left=0, Stayed=1을 학습 기준인 Left=1, Stayed=0으로 뒤집는다.
    target = encoded_target.map(PROCESSED_TARGET_MAPPING)

    # pandas가 CSV 저장 인덱스를 Unnamed: 0 같은 이름으로 읽으므로 입력에서 제외한다.
    saved_index_columns = [
        column for column in data.columns if column.startswith("Unnamed:")
    ]
    features = data.drop(columns=[TARGET_COLUMN, *saved_index_columns])

    # 원핫 인코딩 결과가 True/False로 저장된 경우 모델 입력용 0/1로 변환한다.
    bool_columns = features.select_dtypes(include="bool").columns
    features[bool_columns] = features[bool_columns].astype("int8")

    # train_processed.csv가 완전히 수치화됐는지 마지막으로 검증한다.
    non_numeric_columns = features.select_dtypes(exclude="number").columns.tolist()
    if non_numeric_columns:
        raise ValueError(
            "전처리되지 않은 피처가 있습니다: " + ", ".join(non_numeric_columns)
        )

    return features, target.astype("int8").rename(TARGET_COLUMN)


def make_shared_splits(
    features: pd.DataFrame,
    target: pd.Series,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """모든 모델이 함께 사용할 60/20/20 계층 분할을 생성한다."""

    # 먼저 전체의 20%를 최종 테스트셋으로 분리한다.
    # stratify를 사용해 퇴사/재직 비율을 원본과 비슷하게 유지한다.
    development_features, test_features, development_target, test_target = train_test_split(
        features,
        target,
        test_size=FINAL_TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    # 남은 개발 데이터의 25%를 검증셋으로 분리한다.
    # 결과적으로 전체 데이터는 학습 60%, 검증 20%, 최종 테스트 20%가 된다.
    train_features, validation_features, train_target, validation_target = train_test_split(
        development_features,
        development_target,
        test_size=VALIDATION_SIZE_WITHIN_DEVELOPMENT,
        random_state=RANDOM_STATE,
        stratify=development_target,
    )
    return (
        train_features,
        validation_features,
        test_features,
        train_target,
        validation_target,
        test_target,
    )


def create_common_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """현재 학습 분할에만 fit되는 공통 전처리기를 생성한다."""

    numeric_columns = features.select_dtypes(include="number").columns.tolist()
    categorical_columns = features.select_dtypes(exclude="number").columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def create_training_pipeline(
    model: Any,
    features: pd.DataFrame,
) -> Pipeline:
    """팀원이 만든 모델 앞에 공통 전처리 단계를 연결한다."""

    # fit 호출 시 전처리기 학습 후 모델 학습이 순서대로 실행된다.
    # 따라서 검증/테스트 데이터의 정보가 전처리에 미리 사용되지 않는다.
    return Pipeline(
        steps=[
            ("preprocessor", create_common_preprocessor(features)),
            ("model", model),
        ]
    )

def load_model_factories(
    selected: list[str] | tuple[str, ...] | None = None,
) -> tuple[dict[str, Callable[[], Any]], dict[str, str]]:
    """구현이 끝난 모델 생성 함수를 불러오고 미완성 모델은 따로 기록한다."""

    # selected가 없으면 등록된 네 모델을 모두 불러온다.
    requested = list(selected) if selected is not None else list(MODEL_SPECS)
    unknown = sorted(set(requested) - set(MODEL_SPECS))
    if unknown:
        raise ValueError(f"등록되지 않은 모델입니다: {', '.join(unknown)}")

    factories: dict[str, Callable[[], Any]] = {}
    unavailable: dict[str, str] = {}
    package = __package__ or "src.models.ml"

    # 팀원이 아직 구현하지 않은 빈 파일이나 설치되지 않은 라이브러리가 있어도
    # 다른 모델의 학습까지 중단되지 않도록 모델별로 예외를 처리한다.
    for model_name in requested:
        module_name, factory_name = MODEL_SPECS[model_name]
        try:
            module = import_module(f"{package}.{module_name}")
            factory = getattr(module, factory_name)
        except (ImportError, AttributeError) as exc:
            unavailable[model_name] = str(exc)
        else:
            factories[model_name] = factory

    return factories, unavailable


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
        metrics = evaluate_model(
            model,
            validation_features,
            validation_target,
        )
        metrics.update(
            {
                "model": model_name,
                "train_seconds": elapsed,
                "artifact_path": str(MODELS_DIR / f"{model_name}.joblib"),
            }
        )
        results.append(metrics)
        fitted_models[model_name] = model

    if not results:
        raise RuntimeError("학습할 수 있는 머신러닝 모델이 없습니다.")

    # ROC-AUC를 우선 기준으로, 동률이면 F1이 높은 모델을 위에 배치한다.
    leaderboard = (
        pd.DataFrame(results)[RESULT_COLUMNS]
        .sort_values(["roc_auc", "f1"], ascending=False)
        .reset_index(drop=True)
    )
    return leaderboard, fitted_models


def refit_best_model(
    best_model_name: str,
    factory: Callable[[], Any],
    development_features: pd.DataFrame,
    development_target: pd.Series,
    test_features: pd.DataFrame,
    test_target: pd.Series,
) -> tuple[Pipeline, dict[str, Any]]:
    """선정된 모델을 학습+검증 데이터로 재학습하고 최종 테스트를 한 번 수행한다."""

    # 검증 과정에서 선택된 모델을 더 많은 개발 데이터로 처음부터 다시 학습한다.
    model = create_training_pipeline(factory(), development_features)
    started_at = perf_counter()
    model.fit(development_features, development_target)
    elapsed = perf_counter() - started_at

    metrics = evaluate_model(model, test_features, test_target)
    metrics.update(
        {
            "model": best_model_name,
            "train_seconds": elapsed,
        }
    )
    return model, metrics


def save_training_artifacts(
    leaderboard: pd.DataFrame,
    candidate_models: dict[str, Pipeline],
    best_model: Pipeline,
    final_metrics: dict[str, Any],
) -> None:
    """공통 성능표, 후보 모델, 최종 최고 모델과 테스트 결과를 저장한다."""

    # 산출물 폴더가 없으면 자동으로 생성한다.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 검증 단계에서 학습된 각 후보 모델을 모델 이름으로 저장한다.
    for model_name, model in candidate_models.items():
        joblib.dump(model, MODELS_DIR / f"{model_name}.joblib")

    # 최종 선정 모델과 Streamlit에서 읽을 CSV 리포트를 저장한다.
    joblib.dump(best_model, BEST_MODEL_PATH)
    leaderboard.to_csv(LEADERBOARD_PATH, index=False)
    pd.DataFrame([final_metrics]).to_csv(FINAL_METRICS_PATH, index=False)


def run_training(
    selected: list[str] | tuple[str, ...] | None = None,
    *,
    save_artifacts: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, str]]:
    """공통 ML 학습 흐름을 실행하고 검증표와 최종 테스트 결과를 반환한다."""

    features, target = load_training_data()

    (
        train_features,
        validation_features,
        test_features,
        train_target,
        validation_target,
        test_target,
    ) = make_shared_splits(features, target)

    factories, unavailable = load_model_factories(selected)

    leaderboard, candidate_models = train_candidates(
        factories,
        train_features,
        train_target,
        validation_features,
        validation_target,
    )

    best_model_name = str(leaderboard.iloc[0]["model"])

    development_features = pd.concat([train_features, validation_features]).sort_index()
    development_target = pd.concat([train_target, validation_target]).sort_index()
    best_model, final_metrics = refit_best_model(
        best_model_name,
        factories[best_model_name],
        development_features,
        development_target,
        test_features,
        test_target,
    )

    if save_artifacts:
        save_training_artifacts(
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

    print(f"\n검증 결과표: {LEADERBOARD_PATH}")
    print(f"최고 모델: {BEST_MODEL_PATH}")
    print(f"최종 테스트 결과: {FINAL_METRICS_PATH}")


if __name__ == "__main__":
    main()
