"""사용 가능한 머신러닝 모델을 공통 조건으로 학습·비교·저장하는 모듈."""

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
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"
MODELS_DIR = PROJECT_ROOT / "artifacts" / "models"
REPORTS_DIR = PROJECT_ROOT / "artifacts" / "reports"
LEADERBOARD_PATH = REPORTS_DIR / "ml_leaderboard.csv"
FINAL_METRICS_PATH = REPORTS_DIR / "best_ml_test_metrics.csv"
BEST_MODEL_PATH = MODELS_DIR / "best_ml_model.joblib"

# 전체 데이터의 20%는 최종 테스트용으로 끝까지 분리해 둔다.
# 나머지 80% 중 25%를 검증용으로 사용하면 전체 비율은 60:20:20이 된다.
FINAL_TEST_SIZE = 0.20
VALIDATION_SIZE_WITHIN_DEVELOPMENT = 0.25
ID_COLUMNS = ("Employee ID", "EmployeeNumber")

# 퇴사 여부를 모델이 사용할 이진 숫자로 명시적으로 변환한다.
# LabelEncoder의 알파벳 정렬에 맡기지 않아 타깃 방향이 뒤집히는 문제를 방지한다.
TARGET_MAPPING = {"Left": 1, "Stayed": 0}

# 각 모델 파일과 해당 파일이 제공해야 하는 생성 함수 이름이다.
# 팀원은 담당 모델 파일에 아래 생성 함수만 구현하면 공통 학습에 자동으로 포함된다.
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

# 모든 모델이 동일한 순서와 이름으로 결과표를 만들도록 출력 컬럼을 고정한다.
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
    data_path: Path = RAW_DATA_PATH,
) -> tuple[pd.DataFrame, pd.Series]:
    """원본 데이터를 불러오고 퇴사 타깃을 Left=1로 명시 변환한다."""

    # 잘못된 실행 경로나 누락된 데이터 파일을 초기에 확인한다.
    if not data_path.exists():
        raise FileNotFoundError(f"학습 데이터가 없습니다: {data_path}")

    data = pd.read_csv(data_path)
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"타깃 컬럼이 없습니다: {TARGET_COLUMN}")

    # Left와 Stayed 이외의 예상하지 못한 값이 있으면 잘못 학습하지 않고 중단한다.
    unexpected_labels = set(data[TARGET_COLUMN].dropna().unique()) - set(TARGET_MAPPING)
    if unexpected_labels:
        labels = ", ".join(map(str, sorted(unexpected_labels)))
        raise ValueError(f"예상하지 못한 타깃 값입니다: {labels}")

    # 퇴사자는 1, 재직자는 0으로 변환한다.
    target = data[TARGET_COLUMN].map(TARGET_MAPPING)
    if target.isna().any():
        raise ValueError("타깃에 결측치 또는 변환되지 않은 값이 있습니다.")

    # 타깃과 직원 식별자는 예측 입력 피처에서 제외한다.
    drop_columns = [TARGET_COLUMN, *[column for column in ID_COLUMNS if column in data.columns]]
    features = data.drop(columns=drop_columns)
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

    # 데이터 타입을 기준으로 수치형과 범주형 컬럼을 자동 구분한다.
    numeric_columns = features.select_dtypes(include="number").columns.tolist()
    categorical_columns = features.select_dtypes(exclude="number").columns.tolist()

    # 수치형 결측치는 학습 데이터의 중앙값으로 채운다.
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )

    # 범주형 결측치는 최빈값으로 채우고 원핫 인코딩한다.
    # 검증/테스트에 처음 등장한 범주는 오류 없이 무시한다.
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

    # 두 종류의 전처리를 원래 컬럼에 각각 적용한 뒤 하나의 배열로 결합한다.
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

    # 1. 원본 데이터 로드 및 타깃 변환
    features, target = load_training_data()

    # 2. 모든 모델이 함께 사용할 데이터 분할 생성
    (
        train_features,
        validation_features,
        test_features,
        train_target,
        validation_target,
        test_target,
    ) = make_shared_splits(features, target)

    # 3. 구현이 완료된 모델 생성 함수만 불러오기
    factories, unavailable = load_model_factories(selected)

    # 4. 동일한 학습/검증 데이터로 후보 모델 비교
    leaderboard, candidate_models = train_candidates(
        factories,
        train_features,
        train_target,
        validation_features,
        validation_target,
    )

    # 5. 검증 ROC-AUC 1위 모델 선택
    best_model_name = str(leaderboard.iloc[0]["model"])

    # 6. 학습셋과 검증셋을 합쳐 최종 모델 재학습
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

    # 7. 실제 실행에서는 산출물을 저장하고, 테스트에서는 저장을 끌 수 있다.
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
