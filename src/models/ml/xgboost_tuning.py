"""공통 학습 구조를 사용하는 XGBoost 하이퍼파라미터 튜닝 모듈."""

import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data.loader import load_processed_train_test_features
from src.utils.artifact_io import save_tuned_ml_artifacts
from src.utils.constants import RANDOM_STATE
from src.utils.metrics import evaluate_sklearn_model
from src.utils.ml_training import (
    create_common_preprocessor,
    create_training_pipeline,
    make_train_validation_split,
)
from src.utils.model_promotion import promote_tuned_model
from src.utils.paths import ML_ARTIFACTS_DIR, REPORTS_DIR

# 모든 튜닝 단계에서 동일한 교차검증 조건을 사용한다.
CV_FOLDS = 5
TUNING_LEARNING_RATE = 0.1
TUNING_N_ESTIMATORS = 300
FINAL_LEARNING_RATE = 0.05
FINAL_N_ESTIMATORS = 2000
EARLY_STOPPING_ROUNDS = 50
TUNED_MODEL_PATH = ML_ARTIFACTS_DIR / "xgboost_tuned.joblib"
TUNED_METRICS_PATH = REPORTS_DIR / "xgboost_tuned_metrics.csv"
TUNED_PARAMS_PATH = REPORTS_DIR / "xgboost_tuned_params.json"

# 1단계에서 트리의 복잡도를 결정하는 후보값을 탐색한다.
TREE_PARAM_GRID = {
    "max_depth": [3, 4, 5, 6, 8, 10, 12],
    "max_leaves": [0, 31, 63, 127],
}

# 2단계에서 행과 피처 샘플링 비율을 탐색한다.
SAMPLING_PARAM_GRID = {
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
}


def create_tuning_model() -> XGBClassifier:
    """교차검증 단계에서 사용할 XGBoost 모델을 생성한다."""

    return XGBClassifier(
        objective="binary:logistic",
        n_estimators=TUNING_N_ESTIMATORS,
        learning_rate=TUNING_LEARNING_RATE,
        eval_metric="auc",
        tree_method="hist",
        device="cuda",        # XGBoost 3.x에서는 gpu_hist 대신 hist + cuda를 사용한다.
        random_state=RANDOM_STATE,
        n_jobs=1,        # GridSearchCV가 바깥에서 병렬 처리하므로 모델 하나는 코어 하나만 사용한다.
        verbosity=0,
    )


def add_model_prefix(param_grid: dict) -> dict:
    """XGBoost 파라미터 이름에 공통 Pipeline의 model 접두사를 붙인다."""

    return {f"model__{name}": values for name, values in param_grid.items()}


def remove_model_prefix(params: dict) -> dict:
    """GridSearchCV 결과에서 Pipeline 접두사를 제거한다."""

    prefix = "model__"
    return {
        name.removeprefix(prefix): value
        for name, value in params.items()
    }


def tune_tree_structure(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
    param_grid: dict | None = None,
) -> dict:
    """트리 깊이와 최대 리프 수를 교차검증으로 탐색한다."""

    print()
    print("=" * 70)
    print("1단계: max_depth / max_leaves 튜닝")
    print("=" * 70)

    # 공통 전처리기를 Pipeline에 포함해야 각 CV fold의 학습 부분에만 fit된다.
    pipeline = create_training_pipeline(create_tuning_model(), X_train)
    search_grid = TREE_PARAM_GRID if param_grid is None else param_grid

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=add_model_prefix(search_grid),
        scoring="roc_auc",
        cv=cv,
        n_jobs=1,        # GPU 한 장에 여러 fit을 동시에 올리지 않도록 후보를 순차 학습한다.
        verbose=1,
        refit=True,
    )
    grid_search.fit(X_train, y_train)
    best_params = remove_model_prefix(grid_search.best_params_)
    print("\n최적 트리 구조 파라미터")
    print(best_params)
    print(f"최고 CV ROC-AUC: {grid_search.best_score_:.4f}")
    return best_params


def tune_sampling(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
    tree_params: dict,
    param_grid: dict | None = None,
) -> dict:
    """선정된 트리 구조에서 행·피처 샘플링 비율을 탐색한다."""

    print()
    print("=" * 70)
    print("2단계: subsample / colsample_bytree 튜닝")
    print("=" * 70)

    pipeline = create_training_pipeline(create_tuning_model(), X_train)

    pipeline.set_params(
        **{f"model__{name}": value for name, value in tree_params.items()}
    )
    search_grid = SAMPLING_PARAM_GRID if param_grid is None else param_grid

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=add_model_prefix(search_grid),
        scoring="roc_auc",
        cv=cv,
        n_jobs=1,
        verbose=1,
        refit=True,
    )
    grid_search.fit(X_train, y_train)

    best_params = remove_model_prefix(grid_search.best_params_)
    print("\n최적 샘플링 파라미터")
    print(best_params)
    print(f"최고 CV ROC-AUC: {grid_search.best_score_:.4f}")
    return best_params


def find_best_iteration(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    best_params: dict,
) -> int:
    """공통 validation으로 Early Stopping을 수행해 적정 트리 수를 찾는다."""

    print()
    print("=" * 70)
    print("3단계: Early Stopping")
    print("=" * 70)

    preprocessor = create_common_preprocessor(X_train)
    X_train_processed = preprocessor.fit_transform(X_train, y_train)
    X_valid_processed = preprocessor.transform(X_valid)

    model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=FINAL_N_ESTIMATORS,
        learning_rate=FINAL_LEARNING_RATE,
        **best_params,
        eval_metric="auc",
        tree_method="hist",
        device="cuda",
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(
        X_train_processed,
        y_train,
        eval_set=[(X_valid_processed, y_valid)],
        verbose=False,
    )

    # best_iteration은 0부터 시작하므로 실제 사용할 트리 수는 1을 더한다.
    best_iteration = int(model.best_iteration) + 1
    print(f"최적 트리 수: {best_iteration}")
    print(f"최고 validation ROC-AUC: {float(model.best_score):.4f}")
    return best_iteration


def train_final_model(
    X_full_train: pd.DataFrame,
    y_full_train: pd.Series,
    best_params: dict,
    best_iteration: int,
) -> Pipeline:
    """train_processed.csv 전체로 튜닝된 최종 XGBoost Pipeline을 학습한다."""

    print()
    print("=" * 70)
    print("4단계: 최종 XGBoost 학습")
    print("=" * 70)

    model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=best_iteration,
        learning_rate=FINAL_LEARNING_RATE,
        **best_params,
        eval_metric="auc",
        tree_method="hist",
        device="cuda",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )

    pipeline = create_training_pipeline(model, X_full_train)
    pipeline.fit(X_full_train, y_full_train)
    return pipeline


def run_tuning(
    tree_param_grid: dict | None = None,
    sampling_param_grid: dict | None = None,
) -> tuple[Pipeline, dict, dict, dict, int]:
    """공통 데이터 구조로 전체 XGBoost 튜닝과 최종 평가를 실행한다."""

    X, y, X_test, y_test = load_processed_train_test_features()
    (
        X_train,
        X_valid,
        y_train,
        y_valid,
    ) = make_train_validation_split(X, y)

    print()
    print("=" * 70)
    print("XGBoost 하이퍼파라미터 튜닝")
    print("=" * 70)
    print(f"학습 데이터: {len(X_train):,}개")
    print(f"검증 데이터: {len(X_valid):,}개")
    print(f"외부 최종 테스트 데이터: {len(X_test):,}개")

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    tree_params = tune_tree_structure(
        X_train,
        y_train,
        cv,
        param_grid=tree_param_grid,
    )
    sampling_params = tune_sampling(
        X_train,
        y_train,
        cv,
        tree_params,
        param_grid=sampling_param_grid,
    )
    best_params = {**tree_params, **sampling_params}

    print("\n선정된 하이퍼파라미터")
    print("=" * 70)
    for name, value in best_params.items():
        print(f"{name}: {value}")

    best_iteration = find_best_iteration(
        X_train,
        y_train,
        X_valid,
        y_valid,
        best_params,
    )

    validation_model = train_final_model(
        X_train,
        y_train,
        best_params,
        best_iteration,
    )
    validation_metrics = evaluate_sklearn_model(validation_model, X_valid, y_valid)

    # 모델 선정과 트리 수 결정이 끝났으므로 train_processed.csv 전체로 재학습한다.
    final_model = train_final_model(
        X,
        y,
        best_params,
        best_iteration,
    )

    # test_processed.csv는 튜닝에 사용하지 않고 여기서 마지막으로 한 번만 평가한다.
    metrics = evaluate_sklearn_model(final_model, X_test, y_test)
    return final_model, validation_metrics, metrics, best_params, best_iteration


def main() -> None:
    """터미널에서 전체 XGBoost 튜닝을 실행한다."""

    model, validation_metrics, metrics, best_params, best_iteration = run_tuning()

    saved_params = {
        **best_params,
        "n_estimators": best_iteration,
        "learning_rate": FINAL_LEARNING_RATE,
    }
    save_tuned_ml_artifacts(
        model,
        TUNED_MODEL_PATH,
        "xgboost_tuned",
        metrics,
        TUNED_METRICS_PATH,
        saved_params,
        TUNED_PARAMS_PATH,
    )
    _, promotion_message = promote_tuned_model(
        "xgboost",
        model,
        validation_metrics,
        metrics,
        TUNED_MODEL_PATH,
    )

    print()
    print("=" * 70)
    print("최종 튜닝 XGBoost 결과")
    print("=" * 70)
    print(f"Accuracy          : {metrics['accuracy']:.4f}")
    print(f"Precision         : {metrics['precision']:.4f}")
    print(f"Recall            : {metrics['recall']:.4f}")
    print(f"F1-score          : {metrics['f1']:.4f}")
    print(f"ROC-AUC           : {metrics['roc_auc']:.4f}")
    print(f"Average Precision : {metrics['average_precision']:.4f}")
    print()
    print(
        f"TN={metrics['tn']:,} | FP={metrics['fp']:,} | "
        f"FN={metrics['fn']:,} | TP={metrics['tp']:,}"
    )
    print()
    print(f"선정 파라미터      : {best_params}")
    print(f"최종 n_estimators : {best_iteration}")
    print(f"최종 learning_rate: {FINAL_LEARNING_RATE}")
    print(f"튜닝 모델 저장     : {TUNED_MODEL_PATH}")
    print(f"튜닝 성능 저장     : {TUNED_METRICS_PATH}")
    print(f"튜닝 파라미터 저장 : {TUNED_PARAMS_PATH}")
    print(f"자동 승격 결과     : {promotion_message}")
    print("=" * 70)


if __name__ == "__main__":
    main()
