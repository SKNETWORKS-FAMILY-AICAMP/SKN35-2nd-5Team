"""Optuna를 이용한 CatBoost 하이퍼파라미터 튜닝."""

import argparse
from typing import Any

import optuna
from catboost import CatBoostClassifier
from optuna.trial import Trial
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from src.data.loader import load_processed_train_test_features
from src.utils.artifact_io import save_tuned_ml_artifacts
from src.utils.constants import RANDOM_STATE
from src.utils.metrics import evaluate_sklearn_model
from src.utils.ml_training import create_training_pipeline, make_train_validation_split
from src.utils.model_promotion import promote_tuned_model
from src.utils.paths import ML_ARTIFACTS_DIR, REPORTS_DIR

CV_FOLDS = 3
DEFAULT_TRIALS = 30
DEFAULT_TIMEOUT = 600
TUNED_MODEL_PATH = ML_ARTIFACTS_DIR / "catboost_tuned.joblib"
TUNED_METRICS_PATH = REPORTS_DIR / "catboost_tuned_metrics.csv"
TUNED_PARAMS_PATH = REPORTS_DIR / "catboost_tuned_params.json"


def suggest_params(trial: Trial) -> dict[str, Any]:
    """CatBoost 탐색 공간에서 한 개의 파라미터 조합을 생성한다."""

    return {
        "iterations": trial.suggest_int("iterations", 200, 800, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "depth": trial.suggest_int("depth", 3, 8),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
        "border_count": trial.suggest_categorical("border_count", [32, 64, 128]),
    }


def create_model(params: dict[str, Any]) -> CatBoostClassifier:
    """공통 학습 파이프라인에서 사용할 CatBoost 분류기를 생성한다."""

    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
        **params,
    )


def run_tuning(
    n_trials: int = DEFAULT_TRIALS,
    timeout: int | None = DEFAULT_TIMEOUT,
) -> tuple[
    Pipeline,
    dict[str, float | int],
    dict[str, float | int],
    dict[str, Any],
]:
    """CatBoost를 튜닝하고 validation 및 최종 test 성능을 반환한다."""

    features, target, test_features, test_target = load_processed_train_test_features()
    train_features, valid_features, train_target, valid_target = make_train_validation_split(
        features,
        target,
    )
    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    def objective(trial: Trial) -> float:
        pipeline = create_training_pipeline(create_model(suggest_params(trial)), train_features)
        scores = cross_val_score(
            pipeline,
            train_features,
            train_target,
            cv=cv,
            scoring="roc_auc",
            n_jobs=1,
        )
        return float(scores.mean())

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    best_params = dict(study.best_params)
    validation_model = create_training_pipeline(create_model(best_params), train_features)
    validation_model.fit(train_features, train_target)
    validation_metrics = evaluate_sklearn_model(
        validation_model,
        valid_features,
        valid_target,
    )

    print(f"\nBest CV ROC-AUC : {study.best_value:.4f}")
    print(f"Validation AUC   : {validation_metrics['roc_auc']:.4f}")
    print("Best parameters:")
    for name, value in best_params.items():
        print(f"  {name}: {value}")

    final_model = create_training_pipeline(create_model(best_params), features)
    final_model.fit(features, target)
    test_metrics = evaluate_sklearn_model(final_model, test_features, test_target)
    return final_model, validation_metrics, test_metrics, best_params


def main() -> None:
    """터미널에서 CatBoost Optuna 튜닝을 실행한다."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    args = parser.parse_args()

    model, validation_metrics, metrics, best_params = run_tuning(
        n_trials=args.trials,
        timeout=args.timeout,
    )
    save_tuned_ml_artifacts(
        model,
        TUNED_MODEL_PATH,
        "catboost_tuned",
        metrics,
        TUNED_METRICS_PATH,
        best_params,
        TUNED_PARAMS_PATH,
    )
    _, promotion_message = promote_tuned_model(
        "catboost",
        model,
        validation_metrics,
        metrics,
        TUNED_MODEL_PATH,
    )

    print("\nFinal Test")
    print(f"Accuracy          : {metrics['accuracy']:.4f}")
    print(f"Precision         : {metrics['precision']:.4f}")
    print(f"Recall            : {metrics['recall']:.4f}")
    print(f"F1-score          : {metrics['f1']:.4f}")
    print(f"ROC-AUC           : {metrics['roc_auc']:.4f}")
    print(f"Average Precision : {metrics['average_precision']:.4f}")
    print(
        f"TN={metrics['tn']:,} | FP={metrics['fp']:,} | "
        f"FN={metrics['fn']:,} | TP={metrics['tp']:,}"
    )
    print(f"Saved model       : {TUNED_MODEL_PATH}")
    print(f"Saved metrics     : {TUNED_METRICS_PATH}")
    print(f"Saved parameters  : {TUNED_PARAMS_PATH}")
    print(f"Promotion         : {promotion_message}")


if __name__ == "__main__":
    main()
