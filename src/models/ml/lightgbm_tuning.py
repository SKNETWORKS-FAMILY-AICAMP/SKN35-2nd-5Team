"""Optuna를 이용한 LightGBM 하이퍼파라미터 튜닝."""

import argparse
import json
from typing import Any

import joblib
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from optuna.trial import Trial
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from .train import (
    MODELS_DIR,
    REPORTS_DIR,
    create_training_pipeline,
    load_train_test_data,
    make_train_validation_split,
)
from .utils import RANDOM_STATE, evaluate_model
from .promotion import promote_tuned_model

CV_FOLDS = 5
DEFAULT_TRIALS = 50
DEFAULT_TIMEOUT = 600
TUNED_MODEL_PATH = MODELS_DIR / "lightgbm_tuned.joblib"
TUNED_METRICS_PATH = REPORTS_DIR / "lightgbm_tuned_metrics.csv"
TUNED_PARAMS_PATH = REPORTS_DIR / "lightgbm_tuned_params.json"


def suggest_params(trial: Trial) -> dict[str, Any]:

    max_depth = trial.suggest_categorical("max_depth", [-1, 4, 5, 6, 7, 8])
    max_num_leaves = 128 if max_depth == -1 else min(128, 2**max_depth)

    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, max_num_leaves),
        "max_depth": max_depth,
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
    }


def create_model(params: dict[str, Any]) -> LGBMClassifier:

    return LGBMClassifier(
        boosting_type="gbdt",
        objective="binary",
        metric="auc",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
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

    X, y, X_test, y_test = load_train_test_data()
    X_train, X_valid, y_train, y_valid = make_train_validation_split(X, y)
    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    def objective(trial: Trial) -> float:
        params = suggest_params(trial)
        pipeline = create_training_pipeline(create_model(params), X_train)
        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
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

    best_params = {**study.best_params, "subsample_freq": 1}
    validation_model = create_training_pipeline(
        create_model(best_params),
        X_train,
    )
    validation_model.fit(X_train, y_train)
    validation_metrics = evaluate_model(validation_model, X_valid, y_valid)

    print(f"\nBest CV ROC-AUC : {study.best_value:.4f}")
    print(f"Validation AUC   : {validation_metrics['roc_auc']:.4f}")
    print("Best parameters:")
    for name, value in best_params.items():
        print(f"  {name}: {value}")

    final_model = create_training_pipeline(create_model(best_params), X)
    final_model.fit(X, y)
    test_metrics = evaluate_model(final_model, X_test, y_test)
    return final_model, validation_metrics, test_metrics, best_params


def main() -> None:
    """터미널에서 LightGBM Optuna 튜닝을 실행한다."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    args = parser.parse_args()

    model, validation_metrics, metrics, best_params = run_tuning(
        n_trials=args.trials,
        timeout=args.timeout,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, TUNED_MODEL_PATH)
    pd.DataFrame([{"model": "lightgbm_tuned", **metrics}]).to_csv(
        TUNED_METRICS_PATH,
        index=False,
    )
    TUNED_PARAMS_PATH.write_text(
        json.dumps(best_params, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    promoted, promotion_message = promote_tuned_model(
        "lightgbm",
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
