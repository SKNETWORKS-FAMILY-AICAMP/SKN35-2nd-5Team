"""Tuned Random Forest with preprocessing and stratified cross-validation."""

from __future__ import annotations

import time
from typing import Any

import optuna
from optuna.samplers import TPESampler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.ml.evaluate import evaluate_classifier
from src.utils.constants import RANDOM_STATE


optuna.logging.set_verbosity(optuna.logging.WARNING)


def _validate_feature_lists(
    X_train,
    num_features: list[str],
    cat_features: list[str],
) -> None:
    """Validate that preprocessing columns exist and do not overlap."""
    num_set = set(num_features)
    cat_set = set(cat_features)
    overlap = num_set & cat_set
    if overlap:
        raise ValueError(f"Numeric/categorical feature overlap: {sorted(overlap)}")

    selected = num_set | cat_set
    missing = selected - set(X_train.columns)
    if missing:
        raise ValueError(f"Features not found in training data: {sorted(missing)}")
    if not selected:
        raise ValueError("At least one numeric or categorical feature is required.")


def build_pipeline(
    num_features: list[str],
    cat_features: list[str],
    params: dict[str, Any] | None = None,
) -> Pipeline:
    """Build a leak-safe preprocessing and Random Forest pipeline."""
    classifier_params = dict(params or {})
    classifier_params.setdefault("random_state", RANDOM_STATE)
    classifier_params.setdefault("class_weight", "balanced")
    classifier_params.setdefault("n_jobs", -1)

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            ),
        ]
    )

    transformers = []
    if num_features:
        transformers.append(("num", numeric_transformer, num_features))
    if cat_features:
        transformers.append(("cat", categorical_transformer, cat_features))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=1.0,
        verbose_feature_names_out=False,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(**classifier_params)),
        ]
    )


def tune_hyperparameters(
    X_train,
    y_train,
    num_features: list[str],
    cat_features: list[str],
    n_trials: int = 20,
    cv_splits: int = 5,
) -> tuple[dict[str, Any], float]:
    """Tune Random Forest parameters with reproducible stratified CV.

    Macro F1 is optimized so the minority retained class and majority churn
    class contribute equally to model selection.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1.")
    if cv_splits < 2:
        raise ValueError("cv_splits must be at least 2.")
    _validate_feature_lists(X_train, num_features, cat_features)

    cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    def objective(trial: optuna.Trial) -> float:
        # The forest uses one worker inside each fold; folds are parallelized
        # by cross_val_score to avoid nested n_jobs=-1 oversubscription.
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 450, step=50),
            "max_depth": trial.suggest_int("max_depth", 5, 24),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 12),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 6),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", 0.5]
            ),
            "criterion": trial.suggest_categorical(
                "criterion", ["gini", "entropy", "log_loss"]
            ),
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
            "n_jobs": 1,
        }
        pipeline = build_pipeline(num_features, cat_features, params)
        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1,
            error_score="raise",
        )
        return float(scores.mean())

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = dict(study.best_params)
    best_params.update(
        {
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        }
    )
    return best_params, float(study.best_value)


def _pipeline_feature_importance(
    pipeline: Pipeline,
) -> list[tuple[str, float]] | None:
    """Extract transformed feature names and forest importance from a Pipeline."""
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    if not hasattr(classifier, "feature_importances_"):
        return None
    names = preprocessor.get_feature_names_out().tolist()
    values = classifier.feature_importances_.tolist()
    return sorted(
        zip(names, (float(value) for value in values)),
        key=lambda item: item[1],
        reverse=True,
    )


def train_random_forest_advanced(
    X_train,
    X_test,
    y_train,
    y_test,
    num_features: list[str],
    cat_features: list[str],
    n_trials: int = 20,
    cv_splits: int = 5,
) -> tuple[Pipeline, dict[str, Any]]:
    """Tune, fit, and evaluate the complete Random Forest pipeline."""
    _validate_feature_lists(X_train, num_features, cat_features)
    started = time.monotonic()
    best_params, best_cv_f1_macro = tune_hyperparameters(
        X_train,
        y_train,
        num_features,
        cat_features,
        n_trials=n_trials,
        cv_splits=cv_splits,
    )

    final_pipeline = build_pipeline(num_features, cat_features, best_params)
    final_pipeline.fit(X_train, y_train)
    duration = time.monotonic() - started

    metrics = evaluate_classifier(
        final_pipeline,
        X_train,
        X_test,
        y_train,
        y_test,
        feature_names=None,
    )
    metrics["feature_importance"] = _pipeline_feature_importance(final_pipeline)
    metrics["model_name"] = "Random Forest (Optuna Tuned)"
    metrics["train_time_sec"] = float(duration)
    metrics["best_cv_f1_macro"] = best_cv_f1_macro
    metrics["best_params"] = best_params
    metrics["n_trials"] = int(n_trials)
    metrics["cv_splits"] = int(cv_splits)

    return final_pipeline, metrics
