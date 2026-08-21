"""
ML Model Trainer Orchestrator.
"""

import json
from typing import Dict, Any
import joblib

from src.data.loader import get_train_test_data
from src.ml.decision_tree import train_decision_tree
from src.ml.random_forest import train_random_forest_advanced
from src.ml.xgboost_model import train_xgboost
from src.ml.lightgbm_model import train_lightgbm
from src.utils.paths import ML_MODELS_DIR, RESULTS_DIR


def _train_tuned_random_forest(
    X_train,
    X_test,
    y_train,
    y_test,
    _feature_names,
    n_trials: int,
    cv_splits: int,
):
    """Infer column types and train the preprocessing-aware Random Forest."""
    num_features = X_train.select_dtypes(include="number").columns.tolist()
    cat_features = X_train.select_dtypes(exclude="number").columns.tolist()
    return train_random_forest_advanced(
        X_train,
        X_test,
        y_train,
        y_test,
        num_features=num_features,
        cat_features=cat_features,
        n_trials=n_trials,
        cv_splits=cv_splits,
    )


def train_and_save_all_ml_models(
    random_forest_trials: int = 20,
    random_forest_cv_splits: int = 5,
) -> Dict[str, Any]:
    """Train all 4 ML models, evaluate, and save artifacts."""
    X_train, X_test, y_train, y_test, feature_names = get_train_test_data()
    
    models = {
        "decision_tree": train_decision_tree,
        "random_forest": None,
        "xgboost": train_xgboost,
        "lightgbm": train_lightgbm,
    }
    
    all_metrics = {}
    
    for model_id, train_fn in models.items():
        if model_id == "random_forest":
            model, metrics = _train_tuned_random_forest(
                X_train,
                X_test,
                y_train,
                y_test,
                feature_names,
                n_trials=random_forest_trials,
                cv_splits=random_forest_cv_splits,
            )
        else:
            model, metrics = train_fn(
                X_train, X_test, y_train, y_test, feature_names
            )
        all_metrics[model_id] = metrics
        
        # Save model joblib
        model_path = ML_MODELS_DIR / f"{model_id}.joblib"
        joblib.dump(model, model_path)
        
    # Save combined results
    metrics_path = RESULTS_DIR / "ml_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)
        
    return all_metrics
