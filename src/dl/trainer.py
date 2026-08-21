"""
Deep Learning Trainer Orchestrator.
"""

import time
import json
from typing import Dict, Any
import joblib

from src.data.loader import get_train_test_data
from src.dl.mlp import build_mlp_model
from src.dl.evaluate import evaluate_dl_model
from src.utils.paths import DL_MODELS_DIR, RESULTS_DIR


def train_and_save_dl_model() -> Dict[str, Any]:
    """Train MLP DL model, evaluate, and save artifacts."""
    X_train, X_test, y_train, y_test, feature_names = get_train_test_data()
    
    t0 = time.time()
    model = build_mlp_model()
    model.fit(X_train, y_train)
    duration = time.time() - t0
    
    metrics = evaluate_dl_model(model, X_train, X_test, y_train, y_test, feature_names)
    metrics["model_name"] = "Deep Learning (MLP)"
    metrics["train_time_sec"] = float(duration)
    
    # Save model artifact
    model_path = DL_MODELS_DIR / "mlp.joblib"
    joblib.dump(model, model_path)
    
    # Save result
    metrics_path = RESULTS_DIR / "dl_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"mlp": metrics}, f, ensure_ascii=False, indent=2)
        
    return {"mlp": metrics}
