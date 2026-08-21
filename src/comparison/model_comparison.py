"""
Model Comparison and Benchmark Ranking Utility.
"""

import json
from typing import Dict, Any, List
import pandas as pd

from src.utils.paths import RESULTS_DIR


def load_all_metrics() -> Dict[str, Any]:
    """Load both ML and DL metrics."""
    metrics = {}
    ml_path = RESULTS_DIR / "ml_metrics.json"
    dl_path = RESULTS_DIR / "dl_metrics.json"
    
    if ml_path.exists():
        with open(ml_path, "r", encoding="utf-8") as f:
            metrics.update(json.load(f))
            
    if dl_path.exists():
        with open(dl_path, "r", encoding="utf-8") as f:
            metrics.update(json.load(f))
            
    return metrics


def build_comparison_dataframe(metrics_dict: Dict[str, Any] = None) -> pd.DataFrame:
    """Construct a clean comparison DataFrame across all models."""
    if metrics_dict is None:
        metrics_dict = load_all_metrics()
        
    rows = []
    for m_id, d in metrics_dict.items():
        rows.append({
            "Model ID": m_id,
            "Model Name": d.get("model_name", m_id),
            "Train Accuracy": d.get("train_accuracy", 0.0),
            "Test Accuracy": d.get("test_accuracy", 0.0),
            "Precision (Churn)": d.get("precision", 0.0),
            "Recall (Churn)": d.get("recall", 0.0),
            "F1-Score": d.get("f1_score", 0.0),
            "ROC-AUC": d.get("roc_auc", 0.0),
            "PR-AUC": d.get("pr_auc", 0.0),
            "Train Time (s)": d.get("train_time_sec", 0.0),
        })
        
    df = pd.DataFrame(rows)
    if not df.empty and "ROC-AUC" in df.columns:
        df = df.sort_values(by="ROC-AUC", ascending=False).reset_index(drop=True)
    return df
