"""
SHAP Explainability Analysis Module.
"""

from typing import Tuple, Any
import pandas as pd
import numpy as np
import shap

from src.utils.paths import ML_MODELS_DIR


def compute_shap_values(
    model: Any,
    X_sample: pd.DataFrame,
) -> Tuple[shap.Explainer, np.ndarray]:
    """Compute TreeExplainer or Explainer SHAP values for a sample dataset."""
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    except Exception:
        explainer = shap.Explainer(model, X_sample)
        shap_values = explainer(X_sample)
        
    return explainer, shap_values
