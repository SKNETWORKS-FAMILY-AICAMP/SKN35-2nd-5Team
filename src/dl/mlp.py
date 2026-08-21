"""
Multi-Layer Perceptron (MLP) Deep Learning Model for Churn Prediction.
"""

from typing import Tuple
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.utils.constants import RANDOM_STATE


def build_mlp_model(hidden_layer_sizes=(64, 32), max_iter=200) -> Pipeline:
    """Build a StandardScaled MLP Pipeline."""
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            max_iter=max_iter,
            random_state=RANDOM_STATE,
            early_stopping=True,
            validation_fraction=0.1,
        ))
    ])
    return model
