"""A from-scratch-trained MLP baseline built with scikit-learn primitives."""

from time import perf_counter
from typing import Any

import joblib
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from src.ml.evaluation import evaluate_classifier
from src.ml.preprocessing import build_preprocessor
from src.ml.trainer import make_train_valid_split
from src.utils.constants import RANDOM_STATE, TEST_SIZE
from src.utils.paths import DL_METRICS_PATH, DL_MODEL_PATH, ensure_artifact_dirs


def train_mlp(
    frame: pd.DataFrame,
    *,
    hidden_layer_sizes: tuple[int, ...] = (64, 32),
    max_iter: int = 100,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    save_artifact: bool = True,
) -> tuple[Pipeline, dict[str, Any]]:
    """Train a two-hidden-layer MLP with early stopping."""
    x_train, x_valid, y_train, y_valid = make_train_valid_split(
        frame,
        test_size=test_size,
        random_state=random_state,
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(x_train, dense_output=True)),
            (
                "model",
                MLPClassifier(
                    hidden_layer_sizes=hidden_layer_sizes,
                    activation="relu",
                    solver="adam",
                    batch_size=256,
                    learning_rate_init=0.001,
                    max_iter=max_iter,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=10,
                    random_state=random_state,
                ),
            ),
        ]
    )
    started = perf_counter()
    pipeline.fit(x_train, y_train)
    elapsed = perf_counter() - started
    metrics = evaluate_classifier(pipeline, x_valid, y_valid)
    network = pipeline.named_steps["model"]
    metrics.update(
        {
            "model": "mlp",
            "train_seconds": elapsed,
            "epochs": int(network.n_iter_),
            "final_loss": float(network.loss_),
        }
    )
    if save_artifact:
        ensure_artifact_dirs()
        joblib.dump(pipeline, DL_MODEL_PATH, compress=3)
        pd.DataFrame([metrics]).to_csv(DL_METRICS_PATH, index=False)
    return pipeline, metrics
