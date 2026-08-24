"""Load trained artifacts for model comparison."""

import joblib
import pandas as pd

from src.ml.evaluation import classification_curves, evaluate_classifier
from src.ml.trainer import make_train_valid_split
from src.utils.paths import ML_LEADERBOARD_PATH


def load_leaderboard(path=ML_LEADERBOARD_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError("ML 리더보드가 없습니다. 먼저 ML 학습을 실행하세요.")
    return pd.read_csv(path)


def evaluate_saved_models(frame: pd.DataFrame, leaderboard: pd.DataFrame) -> dict[str, dict]:
    _, x_valid, _, y_valid = make_train_valid_split(frame)
    evaluated: dict[str, dict] = {}
    for row in leaderboard.itertuples(index=False):
        model = joblib.load(row.artifact_path)
        evaluated[row.model] = {
            "metrics": evaluate_classifier(model, x_valid, y_valid),
            "curves": classification_curves(model, x_valid, y_valid),
        }
    return evaluated
