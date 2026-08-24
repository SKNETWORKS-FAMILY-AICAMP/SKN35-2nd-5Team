"""Filesystem paths resolved independently of the current working directory."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DATA_PATH = DATA_DIR / "train.csv"
TEST_DATA_PATH = DATA_DIR / "test.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
ML_LEADERBOARD_PATH = REPORTS_DIR / "ml_leaderboard.csv"
DL_METRICS_PATH = REPORTS_DIR / "dl_metrics.csv"
BEST_ML_MODEL_PATH = MODELS_DIR / "best_ml_model.joblib"
DL_MODEL_PATH = MODELS_DIR / "mlp.joblib"


def ensure_artifact_dirs() -> None:
    """Create runtime output directories when needed."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
