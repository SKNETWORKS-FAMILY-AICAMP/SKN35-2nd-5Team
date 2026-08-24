from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DATA_PATH = DATA_DIR / "train.csv"
TEST_DATA_PATH = DATA_DIR / "test.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
ML_LEADERBOARD_PATH = REPORTS_DIR / "ml_leaderboard.csv"
BEST_ML_MODEL_PATH = MODELS_DIR / "best_ml_model.joblib"


def ensure_artifact_dirs() -> None:
    """모델과 리포트 저장 폴더가 없으면 생성한다."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
