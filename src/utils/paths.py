from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "preprocessing"
RAW_TRAIN_DATA_PATH = RAW_DATA_DIR / "train.csv"
RAW_TEST_DATA_PATH = RAW_DATA_DIR / "test.csv"
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train_processed.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test_processed.csv"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ML_ARTIFACTS_DIR = ARTIFACTS_DIR / "ml"
DL_ARTIFACTS_DIR = ARTIFACTS_DIR / "dl"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

ML_LEADERBOARD_PATH = REPORTS_DIR / "ml_leaderboard.csv"
BEST_ML_TEST_METRICS_PATH = REPORTS_DIR / "best_ml_test_metrics.csv"
DL_METRICS_PATH = REPORTS_DIR / "dl_metrics.csv"
BEST_ML_MODEL_PATH = ML_ARTIFACTS_DIR / "best_ml_model.joblib"

MLP_MODEL_PATH = DL_ARTIFACTS_DIR / "mlp_model.pt"
MLP_PREPROCESSOR_PATH = DL_ARTIFACTS_DIR / "mlp_scaler.pkl"
MLP_BEST_PARAMS_PATH = DL_ARTIFACTS_DIR / "mlp_best_params.pkl"
MLP_THRESHOLD_PATH = DL_ARTIFACTS_DIR / "mlp_threshold.pkl"
MLP_METADATA_PATH = DL_ARTIFACTS_DIR / "mlp_metadata.pkl"


def ensure_artifact_dirs() -> None:
    """ML, DL 모델과 리포트 저장 폴더가 없으면 생성한다."""

    for directory in (ML_ARTIFACTS_DIR, DL_ARTIFACTS_DIR, REPORTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def project_relative_path(path: Path) -> str:
    """프로젝트 내부 경로를 OS와 무관한 상대경로 문자열로 변환한다."""

    return path.relative_to(PROJECT_ROOT).as_posix()
