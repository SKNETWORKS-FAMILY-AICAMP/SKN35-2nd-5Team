"""기존 import 호환성을 위한 프로젝트 경로 별칭."""

from src.utils.paths import (
    ARTIFACTS_DIR,
    DATA_DIR,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    RAW_DATA_DIR,
)

ROOT_DIR = PROJECT_ROOT
RAW_DIR = RAW_DATA_DIR
PROCESSED_DIR = PROCESSED_DATA_DIR

__all__ = ["ARTIFACTS_DIR", "DATA_DIR", "PROCESSED_DIR", "RAW_DIR", "ROOT_DIR"]
# project/data/raw 경로
RAW_DIR = DATA_DIR / "raw"
# project/data/preprocessing 경로
PROCESSED_DIR = DATA_DIR / "preprocessing"

# project/artifacts/dl
DL_MODEL_DIR = ARTIFACTS_DIR / "dl"
