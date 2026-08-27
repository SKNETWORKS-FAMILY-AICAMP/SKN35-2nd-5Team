# csv 파일 또는 학습 완료된 모델 경로 설정
from pathlib import Path

# project 경로
ROOT_DIR = Path(__file__).parent.parent

# project/data 경로
DATA_DIR = ROOT_DIR / "data"
# project/artifacts 경로
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

# project/data/raw 경로
RAW_DIR = DATA_DIR / "raw"
# project/data/preprocessing 경로
PROCESSED_DIR = DATA_DIR / "preprocessing"

# project/artifacts/dl
DL_MODEL_DIR = ARTIFACTS_DIR / "dl"
