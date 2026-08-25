# csv 파일 또는 모델 불러오는 함수 정의(config.py에 경로 설정 후 import로 추가형 함수 정의)
import pandas as pd

from src.config import PROCESSED_DIR, RAW_DIR


def load_raw_train():
    return pd.read_csv(RAW_DIR / "train.csv")


def load_raw_test():
    return pd.read_csv(RAW_DIR / "test.csv")


def load_processed_train():
    return pd.read_csv(PROCESSED_DIR / "train_processed.csv")
