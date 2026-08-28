import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# 프로젝트 루트 및 저장 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_PATH = os.path.join(BASE_DIR, "data", "preprocessing")
ARTIFACTS_PATH = os.path.join(BASE_DIR, "artifacts")

from sklearn.ensemble import RandomForestClassifier 
from src.utils.constants import RANDOM_STATE 

def create_random_forest():
    """Random Forest 모델 객체를 생성하여 반환합니다.

    Returns:
        RandomForestClassifier: 클래스 불균형 및 기본 하이퍼파라미터가 설정된 모델 객체
    """
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return model


