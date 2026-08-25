"""튜닝 전 비교에 사용할 XGBoost 이진 분류 모델 정의."""

from xgboost import XGBClassifier

from .utils import RANDOM_STATE


def create_xgboost() -> XGBClassifier:

    return XGBClassifier(
        objective="binary:logistic",
        n_estimators=200,   # 성능향상할때 조정 가능
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="auc",
        tree_method="hist",         # 히스토그램 알고리즘
        device="cuda",              # XGBoost 3.x 방식으로 NVIDIA GPU 사용
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )
