"""공통 80/20 검증에서 튜닝 전 성능을 비교할 XGBoost 모델 정의."""

from xgboost import XGBClassifier

from src.utils.constants import RANDOM_STATE


def create_xgboost() -> XGBClassifier:
    """퇴사=1, 재직=0인 전처리 데이터용 XGBoost 분류기를 생성한다."""

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
