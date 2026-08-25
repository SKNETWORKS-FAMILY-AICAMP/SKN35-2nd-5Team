"""튜닝 전 비교에 사용할 XGBoost 이진 분류 모델 정의."""

from xgboost import XGBClassifier

from .utils import RANDOM_STATE


def create_xgboost() -> XGBClassifier:
    """직원 퇴사 여부를 예측하는 XGBoost 베이스라인 모델을 생성한다.

    이전 트리의 오차를 다음 트리가 순차적으로 보완하는 부스팅 모델이다.
    하이퍼파라미터 튜닝 전이므로 과도한 최적화 없이 비교 기준값만 설정한다.
    """

    return XGBClassifier(
        objective="binary:logistic",
        n_estimators=200,   # 성능향상할때 조정 가능
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="auc",
        tree_method="hist",         # 히스토그램 알고리즘
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )
