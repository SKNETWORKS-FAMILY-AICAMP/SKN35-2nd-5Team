"""XGBoost model definition."""

from xgboost import XGBClassifier

from .utils import RANDOM_STATE


def create_xgboost() -> XGBClassifier:
    """
    XGBoost 모델을 생성한다.

    이전 Tree가 틀린 데이터를 다음 Tree가
    순차적으로 보완하는 Boosting 기반 모델이다.
    """

    model = XGBClassifier(

        # 생성할 Tree 개수
        n_estimators=200,

        # 각각의 Tree 최대 깊이
        max_depth=6,

        # 이전 Tree의 결과를 다음 Tree가
        # 얼마나 강하게 보완할지 결정
        learning_rate=0.1,

        # 각 Tree 학습 시 전체 데이터의 90% 사용
        subsample=0.9,

        # 각 Tree 학습 시 전체 Feature의 90% 사용
        colsample_bytree=0.9,

        # 이진 분류 확률 성능 평가 방법
        eval_metric="logloss",

        # 동일한 결과 재현
        random_state=RANDOM_STATE,

        # CPU 전체 사용
        n_jobs=-1,
    )

    return model