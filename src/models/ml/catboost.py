"""CatBoost 모델 구현 파일.

데이터 로딩·분할·평가·저장은 ``train.py``가 공통으로 처리한다.
이 파일에서는 아래 import를 사용해 ``create_catboost()``만 구현하면 된다.
"""

from catboost import CatBoostClassifier  # noqa: F401

from src.utils.constants import RANDOM_STATE


def create_catboost():
    """CatBoost 모델 객체를 생성하여 반환합니다.

    Returns:
        CatBoostClassifier: 클래스 불균형 및 기본 하이퍼파라미터가 설정된 모델 객체
    """
    model = CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        eval_metric="AUC",
        random_seed=RANDOM_STATE,
        verbose=0,  # 학습 과정 로그 출력 숨김
    )
    return model
