"""랜덤 포레스트 모델 구현 파일.

데이터 로딩·분할·평가·저장은 ``train.py``가 공통으로 처리한다.
이 파일에서는 아래 import를 사용해 ``create_random_forest()``만 구현하면 된다.
"""

from sklearn.ensemble import RandomForestClassifier  # noqa: F401

from .utils import RANDOM_STATE  # noqa: F401


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