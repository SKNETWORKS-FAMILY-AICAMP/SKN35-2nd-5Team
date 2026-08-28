"""공통 학습 구조에서 사용할 Gradient Boosting 기본 모델 정의."""

from sklearn.ensemble import GradientBoostingClassifier

from src.utils.constants import RANDOM_STATE


def create_gradient_boosting() -> GradientBoostingClassifier:
    """GradientBoostingClassifier 객체를 생성한다.

    트리 기반 모델이므로 로지스틱 회귀와 달리 별도의 스케일링 없이
    train.py의 공통 전처리기(결측치 처리, 원핫 인코딩) 뒤에 바로 연결
    """
    return GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=RANDOM_STATE,
    )
