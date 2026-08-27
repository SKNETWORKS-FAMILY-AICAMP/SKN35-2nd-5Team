"""공통 학습 구조에서 사용할 LightGBM 기본 모델 정의."""

from lightgbm import LGBMClassifier

from src.utils.constants import RANDOM_STATE


def create_lightgbm() -> LGBMClassifier:

    return LGBMClassifier(
        boosting_type="gbdt",
        objective="binary",
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )
