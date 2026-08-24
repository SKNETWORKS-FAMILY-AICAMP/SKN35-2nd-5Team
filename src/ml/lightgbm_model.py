from typing import Any

from src.utils.constants import RANDOM_STATE


def create_lightgbm(*, random_state: int = RANDOM_STATE) -> Any:
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=200,
        learning_rate=0.1,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
    )
