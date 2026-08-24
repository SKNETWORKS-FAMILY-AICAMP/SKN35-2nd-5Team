from collections import OrderedDict

from sklearn.base import BaseEstimator

from src.ml.lightgbm_model import create_lightgbm
from src.ml.logistic_regression import create_logistic_regression
from src.ml.random_forest import create_random_forest
from src.ml.xgboost_model import create_xgboost
from src.utils.constants import RANDOM_STATE


def get_model_candidates(
    *,
    random_state: int = RANDOM_STATE,
    selected: list[str] | tuple[str, ...] | None = None,
) -> tuple[OrderedDict[str, BaseEstimator], dict[str, str]]:
    """선택한 모델을 생성하고 설치되지 않은 모델 정보를 반환한다."""
    factories = OrderedDict(
        [
            ("logistic_regression", create_logistic_regression),
            ("random_forest", create_random_forest),
            ("xgboost", create_xgboost),
            ("lightgbm", create_lightgbm),
        ]
    )
    requested = set(selected or factories)
    models: OrderedDict[str, BaseEstimator] = OrderedDict()
    unavailable: dict[str, str] = {}

    for name, factory in factories.items():
        if name not in requested:
            continue
        try:
            models[name] = factory(random_state=random_state)
        except ImportError as exc:
            unavailable[name] = str(exc)
    return models, unavailable
