"""Model registry that assembles independently defined baselines."""

from collections import OrderedDict

from sklearn.base import BaseEstimator

from src.ml.decision_tree import create_decision_tree
from src.ml.lightgbm_model import create_lightgbm
from src.ml.random_forest import create_random_forest
from src.ml.xgboost_model import create_xgboost
from src.utils.constants import RANDOM_STATE


def get_model_candidates(
    *,
    random_state: int = RANDOM_STATE,
    selected: list[str] | tuple[str, ...] | None = None,
) -> tuple[OrderedDict[str, BaseEstimator], dict[str, str]]:
    """Create fresh baseline estimators and report unavailable optional models."""
    factories = OrderedDict(
        [
            ("decision_tree", create_decision_tree),
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
