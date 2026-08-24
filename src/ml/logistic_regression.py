"""Logistic Regression baseline definition."""

from sklearn.linear_model import LogisticRegression

from src.utils.constants import RANDOM_STATE


def create_logistic_regression(
    *,
    random_state: int = RANDOM_STATE,
) -> LogisticRegression:
    """Return a fresh, untuned Logistic Regression classifier."""
    return LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        random_state=random_state,
    )
