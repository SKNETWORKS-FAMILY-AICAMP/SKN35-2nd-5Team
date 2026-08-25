"""Logistic Regression model definition."""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .utils import RANDOM_STATE


def create_logistic_regression() -> Pipeline:
    """
    Pipeline을 사용하여
    Scaling -> Logistic Regression
    순서로 자동 실행되도록 구성한다.
    """

    model = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    return model