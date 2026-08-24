"""Decision Tree baseline definition."""

from sklearn.tree import DecisionTreeClassifier

from src.utils.constants import RANDOM_STATE


def create_decision_tree(*, random_state: int = RANDOM_STATE) -> DecisionTreeClassifier:
    """Return a fresh, untuned Decision Tree classifier."""
    return DecisionTreeClassifier(
        random_state=random_state,
        class_weight="balanced",
        min_samples_leaf=5,
    )
