from sklearn.ensemble import RandomForestClassifier

from src.utils.constants import RANDOM_STATE


def create_random_forest(*, random_state: int = RANDOM_STATE) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200,
        random_state=random_state,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
