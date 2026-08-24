"""Project-wide constants."""

TARGET_COLUMN = "Attrition"
ID_COLUMN = "Employee ID"
POSITIVE_LABEL = "Left"
NEGATIVE_LABEL = "Stayed"
RANDOM_STATE = 42
TEST_SIZE = 0.2

MODEL_NAMES = (
    "decision_tree",
    "random_forest",
    "xgboost",
    "lightgbm",
)
