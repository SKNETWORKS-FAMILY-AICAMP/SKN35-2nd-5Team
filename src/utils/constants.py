TARGET_COLUMN = "Attrition"
ID_COLUMN = "Employee ID"
POSITIVE_LABEL = "Left"
NEGATIVE_LABEL = "Stayed"
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.2
IN_FEATURES = 28

MODEL_NAMES = (
    "logistic_regression",
    "random_forest",
    "xgboost",
    "lightgbm",
)

ML_RESULT_COLUMNS = [
    "model",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "average_precision",
    "tn",
    "fp",
    "fn",
    "tp",
    "train_seconds",
    "artifact_path",
]
