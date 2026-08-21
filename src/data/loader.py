"""
Data loader module for loading and splitting churn datasets.
"""

from typing import Tuple, List
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.paths import FEATURE_DATA_PATH, RAW_DATA_PATH
from src.utils.constants import RANDOM_STATE, TARGET_COLUMN, LEAKAGE_AND_ID_COLUMNS


def load_raw_dataset(path=RAW_DATA_PATH) -> pd.DataFrame:
    """Load consolidated raw payment user logs."""
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found at {path}")
    return pd.read_csv(path)


def load_feature_dataset(path=FEATURE_DATA_PATH) -> pd.DataFrame:
    """Load prepared churn modeling feature dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Feature dataset not found at {path}")
    return pd.read_csv(path)


def get_train_test_data(
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
    path=FEATURE_DATA_PATH,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """
    Load feature dataset and perform stratified train/test split.
    """
    df = load_feature_dataset(path)
    
    y = df[TARGET_COLUMN].copy()
    drop_cols = [c for c in LEAKAGE_AND_ID_COLUMNS if c in df.columns]
    X = df.drop(columns=drop_cols)
    feature_names = list(X.columns)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    
    return X_train, X_test, y_train, y_test, feature_names
