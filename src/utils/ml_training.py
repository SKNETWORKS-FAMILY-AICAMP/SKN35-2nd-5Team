"""ML 학습 진입점에서 사용하는 데이터 분할·파이프라인 구성 보조 함수."""

from importlib import import_module
from typing import Any, Callable

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.utils.constants import RANDOM_STATE, VAL_SIZE

MODEL_SPECS = {
    "logistic_regression": ("logistic_regression", "create_logistic_regression"),
    "random_forest": ("random_forest", "create_random_forest"),
    "xgboost": ("xgboost", "create_xgboost"),
    "lightgbm": ("lightgbm", "create_lightgbm"),
}


def make_train_validation_split(
    features: pd.DataFrame,
    target: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """학습 데이터를 공통 학습/검증 데이터로 계층 분할한다."""

    return train_test_split(
        features,
        target,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )


def create_common_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """현재 학습 분할에만 fit되는 공통 전처리기를 생성한다."""

    numeric_columns = features.select_dtypes(include="number").columns.tolist()
    categorical_columns = features.select_dtypes(exclude="number").columns.tolist()
    numeric_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def create_training_pipeline(model: Any, features: pd.DataFrame) -> Pipeline:
    """팀원이 만든 모델 앞에 공통 전처리 단계를 연결한다."""

    return Pipeline(
        steps=[
            ("preprocessor", create_common_preprocessor(features)),
            ("model", model),
        ]
    )


def load_model_factories(
    selected: list[str] | tuple[str, ...] | None = None,
) -> tuple[dict[str, Callable[[], Any]], dict[str, str]]:
    """구현된 ML 모델 생성 함수를 불러오고 사용할 수 없는 모델을 기록한다."""

    requested = list(selected) if selected is not None else list(MODEL_SPECS)
    unknown = sorted(set(requested) - set(MODEL_SPECS))
    if unknown:
        raise ValueError(f"등록되지 않은 모델입니다: {', '.join(unknown)}")

    factories: dict[str, Callable[[], Any]] = {}
    unavailable: dict[str, str] = {}
    for model_name in requested:
        module_name, factory_name = MODEL_SPECS[model_name]
        try:
            module = import_module(f"src.models.ml.{module_name}")
            factory = getattr(module, factory_name)
        except (ImportError, AttributeError) as exc:
            unavailable[model_name] = str(exc)
        else:
            factories[model_name] = factory

    return factories, unavailable
