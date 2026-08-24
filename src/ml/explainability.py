import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.load_data.loader import split_features_target
from src.utils.constants import RANDOM_STATE


def shap_feature_importance(
    pipeline,
    frame: pd.DataFrame,
    *,
    max_samples: int = 500,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """인코딩된 피처별 SHAP 절댓값 평균을 계산한다."""
    import shap

    features, _ = split_features_target(frame)
    if len(features) > max_samples:
        features = features.sample(n=max_samples, random_state=random_state)
    preprocessor = pipeline.named_steps["preprocessor"]
    estimator = pipeline.named_steps["model"]
    transformed = preprocessor.transform(features)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    explainer = shap.TreeExplainer(estimator)
    raw_values = explainer.shap_values(transformed)
    if isinstance(raw_values, list):
        values = np.asarray(raw_values[-1])
    else:
        values = np.asarray(raw_values)
        if values.ndim == 3:
            values = values[:, :, -1]
    names = preprocessor.get_feature_names_out()
    if values.shape[1] != len(names):
        raise ValueError("SHAP 결과와 전처리된 피처 수가 일치하지 않습니다.")
    return (
        pd.DataFrame(
            {
                "feature": names,
                "importance_mean": np.abs(values).mean(axis=0),
            }
        )
        .sort_values("importance_mean", ascending=False, ignore_index=True)
    )


def permutation_feature_importance(
    model,
    frame: pd.DataFrame,
    *,
    max_samples: int = 3000,
    n_repeats: int = 3,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """피처를 섞었을 때 발생하는 ROC-AUC 감소량으로 중요도를 계산한다."""
    features, target = split_features_target(frame)
    if len(features) > max_samples:
        sampled = features.sample(n=max_samples, random_state=random_state)
        target = target.loc[sampled.index]
        features = sampled
    result = permutation_importance(
        model,
        features,
        target,
        scoring="roc_auc",
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )
    return (
        pd.DataFrame(
            {
                "feature": features.columns,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False, ignore_index=True)
    )
