import numpy as np

from src.load_data.loader import split_features_target
from src.ml.preprocessing import build_preprocessor, infer_feature_types


def test_preprocessor_handles_numeric_and_categorical(sample_frame):
    features, _ = split_features_target(sample_frame)
    numeric, categorical = infer_feature_types(features)
    transformed = build_preprocessor(features, dense_output=True).fit_transform(features)
    assert "Age" in numeric
    assert "Gender" in categorical
    assert isinstance(transformed, np.ndarray)
    assert transformed.shape[0] == len(sample_frame)
    assert transformed.shape[1] > features.shape[1]
