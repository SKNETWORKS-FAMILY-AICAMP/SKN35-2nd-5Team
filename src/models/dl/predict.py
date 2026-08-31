import joblib
import numpy as np
import torch

from src.models.dl.mlp_model import MLPClassifier
from src.utils.paths import (
    MLP_BEST_PARAMS_PATH,
    MLP_METADATA_PATH,
    MLP_MODEL_PATH,
    MLP_PREPROCESSOR_PATH,
    MLP_THRESHOLD_PATH,
)


def load_mlp_pipeline():
    best_params = joblib.load(MLP_BEST_PARAMS_PATH)
    preprocessor = joblib.load(MLP_PREPROCESSOR_PATH)
    threshold = joblib.load(MLP_THRESHOLD_PATH)
    metadata = joblib.load(MLP_METADATA_PATH)
    model = MLPClassifier(best_params, in_features=int(metadata["in_features"]))

    model.load_state_dict(
        torch.load(MLP_MODEL_PATH, map_location="cpu", weights_only=True)
    )
    model.eval()

    return model, preprocessor, float(threshold)


class MLPPredictionModel:
    """HR 예측 코드에서 scikit-learn 모델처럼 사용할 수 있는 MLP 어댑터."""

    def __init__(self, model, preprocessor, threshold: float, feature_names: list[str]):
        self.model = model
        self.preprocessor = preprocessor
        self.threshold = float(threshold)
        self.feature_names_in_ = np.asarray(feature_names, dtype=object)
        self.classes_ = np.asarray([0, 1])

    def predict_proba(self, frame):
        """직원별 재직/퇴사 확률을 ``[P(0), P(1)]`` 형태로 반환한다."""

        ordered = frame.reindex(columns=self.feature_names_in_.tolist())
        transformed = np.asarray(self.preprocessor.transform(ordered), dtype=np.float32)
        tensor = torch.tensor(transformed, dtype=torch.float32)
        with torch.no_grad():
            probabilities = torch.sigmoid(self.model(tensor)).reshape(-1).cpu().numpy()
        return np.column_stack((1.0 - probabilities, probabilities))

    def predict(self, frame):
        """학습 시 저장한 Recall 중심 임계값으로 퇴사 여부를 분류한다."""

        return (self.predict_proba(frame)[:, 1] >= self.threshold).astype(np.int8)


def load_mlp_prediction_model() -> MLPPredictionModel:
    """저장된 MLP 전체 파이프라인을 HR 서비스용 예측기로 불러온다."""

    model, preprocessor, threshold = load_mlp_pipeline()
    metadata = joblib.load(MLP_METADATA_PATH)
    feature_names = list(metadata["feature_names"])
    return MLPPredictionModel(model, preprocessor, threshold, feature_names)


def predict(new_data_df):
    prediction_model = load_mlp_prediction_model()
    probabilities = prediction_model.predict_proba(new_data_df)[:, 1]
    predictions = prediction_model.predict(new_data_df)
    return predictions, probabilities
