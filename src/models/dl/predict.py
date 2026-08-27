import joblib
import torch

from src.models.dl.mlp_model import MLPClassifier
from src.utils.paths import (
    MLP_BEST_PARAMS_PATH,
    MLP_MODEL_PATH,
    MLP_PREPROCESSOR_PATH,
    MLP_THRESHOLD_PATH,
)


def load_mlp_pipeline():
    best_params = joblib.load(MLP_BEST_PARAMS_PATH)
    scaler = joblib.load(MLP_PREPROCESSOR_PATH)
    threshold = joblib.load(MLP_THRESHOLD_PATH)
    model = MLPClassifier(best_params, in_features=28)

    model.load_state_dict(torch.load(MLP_MODEL_PATH, map_location="cpu"))
    model.eval()

    return model, scaler, threshold


def predict(new_data_df):
    model, scaler, threshold = load_mlp_pipeline()
    X_scaled = scaler.transform(new_data_df)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.sigmoid(logits).squeeze()
        preds = (probs >= threshold).float()

    return preds.numpy(), probs.numpy()
