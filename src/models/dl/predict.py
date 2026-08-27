import joblib
import torch

from src.models.dl.mlp_model import MLPClassifier
from src.utils.constants import IN_FEATURES


def load_mlp_pipeline():
    best_params = joblib.load("artifacts/dl/mlp_best_params.pkl")
    scaler = joblib.load("artifacts/dl/mlp_scaler.pkl")
    threshold = joblib.load("artifacts/dl/mlp_threshold.pkl")
    model = MLPClassifier(best_params, in_features=IN_FEATURES)

    model.load_state_dict(torch.load("artifacts/dl/mlp_model.pt", map_location="cpu"))
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
