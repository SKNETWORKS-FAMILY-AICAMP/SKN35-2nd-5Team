import joblib
import torch

from src.models.dl.mlp_model import MLPClassifier


# 예측에 필요한 3가지(구조, 가중치, 스케일러)를 모두 불러와 반환
def load_mlp_pipeline():
    best_params = joblib.load("artifacts/dl/mlp_best_params.pkl")
    scaler = joblib.load("artifacts/dl/mlp_scaler.pkl")

    model = MLPClassifier(best_params, in_features=41)
    model.load_state_dict(torch.load("artifacts/dl/mlp_model.pt"))
    model.eval()

    return model, scaler


def predict(new_data_df):
    model, scaler = load_mlp_pipeline()
    X_scaled = scaler.transform(new_data_df)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    with torch.no_grad():
        prob = torch.sigmoid(model(X_tensor))
        pred = (prob >= 0.5).float()

    return pred
