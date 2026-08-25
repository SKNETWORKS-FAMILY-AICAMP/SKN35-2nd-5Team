from copy import deepcopy

import joblib
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.data.loader import load_processed_train
from src.models.dl.mlp_model import MLPClassifier, build_trial_model
from src.utils.constants import RANDOM_STATE, TEST_SIZE, VAL_SIZE
from src.utils.metrics import evaluate_model


def prepare_dataloaders(df, target_col="Attrition", batch_size=64):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=VAL_SIZE, stratify=y_temp, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.to_numpy(), dtype=torch.float32).reshape(-1, 1)

    X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_t = torch.tensor(y_val.to_numpy(), dtype=torch.float32).reshape(-1, 1)

    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_t = torch.tensor(y_test.to_numpy(), dtype=torch.float32).reshape(-1, 1)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True
    )

    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False)

    test_loader = DataLoader(
        TensorDataset(X_test_t, y_test_t), batch_size=batch_size, shuffle=False
    )

    return train_loader, val_loader, test_loader, scaler, X_train.shape[1]


def run_optuna_search(train_loader, val_loader, in_features, n_trials=10, epochs=30):
    def objective(trial):
        model = build_trial_model(trial, in_features)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        for epoch in range(epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(X_batch), y_batch)
                loss.backward()
                optimizer.step()

            model.eval()

            val_loss = 0.0

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    val_loss += criterion(model(X_batch), y_batch).item()
            val_loss /= len(val_loader)

            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return val_loss

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
        ),
    )
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def train_final_model(best_params, in_features, train_loader, val_loader, epochs=100, patience=15):
    model = MLPClassifier(best_params, in_features)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=best_params["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val_loss = float("inf")
    patience_counter = 0
    best_weights = None

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                val_loss += criterion(model(X_batch), y_batch).item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model


# 학습된 모델로 test_loader에 대한 예측값(0/1)과 실제 정답을 뽑아내는 함수
def get_predictions(model, test_loader):

    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            output = model(X_batch)
            probs = torch.sigmoid(output)
            preds = (probs >= 0.5).float()

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
            all_probs.extend(probs.cpu().numpy().ravel())

    return all_targets, all_preds, all_probs


def main():
    df = load_processed_train()
    train_loader, val_loader, test_loader, scaler, in_features = prepare_dataloaders(df)

    best_params = run_optuna_search(train_loader, val_loader, in_features)
    model = train_final_model(best_params, in_features, train_loader, val_loader)

    y_true, y_pred, y_proba = get_predictions(model, test_loader)
    evaluate_model(y_true, y_pred, y_proba)

    torch.save(model.state_dict(), "artifacts/dl/mlp_model.pt")
    joblib.dump(scaler, "artifacts/dl/mlp_scaler.pkl")
    joblib.dump(best_params, "artifacts/dl/mlp_best_params.pkl")

    print("모델 학습 완료 및 저장")


if __name__ == "__main__":
    main()
