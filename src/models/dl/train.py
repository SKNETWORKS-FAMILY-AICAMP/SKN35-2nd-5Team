from copy import deepcopy

import joblib
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.data.loader import load_processed_train
from src.models.dl.mlp_model import MLPClassifier
from src.utils.constants import RANDOM_STATE, TEST_SIZE, VAL_SIZE
from src.utils.metrics import evaluate_model


def prepare_dataloaders(df, target_col="Attrition", batch_size=64):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=VAL_SIZE,
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    X_train_t = torch.tensor(
        X_train_scaled,
        dtype=torch.float32,
    )

    y_train_t = torch.tensor(
        y_train.to_numpy(),
        dtype=torch.float32,
    ).reshape(-1, 1)

    X_val_t = torch.tensor(
        X_val_scaled,
        dtype=torch.float32,
    )

    y_val_t = torch.tensor(
        y_val.to_numpy(),
        dtype=torch.float32,
    ).reshape(-1, 1)

    X_test_t = torch.tensor(
        X_test_scaled,
        dtype=torch.float32,
    )

    y_test_t = torch.tensor(
        y_test.to_numpy(),
        dtype=torch.float32,
    ).reshape(-1, 1)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        TensorDataset(X_val_t, y_val_t),
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        TensorDataset(X_test_t, y_test_t),
        batch_size=batch_size,
        shuffle=False,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        scaler,
        X_train.shape[1],
    )


def run_optuna_search(
    train_loader,
    val_loader,
    in_features,
    n_trials=10,
    epochs=30,
):
    def objective(trial):
        n_layers = trial.suggest_int(
            "n_layers",
            1,
            3,
        )

        params = {
            "n_layers": n_layers,
        }

        for i in range(n_layers):
            params[f"n_units_l{i}"] = trial.suggest_int(
                f"n_units_l{i}",
                16,
                128,
            )

            params[f"dropout_l{i}"] = trial.suggest_float(
                f"dropout_l{i}",
                0.1,
                0.5,
            )

        params["lr"] = trial.suggest_float(
            "lr",
            1e-4,
            1e-2,
            log=True,
        )

        model = MLPClassifier(
            params,
            in_features,
        )

        optimizer = optim.Adam(
            model.parameters(),
            lr=params["lr"],
        )

        criterion = nn.BCEWithLogitsLoss()

        for epoch in range(epochs):
            model.train()

            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()

                output = model(X_batch)

                loss = criterion(
                    output,
                    y_batch,
                )

                loss.backward()

                optimizer.step()

            model.eval()

            val_loss = 0.0

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    output = model(X_batch)

                    loss = criterion(
                        output,
                        y_batch,
                    )

                    val_loss += loss.item()

            val_loss /= len(val_loader)

            trial.report(
                val_loss,
                epoch,
            )

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

    study.optimize(
        objective,
        n_trials=n_trials,
    )

    print("\n[ Optuna Best Parameters ]")
    print(study.best_params)

    print(f"Best Validation Loss : {study.best_value:.4f}")

    return study.best_params


def train_final_model(
    best_params,
    in_features,
    train_loader,
    val_loader,
    epochs=100,
    patience=15,
):
    model = MLPClassifier(
        best_params,
        in_features,
    )

    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=best_params["lr"],
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    best_val_loss = float("inf")
    patience_counter = 0
    best_weights = None

    for epoch in range(epochs):
        model.train()

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()

            output = model(X_batch)

            loss = criterion(
                output,
                y_batch,
            )

            loss.backward()
            optimizer.step()

        model.eval()

        val_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                output = model(X_batch)

                loss = criterion(
                    output,
                    y_batch,
                )

                val_loss += loss.item()

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

    # 가장 성능 좋았던 Validation 모델 복원
    if best_weights is not None:
        model.load_state_dict(best_weights)

    print(f"Best Validation Loss : {best_val_loss:.4f}")

    return model


def get_predictions(model, data_loader):
    model.eval()

    all_targets = []
    all_probs = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            output = model(X_batch)

            probs = torch.sigmoid(output)

            all_targets.extend(y_batch.cpu().numpy().ravel())

            all_probs.extend(probs.cpu().numpy().ravel())

    y_true = np.asarray(
        all_targets,
        dtype=np.int32,
    )

    y_proba = np.asarray(
        all_probs,
        dtype=np.float32,
    )

    return y_true, y_proba


def find_best_threshold(
    y_true,
    y_proba,
    min_threshold=0.1,
    max_threshold=0.9,
    step=0.01,
):
    best_threshold = 0.5
    best_f1 = 0.0

    threshold = min_threshold

    while threshold <= max_threshold:
        y_pred = (y_proba >= threshold).astype(int)

        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

        threshold += step

    print("\n[ Threshold Optimization ]")
    print(f"Best Threshold : {best_threshold:.2f}")
    print(f"Best Validation F1 : {best_f1:.4f}")

    return best_threshold


def apply_threshold(y_proba, threshold):
    return (y_proba >= threshold).astype(int)


def main():
    df = load_processed_train()

    (
        train_loader,
        val_loader,
        test_loader,
        scaler,
        in_features,
    ) = prepare_dataloaders(df)

    print("\n[ DataLoader ]")
    print(f"Train batches : {len(train_loader)}")
    print(f"Validation batches : {len(val_loader)}")
    print(f"Test batches : {len(test_loader)}")

    best_params = run_optuna_search(
        train_loader,
        val_loader,
        in_features,
    )

    model = train_final_model(
        best_params,
        in_features,
        train_loader,
        val_loader,
    )

    val_true, val_proba = get_predictions(
        model,
        val_loader,
    )

    best_threshold = find_best_threshold(
        val_true,
        val_proba,
    )

    test_true, test_proba = get_predictions(
        model,
        test_loader,
    )

    test_pred = apply_threshold(
        test_proba,
        best_threshold,
    )

    print("\n" + "=" * 50)
    print("Final Test Evaluation")
    print("=" * 50)

    evaluate_model(
        test_true,
        test_pred,
        test_proba,
    )

    torch.save(
        model.state_dict(),
        "artifacts/dl/mlp_model.pt",
    )

    joblib.dump(
        scaler,
        "artifacts/dl/mlp_scaler.pkl",
    )

    joblib.dump(
        best_params,
        "artifacts/dl/mlp_best_params.pkl",
    )

    joblib.dump(
        best_threshold,
        "artifacts/dl/mlp_threshold.pkl",
    )

    print("\n모델 학습 및 저장 완료")
    print(f"저장된 Threshold : {best_threshold:.2f}")


if __name__ == "__main__":
    main()
