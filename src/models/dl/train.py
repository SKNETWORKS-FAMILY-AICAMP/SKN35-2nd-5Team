import random
import time
from copy import deepcopy

import joblib
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.data.loader import load_processed_train_from_db
from src.models.dl.mlp_model import MLPClassifier
from src.utils.constants import RANDOM_STATE, VAL_SIZE
from src.utils.paths import (
    MLP_BEST_PARAMS_PATH,
    MLP_METADATA_PATH,
    MLP_MODEL_PATH,
    MLP_PREPROCESSOR_PATH,
    MLP_THRESHOLD_PATH,
    ensure_artifact_dirs,
)

# Optuna 로깅 레벨 설정 (경고만 출력)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ==============================================================================
# 0. 시간 포맷 유틸
# ==============================================================================


def format_seconds(seconds: float) -> str:
    """초 단위 float를 H:MM:SS 형태의 문자열로 변환합니다."""
    if seconds < 0 or seconds != seconds:  # NaN 방어
        seconds = 0
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}"


# ==============================================================================
# 1. 시드 고정 및 디바이스(GPU / MPS / CPU) 가속 설정
# ==============================================================================


def set_seed(seed: int = RANDOM_STATE) -> None:
    """모든 라이브러리의 시드를 고정하여 재현성을 보장합니다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device() -> torch.device:
    """사용 가능한 최적의 하드웨어 가속 장치를 반환합니다 (CUDA -> MPS -> CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ==============================================================================
# 3. 데이터 전처리 및 DataLoader 구성
# ==============================================================================


def prepare_dataloaders(train_df, target_col="Attrition", batch_size=128):
    """
    연속형 피처는 StandardScaler로 정규화하고, 0/1 바이너리/원핫 피처는 보존합니다.
    """
    saved_index_cols = [c for c in train_df.columns if c.startswith("Unnamed:")]
    drop_cols = [target_col, *saved_index_cols]

    X = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    y = train_df[target_col]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=VAL_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # 이진(0/1) 피처와 연속형 피처 분리
    binary_cols = [c for c in X_train.columns if set(X_train[c].dropna().unique()).issubset({0, 1})]
    continuous_cols = [c for c in X_train.columns if c not in binary_cols]

    preprocessor = ColumnTransformer(
        transformers=[("continuous", StandardScaler(), continuous_cols)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    X_train_scaled = preprocessor.fit_transform(X_train)
    X_val_scaled = preprocessor.transform(X_val)

    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.to_numpy(), dtype=torch.float32).reshape(-1, 1)

    X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_t = torch.tensor(y_val.to_numpy(), dtype=torch.float32).reshape(-1, 1)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(RANDOM_STATE),
    )
    val_loader = DataLoader(
        TensorDataset(X_val_t, y_val_t),
        batch_size=batch_size * 2,
        shuffle=False,
    )

    in_features = X_train_scaled.shape[1]
    feature_names = list(X.columns)

    return train_loader, val_loader, preprocessor, in_features, feature_names


# ==============================================================================
# 4. Optuna 하이퍼파라미터 탐색 (AdamW, Cosine Scheduler, PR-AUC 최적화)
# ==============================================================================


def run_optuna_search(
    train_df,
    n_trials=50,
    epochs=40,
):
    """
    Optuna를 활용하여 Tabular Deep Learning 최적 하이퍼파라미터를 탐색합니다.
    """
    device = get_device()

    def objective(trial):
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
        n_layers = trial.suggest_int("n_layers", 1, 4)
        use_residual = trial.suggest_categorical("use_residual", [True, False])
        activation = trial.suggest_categorical("activation", ["gelu", "silu", "relu"])

        params = {
            "batch_size": batch_size,
            "n_layers": n_layers,
            "use_residual": use_residual,
            "activation": activation,
        }

        if use_residual:
            hidden_dim = trial.suggest_int("n_units_l0", 64, 320, step=32)
            params["n_units_l0"] = hidden_dim
            for i in range(n_layers):
                params[f"dropout_l{i}"] = trial.suggest_float(f"dropout_l{i}", 0.05, 0.45)
        else:
            for i in range(n_layers):
                params[f"n_units_l{i}"] = trial.suggest_int(f"n_units_l{i}", 64, 320, step=32)
                params[f"dropout_l{i}"] = trial.suggest_float(f"dropout_l{i}", 0.05, 0.45)

        params["lr"] = trial.suggest_float("lr", 5e-4, 1e-2, log=True)
        params["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

        set_seed(RANDOM_STATE)
        train_loader, val_loader, _, in_features, _ = prepare_dataloaders(
            train_df, batch_size=batch_size
        )

        model = MLPClassifier(params, in_features).to(device)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=params["lr"],
            weight_decay=params["weight_decay"],
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=1e-5,
        )
        criterion = nn.BCEWithLogitsLoss()

        best_trial_ap = 0.0

        for epoch in range(epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                X_b = X_batch.to(device)
                y_b = y_batch.to(device)

                optimizer.zero_grad()
                output = model(X_b)
                loss = criterion(output, y_b)
                loss.backward()
                optimizer.step()

            scheduler.step()

            # Validation PR-AUC 평가
            model.eval()
            val_targets = []
            val_probs = []

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_b = X_batch.to(device)
                    output = model(X_b)
                    probs = torch.sigmoid(output)
                    val_targets.extend(y_batch.cpu().numpy().ravel())
                    val_probs.extend(probs.cpu().numpy().ravel())

            val_ap = average_precision_score(val_targets, val_probs)
            if val_ap > best_trial_ap:
                best_trial_ap = val_ap

            trial.report(val_ap, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return best_trial_ap

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
        ),
    )

    # 탐색 시작 시각 기록 (경과/예상 잔여 시간 계산용)
    search_start_time = time.time()

    def log_callback(study, trial):
        now = time.time()
        elapsed = now - search_start_time

        # 완료(성공/가지치기 포함) 처리된 trial 수 기준으로 평균 산출
        # 주의: 이 계산은 study.optimize()가 순차 실행(n_jobs=1, 기본값)일 때만 정확합니다.
        finished_states = (
            optuna.trial.TrialState.COMPLETE,
            optuna.trial.TrialState.PRUNED,
            optuna.trial.TrialState.FAIL,
        )
        n_done = sum(1 for t in study.trials if t.state in finished_states)
        avg_trial_time = elapsed / n_done if n_done > 0 else 0.0
        remaining_trials = max(n_trials - n_done, 0)
        eta_seconds = avg_trial_time * remaining_trials
        estimated_total_seconds = elapsed + eta_seconds

        state = trial.state.name
        value = f"{trial.value:.4f}" if trial.value is not None else "N/A"

        print(
            f"[Trial {trial.number:>3}/{n_trials}] state={state:<8} val={value} "
            f"best={study.best_value:.4f} | "
            f"진행 {n_done}/{n_trials} | "
            f"경과 {format_seconds(elapsed)} | "
            f"평균 {format_seconds(avg_trial_time)}/trial | "
            f"예상 잔여 {format_seconds(eta_seconds)} | "
            f"예상 총 소요 {format_seconds(estimated_total_seconds)}"
        )

    print(f"\n[ Optuna Search Started ] (Device: {device})")
    study.optimize(objective, n_trials=n_trials, callbacks=[log_callback])

    total_elapsed = time.time() - search_start_time
    print("\n[ Optuna Best Parameters ]")
    print(study.best_params)
    print(f"Best Validation PR-AUC : {study.best_value:.4f}")
    print(f"총 탐색 소요 시간      : {format_seconds(total_elapsed)}")

    return study.best_params


# ==============================================================================
# 5. 최종 모델 학습 (PR-AUC 기반 체크포인트 복원 및 조기 종료)
# ==============================================================================


def train_final_model(
    best_params,
    in_features,
    train_loader,
    val_loader,
    epochs=150,
    patience=20,
):
    """
    최적 파라미터로 최종 모델을 학습합니다.
    Optuna 목표 지표와 일치하게 Validation PR-AUC가 최고인 모델 가중치를 복원합니다.
    """
    device = get_device()
    set_seed(RANDOM_STATE)

    model = MLPClassifier(best_params, in_features).to(device)
    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=best_params["lr"],
        weight_decay=best_params.get("weight_decay", 1e-4),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-5,
    )

    best_val_ap = -1.0
    best_val_loss = float("inf")
    patience_counter = 0
    best_weights = None
    best_epoch = 0

    train_start_time = time.time()

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_b = X_batch.to(device)
            y_b = y_batch.to(device)

            optimizer.zero_grad()
            output = model(X_b)
            loss = criterion(output, y_b)
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Validation 평가
        model.eval()
        val_targets = []
        val_probs = []
        val_loss_sum = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_b = X_batch.to(device)
                y_b = y_batch.to(device)
                output = model(X_b)
                loss = criterion(output, y_b)
                val_loss_sum += loss.item()

                probs = torch.sigmoid(output)
                val_targets.extend(y_batch.cpu().numpy().ravel())
                val_probs.extend(probs.cpu().numpy().ravel())

        val_loss = val_loss_sum / len(val_loader)
        val_ap = average_precision_score(val_targets, val_probs)

        # Optuna 목표와 일치하게 PR-AUC 기준으로 최고 체크포인트 저장
        if val_ap > best_val_ap:
            best_val_ap = val_ap
            best_val_loss = val_loss
            best_weights = deepcopy(model.state_dict())
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1

        elapsed = time.time() - train_start_time
        print(
            f"[Epoch {epoch + 1:>3}/{epochs}] val_ap={val_ap:.4f} "
            f"val_loss={val_loss:.4f} best_ap={best_val_ap:.4f} "
            f"경과 {format_seconds(elapsed)}"
        )

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1} (Best Epoch: {best_epoch})")
            break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    total_elapsed = time.time() - train_start_time
    print(f"Best Validation PR-AUC : {best_val_ap:.4f}")
    print(f"Best Validation Loss   : {best_val_loss:.4f}")
    print(f"최종 모델 학습 소요 시간: {format_seconds(total_elapsed)}")

    return model


# ==============================================================================
# 6. 예측 및 임계값(Threshold) 최적화
# ==============================================================================


def get_predictions(model, data_loader):
    """DataLoader로부터 확률값 및 실제 타깃을 추출합니다."""
    device = get_device()
    model.eval()

    all_targets = []
    all_probs = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_b = X_batch.to(device)
            output = model(X_b)
            probs = torch.sigmoid(output)
            all_targets.extend(y_batch.cpu().numpy().ravel())
            all_probs.extend(probs.cpu().numpy().ravel())

    y_true = np.asarray(all_targets, dtype=np.int32)
    y_proba = np.asarray(all_probs, dtype=np.float32)

    return y_true, y_proba


def find_best_threshold(
    y_true, y_proba, min_recall=0.80, min_threshold=0.1, max_threshold=0.9, step=0.01
):
    """
    최소 재현율(Recall >= min_recall) 제약 하에서
    정밀도(Precision)를 극대화하는 최적 임계값을 탐색합니다.
    """
    results = []
    best_threshold = None
    best_precision = -1.0
    best_f1 = -1.0

    threshold = min_threshold
    while threshold <= max_threshold:
        y_pred = (y_proba >= threshold).astype(int)

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        results.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

        if recall >= min_recall and precision > best_precision:
            best_precision = precision
            best_threshold = threshold
            best_f1 = f1

        threshold = round(threshold + step, 10)

    # 제약조건 만족 임계값이 없는 경우 F1 최대 지점으로 Fallback
    if best_threshold is None:
        best_result = max(results, key=lambda r: r["f1"])
        best_threshold = best_result["threshold"]
        best_precision = best_result["precision"]
        best_f1 = best_result["f1"]

    print("\n[ Threshold Optimization ]")
    print(f"Min Recall Constraint : {min_recall}")
    print(f"Best Threshold : {best_threshold:.2f}")
    print(f"Best Precision (recall >= {min_recall}) : {best_precision:.4f}")
    print(f"Validation F1 at Best Threshold : {best_f1:.4f}")

    print("\n[ Threshold Comparison ]")
    print(f"{'Threshold':<12}{'Precision':<12}{'Recall':<12}{'F1':<12}")
    print("-" * 48)

    for result in results:
        t = result["threshold"]
        if abs(t - best_threshold) < 0.001 or abs((t * 100) % 5) < 0.001:
            print(
                f"{t:<12.2f}"
                f"{result['precision']:<12.4f}"
                f"{result['recall']:<12.4f}"
                f"{result['f1']:<12.4f}"
            )

    return best_threshold


def apply_threshold(y_proba, threshold):
    """임계값을 적용하여 이진 예측 라벨을 생성합니다."""
    return (y_proba >= threshold).astype(int)


# ==============================================================================
# 7. 메인 실행 흐름 (전체 파이프라인 및 아티팩트 저장)
# ==============================================================================


def main():
    set_seed(RANDOM_STATE)
    device = get_device()
    print("==================================================")
    print(" Deep Learning (MLP / Tabular ResNet) Training")
    print(f" Acceleration Device : {device}")

    print("\n[ Data Source ] DB (employee_attrition_processed, type='train')")
    train_df = load_processed_train_from_db()

    # 1. 하이퍼파라미터 최적화
    best_params = run_optuna_search(train_df, n_trials=150, epochs=40)

    # 2. 최적 배치 사이즈로 최종 DataLoader 구성
    batch_size = best_params.get("batch_size", 128)
    (
        train_loader,
        val_loader,
        preprocessor,
        in_features,
        feature_names,
    ) = prepare_dataloaders(train_df, batch_size=batch_size)

    # 3. 최종 모델 학습
    model = train_final_model(
        best_params,
        in_features,
        train_loader,
        val_loader,
        epochs=150,
        patience=20,
    )

    print("\n" + "=" * 50)
    print("Training Completed")

    # 4. 검증셋 기준 임계값 최적화
    val_true, val_proba = get_predictions(model, val_loader)
    best_threshold = find_best_threshold(val_true, val_proba, min_recall=0.80)

    # 6. 모델 구성요소와 공통 스키마의 DL 성능 CSV 저장
    metadata = {
        "in_features": in_features,
        "feature_names": feature_names,
        "best_threshold": best_threshold,
        "best_params": best_params,
    }

    ensure_artifact_dirs()
    model_cpu = deepcopy(model).to("cpu")
    torch.save(model_cpu.state_dict(), MLP_MODEL_PATH)
    joblib.dump(preprocessor, MLP_PREPROCESSOR_PATH)
    joblib.dump(dict(best_params), MLP_BEST_PARAMS_PATH)
    joblib.dump(float(best_threshold), MLP_THRESHOLD_PATH)
    joblib.dump(dict(metadata), MLP_METADATA_PATH)


if __name__ == "__main__":
    main()
