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

from src.data.loader import load_processed_train
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

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ==============================================================================
# 0. 설정
# ==============================================================================

TARGET_COL = "Attrition"

# 기존 test.py와의 호환을 유지하면서 학습만 강화합니다.
N_TRIALS = 150
OPTUNA_EPOCHS = 45
FINAL_EPOCHS = 180
FINAL_PATIENCE = 25

# 과적합 방지 / 일반화 강화
DEFAULT_NOISE_STD = 0.01
DEFAULT_MIXUP_ALPHA = 0.15
DEFAULT_FOCAL_MIX = 0.20
DEFAULT_FOCAL_GAMMA = 1.5
DEFAULT_EMA_DECAY = 0.995

# 검증 임계값 정책
MIN_RECALL_FOR_THRESHOLD = 0.72


# ==============================================================================
# 1. 시간 / 시드 / 디바이스
# ==============================================================================


def format_seconds(seconds: float) -> str:
    if seconds < 0 or seconds != seconds:
        seconds = 0

    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    return f"{hours:d}:{minutes:02d}:{secs:02d}"


def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# ==============================================================================
# 2. 전처리
# ==============================================================================


def split_and_preprocess(train_df, target_col=TARGET_COL):
    """
    train/validation split과 preprocessing을 한 번만 수행합니다.

    기존 코드에서는 Optuna trial마다 같은 전처리 작업을 반복했습니다.
    여기서는 전처리 결과를 캐시하여 탐색 속도를 높이고,
    모든 trial이 동일한 split을 사용하도록 명시적으로 고정합니다.
    """
    saved_index_cols = [c for c in train_df.columns if c.startswith("Unnamed:")]

    drop_cols = [target_col, *saved_index_cols]

    X = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns]).copy()

    y = train_df[target_col].astype(np.float32).copy()

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=VAL_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    binary_cols = []
    continuous_cols = []

    for col in X_train.columns:
        values = X_train[col].dropna().unique()

        if len(values) > 0 and set(values).issubset({0, 1}):
            binary_cols.append(col)
        else:
            continuous_cols.append(col)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "continuous",
                StandardScaler(),
                continuous_cols,
            )
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    X_train_scaled = preprocessor.fit_transform(X_train)
    X_val_scaled = preprocessor.transform(X_val)

    X_train_scaled = np.asarray(
        X_train_scaled,
        dtype=np.float32,
    )

    X_val_scaled = np.asarray(
        X_val_scaled,
        dtype=np.float32,
    )

    y_train_np = y_train.to_numpy(dtype=np.float32)
    y_val_np = y_val.to_numpy(dtype=np.float32)

    train_X = torch.tensor(
        X_train_scaled,
        dtype=torch.float32,
    )

    train_y = torch.tensor(
        y_train_np,
        dtype=torch.float32,
    ).reshape(-1, 1)

    val_X = torch.tensor(
        X_val_scaled,
        dtype=torch.float32,
    )

    val_y = torch.tensor(
        y_val_np,
        dtype=torch.float32,
    ).reshape(-1, 1)

    feature_names = list(X.columns)
    in_features = X_train_scaled.shape[1]

    positive_rate = float(y_train_np.mean())

    print("\n[ Dataset ]")
    print(f"Train samples   : {len(train_X):,}")
    print(f"Validation data  : {len(val_X):,}")
    print(f"Input features   : {in_features}")
    print(f"Positive rate    : {positive_rate:.4f}")
    print(f"Binary features  : {len(binary_cols)}")
    print(f"Continuous feats : {len(continuous_cols)}")

    return {
        "train_X": train_X,
        "train_y": train_y,
        "val_X": val_X,
        "val_y": val_y,
        "preprocessor": preprocessor,
        "in_features": in_features,
        "feature_names": feature_names,
        "positive_rate": positive_rate,
    }


def make_loaders(data, batch_size):
    generator = torch.Generator()
    generator.manual_seed(RANDOM_STATE)

    train_loader = DataLoader(
        TensorDataset(
            data["train_X"],
            data["train_y"],
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=False,
    )

    val_loader = DataLoader(
        TensorDataset(
            data["val_X"],
            data["val_y"],
        ),
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    return train_loader, val_loader


# ==============================================================================
# 3. 손실 함수
# ==============================================================================


class HybridBCELoss(nn.Module):
    """
    BCE + Focal Loss 혼합.

    BCE:
        전체적인 확률 학습과 calibration에 유리

    Focal:
        쉽게 분류되는 샘플의 영향력을 줄이고
        경계 근처 / 어려운 샘플에 더 집중

    focal_mix=0 이면 기존 BCE와 동일하게 동작합니다.
    """

    def __init__(
        self,
        focal_mix=DEFAULT_FOCAL_MIX,
        gamma=DEFAULT_FOCAL_GAMMA,
        pos_weight=None,
    ):
        super().__init__()

        self.focal_mix = float(focal_mix)
        self.gamma = float(gamma)

        if pos_weight is None:
            self.register_buffer("pos_weight", None)
        else:
            self.register_buffer(
                "pos_weight",
                torch.tensor(
                    [float(pos_weight)],
                    dtype=torch.float32,
                ),
            )

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction="none",
        )

        bce_loss = bce.mean()

        if self.focal_mix <= 0.0:
            return bce_loss

        pt = torch.exp(-bce)

        focal = ((1.0 - pt).pow(self.gamma) * bce).mean()

        return (1.0 - self.focal_mix) * bce_loss + self.focal_mix * focal


# ==============================================================================
# 4. EMA
# ==============================================================================


class ModelEMA:
    """
    Exponential Moving Average.

    학습 중 순간적으로 좋은 방향으로 튄 weight보다
    여러 step의 평균적인 weight를 사용하여 일반화 성능을 높입니다.
    """

    def __init__(self, model, decay=DEFAULT_EMA_DECAY):
        self.decay = float(decay)

        self.shadow = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            shadow = self.shadow[name]
            shadow.mul_(self.decay)
            shadow.add_(
                param.detach(),
                alpha=1.0 - self.decay,
            )

    def apply_to(self, model):
        backup = {}

        with torch.no_grad():
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue

                backup[name] = param.detach().clone()
                param.copy_(self.shadow[name])

        return backup

    @staticmethod
    def restore(model, backup):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in backup:
                    param.copy_(backup[name])


# ==============================================================================
# 5. Mixup / 입력 노이즈
# ==============================================================================


def apply_feature_noise(X, noise_std):
    if noise_std <= 0.0:
        return X

    return X + torch.randn_like(X) * noise_std


def apply_mixup(X, y, alpha):
    if alpha <= 0.0 or X.size(0) < 2:
        return X, y

    lam = np.random.beta(alpha, alpha)

    if lam <= 0.0 or lam >= 1.0:
        return X, y

    indices = torch.randperm(
        X.size(0),
        device=X.device,
    )

    X_mixed = lam * X + (1.0 - lam) * X[indices]

    y_mixed = lam * y + (1.0 - lam) * y[indices]

    return X_mixed, y_mixed


# ==============================================================================
# 6. 평가
# ==============================================================================


@torch.no_grad()
def predict_loader(model, data_loader, device):
    model.eval()

    targets = []
    probs = []

    for X_batch, y_batch in data_loader:
        X_batch = X_batch.to(device)
        output = model(X_batch)

        batch_probs = torch.sigmoid(output)

        targets.extend(y_batch.cpu().numpy().ravel())

        probs.extend(batch_probs.cpu().numpy().ravel())

    y_true = np.asarray(
        targets,
        dtype=np.int32,
    )

    y_proba = np.asarray(
        probs,
        dtype=np.float32,
    )

    return y_true, y_proba


def calculate_metrics(
    y_true,
    y_proba,
    threshold=0.50,
):
    y_pred = (y_proba >= threshold).astype(np.int32)

    return {
        "ap": average_precision_score(
            y_true,
            y_proba,
        ),
        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
    }


# ==============================================================================
# 7. Optuna
# ==============================================================================


def run_optuna_search(
    data,
    n_trials=N_TRIALS,
    epochs=OPTUNA_EPOCHS,
):
    device = get_device()

    train_X = data["train_X"]
    train_y = data["train_y"]
    val_X = data["val_X"]
    val_y = data["val_y"]

    positive_rate = data["positive_rate"]

    train_start_time = time.time()

    def objective(trial):
        set_seed(RANDOM_STATE + trial.number)

        batch_size = trial.suggest_categorical(
            "batch_size",
            [64, 128, 256],
        )

        n_layers = trial.suggest_int(
            "n_layers",
            2,
            5,
        )

        use_residual = trial.suggest_categorical(
            "use_residual",
            [True, False],
        )

        activation = trial.suggest_categorical(
            "activation",
            ["gelu", "silu", "relu"],
        )

        noise_std = trial.suggest_float(
            "noise_std",
            0.0,
            0.04,
        )

        mixup_alpha = trial.suggest_float(
            "mixup_alpha",
            0.0,
            0.35,
        )

        focal_mix = trial.suggest_float(
            "focal_mix",
            0.0,
            0.40,
        )

        focal_gamma = trial.suggest_float(
            "focal_gamma",
            0.5,
            3.0,
        )

        ema_decay = trial.suggest_float(
            "ema_decay",
            0.990,
            0.9995,
        )

        if use_residual:
            hidden_dim = trial.suggest_int(
                "n_units_l0",
                96,
                384,
                step=32,
            )

            params = {
                "batch_size": batch_size,
                "n_layers": n_layers,
                "n_units_l0": hidden_dim,
                "use_residual": True,
                "activation": activation,
            }

            for i in range(n_layers):
                params[f"dropout_l{i}"] = trial.suggest_float(
                    f"dropout_l{i}",
                    0.03,
                    0.30,
                )

        else:
            params = {
                "batch_size": batch_size,
                "n_layers": n_layers,
                "use_residual": False,
                "activation": activation,
            }

            for i in range(n_layers):
                params[f"n_units_l{i}"] = trial.suggest_int(
                    f"n_units_l{i}",
                    96,
                    384,
                    step=32,
                )

                params[f"dropout_l{i}"] = trial.suggest_float(
                    f"dropout_l{i}",
                    0.03,
                    0.30,
                )

        params["lr"] = trial.suggest_float(
            "lr",
            2e-4,
            4e-3,
            log=True,
        )

        params["weight_decay"] = trial.suggest_float(
            "weight_decay",
            1e-6,
            3e-3,
            log=True,
        )

        params["noise_std"] = noise_std
        params["mixup_alpha"] = mixup_alpha
        params["focal_mix"] = focal_mix
        params["focal_gamma"] = focal_gamma
        params["ema_decay"] = ema_decay

        train_loader = DataLoader(
            TensorDataset(train_X, train_y),
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(RANDOM_STATE + trial.number),
            num_workers=0,
        )

        val_loader = DataLoader(
            TensorDataset(val_X, val_y),
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=0,
        )

        model = MLPClassifier(
            params,
            data["in_features"],
        ).to(device)

        # 현재 데이터는 기존 DL 결과에서 이미 precision/recall 균형이
        # 괜찮은 편이므로 과도한 pos_weight를 걸지 않습니다.
        criterion = HybridBCELoss(
            focal_mix=focal_mix,
            gamma=focal_gamma,
            pos_weight=None,
        ).to(device)

        optimizer = optim.AdamW(
            model.parameters(),
            lr=params["lr"],
            weight_decay=params["weight_decay"],
            betas=(0.9, 0.999),
        )

        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=params["lr"],
            epochs=epochs,
            steps_per_epoch=max(
                len(train_loader),
                1,
            ),
            pct_start=trial.suggest_float(
                "onecycle_pct_start",
                0.05,
                0.25,
            ),
            anneal_strategy="cos",
            div_factor=10.0,
            final_div_factor=100.0,
        )

        ema = ModelEMA(
            model,
            decay=ema_decay,
        )

        best_ap = -1.0
        best_state = None

        for epoch in range(epochs):
            model.train()

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                X_batch = apply_feature_noise(
                    X_batch,
                    noise_std,
                )

                X_batch, y_batch = apply_mixup(
                    X_batch,
                    y_batch,
                    mixup_alpha,
                )

                optimizer.zero_grad(set_to_none=True)

                output = model(X_batch)
                loss = criterion(
                    output,
                    y_batch,
                )

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()
                scheduler.step()

                ema.update(model)

            backup = ema.apply_to(model)

            y_true, y_proba = predict_loader(
                model,
                val_loader,
                device,
            )

            ema.restore(model, backup)

            val_ap = average_precision_score(
                y_true,
                y_proba,
            )

            if val_ap > best_ap:
                best_ap = val_ap

                # EMA weights를 별도로 복사
                ema_best = {name: value.detach().clone() for name, value in ema.shadow.items()}

                best_state = ema_best

            trial.report(
                best_ap,
                epoch,
            )

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return best_ap

    sampler = optuna.samplers.TPESampler(
        seed=RANDOM_STATE,
        multivariate=True,
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=12,
        ),
    )

    print(f"\n[ Optuna Search Started ] (Device: {device}, Trials: {n_trials})")

    def log_callback(study_obj, trial):
        elapsed = time.time() - train_start_time

        done_states = {
            optuna.trial.TrialState.COMPLETE,
            optuna.trial.TrialState.PRUNED,
            optuna.trial.TrialState.FAIL,
        }

        n_done = sum(1 for t in study_obj.trials if t.state in done_states)

        avg_trial_time = elapsed / n_done if n_done > 0 else 0.0

        remaining = max(
            n_trials - n_done,
            0,
        )

        eta = avg_trial_time * remaining

        value = f"{trial.value:.4f}" if trial.value is not None else "N/A"

        best_value = study_obj.best_value if study_obj.best_trial else float("nan")

        print(
            f"[Trial {trial.number:>3}/{n_trials}] "
            f"state={trial.state.name:<8} "
            f"val={value} "
            f"best={best_value:.4f} | "
            f"진행 {n_done}/{n_trials} | "
            f"경과 {format_seconds(elapsed)} | "
            f"ETA {format_seconds(eta)}"
        )

    study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=[log_callback],
    )

    total_elapsed = time.time() - train_start_time

    print("\n[ Optuna Best Parameters ]")
    print(study.best_params)
    print(f"Best Validation PR-AUC : {study.best_value:.4f}")
    print(f"총 탐색 소요 시간      : {format_seconds(total_elapsed)}")

    return study.best_params


# ==============================================================================
# 8. 최종 학습
# ==============================================================================


def train_final_model(
    best_params,
    data,
    epochs=FINAL_EPOCHS,
    patience=FINAL_PATIENCE,
):
    device = get_device()

    set_seed(RANDOM_STATE)

    train_loader, val_loader = make_loaders(
        data,
        batch_size=best_params.get(
            "batch_size",
            128,
        ),
    )

    model = MLPClassifier(
        best_params,
        data["in_features"],
    ).to(device)

    criterion = HybridBCELoss(
        focal_mix=best_params.get(
            "focal_mix",
            DEFAULT_FOCAL_MIX,
        ),
        gamma=best_params.get(
            "focal_gamma",
            DEFAULT_FOCAL_GAMMA,
        ),
        pos_weight=None,
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=best_params["lr"],
        weight_decay=best_params.get(
            "weight_decay",
            1e-4,
        ),
        betas=(0.9, 0.999),
    )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=best_params["lr"],
        epochs=epochs,
        steps_per_epoch=max(
            len(train_loader),
            1,
        ),
        pct_start=best_params.get(
            "onecycle_pct_start",
            0.15,
        ),
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=100.0,
    )

    noise_std = best_params.get(
        "noise_std",
        DEFAULT_NOISE_STD,
    )

    mixup_alpha = best_params.get(
        "mixup_alpha",
        DEFAULT_MIXUP_ALPHA,
    )

    ema_decay = best_params.get(
        "ema_decay",
        DEFAULT_EMA_DECAY,
    )

    ema = ModelEMA(
        model,
        decay=ema_decay,
    )

    best_val_ap = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_ema_state = None

    train_start_time = time.time()

    for epoch in range(epochs):
        model.train()

        train_losses = []

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            X_batch = apply_feature_noise(
                X_batch,
                noise_std,
            )

            X_batch, y_batch = apply_mixup(
                X_batch,
                y_batch,
                mixup_alpha,
            )

            optimizer.zero_grad(set_to_none=True)

            output = model(X_batch)

            loss = criterion(
                output,
                y_batch,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()
            scheduler.step()

            ema.update(model)

            train_losses.append(float(loss.detach().cpu()))

        # EMA weight로 validation
        backup = ema.apply_to(model)

        model.eval()

        val_targets = []
        val_probs = []
        val_loss_sum = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                output = model(X_batch)

                val_loss = criterion(
                    output,
                    y_batch,
                )

                val_loss_sum += float(val_loss.detach().cpu())

                probs = torch.sigmoid(output)

                val_targets.extend(y_batch.cpu().numpy().ravel())

                val_probs.extend(probs.cpu().numpy().ravel())

        current_val_loss = val_loss_sum / len(val_loader)

        current_val_ap = average_precision_score(
            val_targets,
            val_probs,
        )

        if current_val_ap > best_val_ap:
            best_val_ap = current_val_ap
            best_val_loss = current_val_loss
            best_epoch = epoch + 1
            patience_counter = 0

            best_ema_state = {name: value.detach().clone() for name, value in ema.shadow.items()}
        else:
            patience_counter += 1

        # 현재 EMA를 실제 모델에도 복구
        ema.restore(model, backup)

        elapsed = time.time() - train_start_time

        mean_train_loss = float(np.mean(train_losses)) if train_losses else 0.0

        print(
            f"[Epoch {epoch + 1:>3}/{epochs}] "
            f"train_loss={mean_train_loss:.4f} "
            f"val_ap={current_val_ap:.4f} "
            f"val_loss={current_val_loss:.4f} "
            f"best_ap={best_val_ap:.4f} "
            f"lr={scheduler.get_last_lr()[0]:.6f} "
            f"경과 {format_seconds(elapsed)}"
        )

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1} (Best Epoch: {best_epoch})")
            break

    if best_ema_state is not None:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in best_ema_state:
                    param.copy_(best_ema_state[name])

    total_elapsed = time.time() - train_start_time

    print(f"Best Validation PR-AUC : {best_val_ap:.4f}")
    print(f"Best Validation Loss   : {best_val_loss:.4f}")
    print(f"Best Epoch             : {best_epoch}")
    print(f"최종 모델 학습 소요 시간: {format_seconds(total_elapsed)}")

    return model, val_loader


# ==============================================================================
# 9. Threshold 최적화
# ==============================================================================


def find_best_threshold(
    y_true,
    y_proba,
    min_recall=MIN_RECALL_FOR_THRESHOLD,
    min_threshold=0.20,
    max_threshold=0.80,
    step=0.005,
):
    """
    최소 Recall 조건을 만족하는 구간에서 Precision과 F1을 함께 고려합니다.

    Precision만 최대화하면 극단적으로 높은 threshold를 선택할 수 있으므로
    다음 score를 사용합니다.

        selection_score = 0.65 * precision + 0.35 * f1

    같은 score라면 F1이 높은 threshold를 우선합니다.
    """

    results = []

    threshold = min_threshold

    while threshold <= max_threshold + 1e-9:
        threshold = round(threshold, 4)

        y_pred = (y_proba >= threshold).astype(np.int32)

        precision = precision_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        selection_score = 0.65 * precision + 0.35 * f1

        results.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "score": selection_score,
            }
        )

        threshold += step

    feasible = [r for r in results if r["recall"] >= min_recall]

    if feasible:
        best = max(
            feasible,
            key=lambda r: (
                r["score"],
                r["f1"],
                r["precision"],
            ),
        )
    else:
        best = max(
            results,
            key=lambda r: (
                r["f1"],
                r["precision"],
            ),
        )

    print("\n[ Threshold Optimization ]")
    print(f"Min Recall Constraint : {min_recall:.2f}")
    print(f"Best Threshold        : {best['threshold']:.3f}")
    print(f"Precision             : {best['precision']:.4f}")
    print(f"Recall                : {best['recall']:.4f}")
    print(f"F1                    : {best['f1']:.4f}")

    print("\n[ Threshold Comparison ]")
    print(f"{'Threshold':<12}{'Precision':<12}{'Recall':<12}{'F1':<12}")
    print("-" * 48)

    for result in results:
        is_best = abs(result["threshold"] - best["threshold"]) < 1e-9

        every_5_percent = abs((result["threshold"] * 100) % 5) < 1e-9

        if is_best or every_5_percent:
            print(
                f"{result['threshold']:<12.3f}"
                f"{result['precision']:<12.4f}"
                f"{result['recall']:<12.4f}"
                f"{result['f1']:<12.4f}"
            )

    return float(best["threshold"])


# ==============================================================================
# 10. Main
# ==============================================================================


def main():
    set_seed(RANDOM_STATE)

    device = get_device()

    print("==================================================")
    print(" Deep Learning (Improved MLP / Tabular ResNet)")
    print(f" Acceleration Device : {device}")
    print("==================================================")

    train_df = load_processed_train()

    # --------------------------------------------------------------------------
    # 1. Split + preprocessing
    # --------------------------------------------------------------------------
    data = split_and_preprocess(
        train_df,
        target_col=TARGET_COL,
    )

    # --------------------------------------------------------------------------
    # 2. Optuna
    # --------------------------------------------------------------------------
    best_params = run_optuna_search(
        data,
        n_trials=N_TRIALS,
        epochs=OPTUNA_EPOCHS,
    )

    # --------------------------------------------------------------------------
    # 3. 최종 모델
    # --------------------------------------------------------------------------
    model, val_loader = train_final_model(
        best_params,
        data,
        epochs=FINAL_EPOCHS,
        patience=FINAL_PATIENCE,
    )

    # --------------------------------------------------------------------------
    # 4. Validation threshold optimization
    # --------------------------------------------------------------------------
    val_true, val_proba = predict_loader(
        model,
        val_loader,
        device,
    )

    best_threshold = find_best_threshold(
        val_true,
        val_proba,
        min_recall=MIN_RECALL_FOR_THRESHOLD,
    )

    final_val_metrics = calculate_metrics(
        val_true,
        val_proba,
        threshold=best_threshold,
    )

    print("\n==================================================")
    print(" Final Validation Evaluation")
    print("==================================================")
    print(f"PR-AUC     : {final_val_metrics['ap']:.4f}")
    print(f"Precision  : {final_val_metrics['precision']:.4f}")
    print(f"Recall     : {final_val_metrics['recall']:.4f}")
    print(f"F1         : {final_val_metrics['f1']:.4f}")
    print(f"Threshold  : {best_threshold:.3f}")

    # --------------------------------------------------------------------------
    # 5. Artifact 저장
    # --------------------------------------------------------------------------
    metadata = {
        "in_features": data["in_features"],
        "feature_names": data["feature_names"],
        "best_threshold": best_threshold,
        "best_params": dict(best_params),
        "validation_pr_auc": final_val_metrics["ap"],
        "validation_precision": final_val_metrics["precision"],
        "validation_recall": final_val_metrics["recall"],
        "validation_f1": final_val_metrics["f1"],
        "training_strategy": {
            "optimizer": "AdamW",
            "scheduler": "OneCycleLR",
            "loss": "Hybrid BCE + Focal",
            "ema": True,
            "mixup": True,
            "feature_noise": True,
            "gradient_clipping": 1.0,
        },
    }

    ensure_artifact_dirs()

    model_cpu = deepcopy(model).to("cpu")

    torch.save(
        model_cpu.state_dict(),
        MLP_MODEL_PATH,
    )

    joblib.dump(
        data["preprocessor"],
        MLP_PREPROCESSOR_PATH,
    )

    joblib.dump(
        dict(best_params),
        MLP_BEST_PARAMS_PATH,
    )

    joblib.dump(
        float(best_threshold),
        MLP_THRESHOLD_PATH,
    )

    joblib.dump(
        dict(metadata),
        MLP_METADATA_PATH,
    )

    print("\n==================================================")
    print(" Training Completed")
    print("==================================================")
    print(f"Model saved      : {MLP_MODEL_PATH}")
    print(f"Preprocessor     : {MLP_PREPROCESSOR_PATH}")
    print(f"Best parameters  : {MLP_BEST_PARAMS_PATH}")
    print(f"Threshold        : {MLP_THRESHOLD_PATH}")
    print(f"Metadata         : {MLP_METADATA_PATH}")


if __name__ == "__main__":
    main()
