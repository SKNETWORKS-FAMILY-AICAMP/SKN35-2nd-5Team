import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

from src.data.loader import load_processed_test_from_db
from src.models.dl.mlp_model import MLPClassifier
from src.utils.paths import (
    DL_ARTIFACTS_DIR,
    REPORTS_DIR,
)

# ==============================================================================
# 1. Device
# ==============================================================================


def get_device() -> torch.device:
    """사용 가능한 최적의 디바이스를 반환합니다."""

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# ==============================================================================
# 2. 학습 완료 모델 및 설정 로드
# ==============================================================================


def load_model_artifacts():
    """
    train.py에서 저장한 모델과
    Test 추론에 필요한 아티팩트를 로드합니다.
    """

    model_path = DL_ARTIFACTS_DIR / "mlp_model.pt"

    preprocessor_path = DL_ARTIFACTS_DIR / "mlp_scaler.pkl"

    best_params_path = DL_ARTIFACTS_DIR / "mlp_best_params.pkl"

    threshold_path = DL_ARTIFACTS_DIR / "mlp_threshold.pkl"

    metadata_path = DL_ARTIFACTS_DIR / "mlp_metadata.pkl"

    required_files = [
        model_path,
        preprocessor_path,
        best_params_path,
        threshold_path,
        metadata_path,
    ]

    missing_files = [path for path in required_files if not path.exists()]

    if missing_files:
        missing = "\n".join(f"- {path}" for path in missing_files)

        raise FileNotFoundError(
            f"학습된 모델 아티팩트를 찾을 수 없습니다.\n먼저 train.py를 실행하세요.\n{missing}"
        )

    # --------------------------------------------------------------------------
    # Artifact Load
    # --------------------------------------------------------------------------

    best_params = joblib.load(best_params_path)

    preprocessor = joblib.load(preprocessor_path)

    best_threshold = float(joblib.load(threshold_path))

    metadata = joblib.load(metadata_path)

    in_features = metadata["in_features"]

    # --------------------------------------------------------------------------
    # Model Load
    # --------------------------------------------------------------------------

    device = get_device()

    model = MLPClassifier(
        best_params,
        in_features,
    ).to(device)

    state_dict = torch.load(
        model_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(state_dict)

    model.eval()

    return (
        model,
        preprocessor,
        best_params,
        best_threshold,
        metadata,
    )


# ==============================================================================
# 3. Test 데이터 준비
# ==============================================================================


def prepare_test_data(
    test_df,
    preprocessor,
    target_col="Attrition",
):
    """
    train.py에서 저장한 preprocessor를 사용하여
    Test 데이터를 변환합니다.

    전처리기는 절대 fit하지 않습니다.
    """

    saved_index_cols = [column for column in test_df.columns if column.startswith("Unnamed:")]

    drop_cols = [
        target_col,
        *saved_index_cols,
    ]

    X_test = test_df.drop(columns=[column for column in drop_cols if column in test_df.columns])

    y_test = test_df[target_col]

    # 저장된 preprocessor를 그대로 사용
    X_test_scaled = preprocessor.transform(X_test)

    X_test_t = torch.tensor(
        X_test_scaled,
        dtype=torch.float32,
    )

    y_test = np.asarray(
        y_test.to_numpy(),
        dtype=np.int32,
    )

    return X_test_t, y_test


# ==============================================================================
# 4. Test DataLoader
# ==============================================================================


def create_test_loader(
    X_test,
    y_test,
    batch_size=128,
):
    """
    Test 데이터용 DataLoader를 구성합니다.
    """

    X_test_t = X_test

    y_test_t = torch.tensor(
        y_test,
        dtype=torch.float32,
    ).reshape(-1, 1)

    return DataLoader(
        TensorDataset(
            X_test_t,
            y_test_t,
        ),
        batch_size=batch_size * 2,
        shuffle=False,
    )


# ==============================================================================
# 5. Test Prediction
# ==============================================================================


def get_predictions(
    model,
    test_loader,
):
    """
    Test 데이터에 대한 실제값과 예측 확률을 반환합니다.
    """

    device = get_device()

    model.eval()

    all_targets = []
    all_probs = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)

            output = model(X_batch)

            probabilities = torch.sigmoid(output).cpu().numpy().ravel()

            all_probs.extend(probabilities)

            all_targets.extend(y_batch.cpu().numpy().ravel())

    y_true = np.asarray(
        all_targets,
        dtype=np.int32,
    )

    y_proba = np.asarray(
        all_probs,
        dtype=np.float32,
    )

    return y_true, y_proba


# ==============================================================================
# 6. Test Evaluation
# ==============================================================================


def evaluate_test(
    y_true,
    y_proba,
    threshold,
):
    """
    Validation에서 결정된 threshold를 사용하여
    독립 Test 데이터의 최종 성능을 계산합니다.
    """

    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "model": "mlp",
        "threshold": float(threshold),
        "accuracy": accuracy_score(
            y_true,
            y_pred,
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
        "roc_auc": roc_auc_score(
            y_true,
            y_proba,
        ),
        "average_precision": average_precision_score(
            y_true,
            y_proba,
        ),
    }

    return metrics, y_pred


# ==============================================================================
# 7. CSV 저장
# ==============================================================================


def save_test_metrics(metrics):
    """
    Test 성능 결과를 CSV로 저장합니다.
    """

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_path = REPORTS_DIR / "mlp_test_metrics.csv"

    metrics_df = pd.DataFrame([metrics])

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    return metrics_path


# ==============================================================================
# 8. Main
# ==============================================================================


def main():

    print("==================================================")

    print(" Deep Learning (MLP / Tabular ResNet) Test")

    print(f" Evaluation Device : {get_device()}")

    print("==================================================")

    # ==========================================================================
    # 1. 모델 및 Artifact 로드
    # ==========================================================================

    (
        model,
        preprocessor,
        best_params,
        best_threshold,
        metadata,
    ) = load_model_artifacts()

    batch_size = best_params.get(
        "batch_size",
        128,
    )

    print("\n[ Loaded Model ]")

    print(f"Input features : {metadata['in_features']}")

    print(f"Threshold      : {best_threshold:.2f}")

    print(f"Batch size     : {batch_size}")

    # ==========================================================================
    # 2. Test 데이터 로드
    # ==========================================================================

    print("\n[ Data Source ] DB (employee_attrition_processed, type='test')")
    test_df = load_processed_test_from_db()

    X_test, y_test = prepare_test_data(
        test_df,
        preprocessor,
    )

    test_loader = create_test_loader(
        X_test,
        y_test,
        batch_size=batch_size,
    )

    print("\n[ Test Data ]")

    print(f"Test samples : {len(y_test)}")

    print(f"Test batches : {len(test_loader)}")

    # ==========================================================================
    # 3. Test Prediction
    # ==========================================================================

    y_true, y_proba = get_predictions(
        model,
        test_loader,
    )

    # ==========================================================================
    # 4. Test Evaluation
    # ==========================================================================

    metrics, y_pred = evaluate_test(
        y_true,
        y_proba,
        best_threshold,
    )

    print("\n" + "=" * 50)

    print("Final Test Evaluation")

    print("=" * 50)

    print(f"Accuracy          : {metrics['accuracy']:.4f}")

    print(f"Precision         : {metrics['precision']:.4f}")

    print(f"Recall            : {metrics['recall']:.4f}")

    print(f"F1 Score          : {metrics['f1']:.4f}")

    print(f"ROC-AUC           : {metrics['roc_auc']:.4f}")

    print(f"Average Precision : {metrics['average_precision']:.4f}")

    print(f"Threshold         : {metrics['threshold']:.2f}")

    # ==========================================================================
    # 5. CSV 저장
    # ==========================================================================

    metrics_path = save_test_metrics(metrics)

    print("\nTest 평가 완료")

    print(f"Test 성능 리포트 : {metrics_path}")


if __name__ == "__main__":
    main()
