"""Train and evaluate all machine learning models."""

from pathlib import Path
from time import perf_counter

import pandas as pd

from .logistic_regression import create_logistic_regression
from .random_forest import create_random_forest
from .xgboost import create_xgboost
from .lightgbm import create_lightgbm

from .utils import (
    load_data,
    split_data,
    evaluate_model,
)


def main() -> None:
    """
    전체 머신러닝 학습 파이프라인.

    1. 데이터 로드
    2. Train / Test 분리
    3. 모델 생성
    4. 모델 학습
    5. 모델 평가
    6. 모델 성능 비교
    """

    # ========================================================
    # 1. 데이터 경로 설정
    # ========================================================

    data_path = Path(
        "data/processed/train_processed.csv"
    )

    # ========================================================
    # 2. 데이터 불러오기
    # ========================================================

    X, y = load_data(data_path)

    print()
    print("=" * 70)
    print("Dataset Information")
    print("=" * 70)

    print(
        f"Number of samples : {len(X):,}"
    )

    print(
        f"Number of features: {X.shape[1]}"
    )

    print()

    print("Target distribution")

    print(
        y.value_counts()
        .sort_index()
        .to_string()
    )

    print()

    print("Target ratio")

    print(
        y.value_counts(normalize=True)
        .sort_index()
        .round(4)
        .to_string()
    )

    # ========================================================
    # 3. Train / Test 분리
    # ========================================================

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = split_data(
        X,
        y,
    )

    print()
    print("=" * 70)
    print("Train / Test Split")
    print("=" * 70)

    print(
        f"Train samples : {len(X_train):,}"
    )

    print(
        f"Test samples  : {len(X_test):,}"
    )

    print()

    # ========================================================
    # 4. 사용할 모델 생성
    # ========================================================

    models = {
        "LogisticRegression":
            create_logistic_regression(),

        "RandomForest":
            create_random_forest(),

        "XGBoost":
            create_xgboost(),

        "LightGBM":
            create_lightgbm(),
    }

    results = []

    # ========================================================
    # 5. 모델 학습 및 평가
    # ========================================================

    for model_name, model in models.items():

        print()
        print("=" * 70)
        print(f"Training : {model_name}")
        print("=" * 70)

        start_time = perf_counter()

        # ----------------------------------------------------
        # 모델 학습
        # ----------------------------------------------------

        model.fit(
            X_train,
            y_train,
        )

        # ----------------------------------------------------
        # 모델 평가
        # ----------------------------------------------------

        metrics = evaluate_model(
            model,
            X_test,
            y_test,
        )

        elapsed_time = (
            perf_counter()
            - start_time
        )

        metrics["model"] = model_name
        metrics["time_sec"] = elapsed_time

        results.append(metrics)

        # ----------------------------------------------------
        # 현재 모델 결과 출력
        # ----------------------------------------------------

        print(
            f"Accuracy          : "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"Precision         : "
            f"{metrics['precision']:.4f}"
        )

        print(
            f"Recall            : "
            f"{metrics['recall']:.4f}"
        )

        print(
            f"F1-score          : "
            f"{metrics['f1']:.4f}"
        )

        print(
            f"ROC-AUC           : "
            f"{metrics['roc_auc']:.4f}"
        )

        print(
            f"Average Precision : "
            f"{metrics['average_precision']:.4f}"
        )

        print()

        print(
            "Confusion Matrix"
        )

        print(
            f"TN={metrics['tn']:,} | "
            f"FP={metrics['fp']:,} | "
            f"FN={metrics['fn']:,} | "
            f"TP={metrics['tp']:,}"
        )

        print()

        print(
            f"Elapsed Time : "
            f"{elapsed_time:.2f} sec"
        )

    # ========================================================
    # 6. 모든 모델 결과 DataFrame으로 변환
    # ========================================================

    result_df = pd.DataFrame(
        results
    )

    # 컬럼 출력 순서 설정
    result_df = result_df[
        [
            "model",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "average_precision",
            "tn",
            "fp",
            "fn",
            "tp",
            "time_sec",
        ]
    ]

    # ========================================================
    # 7. ROC-AUC 기준으로 모델 순위 정렬
    # ========================================================

    result_df = result_df.sort_values(
        by="roc_auc",
        ascending=False,
    ).reset_index(
        drop=True
    )

    # ========================================================
    # 8. 최종 비교 결과 출력
    # ========================================================

    print()
    print()
    print("=" * 100)
    print("Final Model Comparison")
    print("=" * 100)

    print(
        result_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # ========================================================
    # 9. 가장 높은 ROC-AUC 모델 출력
    # ========================================================

    best_model = result_df.iloc[0]

    print()
    print("=" * 100)

    print(
        f"Best ROC-AUC Model : "
        f"{best_model['model']}"
    )

    print(
        f"ROC-AUC            : "
        f"{best_model['roc_auc']:.4f}"
    )

    print(
        f"Recall             : "
        f"{best_model['recall']:.4f}"
    )

    print(
        f"F1-score           : "
        f"{best_model['f1']:.4f}"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()