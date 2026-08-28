"""CatBoost 단독 테스트 및 튜닝 스크립트 (파일 생성 없음)."""

from pathlib import Path
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, roc_auc_score

from .utils import RANDOM_STATE, TARGET_COLUMN

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRAIN_DATA_PATH = PROJECT_ROOT / "data" / "preprocessing" / "train_processed.csv"


def run_tuning():
    """CatBoost 모델 튜닝 및 터미널 성과 측정."""
    if not TRAIN_DATA_PATH.exists():
        raise FileNotFoundError(f"전처리 데이터가 없습니다: {TRAIN_DATA_PATH}")

    data = pd.read_csv(TRAIN_DATA_PATH)

    # 타깃 및 피처 분리
    y = pd.to_numeric(data[TARGET_COLUMN], errors="coerce").astype("int8")
    saved_index_cols = [c for c in data.columns if c.startswith("Unnamed:")]
    X = data.drop(columns=[TARGET_COLUMN, *saved_index_cols])

    # bool 타입을 int8로 변환 및 원핫 인코딩
    bool_cols = X.select_dtypes(include="bool").columns
    X[bool_cols] = X[bool_cols].astype("int8")
    X = pd.get_dummies(X, drop_first=True)

    # 80:20 계층 분할
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    print("🚀 [CatBoost] 단독 파인튜닝 탐색을 시작합니다...")

    param_grid = {
    "iterations": [400, 600],
    "learning_rate": [0.03, 0.05],
    "depth": [3, 4, 5],
    "l2_leaf_reg": [1, 3, 5],  # 과적합 방지 규제항 추가
    }

    base_model = CatBoostClassifier(
        eval_metric="AUC",
        random_seed=RANDOM_STATE,
        verbose=0,
    )

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=3,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    
    # 평가 수행
    y_pred = best_model.predict(X_val)
    y_proba = best_model.predict_proba(X_val)[:, 1]

    print("\n" + "=" * 50)
    print("✨ CatBoost 최적 파라미터:", grid_search.best_params_)
    print("=" * 50)
    print(classification_report(y_val, y_pred))
    print(f"🔥 ROC-AUC Score: {roc_auc_score(y_val, y_proba):.4f}")
    print("=" * 50)


if __name__ == "__main__":
    run_tuning()