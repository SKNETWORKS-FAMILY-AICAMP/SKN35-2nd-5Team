"""Random Forest 하이퍼파라미터 파인튜닝 스크립트."""

from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, roc_auc_score

from .utils import RANDOM_STATE, TARGET_COLUMN

# 프로젝트 최상위 경로 설정
PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRAIN_DATA_PATH = PROJECT_ROOT / "data" / "preprocessing" / "train_processed.csv"
REPORTS_DIR = PROJECT_ROOT / "artifacts" / "reports"


def run_tuning():
    """GridSearchCV를 활용해 Random Forest 최적의 파라미터를 탐색합니다."""
    if not TRAIN_DATA_PATH.exists():
        raise FileNotFoundError(f"전처리 데이터가 없습니다: {TRAIN_DATA_PATH}")

    data = pd.read_csv(TRAIN_DATA_PATH)
    
    # Target / Feature 분리 및 bool 변수 수치화
    y = pd.to_numeric(data[TARGET_COLUMN], errors="coerce").astype("int8")
    saved_index_cols = [c for c in data.columns if c.startswith("Unnamed:")]
    X = data.drop(columns=[TARGET_COLUMN, *saved_index_cols])
    
    bool_cols = X.select_dtypes(include="bool").columns
    X[bool_cols] = X[bool_cols].astype("int8")
    
    # 원-핫 인코딩 안전장치
    X = pd.get_dummies(X, drop_first=True)

    # Train / Validation Split (20% 검증)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    print("🔍 [GridSearchCV] Random Forest 파인튜닝 시작...")
    
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 12, 15],
        "min_samples_split": [2, 5],
        "max_features": ["sqrt", "log2"],
    }

    base_model = RandomForestClassifier(
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
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
    print(f"\n✨ 최적 파라미터 조합: {grid_search.best_params_}")

    # 검증셋 평가
    y_pred = best_model.predict(X_val)
    y_proba = best_model.predict_proba(X_val)[:, 1]

    report_text = "=== Fine-Tuned Random Forest Evaluation ===\n\n"
    report_text += f"Best Params: {grid_search.best_params_}\n\n"
    report_text += classification_report(y_val, y_pred)
    report_text += f"\nROC-AUC Score: {roc_auc_score(y_val, y_proba):.4f}\n"

    print("\n" + report_text)

    # 튜닝 결과 저장
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    result_path = REPORTS_DIR / "rf_tuning_result.txt"
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(report_text)
        
    print(f"📊 튜닝 결과 리포트 저장 완료: {result_path}")


if __name__ == "__main__":
    run_tuning()