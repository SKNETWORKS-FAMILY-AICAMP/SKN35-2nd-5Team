import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_PATH = os.path.join(BASE_DIR, "data", "preprocessing")
ARTIFACTS_PATH = os.path.join(BASE_DIR, "artifacts")

def load_preprocessed_data(file_name="train_processed"):
    if not file_name.endswith('.csv'):
        file_name += '.csv'
    file_full_path = os.path.join(DATA_PATH, file_name)
    return pd.read_csv(file_full_path)

def run_fine_tuning_pipeline(data, target_col="Attrition"):
    drop_cols = [target_col]
    if "Employee ID" in data.columns:
        drop_cols.append("Employee ID")
        
    X = pd.get_dummies(data.drop(columns=drop_cols), drop_first=True)
    y = data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("🔍 [GridSearchCV] 최적의 하이퍼파라미터 탐색 중...")
    
    # 파인튜닝할 탐색 범주 설정
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [12, 18, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    }

    base_rf = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)

    # 3-Fold Cross Validation으로 하이퍼파라미터 수색
    grid_search = GridSearchCV(
        estimator=base_rf,
        param_grid=param_grid,
        cv=3,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"\n✨ 최적 하이퍼파라미터: {grid_search.best_params_}")

    # 최종 평가
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    report_text = f"=== Fine-Tuned Random Forest Model Evaluation ===\n\n"
    report_text += f"Best Params: {grid_search.best_params_}\n\n"
    report_text += classification_report(y_test, y_pred)
    report_text += f"\nROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}\n"

    print("\n" + report_text)

    # 파일 저장
    os.makedirs(ARTIFACTS_PATH, exist_ok=True)
    joblib.dump(best_model, os.path.join(ARTIFACTS_PATH, "random_forest_tuned_model.pkl"))
    
    with open(os.path.join(ARTIFACTS_PATH, "result_randomforest_tuned.txt"), "w", encoding="utf-8") as f:
        f.write(report_text)

if __name__ == "__main__":
    df = load_preprocessed_data("train_processed")
    run_fine_tuning_pipeline(df)