import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# 프로젝트 루트 및 저장 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_PATH = os.path.join(BASE_DIR, "data", "preprocessing")
ARTIFACTS_PATH = os.path.join(BASE_DIR, "artifacts")

def load_preprocessed_data(file_name="train_processed"):
    if not file_name.endswith('.csv'):
        file_name += '.csv'
        
    file_full_path = os.path.join(DATA_PATH, file_name)
    if not os.path.exists(file_full_path):
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {file_full_path}")
    
    df = pd.read_csv(file_full_path)
    print(f" 성공적으로 데이터를 로드했습니다: {df.shape}")
    return df

def run_pipeline(data, target_col="Attrition"):
    # 1. Feature(X)와 Target(y) 분리
    drop_cols = [target_col]
    if "Employee ID" in data.columns:
        drop_cols.append("Employee ID")
        
    X = data.drop(columns=drop_cols)
    y = data[target_col]

    # 원-핫 인코딩
    X = pd.get_dummies(X, drop_first=True)

    # 2. Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Random Forest 모델 학습
    print(" Random Forest 모델 학습 시작...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # 4. 평가 예측
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    report_text = f"=== Random Forest Model Evaluation ===\n\n"
    report_text += classification_report(y_test, y_pred)
    report_text += f"\nROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}\n"

    print("\n" + report_text)

    # 5. 폴더 생성 (artifacts)
    os.makedirs(ARTIFACTS_PATH, exist_ok=True)

    # 6. .pkl 모델 저장
    model_save_path = os.path.join(ARTIFACTS_PATH, "random_forest_model.pkl")
    joblib.dump(model, model_save_path)
    print(f" 모델 파일이 저장되었습니다: {model_save_path}")

    # 7. result_randomforest.txt 결과 저장
    result_txt_path = os.path.join(ARTIFACTS_PATH, "result_randomforest.txt")
    with open(result_txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f" 결과 텍스트 파일이 저장되었습니다: {result_txt_path}")

    # 8. ★ 피처 중요도(Top 10) 추출 및 시각화 저장 ★
    importances = pd.Series(model.feature_importances_, index=X.columns)
    top10_features = importances.sort_values(ascending=True).tail(10) # 상위 10개

    plt.figure(figsize=(10, 6))
    top10_features.plot(kind='barh', color='skyblue')
    plt.title('Top 10 Feature Importances (Random Forest)', fontsize=14)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.tight_layout()

    # 이미지 파일 저장
    img_save_path = os.path.join(ARTIFACTS_PATH, "rf_feature_importance.png")
    plt.savefig(img_save_path, dpi=300)
    plt.close()
    print(f"📊 피처 중요도 그래프가 저장되었습니다: {img_save_path}")

if __name__ == "__main__":
    DATA_FILE_NAME = "train_processed"
    
    try:
        df = load_preprocessed_data(DATA_FILE_NAME)
        run_pipeline(df, target_col="Attrition")
        print("\n 모든 작업 및 시각화 저장이 완료되었습니다! 🎉")
    except Exception as e:
        print(f" 실행 중 오류 발생: {e}")