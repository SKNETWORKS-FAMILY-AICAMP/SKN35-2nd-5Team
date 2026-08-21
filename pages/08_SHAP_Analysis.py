import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.utils.paths import ML_MODELS_DIR, RESULTS_DIR
from src.data.loader import get_train_test_data

st.set_page_config(page_title="08. SHAP Analysis", page_icon="🔍", layout="wide")

def main():
    st.title("🔍 08. 모델 해석 및 피처 기여도 (SHAP / Feature Importance)")
    st.markdown("XGBoost / LightGBM 등 최고 성능 모델이 어떤 피처를 근거로 이탈을 판단했는지 해석합니다.")
    
    model_choice = st.selectbox("해석할 모델 선택", ["lightgbm", "xgboost", "random_forest", "decision_tree"])
    model_path = ML_MODELS_DIR / f"{model_choice}.joblib"
    
    if not model_path.exists():
        st.warning(f"{model_choice} 모델이 아직 학습되지 않았습니다. 05 페이지에서 학습을 먼저 진행해주세요.")
        return
        
    model = joblib.load(model_path)
    X_train, X_test, y_train, y_test, feature_names = get_train_test_data()
    
    if hasattr(model, "feature_importances_"):
        st.subheader(f"📊 {model_choice.upper()} Feature Importance Top 15")
        fi = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False).head(15)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x=fi.values, y=fi.index, ax=ax, palette="viridis")
        ax.set_title(f"Top 15 Important Features ({model_choice})")
        ax.set_xlabel("Importance")
        st.pyplot(fig)
        
        st.info("""
        **💡 핵심 이탈 결정 요인**:
        1. **`obs_decay_ratio` / `obs_w2_events`**: 2주차 학습량 급감 여부가 이탈 예측에 가장 결정적인 영향
        2. **`obs_recency_days`**: 관측 기간 종료 시점 기준 최근 며칠간 미접속 상태인지가 직결
        3. **`obs_active_days`**: 14일 중 실제 접속한 일수가 적을수록 이탈 확률 급상승
        """)

if __name__ == "__main__":
    main()
