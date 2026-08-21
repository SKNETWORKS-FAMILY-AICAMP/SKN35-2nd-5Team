import streamlit as st
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.utils.paths import RESULTS_DIR
from src.ml.trainer import train_and_save_all_ml_models

st.set_page_config(page_title="05. ML Training", page_icon="🤖", layout="wide")

def load_ml_metrics():
    path = RESULTS_DIR / "ml_metrics.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def main():
    st.title("🤖 05. 머신러닝 4종 모델 학습 및 평가")
    st.markdown("Decision Tree, Random Forest, XGBoost, LightGBM 모델을 학습하고 성능을 비교합니다.")
    
    if st.button("🚀 ML 4종 모델 일괄 학습 실행"):
        with st.spinner("4개 머신러닝 모델 학습 중... (약 5~10초 소요)"):
            train_and_save_all_ml_models()
        st.success("✅ 학습 완료!")
        
    metrics_data = load_ml_metrics()
    if not metrics_data:
        st.info("아직 학습 결과가 없습니다. 위의 'ML 4종 모델 일괄 학습 실행' 버튼을 눌러주세요.")
        return
        
    models = ["decision_tree", "random_forest", "xgboost", "lightgbm"]
    
    # 1. 메트릭 테이블
    st.subheader("1. 🏆 모델별 성능 지표")
    rows = []
    for m in models:
        if m in metrics_data:
            d = metrics_data[m]
            rows.append({
                "Model": d.get("model_name", m),
                "Train Acc": f"{d.get('train_accuracy', 0):.4f}",
                "Test Acc": f"{d.get('test_accuracy', 0):.4f}",
                "Precision": f"{d.get('precision', 0):.4f}",
                "Recall": f"{d.get('recall', 0):.4f}",
                "F1-Score": f"{d.get('f1_score', 0):.4f}",
                "ROC-AUC": f"{d.get('roc_auc', 0):.4f}" if d.get("roc_auc") else "N/A",
                "Train Time (s)": f"{d.get('train_time_sec', 0):.2f}"
            })
    df_metrics = pd.DataFrame(rows).set_index("Model")
    st.dataframe(df_metrics, use_container_width=True)
    
    # Decision Tree 과적합 코멘트
    dt = metrics_data.get("decision_tree")
    if dt:
        st.warning(f"💡 **Decision Tree 오버피팅 관측**: Train Accuracy **{dt['train_accuracy']*100:.2f}%** 대비 Test Accuracy **{dt['test_accuracy']*100:.2f}%** 및 ROC-AUC **{dt['roc_auc']:.3f}**로 훈련 데이터에 심하게 과적합되었음을 확인할 수 있습니다. Random Forest 및 부스팅 모델에서 일반화 성능이 대폭 향상됩니다.")

    st.divider()

    # 2. Confusion Matrix
    st.subheader("2. 🟦 Confusion Matrix 비교")
    cols_cm = st.columns(4)
    for i, m in enumerate(models):
        if m in metrics_data and "confusion_matrix" in metrics_data[m]:
            cm = np.array(metrics_data[m]["confusion_matrix"])
            with cols_cm[i]:
                fig, ax = plt.subplots(figsize=(4, 4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                            xticklabels=["Retained", "Churned"],
                            yticklabels=["Retained", "Churned"], ax=ax)
                ax.set_title(metrics_data[m]["model_name"])
                st.pyplot(fig)

if __name__ == "__main__":
    main()
