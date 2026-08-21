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
from src.dl.trainer import train_and_save_dl_model

st.set_page_config(page_title="06. DL Training", page_icon="🧠", layout="wide")

def load_dl_metrics():
    path = RESULTS_DIR / "dl_metrics.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def main():
    st.title("🧠 06. 딥러닝 (Multi-Layer Perceptron) 학습 및 평가")
    st.markdown("표준 정규화(StandardScaler) 및 다층 퍼셉트론(MLP) 신경망 구조로 이탈 예측을 수행합니다.")
    
    if st.button("🚀 DL (MLP) 모델 학습 실행"):
        with st.spinner("딥러닝 신경망 모델 학습 중..."):
            train_and_save_dl_model()
        st.success("✅ 학습 완료!")
        
    metrics_data = load_dl_metrics()
    if not metrics_data:
        st.info("아직 딥러닝 결과가 없습니다. 위의 버튼을 눌러 학습을 실행해주세요.")
        return
        
    mlp_data = metrics_data.get("mlp", {})
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Test Accuracy", f"{mlp_data.get('test_accuracy', 0):.4f}")
    col2.metric("F1-Score", f"{mlp_data.get('f1_score', 0):.4f}")
    col3.metric("ROC-AUC", f"{mlp_data.get('roc_auc', 0):.4f}")
    col4.metric("Train Time", f"{mlp_data.get('train_time_sec', 0):.2f}s")
    
    st.divider()
    
    if "confusion_matrix" in mlp_data:
        st.subheader("🟦 Confusion Matrix")
        cm = np.array(mlp_data["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", cbar=False,
                    xticklabels=["Retained", "Churned"],
                    yticklabels=["Retained", "Churned"], ax=ax)
        st.pyplot(fig)

if __name__ == "__main__":
    main()
