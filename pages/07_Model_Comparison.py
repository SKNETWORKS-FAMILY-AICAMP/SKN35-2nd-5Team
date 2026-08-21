import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.comparison.model_comparison import load_all_metrics, build_comparison_dataframe

st.set_page_config(page_title="07. Model Comparison", page_icon="🏆", layout="wide")

def main():
    st.title("🏆 07. 전체 모델(ML & DL) 종합 벤치마크 비교")
    st.markdown("Decision Tree, Random Forest, XGBoost, LightGBM, MLP의 성능 지표와 ROC/PR Curve를 종합 비교합니다.")
    
    all_metrics = load_all_metrics()
    if not all_metrics:
        st.warning("비교할 모델 결과가 없습니다. ML 또는 DL 학습 페이지에서 먼저 학습을 완료해주세요.")
        return
        
    df_comp = build_comparison_dataframe(all_metrics)
    
    st.subheader("1. 🥇 모델 종합 랭킹 (ROC-AUC 기준 정렬)")
    st.dataframe(
        df_comp.style.highlight_max(subset=["ROC-AUC", "PR-AUC", "F1-Score", "Test Accuracy"], color="#55efc4")
               .format({"Train Accuracy": "{:.4f}", "Test Accuracy": "{:.4f}", "Precision (Churn)": "{:.4f}",
                        "Recall (Churn)": "{:.4f}", "F1-Score": "{:.4f}", "ROC-AUC": "{:.4f}", "PR-AUC": "{:.4f}",
                        "Train Time (s)": "{:.2f}"}),
        use_container_width=True
    )
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("2. 📈 ROC Curve 종합 비교")
        fig, ax = plt.subplots(figsize=(7, 5))
        for m_id, d in all_metrics.items():
            if d.get("roc_curve"):
                roc = d["roc_curve"]
                auc = d.get("roc_auc", 0)
                ax.plot(roc["fpr"], roc["tpr"], label=f"{d.get('model_name', m_id)} (AUC={auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        st.pyplot(fig)
        
    with col2:
        st.subheader("3. 📉 Precision-Recall Curve 종합 비교")
        fig, ax = plt.subplots(figsize=(7, 5))
        for m_id, d in all_metrics.items():
            if d.get("pr_curve"):
                pr = d["pr_curve"]
                auc = d.get("pr_auc", 0)
                ax.plot(pr["recall"], pr["precision"], label=f"{d.get('model_name', m_id)} (AUC={auc:.3f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend()
        st.pyplot(fig)

if __name__ == "__main__":
    main()
