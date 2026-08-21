import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.loader import load_feature_dataset

st.set_page_config(page_title="02. Churn Definition", page_icon="🎯", layout="wide")

def main():
    st.title("🎯 02. 고객 이탈(Churn) 정의 및 타깃 설계")
    
    st.markdown("""
    ### 📌 이탈(Churn)의 비즈니스 정의
    에듀테크 산타토익 결제 고객은 **단발성 결제** 후 만료되거나 **환불**하는 경우가 발생합니다.
    
    ```text
    전체 유료 결제 고객 (23,789명)
    ├── [1] 환불(Refund) 이탈: 1,096명 (4.61%) ──▶ Churn = 1
    ├── [2] 미갱신(Non-renewal) 이탈: 20,519명 (86.25%) ──▶ Churn = 1
    └── [3] 재결제/구독 유지(Retained): 2,174명 (9.14%) ──▶ Retained = 0
    ```
    """)
    
    df = load_feature_dataset()
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("📊 타깃 클래스 분포")
        churn_counts = df["is_churn"].value_counts()
        labels = ["Churned (이탈)", "Retained (유지)"]
        sizes = [churn_counts[1], churn_counts[0]]
        
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(sizes, labels=labels, autopct="%1.1f%%", colors=["#ff7675", "#55efc4"], startangle=140)
        ax.axis("equal")
        st.pyplot(fig)
        
    with col2:
        st.subheader("💡 불균형 데이터셋 대응 전략")
        st.info("""
        - **불균형 비율**: 이탈(90.9%) vs 유지(9.1%)
        - **대응 기법**:
          - **Random Forest**: `class_weight='balanced'`
          - **XGBoost**: `scale_pos_weight` 조정
          - **LightGBM**: `is_unbalance=True`
        - **평가 지표**: 단순 Accuracy 대신 **PR-AUC, ROC-AUC, F1-Score, Recall** 중점 평가
        """)

if __name__ == "__main__":
    main()
