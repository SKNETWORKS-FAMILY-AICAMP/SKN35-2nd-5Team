import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.loader import load_feature_dataset
from src.analysis.eda import compute_correlations

st.set_page_config(page_title="03. Feature Engineering", page_icon="⚙️", layout="wide")

def main():
    st.title("⚙️ 03. 피처 엔지니어링 & 상관관계")
    st.markdown("결제 시점($T_0$) 기준 초기 14일 관측 윈도우에서 추출한 피처들을 확인합니다.")
    
    df = load_feature_dataset()
    
    st.subheader("1. 📋 피처 데이터셋 미리보기 (23,789 $\times$ 35)")
    st.dataframe(df.head(20), use_container_width=True)
    
    st.divider()
    
    st.subheader("2. 📈 이탈 타깃(`is_churn`)과의 피처 상관계수 순위")
    corrs = compute_correlations(df)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.dataframe(corrs.to_frame("상관계수 (Correlation)"), use_container_width=True)
        
    with col2:
        fig, ax = plt.subplots(figsize=(8, 10))
        sns.barplot(x=corrs.values, y=corrs.index, ax=ax, palette="coolwarm")
        ax.set_title("Feature Correlations with is_churn")
        ax.set_xlabel("Correlation")
        st.pyplot(fig)

if __name__ == "__main__":
    main()
