import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.loader import load_feature_dataset
from src.clustering.kmeans import perform_kmeans_clustering, get_cluster_summary

st.set_page_config(page_title="04. User Clustering", page_icon="👥", layout="wide")

def main():
    st.title("👥 04. 고객 행동 군집 분석 (K-Means Clustering)")
    st.markdown("학습 행동 패턴(총 활동량, 접속 일수, 감소율, 문제 풀이수 등)을 바탕으로 고객 군집을 프로파일링합니다.")
    
    df = load_feature_dataset()
    
    k = st.slider("군집 개수 (K)", min_value=2, max_value=6, value=4)
    
    if st.button("군집화 실행"):
        with st.spinner("K-Means 군집화 진행 중..."):
            df_clustered, kmeans, scaler = perform_kmeans_clustering(df, n_clusters=k)
            summary = get_cluster_summary(df_clustered)
            
            st.subheader("1. 📊 군집별 평균 프로필 및 이탈률")
            st.dataframe(summary.style.format("{:.2f}"), use_container_width=True)
            
            st.subheader("2. 📉 군집별 이탈률 비교")
            fig, ax = plt.subplots(figsize=(8, 4))
            summary["is_churn"].plot(kind="bar", ax=ax, color="#e17055")
            ax.set_ylabel("이탈률 (Churn Rate)")
            ax.set_title("Churn Rate by Cluster")
            st.pyplot(fig)

if __name__ == "__main__":
    main()
