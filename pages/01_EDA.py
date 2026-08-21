import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.loader import load_feature_dataset
from src.analysis.eda import compute_dataset_overview, compare_churn_groups, compute_correlations

st.set_page_config(page_title="01. EDA", page_icon="📊", layout="wide")

def main():
    st.title("📊 01. 탐색적 데이터 분석 (EDA)")
    st.markdown("가입 및 결제 고객 23,789명의 기본 통계 및 이탈/잔존 집단 간 행동 차이를 탐색합니다.")
    
    df = load_feature_dataset()
    overview = compute_dataset_overview(df)
    
    # 1. 상단 지표 카드
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("총 결제 고객 수", f"{overview['total_users']:,}명")
    col2.metric("이탈 고객 (Churn)", f"{overview['churn_users']:,}명", f"{overview['churn_rate']:.1f}%", delta_color="inverse")
    col3.metric("유지 고객 (Retained)", f"{overview['retained_users']:,}명", f"{overview['retained_rate']:.1f}%")
    col4.metric("추출된 모델링 피처", f"{overview['num_features']}개")
    
    st.divider()
    
    # 2. 이탈 vs 잔존 행동 비교
    st.subheader("🔍 잔존 고객 vs 이탈 고객의 초기 14일 학습 패턴 비교")
    stats_df = compare_churn_groups(df)
    st.dataframe(stats_df, use_container_width=True)
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("##### 2주차 활동량 분포 (`obs_w2_events`)")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(x="is_churn", y="obs_w2_events", data=df, ax=ax, showfliers=False, palette="Set2")
        ax.set_xticklabels(["Retained (0)", "Churned (1)"])
        ax.set_title("Week 2 Activity by Churn Status")
        st.pyplot(fig)
        
    with col_b:
        st.markdown("##### 최근 미접속 일수 분포 (`obs_recency_days`)")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(x="is_churn", y="obs_recency_days", data=df, ax=ax, palette="Set1")
        ax.set_xticklabels(["Retained (0)", "Churned (1)"])
        ax.set_title("Recency Days by Churn Status")
        st.pyplot(fig)

if __name__ == "__main__":
    main()
