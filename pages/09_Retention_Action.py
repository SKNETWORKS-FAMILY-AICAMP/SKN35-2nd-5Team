import streamlit as st
import pandas as pd
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.loader import load_feature_dataset
from src.retention.retention_action import generate_user_retention_plan

st.set_page_config(page_title="09. Retention Action", page_icon="💌", layout="wide")

def main():
    st.title("💌 09. 개인화 맞춤형 리텐션(CRM) 액션 & LLM 메시징")
    st.markdown("이탈 고위험 고객의 학습 패턴을 진단하고, 맞춤형 리텐션 전략 및 CRM 푸시 메시지를 생성합니다.")
    
    df = load_feature_dataset()
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("1. 👤 대상 고객 선택")
        user_list = df[df["is_churn"] == 1]["user_id"].head(50).tolist()
        selected_user = st.selectbox("진단할 이탈 위험 고객 선택", user_list)
        
        user_row = df[df["user_id"] == selected_user].iloc[0].to_dict()
        st.markdown(f"""
        - **총 활동 이벤트**: {user_row['obs_total_events']:,}회
        - **2주차 활동량**: {user_row['obs_w2_events']:,}회 (감소비율: {user_row['obs_decay_ratio']:.2f})
        - **접속 일수**: {user_row['obs_active_days']}일 / 14일
        - **최근 미접속 일수**: {user_row['obs_recency_days']:.1f}일
        - **문제 풀이 수**: {user_row['obs_respond_count']}회
        """)
        
    with col2:
        st.subheader("2. 🩺 위험 유형 진단 & 처방")
        plan = generate_user_retention_plan(user_row)
        diag = plan["risk_diagnosis"]
        
        st.error(f"**위험 유형**: {diag['risk_title']}")
        st.write(f"**상세 진단**: {diag['description']}")
        st.success(f"**추천 리텐션 액션**: {diag['recommended_action']}")
        
        st.divider()
        st.subheader("3. 📱 LLM 생성 맞춤형 CRM 푸시 메시지")
        st.code(plan["crm_push_message"], language="text")

if __name__ == "__main__":
    main()
