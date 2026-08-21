import streamlit as st

st.set_page_config(
    page_title="고객 이탈 예측 & 리텐션 대시보드",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

def main():
    st.title("🎯 AI 기반 가입 고객 이탈(Churn) 예측 & 리텐션 액션 시스템")
    st.markdown("### SKN35-2nd-5Team End-to-End MLOps Platform")
    
    st.markdown("""
    ---
    #### 📌 프로젝트 개요 및 워크플로우
    본 시스템은 에듀테크 유료 결제 고객(23,789명)의 초기 14일 학습 행동 데이터를 분석하여, 
    만료 시점의 이탈 위험을 조기에 진단하고 개인화된 맞춤형 리텐션(CRM) 액션을 제안하는 플랫폼입니다.
    
    좌측 사이드바에서 원하는 단계의 분석 및 모델링 페이지를 선택하세요.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **1단계: 데이터 분석 & 탐색**
        - **01. EDA**: 결제 고객 행동 패턴 심층 분석
        - **02. Churn Definition**: 환불 및 미갱신 이탈 기준 정의
        - **03. Feature Engineering**: 14일 관측 윈도우 피처셋
        """)
        
    with col2:
        st.success("""
        **2단계: 고객 군집화 & 모델 학습**
        - **04. User Clustering**: K-Means 행동 군집 프로파일링
        - **05. ML Training**: Decision Tree, RF, XGB, LGBM 학습
        - **06. DL Training**: Multi-Layer Perceptron (MLP) 학습
        """)
        
    with col3:
        st.warning("""
        **3단계: 모델 비교 & 설명 & CRM 액션**
        - **07. Model Comparison**: ROC/PR Curve 및 종합 랭킹
        - **08. SHAP Analysis**: 모델 예측 근거 XAI 해석
        - **09. Retention Action**: 위험 유형 진단 & LLM CRM 생성
        """)
        
    st.divider()
    st.caption("Developed by SK Networks AI Camp 35기 2차 프로젝트 5팀")

if __name__ == "__main__":
    main()
