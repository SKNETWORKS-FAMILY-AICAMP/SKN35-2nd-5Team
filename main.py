import streamlit as st

from src.data.loader import load_raw_train
from streamlit_ui import apply_page_style, page_header

st.set_page_config(
    page_title="퇴사 예측 연구소",
    page_icon="🌿",
    layout="wide",
)

apply_page_style()
page_header(
    "직원 퇴사 예측 연구",
    "퇴사 예측 연구소 🌿",
    "직원 데이터를 차분히 살펴보고, 여러 모델의 결과와 개선 시나리오를 한곳에서 확인해요.",
)

try:
    data = load_raw_train()
    attrition_rate = data["Attrition"].eq("Left").mean()
    columns = st.columns(4)
    columns[0].metric("학습 직원", f"{len(data):,}명")
    columns[1].metric("입력 항목", f"{len(data.columns) - 2}개")
    columns[2].metric("퇴사 비율", f"{attrition_rate:.1%}")
    columns[3].metric("결측값", f"{int(data.isna().sum().sum()):,}개")
except Exception as exc:
    st.error(f"학습 데이터를 불러오지 못했어요: {exc}")

st.subheader("둘러보기")
st.markdown(
    """
    <div class="nav-grid">
        <a class="nav-card" href="/ML_Comparison" target="_self">
            <div class="nav-card-icon">🌳</div>
            <div class="nav-card-title">머신러닝 비교</div>
            <div class="nav-card-description">기본·튜닝 모델의 검증 성능을 나란히 봐요.</div>
        </a>
        <a class="nav-card" href="/DL_Performance" target="_self">
            <div class="nav-card-icon">🧠</div>
            <div class="nav-card-title">딥러닝 성능</div>
            <div class="nav-card-description">MLP 학습 결과가 저장되면 바로 표시해요.</div>
        </a>
        <a class="nav-card" href="/ML_vs_DL" target="_self">
            <div class="nav-card-icon">⚖️</div>
            <div class="nav-card-title">머신러닝과 딥러닝 비교</div>
            <div class="nav-card-description">두 계열의 최고 모델을 같은 지표로 비교해요.</div>
        </a>
        <a class="nav-card" href="/Final_Scenario_Test" target="_self">
            <div class="nav-card-icon">✨</div>
            <div class="nav-card-title">시나리오</div>
            <div class="nav-card-description">직원 조건을 바꿔 퇴사 확률 변화를 살펴봐요.</div>
        </a>
    </div>
    """,
    unsafe_allow_html=True,
)
