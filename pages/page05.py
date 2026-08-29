import streamlit as st

from src.employee_dashboard import EMERALD, get_employees, page_intro, render_executive
from streamlit_ui import apply_page_style, page_header, top_navigation

st.set_page_config(page_title="인사 구조 안정도 · TalentGuard AI", layout="wide")
apply_page_style()
top_navigation("stability")
page_header(
    "HR ADMIN · ORGANIZATION STABILITY",
    "인사 구조 안정도",
    "직무별 위험 신호와 전사 인력 안정성을 모니터링합니다.",
)
try:
    employees = get_employees()
except Exception as exc:
    st.error(f"직원 데이터 또는 예측 모델을 불러오지 못했습니다: {exc}")
    st.stop()
page_intro(
    5,
    "HR ADMIN",
    "부서별 모니터링",
    "고위험 인원과 직무별 평균 퇴사 위험을 확인해 선제 대응 대상을 찾습니다.",
    EMERALD,
)
render_executive(employees)
