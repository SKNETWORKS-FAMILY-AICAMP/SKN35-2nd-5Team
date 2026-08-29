import streamlit as st

from src.employee_dashboard import CYAN, get_employees, page_intro, render_salary
from streamlit_ui import apply_page_style, page_header, top_navigation

st.set_page_config(page_title="연봉 협상 지원 · TalentGuard AI", layout="wide")
apply_page_style()
top_navigation("salary")
page_header(
    "HR ADMIN · COMPENSATION",
    "연봉 협상 지원",
    "개별 직원 ID를 선택해 퇴사 위험과 인재 가치를 확인합니다.",
)
try:
    employees = get_employees()
except Exception as exc:
    st.error(f"직원 데이터 또는 예측 모델을 불러오지 못했습니다: {exc}")
    st.stop()
page_intro(
    2,
    "HR ADMIN",
    "개별 ID 시뮬레이션",
    "LightGBM이 계산한 퇴사 확률과 사내 인재 가치 지수를 협상 판단에 활용합니다.",
    CYAN,
)
render_salary(employees)
