import streamlit as st

from src.employee_dashboard import AMBER, get_employees, page_intro, render_people_decision
from streamlit_ui import apply_page_style, page_header, top_navigation

st.set_page_config(page_title="인사 지원 · TalentGuard AI", layout="wide")
apply_page_style()
top_navigation("people")
page_header(
    "HR ADMIN · PEOPLE DECISION",
    "인사 지원",
    "동일 직무 그룹에서 승진·발령·검토 우선순위를 비교합니다.",
)
try:
    employees = get_employees()
except Exception as exc:
    st.error(f"직원 데이터 또는 예측 모델을 불러오지 못했습니다: {exc}")
    st.stop()
page_intro(
    4,
    "HR ADMIN",
    "승진·발령 우선순위",
    "인재 가치와 잔류 가능성을 함께 반영한 복합 점수로 검토 순위를 제공합니다.",
    AMBER,
)
render_people_decision(employees)
