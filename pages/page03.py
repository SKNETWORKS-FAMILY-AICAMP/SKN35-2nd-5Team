import streamlit as st

from src.employee_dashboard import VIOLET, get_employees, page_intro, render_team
from streamlit_ui import apply_page_style, page_header, top_navigation

st.set_page_config(page_title="프로젝트 팀 구성 지원 · TalentGuard AI", layout="wide")
apply_page_style()
top_navigation("team")
page_header(
    "HR ADMIN · TEAM PLANNING",
    "프로젝트별 팀 구성 지원",
    "팀원을 변경하며 프로젝트 팀의 안정도와 인재 구성을 비교합니다.",
)
try:
    employees = get_employees()
except Exception as exc:
    st.error(f"직원 데이터 또는 예측 모델을 불러오지 못했습니다: {exc}")
    st.stop()
page_intro(
    3,
    "HR ADMIN",
    "팀원 변경 시뮬레이션",
    "후보 직무와 팀원을 바꾸면서 평균 퇴사 위험과 팀 적합도를 확인합니다.",
    VIOLET,
)
render_team(employees)
