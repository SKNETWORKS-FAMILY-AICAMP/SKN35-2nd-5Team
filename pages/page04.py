"""Page 4: 승진·발령 우선순위."""

import streamlit as st

from src.employee_dashboard import get_employees, render_people_decision
from streamlit_ui import apply_page_style, page_header, top_navigation

st.set_page_config(page_title="TalentGuard AI | 인사 지원", layout="wide")
apply_page_style(); top_navigation("people")
page_header("HR ADMIN · PEOPLE DECISION", "인사 지원", "동일 직무 그룹에서 승진·발령·검토 우선순위를 비교합니다.")
try:
    render_people_decision(get_employees())
except Exception as error:
    st.error(f"직원 데이터를 불러오지 못했습니다. DB 연결 설정을 확인해 주세요. ({error})")
