"""Page 2: 연봉 협상 지원."""

import streamlit as st

from src.employee_dashboard import get_employees, render_salary
from streamlit_ui import apply_page_style, page_header, top_navigation

st.set_page_config(page_title="TalentGuard AI | 연봉 협상", layout="wide")
apply_page_style(); top_navigation("salary")
page_header("HR ADMIN · COMPENSATION", "연봉 협상 지원", "개별 직원 ID를 선택해 퇴사 위험과 인재 가치를 확인합니다.")
try:
    render_salary(get_employees())
except Exception as error:
    st.error(f"직원 데이터를 불러오지 못했습니다. DB 연결 설정을 확인해 주세요. ({error})")
