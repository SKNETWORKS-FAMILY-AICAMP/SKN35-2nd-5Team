"""Page 5: 부서별 인사 구조 안정도."""

import streamlit as st

from src.employee_dashboard import get_employees, render_executive
from streamlit_ui import apply_page_style, page_header, top_navigation

st.set_page_config(page_title="TalentGuard AI | 구조 안정도", layout="wide")
apply_page_style(); top_navigation("stability")
page_header("HR ADMIN · ORGANIZATION STABILITY", "인사 구조 안정도", "직무별 위험 신호와 전사 인력 안정성을 모니터링합니다.")
try:
    render_executive(get_employees())
except Exception as error:
    st.error(f"직원 데이터를 불러오지 못했습니다. DB 연결 설정을 확인해 주세요. ({error})")
