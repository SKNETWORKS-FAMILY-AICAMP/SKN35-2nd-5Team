"""Page 3: 프로젝트별 팀 구성 지원."""

import streamlit as st

from src.employee_dashboard import get_employees, render_team
from streamlit_ui import apply_page_style, page_header, top_navigation

st.set_page_config(page_title="TalentGuard AI | 팀 구성", layout="wide")
apply_page_style(); top_navigation("team")
page_header("HR ADMIN · TEAM PLANNING", "프로젝트별 팀 구성 지원", "팀원을 변경하며 프로젝트 팀의 안정도와 인재 구성을 비교합니다.")
try:
    render_team(get_employees())
except Exception as error:
    st.error(f"직원 데이터를 불러오지 못했습니다. DB 연결 설정을 확인해 주세요. ({error})")
