"""Page 1: 시작 화면과 관리자 역할 선택."""

import streamlit as st

from streamlit_ui import apply_page_style, top_navigation

st.set_page_config(page_title="TalentGuard AI | 역할 선택", page_icon="HR", layout="wide")
apply_page_style()
top_navigation("start")
st.markdown(
    """
    <section class="start-hero"><div class="start-inner">
      <div class="start-badge">● AI-POWERED HR INTELLIGENCE PLATFORM</div>
      <h1>기업의 인재를<br><span>지켜드립니다.</span></h1>
      <p class="start-desc">핵심 기술 인력의 이탈 위험을 사전에 감지하고, 데이터 기반 의사결정으로<br>소중한 인재를 지키는 스마트 HR 모니터링 플랫폼입니다.</p>
      <div class="start-actions">
        <a class="role-card hr" href="/page02" target="_self"><small>HR ADMIN</small><strong>인사팀 담당자</strong><b>→</b></a>
        <a class="role-card dev" href="/page06" target="_self"><small>DEV ADMIN</small><strong>개발 관리자</strong><b>→</b></a>
      </div>
    </div></section>
    """,
    unsafe_allow_html=True,
)
