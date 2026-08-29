"""Page 1: 시작 화면과 관리자 역할 선택."""

import streamlit as st

from streamlit_ui import apply_page_style, top_navigation

st.set_page_config(page_title="TalentGuard AI | 역할 선택", page_icon="HR", layout="wide")
apply_page_style()
top_navigation("start")
st.markdown(
    """
    <section class="start-hero"><div class="start-inner">
      <div class="start-badge">● TalentShield · 인트라넷 시스템 v2.4</div>
      <h1>기업의 인재를<br><span>지켜드립니다.</span></h1>
      <p class="start-desc">핵심 기술 인재의 이탈 신호를 사전에 감지하고,<br>데이터 기반의 선제적 인사 전략을 제공합니다.</p>
      <div class="start-actions">
        <a class="role-card hr" href="/page02" target="_self"><small>HR MANAGER</small><strong>인사팀 관리자</strong><b>›</b><p>연봉협상 · 팀구성 · 인사지원 · 안정도 모니터링</p></a>
        <a class="role-card dev" href="/page06" target="_self"><small>DEV MANAGER</small><strong>개발관리 관리자</strong><b>›</b><p>ML/DL 모델 성능 평가 · 예측 모델 비교 분석</p></a>
      </div>
      <div class="start-stats"><div><strong>1,247명</strong><small>모니터링 인원</small></div><div><strong>88.2%</strong><small>위험 감지 정확도</small></div><div><strong>71.4%</strong><small>사전 예방 성공률</small></div></div>
    </div></section>
    """,
    unsafe_allow_html=True,
)
