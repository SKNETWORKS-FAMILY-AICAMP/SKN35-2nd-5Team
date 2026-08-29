# ruff: noqa: E501

import streamlit as st

from streamlit_ui import apply_page_style, top_navigation

st.set_page_config(page_title="관리자 역할 선택 · TalentGuard AI", layout="wide")
apply_page_style()
top_navigation("start")

st.markdown(
    """
    <section class="start-hero">
        <div class="start-hero-inner">
            <div class="start-badge">AI-POWERED HR INTELLIGENCE PLATFORM</div>
            <h1>기업의 인재를<br><span class="start-gradient">지켜드립니다.</span></h1>
            <div class="start-description">핵심 기술 인력의 이탈 위험을 사전에 감지하고, 데이터 기반 의사결정으로<br>소중한 인재를 지키는 스마트 HR 모니터링 플랫폼입니다.</div>
            <div class="start-actions">
                <a class="start-action" style="--accent:#22D3EE" href="/page02" target="_self">
                    <div class="start-action-code">HR ADMIN</div>
                    <div class="start-action-title"><span>인사팀 담당자</span><span>→</span></div>
                </a>
                <a class="start-action" style="--accent:#A78BFA" href="/page06" target="_self">
                    <div class="start-action-code">DEV ADMIN</div>
                    <div class="start-action-title"><span>개발 관리자</span><span>→</span></div>
                </a>
            </div>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)
