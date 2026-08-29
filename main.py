"""접속 유형(인사팀 / 기술개발팀·관리자)을 고르는 랜딩 페이지.

시간 관계상 실제 로그인 인증은 만들지 않고, 이 프로젝트의 핵심 기능인 "모델 학습과
퇴사율 예측"에 집중하기 위해 접속 유형만 나누는 간단한 분기 화면으로 대신한다.
"""

import streamlit as st
import streamlit.components.v1 as components

from streamlit_ui import apply_page_style

st.set_page_config(page_title="퇴사 예측 연구소 · STAYON", layout="wide")
apply_page_style()

st.session_state.setdefault("role", None)
st.session_state.setdefault("workspace_tab", "salary")

st.markdown(
    """
    <div class="hero-wrap">
        <span class="hero-eyebrow">EMPLOYEE ATTRITION INTELLIGENCE</span>
        <h1 class="hero-title">직원 퇴사 예측 연구소</h1>
        <p class="hero-desc">
            머신러닝·딥러닝으로 학습한 퇴사 예측 모델을 바탕으로, 연봉 협상부터 팀 구성,
            인사발령, 전사 인사 구조 안정도까지 한곳에서 확인할 수 있어요.
            먼저 접속 유형을 선택해 주세요.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns(2, gap="large")

with left:
    with st.container(key="role-panel-hr"):
        st.markdown(
            """
            <div class="role-card">
                <div class="role-card-icon">🧑‍💼</div>
                <div class="role-card-title">인사팀으로 접속</div>
                <div class="role-card-desc">
                    채용·보상·조직 운영 등 사람과 관련된 실무 의사결정을 지원하는 화면이에요.
                </div>
                <ul class="role-card-points">
                    <li>연봉 협상 · 인재 가치 대비 퇴사 위험 확인</li>
                    <li>프로젝트 팀 구성 · 팀 안정도 진단</li>
                    <li>인사발령 · 승진 · 구조조정 후보 추천</li>
                    <li>전사 인사 구조 안정도 모니터링</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("인사팀으로 접속", key="role_hr_btn", type="primary", width="stretch"):
            st.session_state["role"] = "hr"
            st.session_state["workspace_tab"] = "salary"
            st.switch_page("pages/01_Workspace.py")

with right:
    with st.container(key="role-panel-admin"):
        st.markdown(
            """
            <div class="role-card">
                <div class="role-card-icon">🛠️</div>
                <div class="role-card-title">기술개발팀(관리자)로 접속</div>
                <div class="role-card-desc">
                    인사팀 화면을 모두 확인하면서, 모델을 만든 입장에서 학습 결과와
                    성능 근거까지 함께 점검하는 화면이에요.
                </div>
                <ul class="role-card-points">
                    <li>인사팀 화면 전체 열람 가능</li>
                    <li>ML 4종 · DL(MLP) 성능 비교</li>
                    <li>최종 모델 선정 근거 · 튜닝 하이퍼파라미터</li>
                    <li>퇴사율 측정에 사용한 학습 기준 확인</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("기술개발팀(관리자)로 접속", key="role_admin_btn", width="stretch"):
            st.session_state["role"] = "admin"
            st.session_state["workspace_tab"] = "salary"
            st.switch_page("pages/01_Workspace.py")

# 두 카드 높이 맞추기: Streamlit이 컬럼 내부를 여러 겹의 flex/block 래퍼로 감싸는
# 방식 때문에, 순수 CSS(flex-grow, align-items: stretch)만으로는 "인사팀으로 접속"
# 카드와 "기술개발팀(관리자)로 접속" 카드처럼 내용 길이가 다른 두 카드의 높이를
# 안정적으로 맞출 수 없었다. 대신 렌더 후 실제 카드 높이를 JS로 측정해서 더 큰 쪽에
# 맞춰 min-height를 직접 지정한다. 화면 폭이 바뀌어 줄바꿈 수가 달라져도 resize 때마다
# 다시 계산하므로 항상 두 "접속" 버튼이 같은 줄에 나란히 놓인다.
components.html(
    """
    <script>
    function equalizeRoleCards() {
        const doc = window.parent.document;
        const cards = doc.querySelectorAll('.role-card');
        if (cards.length < 2) return;
        cards.forEach((c) => { c.style.minHeight = '0px'; });
        let maxHeight = 0;
        cards.forEach((c) => { maxHeight = Math.max(maxHeight, c.getBoundingClientRect().height); });
        cards.forEach((c) => { c.style.minHeight = maxHeight + 'px'; });
    }
    equalizeRoleCards();
    setTimeout(equalizeRoleCards, 150);
    setTimeout(equalizeRoleCards, 500);
    if (!window.parent.__roleCardResizeBound) {
        window.parent.__roleCardResizeBound = true;
        window.parent.addEventListener('resize', () => setTimeout(equalizeRoleCards, 80));
    }
    </script>
    """,
    height=0,
)

st.markdown(
    '<div class="section-divider-thin"></div>'
    '<p class="muted" style="text-align:center; font-size:.82rem;">'
    "실제 로그인 없이 화면 구성만 나눈 데모 진입점이에요. "
    "핵심 기능인 모델 학습·퇴사율 예측은 두 접속 유형 모두 동일한 데이터를 사용합니다."
    "</p>",
    unsafe_allow_html=True,
)
