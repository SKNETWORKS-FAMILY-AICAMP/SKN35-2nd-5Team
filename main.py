"""앱 진입 시 Page 1 관리자 역할 선택 화면으로 이동한다."""

import streamlit as st

<<<<<<< Updated upstream
from src.analysis.eda import build_eda_report
from src.load_data.loader import load_train_data

st.set_page_config(
    page_title="Employee Attrition Lab",
    page_icon="📊",
    layout="wide",
)

st.title("Employee Attrition Modeling Lab")
st.caption("직원 이탈 데이터 분석부터 모델 비교와 리텐션 액션까지 연결하는 프로젝트")

try:
    data = load_train_data()
    report = build_eda_report(data)
    columns = st.columns(4)
    columns[0].metric("학습 데이터", f"{report['rows']:,}행")
    columns[1].metric("피처 수", f"{report['columns'] - 2}개")
    columns[2].metric("이탈률", f"{report['attrition_rate']:.1%}")
    columns[3].metric("결측치", f"{report['missing_values']:,}개")
except (FileNotFoundError, ValueError) as exc:
    st.error(str(exc))

st.subheader("워크플로")
st.markdown(
    """
1. **EDA** — 데이터 품질, 분포, 그룹별 이탈률 확인
2. **ML Training** — Logistic Regression, Random Forest, XGBoost, LightGBM 비교
3. **Model Comparison** — ROC/PR 곡선과 리더보드 확인
4. **SHAP Analysis** — 최고 모델의 중요 피처 해석
5. **Retention Action** — 고위험 직원 후보와 검토 액션 제안
"""
)
st.info("왼쪽 사이드바에서 분석 단계를 선택하세요. 학습 결과는 artifacts/에 저장됩니다.")
=======
st.set_page_config(page_title="TalentGuard AI", layout="wide")
st.switch_page("pages/page01.py")
>>>>>>> Stashed changes
