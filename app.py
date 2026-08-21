"""Main Streamlit entry point for the ML-focused churn dashboard."""

import streamlit as st

from src.utils.streamlit_ui import apply_korean_font_css


st.set_page_config(
    page_title="고객 이탈 예측 ML 대시보드",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    apply_korean_font_css()
    st.title("🎯 가입 고객 이탈 예측 ML 대시보드")
    st.caption("SKN35-2nd-5Team · Machine Learning Scope")

    st.markdown(
        """
        결제 고객 **23,789명**의 최초 결제 후 14일 행동 피처를 바탕으로
        이탈 패턴을 탐색하고, 4개 머신러닝 모델의 성능을 비교합니다.

        현재는 검증이 끝난 **EDA, 이탈 정의, 피처 분석, 고객 군집화,
        ML 모델 성능 비교**를 제공합니다. 왼쪽 사이드바에서 `01`~`05`
        페이지를 선택하세요.
        """
    )

    eda_column, ml_column, roadmap_column = st.columns(3)
    with eda_column:
        st.success(
            """
            **✅ 01~02. 데이터·타깃 이해**

            - 탐색적 데이터 분석
            - 이탈/유지 고객 분포
            - 환불·미갱신 타깃 정의
            """
        )
    with ml_column:
        st.success(
            """
            **✅ 03~04. 피처·고객 군집**

            - 14일 관측 피처와 상관관계
            - K-Means 고객 행동 군집
            - 군집별 이탈률 프로파일
            """
        )
    with roadmap_column:
        st.info(
            """
            **✅ 05. ML 성능 비교**

            Decision Tree, Random Forest, XGBoost, LightGBM의 성능과
            ROC/PR Curve를 비교합니다.
            """
        )

    st.info("🚧 06 DL · 07 전체 모델 종합 비교 · 08 SHAP · 09 리텐션 액션은 준비 중입니다.")

    st.divider()
    st.caption("Developed by SK Networks AI Camp 35기 2차 프로젝트 5팀")


if __name__ == "__main__":
    main()
