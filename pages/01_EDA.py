"""Exploratory data analysis page."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.eda import compare_churn_groups, compute_dataset_overview
from src.data.loader import load_feature_dataset
from src.utils.streamlit_ui import apply_korean_font_css, configure_matplotlib_korean


st.set_page_config(page_title="01. EDA", page_icon="📊", layout="wide")


@st.cache_data(show_spinner="피처 데이터를 불러오는 중입니다...")
def load_data():
    return load_feature_dataset()


def draw_group_boxplot(data, feature: str, title: str):
    figure, axis = plt.subplots(figsize=(6, 4))
    retained = data.loc[data["is_churn"] == 0, feature].dropna()
    churned = data.loc[data["is_churn"] == 1, feature].dropna()
    boxplot = axis.boxplot(
        [retained, churned],
        tick_labels=["Retained (0)", "Churned (1)"],
        showfliers=False,
        patch_artist=True,
    )
    for patch, color in zip(boxplot["boxes"], ["#55efc4", "#ff7675"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    axis.set_title(title)
    axis.set_ylabel(feature)
    figure.tight_layout()
    return figure


def main() -> None:
    apply_korean_font_css()
    configure_matplotlib_korean()
    st.title("📊 01. 탐색적 데이터 분석 (EDA)")
    st.markdown("결제 고객 23,789명의 기본 통계와 이탈·유지 집단의 행동 차이를 탐색합니다.")

    try:
        dataframe = load_data()
    except Exception as exc:
        st.error(f"피처 데이터셋을 불러오지 못했습니다: {exc}")
        st.stop()

    overview = compute_dataset_overview(dataframe)
    columns = st.columns(4)
    columns[0].metric("총 결제 고객", f"{overview['total_users']:,}명")
    columns[1].metric(
        "이탈 고객", f"{overview['churn_users']:,}명",
        f"{overview['churn_rate']:.1f}%", delta_color="inverse",
    )
    columns[2].metric(
        "유지 고객", f"{overview['retained_users']:,}명",
        f"{overview['retained_rate']:.1f}%",
    )
    columns[3].metric("데이터 컬럼", f"{overview['num_features']}개")

    st.divider()
    st.subheader("🔍 유지 고객 vs 이탈 고객의 초기 14일 학습 패턴")
    st.dataframe(compare_churn_groups(dataframe).round(3), width="stretch")

    left, right = st.columns(2)
    with left:
        st.markdown("##### 2주차 활동량 분포")
        figure = draw_group_boxplot(
            dataframe, "obs_w2_events", "Week 2 Activity by Churn Status"
        )
        st.pyplot(figure, width="stretch")
        plt.close(figure)
    with right:
        st.markdown("##### 최근 미접속 일수 분포")
        figure = draw_group_boxplot(
            dataframe, "obs_recency_days", "Recency Days by Churn Status"
        )
        st.pyplot(figure, width="stretch")
        plt.close(figure)


if __name__ == "__main__":
    main()
