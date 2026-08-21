"""Observation-window feature catalog and target-correlation page."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.eda import compute_correlations
from src.data.loader import load_feature_dataset
from src.utils.constants import LEAKAGE_AND_ID_COLUMNS, TARGET_COLUMN
from src.utils.streamlit_ui import apply_korean_font_css, configure_matplotlib_korean


st.set_page_config(page_title="03. 피처 및 상관관계", page_icon="⚙️", layout="wide")


@st.cache_data(show_spinner="피처 데이터를 불러오는 중입니다...")
def load_features():
    return load_feature_dataset()


def main() -> None:
    apply_korean_font_css()
    configure_matplotlib_korean()
    st.title("⚙️ 03. 피처 생성 및 상관관계")
    st.markdown(
        "최초 결제 시점부터 **14일 관측 윈도우**에서 활동량, 최근성, "
        "콘텐츠 이용 및 주차별 변화 피처를 생성합니다."
    )

    try:
        dataframe = load_features()
    except Exception as exc:
        st.error(f"피처 데이터를 불러오지 못했습니다: {exc}")
        st.stop()

    model_columns = [
        column for column in dataframe.columns
        if column not in set(LEAKAGE_AND_ID_COLUMNS)
    ]
    numeric_model_columns = dataframe[model_columns].select_dtypes("number").columns.tolist()
    columns = st.columns(4)
    columns[0].metric("고객", f"{len(dataframe):,}명")
    columns[1].metric("전체 컬럼", f"{dataframe.shape[1]}개")
    columns[2].metric("학습 입력 피처", f"{len(numeric_model_columns)}개")
    columns[3].metric("결측값", f"{int(dataframe.isna().sum().sum()):,}개")

    st.subheader("🧱 피처 구성")
    feature_catalog = pd.DataFrame(
        [
            ("결제 전 행동", "pre_pay_*", "결제 전 이벤트·활성일·활동 기간·최근성"),
            ("14일 활동량", "obs_total_events, obs_active_days", "전체 활동과 접속 빈도"),
            ("주차별 변화", "obs_w1_events, obs_w2_events", "1주차 대비 2주차 활동 변화"),
            ("최근성", "obs_last_active_day, obs_recency_days", "마지막 활동과 미접속 기간"),
            ("학습 행동", "obs_respond_*, obs_submit_*", "응답·제출 및 답변 기록 비율"),
            ("콘텐츠 행동", "obs_play_*, obs_pause_*", "영상·오디오 재생과 일시정지"),
            ("타깃", TARGET_COLUMN, "환불 또는 비갱신 이탈 여부"),
        ],
        columns=["영역", "대표 컬럼", "설명"],
    )
    st.dataframe(feature_catalog, width="stretch", hide_index=True)

    with st.expander("피처 데이터 미리보기"):
        st.dataframe(dataframe.head(20), width="stretch")

    correlations = compute_correlations(dataframe)
    if correlations.empty:
        st.warning("이탈 타깃과의 상관계수를 계산할 수 없습니다.")
        return
    ranked = correlations.reindex(correlations.abs().sort_values(ascending=False).index)
    top = ranked.head(15).sort_values()

    st.divider()
    st.subheader("📈 이탈 타깃과의 상관관계")
    left, right = st.columns([1, 1.4])
    with left:
        correlation_table = ranked.rename("Correlation").to_frame()
        correlation_table["Abs. Correlation"] = correlation_table["Correlation"].abs()
        st.dataframe(correlation_table.head(15).style.format("{:.4f}"), width="stretch")
    with right:
        figure, axis = plt.subplots(figsize=(8, 7))
        colors = ["#0984e3" if value < 0 else "#d63031" for value in top.values]
        axis.barh(top.index, top.values, color=colors)
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set(xlabel="Correlation", title="Top 15 Correlations with Churn")
        figure.tight_layout()
        st.pyplot(figure, width="stretch")
        plt.close(figure)

    st.caption("상관관계는 인과관계를 의미하지 않으며, 모델 피처 중요도와 함께 해석해야 합니다.")


if __name__ == "__main__":
    main()
