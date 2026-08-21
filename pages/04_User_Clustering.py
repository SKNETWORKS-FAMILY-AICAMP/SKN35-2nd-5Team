"""K-Means customer behavior clustering page."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.clustering.kmeans import get_cluster_summary, perform_kmeans_clustering
from src.data.loader import load_feature_dataset
from src.utils.streamlit_ui import apply_korean_font_css, configure_matplotlib_korean


st.set_page_config(page_title="04. 고객 군집 분석", page_icon="👥", layout="wide")

CLUSTER_FEATURES = [
    "obs_total_events",
    "obs_active_days",
    "obs_decay_ratio",
    "obs_recency_days",
    "obs_respond_count",
    "obs_play_video_count",
]


@st.cache_data(show_spinner="고객 행동 군집을 계산하는 중입니다...")
def run_clustering(cluster_count: int):
    dataframe = load_feature_dataset()
    clustered, model, _scaler = perform_kmeans_clustering(
        dataframe, feature_cols=CLUSTER_FEATURES, n_clusters=cluster_count
    )
    summary = get_cluster_summary(clustered)
    counts = clustered["cluster"].value_counts().sort_index()
    return clustered, summary, counts, float(model.inertia_)


def main() -> None:
    apply_korean_font_css()
    configure_matplotlib_korean()
    st.title("👥 04. 고객 행동 군집 분석")
    st.markdown(
        "이탈 라벨을 군집 입력에 사용하지 않고, 초기 14일 행동 피처만으로 "
        "K-Means 고객 세그먼트를 구성합니다."
    )
    cluster_count = st.slider("군집 개수(K)", min_value=2, max_value=6, value=4)

    try:
        clustered, summary, counts, inertia = run_clustering(cluster_count)
    except Exception as exc:
        st.error(f"고객 군집을 계산하지 못했습니다: {exc}")
        st.stop()

    columns = st.columns(3)
    columns[0].metric("분석 고객", f"{len(clustered):,}명")
    columns[1].metric("군집 수", f"{cluster_count}개")
    columns[2].metric("K-Means Inertia", f"{inertia:,.1f}")

    profile = summary.copy()
    profile.insert(0, "users", counts.reindex(profile.index).fillna(0).astype(int))
    profile["churn_rate"] = profile["is_churn"]
    profile = profile.drop(columns=["is_churn"])
    st.subheader("📋 군집별 행동 프로필")
    st.dataframe(profile.style.format("{:.3f}"), width="stretch")

    left, right = st.columns(2)
    with left:
        st.subheader("📉 군집별 이탈률")
        figure, axis = plt.subplots(figsize=(7, 4))
        bars = axis.bar(
            [f"Cluster {index}" for index in profile.index],
            profile["churn_rate"],
            color=plt.cm.Set2(range(len(profile))),
        )
        axis.set_ylim(0, 1)
        axis.set_ylabel("Churn Rate")
        axis.bar_label(bars, labels=[f"{value:.1%}" for value in profile["churn_rate"]])
        figure.tight_layout()
        st.pyplot(figure, width="stretch")
        plt.close(figure)
    with right:
        st.subheader("🗺️ 활성일·최근성 분포")
        figure, axis = plt.subplots(figsize=(7, 4))
        for cluster_id in sorted(clustered["cluster"].unique()):
            subset = clustered[clustered["cluster"] == cluster_id]
            axis.scatter(
                subset["obs_active_days"], subset["obs_recency_days"],
                s=8, alpha=0.2, label=f"Cluster {cluster_id}",
            )
        axis.set(xlabel="Active Days", ylabel="Recency Days")
        axis.legend(markerscale=2)
        figure.tight_layout()
        st.pyplot(figure, width="stretch")
        plt.close(figure)

    st.info("군집 번호는 우열이나 위험 순위가 아니라 행동 패턴이 유사한 고객 그룹을 뜻합니다.")


if __name__ == "__main__":
    main()
