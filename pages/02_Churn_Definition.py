"""Business churn definition and target-distribution page."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_feature_dataset
from src.utils.streamlit_ui import apply_korean_font_css, configure_matplotlib_korean


st.set_page_config(page_title="02. 이탈 정의", page_icon="🎯", layout="wide")


@st.cache_data(show_spinner="타깃 데이터를 불러오는 중입니다...")
def load_target_summary():
    dataframe = load_feature_dataset()
    required = {"is_churn", "is_refund_churn", "is_non_renewal_churn"}
    missing = required - set(dataframe.columns)
    if missing:
        raise ValueError(f"필수 타깃 열이 없습니다: {sorted(missing)}")
    return dataframe


def main() -> None:
    apply_korean_font_css()
    configure_matplotlib_korean()
    st.title("🎯 02. 고객 이탈 기준 및 타깃 정의")
    st.markdown(
        "최초 결제 이후 **환불**하거나 이용권 만료 후 **재결제하지 않은 고객**을 "
        "이탈 고객으로 정의합니다."
    )

    try:
        dataframe = load_target_summary()
    except Exception as exc:
        st.error(f"타깃 데이터를 불러오지 못했습니다: {exc}")
        st.stop()

    total = len(dataframe)
    refund = int(dataframe["is_refund_churn"].sum())
    non_renewal = int(dataframe["is_non_renewal_churn"].sum())
    churn = int(dataframe["is_churn"].sum())
    retained = total - churn
    overlap = int(
        ((dataframe["is_refund_churn"] == 1) &
         (dataframe["is_non_renewal_churn"] == 1)).sum()
    )

    columns = st.columns(4)
    columns[0].metric("전체 결제 고객", f"{total:,}명")
    columns[1].metric("환불 이탈", f"{refund:,}명", f"{refund / total:.1%}")
    columns[2].metric("미갱신 이탈", f"{non_renewal:,}명", f"{non_renewal / total:.1%}")
    columns[3].metric("유지 고객", f"{retained:,}명", f"{retained / total:.1%}")

    st.subheader("📌 라벨 판정 규칙")
    st.code(
        "is_churn = is_refund_churn OR is_non_renewal_churn\n\n"
        "환불 발생                         → is_refund_churn = 1\n"
        "결제 1회 · 환불 없음 · 재결제 없음 → is_non_renewal_churn = 1\n"
        "재결제/이용 유지                  → is_churn = 0",
        language="text",
    )
    if overlap == 0 and refund + non_renewal == churn:
        st.success("라벨 검증 통과: 두 이탈 유형은 중복되지 않으며 전체 이탈 라벨과 일치합니다.")
    else:
        st.warning(
            f"라벨을 확인해야 합니다. 유형 중복 {overlap:,}명, "
            f"유형 합계 {refund + non_renewal:,}명, 전체 이탈 {churn:,}명"
        )

    left, right = st.columns([1, 1])
    with left:
        st.subheader("📊 이탈·유지 클래스 분포")
        figure, axis = plt.subplots(figsize=(6, 4))
        labels = ["Retained", "Churned"]
        values = [retained, churn]
        bars = axis.bar(labels, values, color=["#55efc4", "#ff7675"])
        axis.set_ylabel("Users")
        axis.set_title("Target Class Distribution")
        axis.bar_label(bars, labels=[f"{value:,}" for value in values])
        figure.tight_layout()
        st.pyplot(figure, width="stretch")
        plt.close(figure)
    with right:
        st.subheader("⚖️ 불균형 데이터 대응")
        st.info(
            f"""
            - 이탈 비율: **{churn / total:.1%}**
            - 유지 비율: **{retained / total:.1%}**
            - 무작위 층화 분할로 Train/Test 비율 유지
            - Random Forest `class_weight="balanced"`
            - XGBoost `scale_pos_weight` 적용
            - Accuracy와 함께 ROC-AUC, PR-AUC, F1, Recall 평가
            """
        )


if __name__ == "__main__":
    main()
