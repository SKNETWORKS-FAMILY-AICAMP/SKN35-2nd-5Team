from pathlib import Path

import pandas as pd
import streamlit as st

from streamlit_ui import apply_page_style, home_button, page_header

st.set_page_config(page_title="딥러닝 성능", layout="wide")

REPORT_PATH = Path("artifacts/reports/mlp_test_metrics.csv")
METRIC_LABELS = {
    "accuracy": "정확도",
    "precision": "정밀도",
    "recall": "재현율",
    "f1": "F1 점수",
    "roc_auc": "ROC 곡선 면적",
    "average_precision": "정밀도-재현율 곡선 면적",
    "epochs": "학습 반복 횟수",
    "final_loss": "최종 손실값",
}


@st.cache_data
def load_dl_report(path: str, modified_time: float) -> pd.DataFrame:
    del modified_time
    return pd.read_csv(path)


apply_page_style()
home_button()
page_header(
    "딥러닝 모델 분석",
    "딥러닝 성능",
    "저장된 MLP의 성능과 학습 기록을 편하게 확인해요.",
)

if not REPORT_PATH.exists():
    st.info(
        "아직 `artifacts/reports/mlp_test_metrics.csv`가 없어요. "
        "MLP 학습이 끝난 뒤 성능 리포트를 저장하면 이 화면에 자동으로 나타납니다."
    )
    st.stop()

report = load_dl_report(str(REPORT_PATH), REPORT_PATH.stat().st_mtime)
required_columns = {"model", "accuracy", "precision", "recall", "f1", "roc_auc"}
missing_columns = required_columns.difference(report.columns)
if missing_columns:
    st.error(f"딥러닝 보고서에 필요한 항목이 없습니다: {', '.join(sorted(missing_columns))}")
    st.stop()

best = report.sort_values(["roc_auc", "f1"], ascending=False).iloc[0]
with st.container(key="stat-bar"):
    summary = st.columns(5)
    summary[0].metric("모델", str(best["model"]).upper())
    summary[1].metric("정확도", f"{best['accuracy']:.4f}")
    summary[2].metric("ROC 곡선 면적", f"{best['roc_auc']:.4f}")
    summary[3].metric("F1 점수", f"{best['f1']:.4f}")
    summary[4].metric("재현율", f"{best['recall']:.4f}")

display_columns = ["model", *[column for column in METRIC_LABELS if column in report.columns]]
display = report[display_columns].rename(columns={"model": "모델", **METRIC_LABELS})
display["모델"] = display["모델"].str.upper()
formats = {
    "정확도": "{:.4f}",
    "정밀도": "{:.4f}",
    "재현율": "{:.4f}",
    "F1 점수": "{:.4f}",
    "ROC 곡선 면적": "{:.4f}",
    "정밀도-재현율 곡선 면적": "{:.4f}",
    "최종 손실값": "{:.4f}",
}

st.subheader("딥러닝 모델 성능")
st.dataframe(
    display.style.format({key: value for key, value in formats.items() if key in display.columns}),
    width="stretch",
    hide_index=True,
)

chart_metrics = [
    column
    for column in [
        "정확도",
        "정밀도",
        "재현율",
        "F1 점수",
        "ROC 곡선 면적",
        "정밀도-재현율 곡선 면적",
    ]
    if column in display.columns
]
st.subheader("평가 지표")
st.bar_chart(display.set_index("모델")[chart_metrics].T)

if {"tn", "fp", "fn", "tp"}.issubset(report.columns):
    st.subheader("혼동행렬 수치")
    confusion = pd.DataFrame(
        [[int(best["tn"]), int(best["fp"])], [int(best["fn"]), int(best["tp"])]],
        index=["실제 0", "실제 1"],
        columns=["예측 0", "예측 1"],
    )
    st.dataframe(confusion, width="stretch")

st.info("이 프로젝트의 공통 예측값은 퇴사=1, 재직=0이에요.")
