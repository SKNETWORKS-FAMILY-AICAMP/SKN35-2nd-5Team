from pathlib import Path

import pandas as pd
import streamlit as st


REPORT_PATH = Path("artifacts/reports/dl_metrics.csv")
METRIC_LABELS = {
    "accuracy": "정확도",
    "precision": "정밀도",
    "recall": "재현율",
    "f1": "F1",
    "roc_auc": "ROC-AUC",
    "average_precision": "PR-AUC",
    "train_seconds": "학습 시간(초)",
    "epochs": "학습 Epoch",
    "final_loss": "최종 Loss",
}


@st.cache_data
def load_dl_report(path: str, modified_time: float) -> pd.DataFrame:
    del modified_time
    return pd.read_csv(path)


st.title("3. 딥러닝 성능표")
st.caption("저장된 딥러닝 모델의 검증 성능과 학습 결과를 확인합니다.")

if not REPORT_PATH.exists():
    st.warning("DL 성능 리포트가 없습니다: artifacts/reports/dl_metrics.csv")
    st.stop()

report = load_dl_report(str(REPORT_PATH), REPORT_PATH.stat().st_mtime)
required_columns = {"model", "accuracy", "precision", "recall", "f1", "roc_auc"}
missing_columns = required_columns.difference(report.columns)
if missing_columns:
    st.error(f"DL 리포트에 필요한 컬럼이 없습니다: {', '.join(sorted(missing_columns))}")
    st.stop()

best = report.sort_values(["roc_auc", "f1"], ascending=False).iloc[0]
summary = st.columns(5)
summary[0].metric("모델", str(best["model"]).upper())
summary[1].metric("정확도", f"{best['accuracy']:.4f}")
summary[2].metric("ROC-AUC", f"{best['roc_auc']:.4f}")
summary[3].metric("F1", f"{best['f1']:.4f}")
summary[4].metric("재현율", f"{best['recall']:.4f}")

display_columns = ["model", *[column for column in METRIC_LABELS if column in report.columns]]
display = report[display_columns].rename(columns={"model": "모델", **METRIC_LABELS})
display["모델"] = display["모델"].str.upper()
formats = {
    "정확도": "{:.4f}",
    "정밀도": "{:.4f}",
    "재현율": "{:.4f}",
    "F1": "{:.4f}",
    "ROC-AUC": "{:.4f}",
    "PR-AUC": "{:.4f}",
    "학습 시간(초)": "{:.2f}",
    "최종 Loss": "{:.4f}",
}

st.subheader("딥러닝 모델 성능")
st.dataframe(
    display.style.format(
        {key: value for key, value in formats.items() if key in display.columns}
    ),
    width="stretch",
    hide_index=True,
)

chart_metrics = [
    column
    for column in ["정확도", "정밀도", "재현율", "F1", "ROC-AUC", "PR-AUC"]
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

st.warning(
    "현재 train_processed.csv에서는 Attrition 1이 Stayed(재직)를 뜻합니다. "
    "퇴사를 양성 클래스로 해석하려면 DL 재학습 전에 타깃 라벨을 통일해야 합니다."
)
