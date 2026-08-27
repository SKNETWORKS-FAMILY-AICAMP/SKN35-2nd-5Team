from pathlib import Path

import pandas as pd
import streamlit as st

from streamlit_ui import apply_page_style, home_button, page_header

st.set_page_config(page_title="머신러닝과 딥러닝 비교", layout="wide")

ML_REPORT_PATH = Path("artifacts/reports/ml_leaderboard.csv")
DL_REPORT_PATH = Path("artifacts/reports/dl_metrics.csv")
MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "mlp": "MLP",
}
METRICS = {
    "accuracy": "정확도",
    "precision": "정밀도",
    "recall": "재현율",
    "f1": "F1 점수",
    "roc_auc": "ROC 곡선 면적",
    "average_precision": "정밀도-재현율 곡선 면적",
}


@st.cache_data
def load_report(path: str, modified_time: float) -> pd.DataFrame:
    del modified_time
    return pd.read_csv(path)


apply_page_style()
home_button()
page_header(
    "최종 모델 비교",
    "최고 머신러닝과 딥러닝 비교",
    "각 계열의 가장 좋은 모델을 같은 평가 지표로 나란히 놓고 살펴봐요.",
)

missing_files = [str(path) for path in (ML_REPORT_PATH, DL_REPORT_PATH) if not path.exists()]
if missing_files:
    st.warning("비교에 필요한 리포트가 없습니다: " + ", ".join(missing_files))
    st.stop()

ml_report = load_report(str(ML_REPORT_PATH), ML_REPORT_PATH.stat().st_mtime)
dl_report = load_report(str(DL_REPORT_PATH), DL_REPORT_PATH.stat().st_mtime)

required = {"model", *METRICS}
if not required.issubset(ml_report.columns) or not required.issubset(dl_report.columns):
    st.error("머신러닝 또는 딥러닝 보고서에 비교 지표가 부족합니다.")
    st.stop()

best_ml = ml_report.sort_values(["roc_auc", "f1"], ascending=False).iloc[0]
best_dl = dl_report.sort_values(["roc_auc", "f1"], ascending=False).iloc[0]
ml_name = MODEL_LABELS.get(str(best_ml["model"]), str(best_ml["model"]))
dl_name = MODEL_LABELS.get(str(best_dl["model"]), str(best_dl["model"]).upper())

roc_delta = float(best_ml["roc_auc"] - best_dl["roc_auc"])
winner = ml_name if roc_delta >= 0 else dl_name
with st.container(key="stat-bar"):
    summary = st.columns(3)
    summary[0].metric("머신러닝 1위", ml_name, f"ROC 곡선 면적 {best_ml['roc_auc']:.4f}")
    summary[1].metric("딥러닝", dl_name, f"ROC 곡선 면적 {best_dl['roc_auc']:.4f}")
    summary[2].metric("ROC 곡선 면적 우세 모델", winner, f"차이 {abs(roc_delta):.4f}")

comparison = pd.DataFrame(
    [
        {
            "구분": "최고 머신러닝",
            "모델": ml_name,
            **{label: best_ml[key] for key, label in METRICS.items()},
        },
        {
            "구분": "딥러닝",
            "모델": dl_name,
            **{label: best_dl[key] for key, label in METRICS.items()},
        },
    ]
)

st.subheader("최종 성능 비교표")
st.dataframe(
    comparison.style.format({label: "{:.4f}" for label in METRICS.values()}).highlight_max(
        subset=list(METRICS.values()), color="#E8F3FF"
    ),
    width="stretch",
    hide_index=True,
)

st.subheader("지표별 비교")
st.bar_chart(comparison.set_index("모델")[list(METRICS.values())])

delta_table = pd.DataFrame(
    {
        "지표": list(METRICS.values()),
        ml_name: [best_ml[key] for key in METRICS],
        dl_name: [best_dl[key] for key in METRICS],
        "머신러닝 - 딥러닝": [best_ml[key] - best_dl[key] for key in METRICS],
    }
)
with st.expander("지표별 차이 보기"):
    st.dataframe(
        delta_table.style.format(
            {ml_name: "{:.4f}", dl_name: "{:.4f}", "머신러닝 - 딥러닝": "{:+.4f}"}
        ),
        width="stretch",
        hide_index=True,
    )

st.info("두 모델 모두 퇴사=1을 양성 클래스로 평가한 보고서일 때 가장 정확한 비교가 됩니다.")
