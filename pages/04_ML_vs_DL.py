from pathlib import Path

import pandas as pd
import streamlit as st


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
    "f1": "F1",
    "roc_auc": "ROC-AUC",
    "average_precision": "PR-AUC",
}


@st.cache_data
def load_report(path: str, modified_time: float) -> pd.DataFrame:
    del modified_time
    return pd.read_csv(path)


st.title("4. 최고 ML vs 딥러닝")
st.caption("머신러닝 1위 모델과 딥러닝 모델을 동일한 핵심 지표로 나란히 비교합니다.")

missing_files = [
    str(path) for path in (ML_REPORT_PATH, DL_REPORT_PATH) if not path.exists()
]
if missing_files:
    st.warning("비교에 필요한 리포트가 없습니다: " + ", ".join(missing_files))
    st.stop()

ml_report = load_report(str(ML_REPORT_PATH), ML_REPORT_PATH.stat().st_mtime)
dl_report = load_report(str(DL_REPORT_PATH), DL_REPORT_PATH.stat().st_mtime)

required = {"model", *METRICS}
if not required.issubset(ml_report.columns) or not required.issubset(dl_report.columns):
    st.error("ML 또는 DL 리포트에 비교 지표가 부족합니다.")
    st.stop()

best_ml = ml_report.sort_values(["roc_auc", "f1"], ascending=False).iloc[0]
best_dl = dl_report.sort_values(["roc_auc", "f1"], ascending=False).iloc[0]
ml_name = MODEL_LABELS.get(str(best_ml["model"]), str(best_ml["model"]))
dl_name = MODEL_LABELS.get(str(best_dl["model"]), str(best_dl["model"]).upper())

summary = st.columns(3)
summary[0].metric("ML 1위", ml_name, f"ROC-AUC {best_ml['roc_auc']:.4f}")
summary[1].metric("딥러닝", dl_name, f"ROC-AUC {best_dl['roc_auc']:.4f}")
roc_delta = float(best_ml["roc_auc"] - best_dl["roc_auc"])
winner = ml_name if roc_delta >= 0 else dl_name
summary[2].metric("ROC-AUC 우세 모델", winner, f"차이 {abs(roc_delta):.4f}")

comparison = pd.DataFrame(
    [
        {
            "구분": "최고 ML",
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
        subset=list(METRICS.values()), color="#d1fae5"
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
        "ML - DL": [best_ml[key] - best_dl[key] for key in METRICS],
    }
)
with st.expander("지표별 차이 보기"):
    st.dataframe(
        delta_table.style.format(
            {ml_name: "{:.4f}", dl_name: "{:.4f}", "ML - DL": "{:+.4f}"}
        ),
        width="stretch",
        hide_index=True,
    )

st.warning(
    "현재 ML 리포트는 퇴사(Left)를 양성 클래스로, DL 전처리 데이터는 재직(Stayed)을 "
    "1로 사용합니다. 정확도와 ROC-AUC는 참고할 수 있지만 정밀도·재현율·F1을 최종 비교하기 "
    "전에는 타깃 정의를 통일해 다시 평가해야 합니다."
)
