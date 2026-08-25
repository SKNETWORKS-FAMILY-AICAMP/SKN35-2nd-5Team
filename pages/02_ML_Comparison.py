from pathlib import Path

import pandas as pd
import streamlit as st


REPORT_PATH = Path("artifacts/reports/ml_leaderboard.csv")
MODEL_ORDER = ["logistic_regression", "random_forest", "xgboost", "lightgbm"]
MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
}
METRIC_LABELS = {
    "accuracy": "정확도",
    "precision": "정밀도",
    "recall": "재현율",
    "f1": "F1",
    "roc_auc": "ROC-AUC",
    "average_precision": "PR-AUC",
    "train_seconds": "학습 시간(초)",
}


@st.cache_data
def load_ml_report(path: str, modified_time: float) -> pd.DataFrame:
    del modified_time
    return pd.read_csv(path)


st.title("2. 머신러닝 4종 비교")
st.caption("동일한 검증 데이터에서 네 가지 머신러닝 모델의 성능을 비교합니다.")

if not REPORT_PATH.exists():
    st.warning("ML 성능 리포트가 없습니다: artifacts/reports/ml_leaderboard.csv")
    st.stop()

leaderboard = load_ml_report(str(REPORT_PATH), REPORT_PATH.stat().st_mtime)
required_columns = {"model", *METRIC_LABELS}
missing_columns = required_columns.difference(leaderboard.columns)
if missing_columns:
    st.error(f"ML 리포트에 필요한 컬럼이 없습니다: {', '.join(sorted(missing_columns))}")
    st.stop()

leaderboard = leaderboard[leaderboard["model"].isin(MODEL_ORDER)].copy()
if leaderboard.empty:
    st.warning("비교할 ML 모델 결과가 없습니다.")
    st.stop()

leaderboard["모델"] = leaderboard["model"].map(MODEL_LABELS)
leaderboard = leaderboard.sort_values(
    ["roc_auc", "f1"], ascending=False, ignore_index=True
)

best = leaderboard.iloc[0]
summary = st.columns(4)
summary[0].metric("최고 모델", best["모델"])
summary[1].metric("최고 ROC-AUC", f"{best['roc_auc']:.4f}")
summary[2].metric("F1", f"{best['f1']:.4f}")
summary[3].metric("학습 시간", f"{best['train_seconds']:.2f}초")

display = leaderboard[["모델", *METRIC_LABELS]].rename(columns=METRIC_LABELS)
metric_columns = ["정확도", "정밀도", "재현율", "F1", "ROC-AUC", "PR-AUC"]

st.subheader("모델별 성능표")
st.dataframe(
    display.style.format(
        {column: "{:.4f}" for column in metric_columns}
        | {"학습 시간(초)": "{:.2f}"}
    ).highlight_max(subset=metric_columns, color="#d1fae5"),
    width="stretch",
    hide_index=True,
)

st.subheader("핵심 지표 비교")
chart_data = display.set_index("모델")[["ROC-AUC", "PR-AUC", "F1", "정확도"]]
st.bar_chart(chart_data)

with st.expander("혼동행렬 수치 보기"):
    confusion_columns = ["model", "tn", "fp", "fn", "tp"]
    if set(confusion_columns).issubset(leaderboard.columns):
        confusion = leaderboard[confusion_columns].copy()
        confusion["model"] = confusion["model"].map(MODEL_LABELS)
        confusion = confusion.rename(
            columns={"model": "모델", "tn": "TN", "fp": "FP", "fn": "FN", "tp": "TP"}
        )
        st.dataframe(confusion, width="stretch", hide_index=True)
    else:
        st.info("저장된 리포트에 혼동행렬 수치가 없습니다.")
