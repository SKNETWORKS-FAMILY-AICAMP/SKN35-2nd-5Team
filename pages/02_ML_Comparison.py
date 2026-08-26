from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from streamlit_ui import apply_page_style, home_button, page_header

st.set_page_config(page_title="머신러닝 모델 비교", page_icon="🌳", layout="wide")

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
    "f1": "F1 점수",
    "roc_auc": "ROC 곡선 면적",
    "average_precision": "정밀도-재현율 곡선 면적",
}


@st.cache_data
def load_ml_report(path: str, modified_time: float) -> pd.DataFrame:
    del modified_time
    return pd.read_csv(path)


apply_page_style()
home_button()
page_header(
    "모델 성능 비교",
    "머신러닝 모델 비교 🌳",
    "같은 검증 데이터에서 기본 모델과 승격된 튜닝 모델을 공정하게 비교해요.",
)

if not REPORT_PATH.exists():
    st.warning("머신러닝 성능 보고서가 없습니다: artifacts/reports/ml_leaderboard.csv")
    st.stop()

leaderboard = load_ml_report(str(REPORT_PATH), REPORT_PATH.stat().st_mtime)
required_columns = {"model", *METRIC_LABELS}
missing_columns = required_columns.difference(leaderboard.columns)
if missing_columns:
    st.error(f"머신러닝 보고서에 필요한 항목이 없습니다: {', '.join(sorted(missing_columns))}")
    st.stop()

leaderboard = leaderboard[leaderboard["model"].isin(MODEL_ORDER)].copy()
if leaderboard.empty:
    st.warning("비교할 머신러닝 모델 결과가 없습니다.")
    st.stop()

leaderboard["모델"] = leaderboard["model"].map(MODEL_LABELS)
leaderboard["버전"] = leaderboard["artifact_path"].fillna("").apply(
    lambda path: "옵튜나 튜닝" if "_tuned.joblib" in str(path) else "기본"
)
leaderboard = leaderboard.sort_values(
    ["roc_auc", "f1"], ascending=False, ignore_index=True
)

best = leaderboard.iloc[0]
summary = st.columns(3)
summary[0].metric("최고 모델", best["모델"])
summary[1].metric("최고 ROC 곡선 면적", f"{best['roc_auc']:.4f}")
summary[2].metric("F1 점수", f"{best['f1']:.4f}")

display = leaderboard[["모델", "버전", *METRIC_LABELS]].rename(columns=METRIC_LABELS)
metric_columns = [
    "정확도",
    "정밀도",
    "재현율",
    "F1 점수",
    "ROC 곡선 면적",
    "정밀도-재현율 곡선 면적",
]

st.subheader("한눈에 보는 성능")
st.dataframe(
    display.style.format({column: "{:.4f}" for column in metric_columns}).highlight_max(
        subset=metric_columns, color="#f4dfe5"
    ),
    width="stretch",
    hide_index=True,
)

st.subheader("핵심 지표 비교")
chart_metrics = ["정확도", "F1 점수", "ROC 곡선 면적", "정밀도-재현율 곡선 면적"]
chart = go.Figure()
colors = ["#A85F72", "#748BAD", "#C58B9B", "#8C7E9F"]

for (_, row), color in zip(display.iterrows(), colors, strict=False):
    model_label = f"{row['모델']} · {row['버전']}"
    values = [float(row[metric]) for metric in chart_metrics]
    chart.add_trace(
        go.Scatter(
            x=chart_metrics,
            y=values,
            name=model_label,
            mode="lines+markers+text",
            text=[f"{value:.4f}" for value in values],
            textposition="top center",
            line={"color": color, "width": 4, "shape": "spline"},
            marker={
                "color": "white",
                "line": {"color": color, "width": 3},
                "size": 11,
            },
            hovertemplate=(
                f"<b>{model_label}</b><br>"
                "%{x}: %{y:.4f}<extra></extra>"
            ),
        )
    )

all_values = display[chart_metrics].astype(float).to_numpy().ravel()
y_min = max(0.0, float(all_values.min()) - 0.03)
y_max = min(1.0, float(all_values.max()) + 0.03)
chart.update_layout(
    height=480,
    margin={"l": 20, "r": 20, "t": 35, "b": 20},
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.72)",
    hovermode="x unified",
    legend={
        "orientation": "h",
        "yanchor": "bottom",
        "y": 1.08,
        "xanchor": "left",
        "x": 0,
    },
    xaxis={"title": None, "showgrid": False},
    yaxis={
        "title": "평가 점수",
        "range": [y_min, y_max],
        "tickformat": ".2f",
        "gridcolor": "#E8DFE5",
        "zeroline": False,
    },
)
st.plotly_chart(chart, width="stretch", config={"displayModeBar": False})
st.caption("점수 차이를 보기 쉽도록 세로축은 현재 모델들의 최솟값과 최댓값 주변으로 표시했어요.")

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
