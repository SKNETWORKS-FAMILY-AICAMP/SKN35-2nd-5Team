"""Page 6: 개발 관리자용 ML/DL 모델 성능 평가."""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from streamlit_ui import apply_page_style, page_header, style_plotly_chart, top_navigation

st.set_page_config(page_title="TalentGuard AI | ML/DL 성능평가", layout="wide")
apply_page_style(); top_navigation("models")
page_header("DEV ADMIN · MODEL PERFORMANCE", "ML / DL 모델 성능평가", "개발 관리자가 실제 저장된 평가 결과와 최종 배포 모델의 선정 근거를 확인합니다.")

REPORTS = Path("artifacts/reports")
ML_LABELS = {"lightgbm": "LightGBM", "gradient_boosting": "Gradient Boosting", "xgboost": "XGBoost", "random_forest": "Random Forest", "logistic_regression": "Logistic Regression"}
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
METRIC_LABELS = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"]


@st.cache_data
def load_reports() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    ml = pd.read_csv(REPORTS / "ml_leaderboard.csv").sort_values("roc_auc", ascending=False).reset_index(drop=True)
    best = pd.read_csv(REPORTS / "best_ml_test_metrics.csv").iloc[0]
    dl = pd.read_csv(REPORTS / "mlp_test_metrics.csv").iloc[0]
    return ml, best, dl


def metric_table(frame: pd.DataFrame) -> pd.io.formats.style.Styler:
    shown = frame.copy()
    shown.insert(0, "순위", range(1, len(shown) + 1))
    shown["모델"] = shown["model"].map(ML_LABELS).fillna(shown["model"])
    shown = shown[["순위", "모델", *METRICS]].rename(columns=dict(zip(METRICS, METRIC_LABELS, strict=True)))
    return shown.style.format({metric: "{:.1%}" for metric in METRIC_LABELS}).background_gradient(subset=METRIC_LABELS, cmap="PuBuGn")


ml, best_ml, mlp = load_reports()
st.markdown('<div class="section-heading" style="--accent:#21d4ee"><span class="chip">ML</span><h2>머신러닝 모델 비교</h2></div>', unsafe_allow_html=True)
with st.container(border=True):
    st.caption(f"성능 지표 상세 테이블 — 1등 모델: {ML_LABELS.get(ml.iloc[0]['model'], ml.iloc[0]['model'])}")
    st.dataframe(metric_table(ml), hide_index=True, use_container_width=True)
with st.container(border=True):
    st.caption("ML 모델 Accuracy / F1 / AUC 시각 비교")
    fig = go.Figure()
    for metric, label, color in [("accuracy", "Accuracy", "#28cee3"), ("f1", "F1-Score", "#9b7df4"), ("roc_auc", "AUC", "#38d0a0")]:
        fig.add_bar(name=label, x=ml["model"].map(ML_LABELS), y=ml[metric] * 100, marker_color=color)
    fig.update_layout(barmode="group", yaxis=dict(range=[60, 100], ticksuffix="%"))
    st.plotly_chart(style_plotly_chart(fig, 300), use_container_width=True)

st.markdown('<div class="section-heading" style="--accent:#9b7df4;margin-top:34px"><span class="chip">DL</span><h2>딥러닝 모델 비교</h2></div>', unsafe_allow_html=True)
dl_frame = pd.DataFrame([mlp])
dl_frame["모델"] = "MLP"
dl_frame.insert(0, "순위", 1)
dl_shown = dl_frame[["순위", "모델", *METRICS]].rename(columns=dict(zip(METRICS, METRIC_LABELS, strict=True)))
with st.container(border=True):
    st.caption("성능 지표 상세 테이블 — 1등 모델: MLP")
    st.dataframe(dl_shown.style.format({metric: "{:.1%}" for metric in METRIC_LABELS}).background_gradient(subset=METRIC_LABELS, cmap="PuBuGn"), hide_index=True, use_container_width=True)
with st.container(border=True):
    st.caption("DL 모델 Accuracy / F1 / AUC 시각 비교")
    fig = go.Figure()
    for metric, label, color in [("accuracy", "Accuracy", "#9b7df4"), ("f1", "F1-Score", "#ff6b86"), ("roc_auc", "AUC", "#ffa20a")]:
        fig.add_bar(name=label, x=["MLP"], y=[float(mlp[metric]) * 100], marker_color=color)
    fig.update_layout(barmode="group", yaxis=dict(range=[60, 100], ticksuffix="%"))
    st.plotly_chart(style_plotly_chart(fig, 270), use_container_width=True)

st.markdown("### 최종 모델 비교 — LightGBM vs MLP")
ml_name = ML_LABELS.get(str(best_ml["model"]), str(best_ml["model"]))


def stat_html(row: pd.Series, metric: str, label: str) -> str:
    return f'<div>{float(row[metric]):.1%}<small>{label}</small></div>'


cards = f"""
<div class="champ-grid">
 <div class="champ" style="--accent:#21d4ee"><span class="tag">ML CHAMPION</span><h3>{ml_name}</h3><div class="champ-stats">{stat_html(best_ml,'accuracy','Accuracy')}{stat_html(best_ml,'f1','F1-Score')}{stat_html(best_ml,'roc_auc','AUC')}</div><ul><li>✓ 빠른 학습 및 추론</li><li>✓ 트리 기반 설명 가능성</li><li>✓ 운영 적용 용이</li></ul></div>
 <div class="champ" style="--accent:#9b7df4"><span class="tag">DL CHAMPION</span><h3>MLP</h3><div class="champ-stats">{stat_html(mlp,'accuracy','Accuracy')}{stat_html(mlp,'f1','F1-Score')}{stat_html(mlp,'roc_auc','AUC')}</div><ul><li>✓ 비선형 관계 학습</li><li>✓ 높은 재현율</li><li>✓ 확률 기반 분류</li></ul></div>
</div>"""
st.markdown(cards, unsafe_allow_html=True)

left, right = st.columns(2)
with left:
    with st.container(border=True):
        st.caption("5개 지표 레이더 비교")
        categories = METRIC_LABELS + [METRIC_LABELS[0]]
        fig = go.Figure()
        for row, name, color in [(best_ml, ml_name, "#21d4ee"), (mlp, "MLP", "#9b7df4")]:
            values = [float(row[m]) * 100 for m in METRICS]
            fig.add_trace(go.Scatterpolar(r=values + [values[0]], theta=categories, fill="toself", name=name, line_color=color))
        fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100], gridcolor="#26334d"), bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(style_plotly_chart(fig, 300), use_container_width=True)
with right:
    with st.container(border=True):
        st.caption("지표별 최종 성능 비교")
        fig = go.Figure()
        fig.add_bar(name=ml_name, x=METRIC_LABELS, y=[float(best_ml[m]) * 100 for m in METRICS], marker_color="#28cee3")
        fig.add_bar(name="MLP", x=METRIC_LABELS, y=[float(mlp[m]) * 100 for m in METRICS], marker_color="#9b7df4")
        fig.update_layout(barmode="group", yaxis=dict(range=[0, 100], ticksuffix="%"))
        st.plotly_chart(style_plotly_chart(fig, 300), use_container_width=True)

auc_gap = (float(best_ml["roc_auc"]) - float(mlp["roc_auc"])) * 100
st.markdown(f'<div class="decision"><h3>🎯 최종 배포 모델: {ml_name}</h3><div class="decision-row"><span>테스트 AUC {float(best_ml["roc_auc"]):.2%}로 MLP보다 {auc_gap:.2f}%p 높습니다.</span><span>Accuracy {float(best_ml["accuracy"]):.1%} · Precision {float(best_ml["precision"]):.1%} · Recall {float(best_ml["recall"]):.1%} · F1 {float(best_ml["f1"]):.1%}</span></div></div>', unsafe_allow_html=True)
