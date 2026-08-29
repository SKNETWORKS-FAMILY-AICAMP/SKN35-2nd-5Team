# ruff: noqa: E501

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from streamlit_ui import apply_page_style, page_header, style_plotly_chart, top_navigation

CYAN, VIOLET, AMBER, EMERALD, ROSE = "#22D3EE", "#A78BFA", "#F59E0B", "#34D399", "#FB7185"
ML_PATH, DL_PATH = Path("artifacts/reports/ml_leaderboard.csv"), Path("artifacts/reports/mlp_test_metrics.csv")
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
LABELS = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1-Score", "roc_auc": "AUC"}
MODELS = {"logistic_regression": "Logistic Regression", "gradient_boosting": "Gradient Boosting", "random_forest": "Random Forest", "xgboost": "XGBoost", "lightgbm": "LightGBM", "mlp": "MLP"}

st.set_page_config(page_title="개발 관리자 모델 성능평가 · TalentGuard AI", layout="wide")


@st.cache_data
def load_report(path: str, modified: float) -> pd.DataFrame:
    del modified
    return pd.read_csv(path)


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["모델"] = result["model"].map(MODELS).fillna(result["model"])
    result = result.sort_values(["roc_auc", "f1"], ascending=False).reset_index(drop=True)
    result.insert(0, "순위", range(1, len(result) + 1))
    return result


def render_group(kind: str, title: str, frame: pd.DataFrame, colors: list[str]) -> None:
    chip = "section-chip violet" if kind == "DL" else "section-chip"
    st.markdown(f'<div class="section-heading"><span class="{chip}">{kind}</span><h2>{title}</h2></div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.caption(f"성능 지표 상세 테이블 — 1등 모델: {frame.iloc[0]['모델']}")
        display = frame[["순위", "모델", *METRICS]].rename(columns=LABELS)
        st.dataframe(display.style.format({value: "{:.1%}" for value in LABELS.values()}).highlight_max(subset=list(LABELS.values()), color="rgba(34,211,238,.18)"), width="stretch", hide_index=True)
    with st.container(border=True):
        st.caption(f"{kind} 모델 Accuracy / F1 / AUC 시각 비교")
        fig = go.Figure()
        for metric, color in zip(["accuracy", "f1", "roc_auc"], colors, strict=True):
            fig.add_bar(x=frame["모델"], y=frame[metric], name=LABELS[metric], marker_color=color)
        floor = max(0, frame[["accuracy", "f1", "roc_auc"]].min().min() - .1)
        style_plotly_chart(fig, 310).update_layout(barmode="group")
        fig.update_yaxes(tickformat=".0%", range=[floor, 1])
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def champion(tag: str, row: pd.Series, color: str, notes: list[str]) -> str:
    metrics = "".join(f'<div class="champion-metric"><b>{row[key]:.1%}</b><span>{LABELS[key]}</span></div>' for key in ["accuracy", "f1", "roc_auc"])
    points = "".join(f"<div>✓ {note}</div>" for note in notes)
    return f'<div class="champion-card" style="--accent:{color}"><span class="champion-tag">{tag}</span><div class="champion-name">{row["모델"]}</div><div class="champion-metrics">{metrics}</div><div class="champion-list">{points}</div></div>'


apply_page_style()
top_navigation("models")
page_header("DEV ADMIN · MODEL PERFORMANCE", "ML / DL 모델 성능평가", "개발 관리자가 실제 저장된 평가 결과와 최종 배포 모델의 선정 근거를 확인합니다.")

missing = [str(path) for path in [ML_PATH, DL_PATH] if not path.exists()]
if missing:
    st.warning("비교에 필요한 모델 리포트가 없습니다: " + ", ".join(missing))
    st.stop()
ml, dl = prepare(load_report(str(ML_PATH), ML_PATH.stat().st_mtime)), prepare(load_report(str(DL_PATH), DL_PATH.stat().st_mtime))
if not {"model", *METRICS}.issubset(ml.columns) or not {"model", *METRICS}.issubset(dl.columns):
    st.error("모델 리포트에 Accuracy, Precision, Recall, F1, AUC가 모두 필요합니다.")
    st.stop()

render_group("ML", "머신러닝 모델 비교", ml, [CYAN, VIOLET, EMERALD])
st.markdown('<div class="flow-divider">SCROLL DOWN</div>', unsafe_allow_html=True)
render_group("DL", "딥러닝 모델 비교", dl, [VIOLET, ROSE, AMBER])
st.markdown('<div class="flow-divider">FINAL COMPARISON</div>', unsafe_allow_html=True)

best_ml, best_dl = ml.iloc[0], dl.iloc[0]
st.markdown(f'<div class="section-heading"><h2>최종 모델 비교 — {best_ml["모델"]} vs {best_dl["모델"]}</h2></div>', unsafe_allow_html=True)
left, right = st.columns(2)
with left:
    st.markdown(champion("ML CHAMPION", best_ml, CYAN, ["빠른 학습 및 추론", "트리 기반 설명 가능성", "운영 적용 용이"]), unsafe_allow_html=True)
with right:
    st.markdown(champion("DL CHAMPION", best_dl, VIOLET, ["비선형 관계 학습", "은닉 표현 자동 추출", "확률 기반 분류"]), unsafe_allow_html=True)

comparison = pd.DataFrame({"지표": [LABELS[key] for key in METRICS], best_ml["모델"]: [best_ml[key] for key in METRICS], best_dl["모델"]: [best_dl[key] for key in METRICS]})
radar, bars = st.columns(2)
with radar:
    with st.container(border=True):
        st.caption("5개 지표 레이더 비교")
        theta = comparison["지표"].tolist()
        fig = go.Figure()
        for row, color in [(best_ml, CYAN), (best_dl, VIOLET)]:
            values = [row[key] * 100 for key in METRICS]
            fig.add_trace(go.Scatterpolar(r=values + values[:1], theta=theta + theta[:1], fill="toself", name=row["모델"], line_color=color, opacity=.75))
        style_plotly_chart(fig, 320).update_layout(polar={"bgcolor": "rgba(0,0,0,0)", "radialaxis": {"visible": True, "range": [0, 100], "gridcolor": "rgba(148,163,184,.12)"}, "angularaxis": {"gridcolor": "rgba(148,163,184,.12)"}})
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
with bars:
    with st.container(border=True):
        st.caption("지표별 최종 성능 비교")
        fig = go.Figure()
        fig.add_bar(x=comparison["지표"], y=comparison[best_ml["모델"]], name=best_ml["모델"], marker_color=CYAN)
        fig.add_bar(x=comparison["지표"], y=comparison[best_dl["모델"]], name=best_dl["모델"], marker_color=VIOLET)
        style_plotly_chart(fig, 320).update_layout(barmode="group")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

auc_gap = float(best_ml["roc_auc"] - best_dl["roc_auc"])
selected, other = (best_ml, best_dl) if auc_gap >= 0 else (best_dl, best_ml)
st.markdown(f'<div class="decision-box"><div class="panel-title">🎯 최종 배포 모델: {selected["모델"]}</div><div class="insight-grid"><div class="insight-card">테스트 AUC <b>{selected["roc_auc"]:.2%}</b>로 {other["모델"]}보다 {abs(auc_gap):.2%}p 높습니다.</div><div class="insight-card">Accuracy {selected["accuracy"]:.2%} · Precision {selected["precision"]:.2%} · Recall {selected["recall"]:.2%} · F1 {selected["f1"]:.2%}</div></div></div>', unsafe_allow_html=True)
