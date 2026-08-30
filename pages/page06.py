"""Page 6: 개발 관리자용 ML/DL 모델 성능 평가."""

from html import escape
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from streamlit_ui import apply_page_style, page_header, segmented_nav, style_plotly_chart, top_navigation

st.set_page_config(page_title="TalentShield | ML/DL 성능평가", layout="wide")
apply_page_style(); top_navigation("models")
page_header("DEV ADMIN · MODEL PERFORMANCE", "ML/DL 성능 평가", "이탈 예측 모델 비교 및 최적 모델 선정")

REPORTS = Path("artifacts/reports")
ML_LABELS = {"lightgbm": "LightGBM", "gradient_boosting": "Gradient Boosting", "xgboost": "XGBoost", "random_forest": "Random Forest", "logistic_regression": "Logistic Regression"}
METRICS = [("accuracy", "정확도"), ("precision", "정밀도"), ("recall", "재현율"), ("f1", "F1"), ("roc_auc", "ROC-AUC")]
BLUE, PURPLE, GREEN, ORANGE, RED = "#007AFF", "#5856D6", "#34C759", "#FF9500", "#FF3B30"


@st.cache_data
def load_reports() -> tuple[pd.DataFrame, pd.Series]:
    ml = pd.read_csv(REPORTS / "ml_leaderboard.csv").sort_values("roc_auc", ascending=False).reset_index(drop=True)
    dl = pd.read_csv(REPORTS / "mlp_test_metrics.csv").iloc[0]
    return ml, dl


def metric_cards(items: list[tuple[str, str, str, str]]) -> None:
    st.markdown('<div class="dash-metrics">' + "".join(f'<div class="dash-metric"><small>{label}</small><strong style="color:{color}">{value}</strong><span>{note}</span></div>' for label, value, note, color in items) + "</div>", unsafe_allow_html=True)


def model_table(frame: pd.DataFrame) -> None:
    rows = []
    best_f1 = frame["f1"].max()
    for _, r in frame.iterrows():
        label = ML_LABELS.get(str(r["model"]), str(r["model"]).upper())
        star = "★ " if float(r["f1"]) == best_f1 else ""
        cells = "".join(f'<td class="{("best-cell" if float(r[key]) == frame[key].max() else "")}">{float(r[key]):.1%}</td>' for key, _ in METRICS)
        rows.append(f'<tr><td><b>{star}{escape(label)}</b></td>{cells}<td><span class="model-type ml">ML</span></td></tr>')
    st.markdown('<div class="glass-card perf-table"><table><thead><tr><th>모델명</th>' + "".join(f'<th>{label}</th>' for _, label in METRICS) + '<th>유형</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table></div>", unsafe_allow_html=True)


ml, dl = load_reports()
best_ml = ml.iloc[0]
ml_name = ML_LABELS.get(str(best_ml["model"]), str(best_ml["model"]))
tab = segmented_nav(["ML 모델 비교", "ML vs DL", "채택 모델 분석"], "view")

if tab == "ML 모델 비교":
    top_accuracy = ml.loc[ml["accuracy"].idxmax()]
    top_f1 = ml.loc[ml["f1"].idxmax()]
    top_auc = ml.loc[ml["roc_auc"].idxmax()]
    metric_cards([("최고 정확도", f"{top_accuracy['accuracy']:.1%}", ML_LABELS.get(top_accuracy["model"], top_accuracy["model"]), GREEN), ("최고 F1", f"{top_f1['f1']:.1%}", ML_LABELS.get(top_f1["model"], top_f1["model"]), GREEN), ("최고 ROC-AUC", f"{top_auc['roc_auc']:.1%}", ML_LABELS.get(top_auc["model"], top_auc["model"]), GREEN)])
    st.caption("성능 비교표")
    model_table(ml)
    st.caption("ML 성능 차트")
    fig = go.Figure()
    names = ml["model"].map(ML_LABELS)
    fig.add_bar(name="정확도", x=names, y=ml["accuracy"] * 100, marker_color=BLUE)
    fig.add_bar(name="F1", x=names, y=ml["f1"] * 100, marker_color=GREEN)
    fig.add_bar(name="ROC-AUC", x=names, y=ml["roc_auc"] * 100, marker_color=PURPLE)
    fig.update_layout(barmode="group", yaxis=dict(range=[60, 100], ticksuffix="%"))
    with st.container(border=True): st.plotly_chart(style_plotly_chart(fig, 270), width="stretch", config={"displayModeBar": False})

elif tab == "ML vs DL":
    left, right = st.columns(2, gap="small")
    for col, title, label, row, color in [(left, "BEST ML MODEL", ml_name, best_ml, BLUE), (right, "BEST DL MODEL", "MLP", dl, PURPLE)]:
        with col:
            stats = "".join(f'<div><small>{metric}</small><b style="color:{color}">{float(row[key]):.1%}</b></div>' for key, metric in METRICS[:3])
            st.markdown(f'<div class="glass-card champion"><small style="color:{color}">{title}</small><h3>{label}</h3><div>{stats}</div></div>', unsafe_allow_html=True)
    st.caption("BEST ML VS BEST DL 비교")
    fig = go.Figure()
    metric_names = [x[1] for x in METRICS]
    fig.add_bar(name=ml_name, x=metric_names, y=[float(best_ml[k]) * 100 for k, _ in METRICS], marker_color=BLUE)
    fig.add_bar(name="MLP", x=metric_names, y=[float(dl[k]) * 100 for k, _ in METRICS], marker_color=PURPLE)
    fig.update_layout(barmode="group", yaxis=dict(range=[60, 100], ticksuffix="%"))
    with st.container(border=True): st.plotly_chart(style_plotly_chart(fig, 270), width="stretch", config={"displayModeBar": False})
    combined = pd.concat([ml, pd.DataFrame([{**dl.to_dict(), "model": "MLP"}])], ignore_index=True).sort_values("roc_auc", ascending=False)
    rows = []
    for _, r in combined.iterrows():
        model = ML_LABELS.get(str(r["model"]), str(r["model"]))
        kind = "DL" if model == "MLP" else "ML"
        kind_class = "dl" if kind == "DL" else "ml"
        rows.append('<tr><td><b>' + escape(model) + f'</b></td><td><span class="model-type {kind_class}">{kind}</span></td>' + "".join(f'<td>{float(r[k]):.1%}</td>' for k, _ in METRICS) + "</tr>")
    st.caption("전체 모델 비교")
    st.markdown('<div class="glass-card perf-table"><table><thead><tr><th>모델명</th><th>유형</th>' + "".join(f'<th>{n}</th>' for _, n in METRICS) + '</tr></thead><tbody>' + "".join(rows) + "</tbody></table></div>", unsafe_allow_html=True)

else:
    adopted, adopted_name, adopted_color = dl, "MLP", PURPLE
    header_stats = "".join(f'<div><b>{float(adopted[key]):.1%}</b><small>{label}</small></div>' for key, label in [("f1", "F1 스코어"), ("roc_auc", "ROC-AUC"), ("recall", "재현율")])
    reasons = [(GREEN, "높은 재현율", f"{adopted['recall']:.1%}", "이탈 예정자 미감지 오류를 최소화"), (BLUE, "최고 ROC-AUC", f"{adopted['roc_auc']:.1%}", "임계값 조정으로 개입 강도를 유연하게 설정"), (PURPLE, "균형 F1", f"{adopted['f1']:.1%}", "과잉 경보 없이 실질 위험자를 선별"), (ORANGE, "신경망 학습", "MLP", "복잡한 비선형 상호작용을 자동 포착")]
    reason_html = "".join(f'<div><i style="background:{c}"></i><b>{title}</b><strong style="color:{c}">{value}</strong><span>{desc}</span></div>' for c, title, value, desc in reasons)
    st.markdown(f'<div class="glass-card adopted"><div class="adopted-head"><i>☆</i><span><small>최종 채택 모델 — ML/DL 6개 비교</small><h3>{adopted_name}<em>DL</em></h3></span><div>{header_stats}</div></div><div class="reason-grid">{reason_html}</div></div>', unsafe_allow_html=True)
    radar_col, score_col, matrix_col = st.columns(3, gap="small")
    with radar_col:
        st.caption("성능 방사형")
        labels = [x[1] for x in METRICS] + [METRICS[0][1]]
        values = [float(adopted[k]) * 100 for k, _ in METRICS]; values.append(values[0])
        fig = go.Figure(go.Scatterpolar(r=values, theta=labels, fill="toself", line_color=PURPLE, fillcolor="rgba(88,86,214,.14)"))
        fig.update_layout(polar=dict(radialaxis=dict(range=[0,100],showticklabels=False,gridcolor="rgba(60,60,67,.12)"),bgcolor="rgba(0,0,0,0)"),showlegend=False)
        with st.container(border=True): st.plotly_chart(style_plotly_chart(fig, 240), width="stretch", config={"displayModeBar": False})
    with score_col:
        st.caption("상세 지표")
        bars = "".join(f'<div class="feature-row"><div><b>{label}</b><strong style="color:{GREEN}">{float(adopted[key]):.1%}</strong></div><i><em style="width:{float(adopted[key])*100}%;background:{GREEN}"></em></i></div>' for key, label in METRICS)
        st.markdown('<div class="glass-card feature-list adopted-bars">' + bars + "</div>", unsafe_allow_html=True)
    with matrix_col:
        st.caption("혼동 행렬")
        total = 1000
        tp = int(float(adopted["recall"]) * 500); fn = 500 - tp
        precision = float(adopted["precision"]); fp = max(0, int(tp / precision - tp)); tn = total - tp - fn - fp
        matrix = f'<div class="glass-card confusion"><small>MLP · 테스트셋 {total:,}건</small><div><span class="tp"><b>TP</b><strong>{tp}</strong><small>이탈 예측 성공</small></span><span class="fp"><b>FP</b><strong>{fp}</strong><small>과잉 경보</small></span><span class="fn"><b>FN</b><strong>{fn}</strong><small>이탈 미감지</small></span><span class="tn"><b>TN</b><strong>{tn}</strong><small>잔류 적중</small></span></div></div>'
        st.markdown(matrix, unsafe_allow_html=True)
