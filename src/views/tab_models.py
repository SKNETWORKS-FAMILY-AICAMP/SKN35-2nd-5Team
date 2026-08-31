"""05 · 모델 성능 평가 탭 (기술개발팀·관리자 전용).

ML 4종과 DL(MLP)의 성능을 같은 화면에서 비교하고, 어떤 근거로 최종 모델을 골랐는지
(하이퍼파라미터, 리더보드) 함께 보여준다. 인사팀 화면에는 노출하지 않는다. 표·통계
카드·단순 막대 그래프는 커스텀 HTML/CSS로 그리고, 모델 4~5개 × 지표 4개를 한번에
겹쳐 보는 복합 비교 차트만 Plotly를 사용한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from streamlit_ui import (
    alert_box,
    model_card_grid,
    render_table,
    section_heading,
    stat_cards,
    sub_tabs,
    versus_hero,
)

REPORTS_DIR = Path("artifacts/reports")
ML_LEADERBOARD_PATH = REPORTS_DIR / "ml_leaderboard.csv"
BEST_ML_TEST_METRICS_PATH = REPORTS_DIR / "best_ml_test_metrics.csv"
DL_METRICS_PATH = REPORTS_DIR / "mlp_test_metrics.csv"
TUNED_PARAM_PATHS = {
    "LightGBM": REPORTS_DIR / "lightgbm_tuned_params.json",
    "Random Forest": REPORTS_DIR / "random_forest_tuned_params.json",
    "XGBoost": REPORTS_DIR / "xgboost_tuned_params.json",
}

MODEL_ORDER = ["logistic_regression", "random_forest", "xgboost", "lightgbm"]
MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "mlp": "MLP",
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
def _load_csv(path: str, modified_time: float) -> pd.DataFrame:
    del modified_time
    return pd.read_csv(path)


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """레이더 차트의 채우기 색을 선 색상과 맞추기 위한 hex → rgba 변환."""

    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


def _plotly_font() -> dict:
    return {"family": "-apple-system, BlinkMacSystemFont, sans-serif", "color": "#1D1D1F"}


def _chart_type_picker(label: str, options: list[str], key: str) -> str:
    """지표를 원하는 그래프 형태로 바꿔볼 수 있는 세그먼트 선택기."""

    choice = st.segmented_control(label, options, key=key, default=options[0])
    return choice or st.session_state.get(key) or options[0]


def _render_ml_comparison() -> None:
    if not ML_LEADERBOARD_PATH.exists():
        alert_box("warning", "머신러닝 성능 보고서가 없습니다: artifacts/reports/ml_leaderboard.csv")
        return

    leaderboard = _load_csv(str(ML_LEADERBOARD_PATH), ML_LEADERBOARD_PATH.stat().st_mtime)
    required_columns = {"model", *METRIC_LABELS}
    missing_columns = required_columns.difference(leaderboard.columns)
    if missing_columns:
        alert_box("danger", f"머신러닝 보고서에 필요한 항목이 없습니다: {', '.join(sorted(missing_columns))}")
        return

    leaderboard = leaderboard[leaderboard["model"].isin(MODEL_ORDER)].copy()
    if leaderboard.empty:
        alert_box("warning", "비교할 머신러닝 모델 결과가 없습니다.")
        return

    leaderboard["모델"] = leaderboard["model"].map(MODEL_LABELS)
    leaderboard["버전"] = leaderboard["artifact_path"].fillna("").apply(
        lambda p: "옵튜나 튜닝" if "_tuned.joblib" in str(p) else "기본"
    )
    leaderboard = leaderboard.sort_values(["roc_auc", "f1"], ascending=False, ignore_index=True)

    best = leaderboard.iloc[0]
    stat_cards(
        [
            {"label": "최고 모델", "value": f"{best['모델']}", "hint": best["버전"], "tone": "safe"},
            {"label": "최고 ROC 곡선 면적", "value": f"{best['roc_auc']:.4f}"},
            {"label": "F1 점수", "value": f"{best['f1']:.4f}"},
        ]
    )

    display = leaderboard[["모델", "버전", *METRIC_LABELS]].rename(columns=METRIC_LABELS)
    metric_columns = list(METRIC_LABELS.values())

    st.markdown("**한눈에 보는 성능**")
    best_per_metric = {m: display[m].max() for m in metric_columns}
    model_cards = [
        {
            "name": f"{row['모델']} · {row['버전']}",
            "metrics": [(m, f"{row[m]:.4f}") for m in metric_columns],
            "best": index == 0,
        }
        for index, (_, row) in enumerate(display.iterrows())
    ]
    model_card_grid(model_cards)
    with st.expander("모델별 상세 지표 표 보기"):
        render_table(
            display,
            formats={c: "{:.4f}" for c in metric_columns},
            badges={
                m: (lambda v, m=m: (f"{v:.4f}", "safe" if v == best_per_metric[m] else "neutral"))
                for m in metric_columns
            },
        )

    st.markdown("**핵심 지표 비교**")
    chart_metrics = ["정확도", "F1 점수", "ROC 곡선 면적", "정밀도-재현율 곡선 면적"]
    colors = ["#0071E3", "#5E5CE6", "#32ADE6", "#6E6E73", "#98989D"]
    series = [
        (f"{row['모델']} · {row['버전']}", [float(row[m]) for m in chart_metrics], color)
        for (_, row), color in zip(display.iterrows(), colors, strict=False)
    ]
    all_values = display[chart_metrics].astype(float).to_numpy().ravel()
    y_min = max(0.0, float(all_values.min()) - 0.03)
    y_max_line = min(1.0, float(all_values.max()) + 0.03)
    y_max_bar = min(1.08, float(all_values.max()) + 0.09)

    ml_chart_type = _chart_type_picker(
        "그래프 종류", ["막대 그래프", "레이더 차트", "라인 차트"], key="model_ml_chart_type"
    )

    chart = go.Figure()
    if ml_chart_type == "레이더 차트":
        # 모델이 4~5개라 겹쳐 채우면 지저분해지므로, 선만 겹쳐 각 모델의 지표
        # 프로필 모양을 비교한다.
        for label, values, color in series:
            chart.add_trace(
                go.Scatterpolar(
                    r=values + [values[0]],
                    theta=chart_metrics + [chart_metrics[0]],
                    name=label,
                    line={"color": color, "width": 2.5},
                    marker={"size": 5, "color": color},
                    hovertemplate=f"<b>{label}</b><br>%{{theta}}: %{{r:.4f}}<extra></extra>",
                )
            )
        chart.update_layout(
            polar={
                "radialaxis": {"visible": True, "range": [y_min, y_max_line], "gridcolor": "#E8E8ED"},
                "angularaxis": {"gridcolor": "#E8E8ED"},
                "bgcolor": "rgba(255,255,255,0.6)",
            },
            height=460,
            showlegend=True,
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.1, "xanchor": "center", "x": 0.5},
            margin={"l": 50, "r": 50, "t": 30, "b": 30},
            paper_bgcolor="rgba(0,0,0,0)",
            font=_plotly_font(),
        )
    elif ml_chart_type == "라인 차트":
        for label, values, color in series:
            chart.add_trace(
                go.Scatter(
                    x=chart_metrics,
                    y=values,
                    name=label,
                    mode="lines+markers+text",
                    text=[f"{v:.4f}" for v in values],
                    textposition="top center",
                    line={"color": color, "width": 4, "shape": "spline"},
                    marker={"color": "white", "line": {"color": color, "width": 3}, "size": 11},
                    hovertemplate=f"<b>{label}</b><br>%{{x}}: %{{y:.4f}}<extra></extra>",
                )
            )
        chart.update_layout(
            height=440,
            margin={"l": 20, "r": 20, "t": 35, "b": 20},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.9)",
            font=_plotly_font(),
            hovermode="x unified",
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.08, "xanchor": "left", "x": 0},
            xaxis={"title": None, "showgrid": False},
            yaxis={
                "title": "평가 점수",
                "range": [y_min, y_max_line],
                "tickformat": ".2f",
                "gridcolor": "#E8E8ED",
                "zeroline": False,
            },
        )
    else:  # 막대 그래프 (기본)
        for label, values, color in series:
            chart.add_trace(
                go.Bar(
                    x=chart_metrics,
                    y=values,
                    name=label,
                    marker={"color": color},
                    text=[f"{v:.3f}" for v in values],
                    textposition="outside",
                    textfont={"size": 11},
                    hovertemplate=f"<b>{label}</b><br>%{{x}}: %{{y:.4f}}<extra></extra>",
                )
            )
        chart.update_layout(
            height=440,
            barmode="group",
            bargap=.24,
            bargroupgap=.08,
            margin={"l": 20, "r": 20, "t": 35, "b": 20},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.9)",
            font=_plotly_font(),
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.08, "xanchor": "left", "x": 0},
            xaxis={"title": None, "showgrid": False},
            yaxis={
                "title": "평가 점수",
                "range": [y_min, y_max_bar],
                "tickformat": ".2f",
                "gridcolor": "#E8E8ED",
                "zeroline": False,
            },
        )
    st.plotly_chart(chart, width="stretch", config={"displayModeBar": False})

    with st.expander("혼동행렬 수치 보기"):
        confusion_columns = ["model", "tn", "fp", "fn", "tp"]
        if set(confusion_columns).issubset(leaderboard.columns):
            confusion = leaderboard[confusion_columns].copy()
            confusion["model"] = confusion["model"].map(MODEL_LABELS)
            confusion = confusion.rename(columns={"model": "모델", "tn": "TN", "fp": "FP", "fn": "FN", "tp": "TP"})
            render_table(confusion)
        else:
            alert_box("info", "저장된 리포트에 혼동행렬 수치가 없습니다.")

    with st.expander("⚙️ 학습 설정 · 튜닝 하이퍼파라미터 보기"):
        _render_training_details()


def _render_dl_performance() -> None:
    if not DL_METRICS_PATH.exists():
        alert_box("info", "아직 `artifacts/reports/mlp_test_metrics.csv`가 없어요. MLP 학습이 끝난 뒤 자동으로 나타납니다.")
        return

    report = _load_csv(str(DL_METRICS_PATH), DL_METRICS_PATH.stat().st_mtime)
    required_columns = {"model", "accuracy", "precision", "recall", "f1", "roc_auc"}
    missing_columns = required_columns.difference(report.columns)
    if missing_columns:
        alert_box("danger", f"딥러닝 보고서에 필요한 항목이 없습니다: {', '.join(sorted(missing_columns))}")
        return

    best = report.sort_values(["roc_auc", "f1"], ascending=False).iloc[0]
    stat_cards(
        [
            {"label": "모델", "value": str(best["model"]).upper()},
            {"label": "정확도", "value": f"{best['accuracy']:.4f}"},
            {"label": "ROC 곡선 면적", "value": f"{best['roc_auc']:.4f}"},
            {"label": "F1 점수", "value": f"{best['f1']:.4f}"},
            {"label": "재현율", "value": f"{best['recall']:.4f}"},
        ]
    )

    extra_labels = {**METRIC_LABELS, "epochs": "학습 반복 횟수", "final_loss": "최종 손실값", "threshold": "분류 임계값"}
    display_columns = ["model", *[c for c in extra_labels if c in report.columns]]
    display = report[display_columns].rename(columns={"model": "모델", **extra_labels})
    display["모델"] = display["모델"].astype(str).str.upper()
    formats = {label: "{:.4f}" for key, label in extra_labels.items() if label in display.columns}

    st.markdown("**딥러닝 모델 성능**")
    render_table(display, formats=formats)

    chart_metrics = [c for c in METRIC_LABELS.values() if c in display.columns]
    st.markdown("**평가 지표**")
    best_row = display.iloc[0]
    metric_values = [float(best_row[m]) * 100 for m in chart_metrics]

    dl_chart_type = _chart_type_picker(
        "그래프 종류", ["레이더 차트", "막대 그래프"], key="model_dl_chart_type"
    )

    dl_chart = go.Figure()
    if dl_chart_type == "막대 그래프":
        dl_chart.add_trace(
            go.Bar(
                x=chart_metrics,
                y=metric_values,
                marker={"color": "#5E5CE6"},
                text=[f"{v:.1f}" for v in metric_values],
                textposition="outside",
                hovertemplate="%{x}: %{y:.1f}<extra></extra>",
            )
        )
        dl_chart.update_layout(
            height=380,
            margin={"l": 20, "r": 20, "t": 35, "b": 20},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.9)",
            font=_plotly_font(),
            showlegend=False,
            xaxis={"title": None, "showgrid": False},
            yaxis={
                "title": "점수 (0~100)",
                "range": [0, min(112, float(max(metric_values)) + 12)],
                "gridcolor": "#E8E8ED",
                "zeroline": False,
            },
        )
    else:
        dl_chart.add_trace(
            go.Scatterpolar(
                r=metric_values + [metric_values[0]],
                theta=chart_metrics + [chart_metrics[0]],
                fill="toself",
                name="MLP",
                line={"color": "#5E5CE6", "width": 3},
                fillcolor=_hex_to_rgba("#5E5CE6", 0.25),
                hovertemplate="%{theta}: %{r:.1f}<extra></extra>",
            )
        )
        dl_chart.update_layout(
            polar={
                "radialaxis": {"visible": True, "range": [0, 100], "gridcolor": "#E8E8ED"},
                "angularaxis": {"gridcolor": "#E8E8ED"},
                "bgcolor": "rgba(255,255,255,0.6)",
            },
            showlegend=False,
            margin={"l": 50, "r": 50, "t": 30, "b": 30},
            height=420,
            paper_bgcolor="rgba(0,0,0,0)",
            font=_plotly_font(),
        )
    st.plotly_chart(dl_chart, width="stretch", config={"displayModeBar": False})

    if {"tn", "fp", "fn", "tp"}.issubset(report.columns):
        st.markdown("**혼동행렬**")
        z = [[int(best["tn"]), int(best["fp"])], [int(best["fn"]), int(best["tp"])]]
        heat = go.Figure(
            data=go.Heatmap(
                z=z,
                x=["예측 0 (재직)", "예측 1 (퇴사)"],
                y=["실제 0 (재직)", "실제 1 (퇴사)"],
                colorscale=[[0, "#F5F5F7"], [1, "#5E5CE6"]],
                showscale=False,
                text=z,
                texttemplate="%{text:,}",
                textfont={"size": 18, "color": "#1D1D1F"},
                hovertemplate="%{y} · %{x}: %{z:,}<extra></extra>",
            )
        )
        heat.update_layout(
            height=300,
            margin={"l": 10, "r": 10, "t": 10, "b": 10},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"family": "-apple-system, BlinkMacSystemFont, sans-serif", "color": "#1D1D1F"},
            xaxis={"side": "top"},
            yaxis={"autorange": "reversed"},
        )
        st.plotly_chart(heat, width="stretch", config={"displayModeBar": False})

    alert_box("info", "이 프로젝트의 공통 예측값은 퇴사=1, 재직=0이에요.")


def _render_ml_vs_dl() -> None:
    missing_files = [str(p) for p in (BEST_ML_TEST_METRICS_PATH, DL_METRICS_PATH) if not p.exists()]
    if missing_files:
        alert_box("warning", "비교에 필요한 리포트가 없습니다: " + ", ".join(missing_files))
        return

    ml_report = _load_csv(str(BEST_ML_TEST_METRICS_PATH), BEST_ML_TEST_METRICS_PATH.stat().st_mtime)
    dl_report = _load_csv(str(DL_METRICS_PATH), DL_METRICS_PATH.stat().st_mtime)
    required = {"model", *METRIC_LABELS}
    if not required.issubset(ml_report.columns) or not required.issubset(dl_report.columns):
        alert_box("danger", "머신러닝 또는 딥러닝 보고서에 비교 지표가 부족합니다.")
        return

    best_ml = ml_report.sort_values(["roc_auc", "f1"], ascending=False).iloc[0]
    best_dl = dl_report.sort_values(["roc_auc", "f1"], ascending=False).iloc[0]
    ml_name = MODEL_LABELS.get(str(best_ml["model"]), str(best_ml["model"]))
    dl_name = MODEL_LABELS.get(str(best_dl["model"]), str(best_dl["model"]).upper())

    roc_delta = float(best_ml["roc_auc"] - best_dl["roc_auc"])
    winner = ml_name if roc_delta >= 0 else dl_name
    versus_hero(
        ml_name,
        f"{best_ml['roc_auc']:.4f}",
        dl_name,
        f"{best_dl['roc_auc']:.4f}",
        winner=winner,
    )
    stat_cards(
        [
            {"label": "머신러닝 1위", "value": ml_name, "hint": f"ROC {best_ml['roc_auc']:.4f}"},
            {"label": "딥러닝", "value": dl_name, "hint": f"ROC {best_dl['roc_auc']:.4f}"},
            {"label": "ROC 곡선 면적 우세 모델", "value": winner, "hint": f"차이 {abs(roc_delta):.4f}", "tone": "safe"},
        ]
    )

    comparison = pd.DataFrame(
        [
            {"구분": "최고 머신러닝", "모델": ml_name, **{label: best_ml[key] for key, label in METRIC_LABELS.items()}},
            {"구분": "딥러닝", "모델": dl_name, **{label: best_dl[key] for key, label in METRIC_LABELS.items()}},
        ]
    )
    st.markdown("**최종 성능 비교표**")
    render_table(comparison, formats={label: "{:.4f}" for label in METRIC_LABELS.values()})

    st.markdown("**지표별 비교**")
    metric_labels_list = list(METRIC_LABELS.values())
    ml_values = [float(best_ml[key]) * 100 for key in METRIC_LABELS]
    dl_values = [float(best_dl[key]) * 100 for key in METRIC_LABELS]

    vs_chart_type = _chart_type_picker(
        "그래프 종류", ["레이더 차트", "막대 그래프"], key="model_vs_chart_type"
    )

    overlay = go.Figure()
    if vs_chart_type == "막대 그래프":
        overlay.add_trace(
            go.Bar(
                x=metric_labels_list,
                y=ml_values,
                name=ml_name,
                marker={"color": "#0071E3"},
                text=[f"{v:.1f}" for v in ml_values],
                textposition="outside",
                hovertemplate=f"<b>{ml_name}</b><br>%{{x}}: %{{y:.1f}}<extra></extra>",
            )
        )
        overlay.add_trace(
            go.Bar(
                x=metric_labels_list,
                y=dl_values,
                name=dl_name,
                marker={"color": "#5E5CE6"},
                text=[f"{v:.1f}" for v in dl_values],
                textposition="outside",
                hovertemplate=f"<b>{dl_name}</b><br>%{{x}}: %{{y:.1f}}<extra></extra>",
            )
        )
        overlay.update_layout(
            height=440,
            barmode="group",
            bargap=.28,
            bargroupgap=.1,
            margin={"l": 20, "r": 20, "t": 35, "b": 20},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.9)",
            font=_plotly_font(),
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.08, "xanchor": "center", "x": 0.5},
            xaxis={"title": None, "showgrid": False},
            yaxis={
                "title": "점수 (0~100)",
                "range": [0, min(112, float(max(ml_values + dl_values)) + 12)],
                "gridcolor": "#E8E8ED",
                "zeroline": False,
            },
        )
    else:
        # 지표 6개를 각각 막대쌍으로 나열하는 대신, ML·DL 두 모델의 전체 성능 프로필을
        # 하나의 레이더에 겹쳐 그려서 어느 지표에서 서로 강점이 갈리는지 한 번에 보여준다.
        overlay.add_trace(
            go.Scatterpolar(
                r=ml_values + [ml_values[0]],
                theta=metric_labels_list + [metric_labels_list[0]],
                fill="toself",
                name=ml_name,
                line={"color": "#0071E3", "width": 3},
                fillcolor=_hex_to_rgba("#0071E3", 0.22),
                hovertemplate="%{theta}: %{r:.1f}<extra>" + ml_name + "</extra>",
            )
        )
        overlay.add_trace(
            go.Scatterpolar(
                r=dl_values + [dl_values[0]],
                theta=metric_labels_list + [metric_labels_list[0]],
                fill="toself",
                name=dl_name,
                line={"color": "#5E5CE6", "width": 3},
                fillcolor=_hex_to_rgba("#5E5CE6", 0.22),
                hovertemplate="%{theta}: %{r:.1f}<extra>" + dl_name + "</extra>",
            )
        )
        overlay.update_layout(
            polar={
                "radialaxis": {"visible": True, "range": [0, 100], "gridcolor": "#E8E8ED"},
                "angularaxis": {"gridcolor": "#E8E8ED"},
                "bgcolor": "rgba(255,255,255,0.6)",
            },
            showlegend=True,
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.05, "xanchor": "center", "x": 0.5},
            margin={"l": 50, "r": 50, "t": 30, "b": 30},
            height=460,
            paper_bgcolor="rgba(0,0,0,0)",
            font=_plotly_font(),
        )
    st.plotly_chart(overlay, width="stretch", config={"displayModeBar": False})

    alert_box("info", "두 모델 모두 퇴사=1을 양성 클래스로 평가한 보고서일 때 가장 정확한 비교가 됩니다.")


def _render_training_details() -> None:
    """튜닝 하이퍼파라미터 · 공통 학습 기준. '머신러닝 비교' 탭 안의 접이식 상세로 표시한다.

    이전 버전에는 '학습 인사이트'라는 별도 하위 탭이었지만, 위쪽 최고 모델 카드와
    내용이 겹치고 흐름이 끊겨 이 탭 안의 부가 설명(expander)으로 옮겼다.
    """

    st.caption("퇴사율 측정 근거 · 각 모델이 어떤 설정으로 학습됐는지 확인해요.")
    st.markdown("**튜닝 하이퍼파라미터**")
    param_columns = st.columns(len(TUNED_PARAM_PATHS))
    for column, (name, path) in zip(param_columns, TUNED_PARAM_PATHS.items(), strict=False):
        with column:
            st.markdown(f"**{name}**")
            if path.exists():
                try:
                    params = json.loads(path.read_text(encoding="utf-8"))
                    st.json(params, expanded=False)
                except Exception:
                    st.caption("파일을 읽지 못했어요.")
            else:
                st.caption("튜닝 결과 파일이 없어요.")

    st.markdown("**공통 학습 기준**")
    st.markdown(
        """
        <ul style="margin:0; padding-left:1.1rem; color:var(--muted); font-size:.9rem; line-height:1.7;">
            <li>학습 60% · 검증 20% · 최종 테스트 20%로 계층 분할, 동일 random seed 사용</li>
            <li>수치형은 중앙값 대치 후 표준화, 범주형은 최빈값 대치 후 원-핫 인코딩</li>
            <li>검증 ROC-AUC를 우선 기준, F1 점수를 보조 기준으로 최종 모델 선정</li>
            <li>딥러닝(MLP)은 별도 스케일러·분류 임계값을 저장해 동일 조건으로 평가</li>
        </ul>
        """,
        unsafe_allow_html=True,
    )


def render() -> None:
    section_heading(
        "05 · MODEL OPERATIONS (관리자 전용)",
        "Model Intelligence",
        "Know which model makes the better decision. "
        "ML·DL 모델별 성능과 학습 근거를 확인해요. 인사팀 화면에는 표시되지 않아요.",
    )

    active = sub_tabs(
        [
            ("ml", "머신러닝 비교"),
            ("dl", "딥러닝 성능"),
            ("vs", "ML vs DL 비교"),
        ],
        state_key="model_eval_tab",
    )

    if active == "ml":
        _render_ml_comparison()
    elif active == "dl":
        _render_dl_performance()
    else:
        _render_ml_vs_dl()
