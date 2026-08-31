"""01 · 연봉 협상 지원 탭.

퇴사 예측률과 인재 가치 지수를 함께 보여줘서, "일은 잘하는데 나갈 확률이 높은 사람"과
"굳이 더 안 줘도 되는 사람"을 구분해 협상 우선순위를 정하도록 돕는다. 추가로 조건을
바꿔보는 What-if 시뮬레이터는 화면 하단에 묻히지 않도록, 우하단에 떠 있는 Floating
Action Button을 눌러 모달(st.dialog)로 열람하는 방식으로 제공한다. 값 표시는 대부분
순수 HTML/CSS 컴포넌트로 그리고, 선택·폼처럼 실제 입력이 필요한 부분만 Streamlit
위젯을 사용한다.
"""

from __future__ import annotations

import html
import math

import pandas as pd
import streamlit as st

from src.data.prediction import attrition_probability, prepare_model_input
from src.utils.hr_metrics import (
    FEATURE_LABELS,
    nice_wheel_options,
    risk_badge_tone,
    risk_label,
    translate,
)
from streamlit_ui import (
    alert_box,
    employee_picker,
    hbar_chart,
    section_heading,
    stat_cards,
)
from wheel_picker import wheel_picker_component

SIMULATION_FEATURES = [
    "Monthly Income",
    "Work-Life Balance",
    "Job Satisfaction",
    "Number of Promotions",
    "Overtime",
    "Job Level",
    "Remote Work",
]


def _score_items(employee: pd.Series) -> list[tuple[str, str, float]]:
    """레퍼런스의 8축 패널에 실제 직원 값을 0~100 점수로 환산한다."""

    level_score = {"Entry": 25, "Mid": 55, "Senior": 78, "Manager": 92}.get(
        str(employee.get("Job Level", "")), 50
    )
    tenure_score = min(100.0, float(employee.get("Years at Company", 0)) / 30 * 100)
    return [
        ("학력", "학력 전문성", float(employee.get("학력 점수", 50))),
        ("성과평가", "최근 성과 점수", float(employee.get("성과 점수", 50))),
        ("회사평판", "기업 만족도", float(employee.get("평판 점수", 50))),
        ("재직기간", "재직 기간", float(employee.get("경력 점수", 50))),
        ("리더십기회", "성장 기회", float(employee.get("리더십 점수", 50))),
        ("근무연수", "누적 근속", tenure_score),
        ("직급수준", "현재 직급", float(level_score)),
        ("직원인정", "조직 내 인정도", float(employee.get("인정 점수", 50))),
    ]


def _radar_svg(items: list[tuple[str, str, float]]) -> str:
    cx, cy, radius, count = 150.0, 132.0, 76.0, len(items)

    def point(index: int, distance: float) -> tuple[float, float]:
        angle = math.radians(index * 360 / count - 90)
        return cx + distance * math.cos(angle), cy + distance * math.sin(angle)

    rings = []
    for level in (0.25, 0.5, 0.75, 1.0):
        points = " ".join(f"{x:.1f},{y:.1f}" for x, y in (point(i, radius * level) for i in range(count)))
        rings.append(f'<polygon points="{points}" fill="none" stroke="#E4EAF2" stroke-width="1"/>')
    axes = "".join(
        f'<line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" stroke="#E4EAF2" stroke-width="1"/>'
        for x, y in (point(i, radius) for i in range(count))
    )
    data_points = [point(i, radius * max(0, min(100, item[2])) / 100) for i, item in enumerate(items)]
    polygon = " ".join(f"{x:.1f},{y:.1f}" for x, y in data_points)
    dots = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.6" fill="#EF4444"/>' for x, y in data_points)
    labels = []
    for i, (label, _, _) in enumerate(items):
        x, y = point(i, radius + 23)
        labels.append(
            f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="middle" dominant-baseline="middle" '
            f'font-size="9" fill="#667085">{html.escape(label)}</text>'
        )
    return (
        '<svg viewBox="0 0 300 265" width="100%" height="238" aria-label="역량 방사형 분석">'
        + "".join(rings)
        + axes
        + f'<polygon points="{polygon}" fill="rgba(239,68,68,.12)" stroke="#EF4444" stroke-width="1.5"/>'
        + dots
        + "".join(labels)
        + "</svg>"
    )


def _gauge_svg(value: float) -> str:
    value = max(0.0, min(100.0, value))
    circumference = math.pi * 46
    filled = circumference * value / 100
    color = "#EF4444" if value >= 60 else "#F59E0B" if value >= 35 else "#22C55E"
    return f"""
    <svg viewBox="0 0 132 83" width="132" height="83" aria-label="퇴사 위험도 {value:.0f}">
      <path d="M20 65 A46 46 0 0 1 112 65" fill="none" stroke="#E8EDF4" stroke-width="10" stroke-linecap="round"/>
      <path d="M20 65 A46 46 0 0 1 112 65" fill="none" stroke="{color}" stroke-width="10" stroke-linecap="round"
            stroke-dasharray="{filled:.1f} {circumference + 10:.1f}"/>
      <text x="66" y="57" text-anchor="middle" font-size="22" font-weight="800" fill="{color}">{value:.0f}</text>
      <text x="66" y="70" text-anchor="middle" font-size="7.5" fill="#98A2B3">이탈 위험도</text>
    </svg>
    """


def _feature_color(value: float) -> str:
    return "#22C55E" if value >= 70 else "#F59E0B" if value >= 40 else "#EF4444"


def _wheel_picker(label: str, options: list[int | float], current: int | float, key: str, is_categorical: bool):
    if is_categorical:
        labels = [translate(v) for v in options]
    else:
        labels = [f"{value:,}" if isinstance(value, int) else f"{value:,.1f}" for value in options]
    selected = wheel_picker_component(
        label=label, options=options, labels=labels, value=current, default=current, key=key
    )
    return type(current)(selected)


@st.dialog("What if we change compensation?", width="medium")
def _scenario_dialog(employee: pd.Series, selected_id, train_data: pd.DataFrame, model) -> None:
    st.caption("이 직원의 조건을 조정하면 예측 퇴사 확률이 어떻게 바뀌는지 미리 확인해요.")
    feature_names = list(getattr(model, "feature_names_in_", []))
    editable = [f for f in SIMULATION_FEATURES if f in train_data.columns]
    if not feature_names or not editable:
        alert_box("info", "시뮬레이션에 필요한 모델 정보를 찾지 못했어요.")
        return

    source_row = employee.copy()
    adjusted_values: dict[str, object] = {}
    with st.form(f"salary_scenario_{selected_id}"):
        widget_columns = st.columns(2)
        for index, feature in enumerate(editable):
            reference = train_data[feature].dropna()
            current = source_row[feature]
            with widget_columns[index % 2]:
                if pd.api.types.is_numeric_dtype(reference):
                    current_value = (
                        int(current) if pd.api.types.is_integer_dtype(reference) else float(current)
                    )
                    options = nice_wheel_options(
                        float(reference.min()), float(reference.max()), current_value
                    )
                    adjusted_values[feature] = _wheel_picker(
                        FEATURE_LABELS.get(feature, feature),
                        options,
                        current_value,
                        key=f"salary_scenario_{selected_id}_{feature}",
                        is_categorical=False,
                    )
                else:
                    options = sorted(reference.astype(str).unique().tolist())
                    current_text = str(current)
                    if current_text not in options:
                        options.append(current_text)
                        options.sort()
                    adjusted_values[feature] = st.selectbox(
                        FEATURE_LABELS.get(feature, feature),
                        options,
                        index=options.index(current_text),
                        format_func=lambda value: translate(value),
                        key=f"salary_scenario_{selected_id}_{feature}",
                    )
        submitted = st.form_submit_button("이 조건으로 다시 예측하기", type="primary", width="stretch")

    if submitted:
        adjusted_row = source_row.copy()
        for feature, value in adjusted_values.items():
            if pd.api.types.is_integer_dtype(train_data[feature]):
                value = int(value)
            adjusted_row[feature] = value
        try:
            baseline_input = prepare_model_input(source_row.to_frame().T, train_data, feature_names)
            adjusted_input = prepare_model_input(adjusted_row.to_frame().T, train_data, feature_names)
            before = attrition_probability(model, baseline_input)
            after = attrition_probability(model, adjusted_input)
        except Exception as exc:  # noqa: BLE001
            alert_box("danger", f"예측 중 오류가 발생했어요: {exc}")
        else:
            delta = after - before
            change_text = "위험 감소" if delta < 0 else "위험 증가" if delta > 0 else "변화 없음"
            stat_cards(
                [
                    {"label": "Before", "value": f"{before:.1%}", "hint": risk_label(before), "tone": risk_badge_tone(before)},
                    {"label": "After", "value": f"{after:.1%}", "hint": f"{delta:+.1%}", "tone": risk_badge_tone(after)},
                    {"label": "Change", "value": change_text},
                ]
            )
            hbar_chart([("Before", before * 100), ("After", after * 100)], max_value=100, value_format="{:.1f}%")
            alert_box("warning", "이 결과는 협상 참고용입니다. 개인에 대한 자동 평가나 불이익 부과에 사용하면 안 돼요.")


def render(employees: pd.DataFrame, train_data: pd.DataFrame, model) -> None:
    section_heading(
        "01 · COMPENSATION INTELLIGENCE",
        "Salary Intelligence",
        "개별 직원 ID를 선택해 퇴사 위험과 인재 가치를 확인합니다.",
    )

    selected_id = employee_picker(employees, key_prefix="salary", with_direct_search=True)
    if selected_id is None:
        alert_box("info", "선택한 조건에 해당하는 직원이 없어요. 부서 또는 직급을 바꿔보세요.")
        return

    employee = employees.loc[employees["Employee ID"].eq(selected_id)].iloc[0]
    risk = float(employee["prediction"])
    talent = float(employee["인재 가치 지수"])

    if risk >= 0.6 and talent >= 70:
        message = "퇴사 위험과 인재 가치가 모두 높습니다. 보상 수준 및 성장 경로 면담을 우선 검토하세요."
    elif risk >= 0.6:
        message = "퇴사 위험이 높습니다. 보상 외 업무 만족도와 워라밸 요인을 함께 확인하세요."
    else:
        message = "현재 잔류 가능성이 비교적 안정적입니다. 성과와 역할 확장 가능성을 중심으로 협상하세요."

    score_items = _score_items(employee)
    feature_rows = "".join(
        (
            '<div class="salary-feature">'
            '<div class="salary-feature-name">'
            f'<span><b>{html.escape(name)}</b><small>{html.escape(subtitle)}</small></span>'
            f'<span style="color:{_feature_color(value)};font-weight:800">{value:.0f}</span>'
            '</div><div class="salary-feature-track">'
            f'<div class="salary-feature-fill" style="width:{max(0, min(100, value)):.1f}%;background:{_feature_color(value)}"></div>'
            '</div></div>'
        )
        for name, subtitle, value in score_items
    )
    risk_pct = risk * 100
    risk_color = _feature_color(100 - risk_pct)
    risk_badge = "위험 — HIGH" if risk >= 0.6 else "주의 — MID" if risk >= 0.35 else "안전 — LOW"

    with st.container(key="salary-intelligence-card"):
        st.markdown(
            f"""
            <div class="salary-intel-head">
              <div class="salary-intel-employee"><span class="reference-label">EMPLOYEE</span><span class="salary-intel-id">#{int(selected_id)}</span></div>
              <div class="salary-intel-meta">
                <span>부서 <b>{html.escape(translate(employee['Job Role']))}</b></span>
                <span>직급 <b>{html.escape(translate(employee['Job Level']))}</b></span>
                <span>연봉 <b>₩{float(employee['Monthly Income']):,.0f}/월</b></span>
              </div>
            </div>
            <div class="salary-intel-body">
              <div class="salary-risk-column">
                <span class="reference-label">이탈 위험도</span>
                <div class="salary-gauge">{_gauge_svg(risk_pct)}</div>
                <div style="text-align:center;margin-top:-.25rem"><span class="risk-chip {'danger' if risk >= .6 else 'warning' if risk >= .35 else 'safe'}">{risk_badge}</span></div>
                <div class="salary-talent-row"><span style="color:#667085">인재 가치</span><b style="color:{risk_color}">{talent:.0f}</b></div>
                <div class="salary-ai-note"><b>AI 권고</b>{html.escape(message)}</div>
              </div>
              <div class="salary-radar-column">
                <div class="reference-label">역량 방사형 분석</div>
                <div class="reference-card-subtitle">8개 이탈 예측 피처 종합 분포</div>
                <div class="salary-radar-wrap">{_radar_svg(score_items)}</div>
              </div>
              <div class="salary-feature-column">
                <div><div class="reference-label">개별 피처 점수</div></div>
                {feature_rows}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.container(key="salary-sim-fab"):
            fab_clicked = st.button(
                "👩🏻‍💼 협상 제안 — 이 조건을 바꾸면 어떻게 될까요?",
                key="salary_sim_fab_btn",
                help="What-if 시뮬레이션을 모달로 열어요",
                width="stretch",
            )
    if fab_clicked:
        _scenario_dialog(employee, selected_id, train_data, model)
