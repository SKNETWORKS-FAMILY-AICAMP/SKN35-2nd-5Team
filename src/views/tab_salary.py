"""01 · 연봉 협상 지원 탭.

퇴사 예측률과 인재 가치 지수를 함께 보여줘서, "일은 잘하는데 나갈 확률이 높은 사람"과
"굳이 더 안 줘도 되는 사람"을 구분해 협상 우선순위를 정하도록 돕는다. 추가로 조건을
바꿔보는 What-if 시뮬레이터는 화면 하단에 묻히지 않도록, 우하단에 떠 있는 Floating
Action Button을 눌러 모달(st.dialog)로 열람하는 방식으로 제공한다. 값 표시는 대부분
순수 HTML/CSS 컴포넌트로 그리고, 선택·폼처럼 실제 입력이 필요한 부분만 Streamlit
위젯을 사용한다.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.data.prediction import attrition_probability, prepare_model_input
from src.utils.hr_metrics import (
    FEATURE_LABELS,
    RISK_FEATURES,
    TALENT_FEATURE_LABELS,
    nice_wheel_options,
    risk_badge_tone,
    risk_label,
    translate,
)
from streamlit_ui import (
    alert_box,
    employee_hero,
    employee_picker,
    feature_pills,
    hbar_chart,
    render_table,
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
        "Protect high-value talent before they leave. "
        "직원을 부서 → 직급 → ID 순으로 좁혀 선택하면, 퇴사 확률과 인재 가치 지수를 함께 확인할 수 있어요.",
    )

    selected_id = employee_picker(employees, key_prefix="salary", with_direct_search=True)
    if selected_id is None:
        alert_box("info", "선택한 조건에 해당하는 직원이 없어요. 부서 또는 직급을 바꿔보세요.")
        return

    employee = employees.loc[employees["Employee ID"].eq(selected_id)].iloc[0]
    risk = float(employee["prediction"])
    talent = float(employee["인재 가치 지수"])
    label = risk_label(risk)
    tone = risk_badge_tone(risk)

    employee_hero(
        employee_id=selected_id,
        department=employee["Job Role"],
        level=employee["Job Level"],
        risk_pct=risk,
        risk_label_text=label.upper(),
        risk_tone=tone,
        talent_score=talent,
        extra_badges=[f"${float(employee['Monthly Income']):,.0f} / mo"],
    )
    st.markdown('<div class="section-spacer-lg"></div>', unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        st.markdown("**Current Profile · 퇴사 예측 활용 항목**")
        feature_pills(RISK_FEATURES, FEATURE_LABELS)
        risk_display = pd.DataFrame(
            {
                "항목": [FEATURE_LABELS[feature] for feature in RISK_FEATURES],
                "현재 값": [translate(employee[feature]) for feature in RISK_FEATURES],
            }
        )
        render_table(risk_display, widths={"항목": "42%", "현재 값": "58%"})
    with right:
        st.markdown("**Talent Value · 인재 가치 구성**")
        feature_pills(list(TALENT_FEATURE_LABELS.keys()), FEATURE_LABELS)
        hbar_chart(
            [
                ("학력", float(employee["학력 점수"])),
                ("성과 평가", float(employee["성과 점수"])),
                ("회사 평판", float(employee["평판 점수"])),
                ("총 경력연수", float(employee["경력 점수"])),
                ("리더십 기회", float(employee["리더십 점수"])),
            ],
            max_value=100,
            color="var(--blue)",
            # 왼쪽 "Current Profile" 표는 6행이라 오른쪽(5개 막대)보다 항상 더 길다.
            # 막대 5개를 이 높이 안에 고르게 펼쳐서, 위쪽에만 몰려 보이지 않게 한다.
            min_height="341px",
        )

    if risk >= 0.6 and talent >= 70:
        message = "퇴사 위험과 인재 가치가 모두 높습니다. 보상 수준 및 성장 경로 면담을 우선 검토하세요."
    elif risk >= 0.6:
        message = "퇴사 위험이 높습니다. 보상 외 업무 만족도와 워라밸 요인을 함께 확인하세요."
    else:
        message = "현재 잔류 가능성이 비교적 안정적입니다. 성과와 역할 확장 가능성을 중심으로 협상하세요."
    alert_box("info", message, title="협상 제안")

    with st.container(key="salary-sim-fab"):
        fab_clicked = st.button("＋ Simulate", key="salary_sim_fab_btn", help="What-if 시뮬레이션을 모달로 열어요")
    if fab_clicked:
        _scenario_dialog(employee, selected_id, train_data, model)
