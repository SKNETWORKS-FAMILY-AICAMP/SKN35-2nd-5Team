"""04 · 인사 구조 안정도 탭.

전사 관점에서 "지금 조직이 얼마나 안정적인가"를 보여준다. 첫 화면은 안정 지수를
하나의 큰 Organization Health 내러티브로 보여주고, 그 아래로 이탈률과 연관이 높은
주요 피처, 근속연차별 퇴사 추세, 부서별 안정도를 순서대로 보여준다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.utils.hr_metrics import (
    DRIVER_FEATURE_CANDIDATES,
    FEATURE_LABELS,
    attrition_rate_by_feature,
    department_options,
    department_overview,
    rank_attrition_drivers,
    stability_index,
    translate,
)
from streamlit_ui import (
    alert_box,
    department_cards,
    hbar_chart,
    section_heading,
)


def render(employees: pd.DataFrame) -> None:
    section_heading(
        "04 · INCLUSION & WELLBEING MONITOR",
        "Organization Health",
        "전사·부서별로 퇴사율을 이해하고, 가장 위험한 그룹과 시기를 감지해 미리 대응할 수 있어요.",
    )

    department_choices = ["전체", *department_options(employees)]
    st.session_state.setdefault("stability_department_filter", "전체")
    selected_department = st.session_state["stability_department_filter"]
    scoped = employees if selected_department == "전체" else employees[employees["Job Role"].eq(selected_department)]

    actual_attrition = scoped["Attrition"].astype(str).eq("Left")
    stability = stability_index(scoped)
    if stability >= 70:
        status_label, status_class = "안전", "safe"
    elif stability >= 50:
        status_label, status_class = "위험", "warning"
    else:
        status_label, status_class = "고위험", "danger"

    department_rates = scoped.groupby("Job Role")["Attrition"].apply(lambda col: col.astype(str).eq("Left").mean())
    avg_dept_attrition = float(department_rates.mean()) if not department_rates.empty else 0.0
    scope_label = "전사" if selected_department == "전체" else translate(selected_department)
    st.markdown(
        f"""
        <div class="reference-card health-score-card">
          <div class="health-score-label">ORGANIZATION HEALTH SCORE ({scope_label})</div>
          <div class="health-score-value">{stability:.0f}</div>
          <span class="risk-chip {status_class}">{status_label}</span>
          <div class="health-score-note">Current threshold: Attrition Risk above 15% puts the org in the Danger zone for stability.</div>
        </div>
        <div class="reference-kpis" style="margin-top:.7rem">
          <div class="reference-card reference-kpi"><div class="reference-label">Actual Attrition</div><div class="reference-kpi-value">{actual_attrition.mean():.1%}</div></div>
          <div class="reference-card reference-kpi"><div class="reference-label">Total Employees</div><div class="reference-kpi-value">{len(scoped):,}</div></div>
          <div class="reference-card reference-kpi"><div class="reference-label">Avg. Dept Attrition</div><div class="reference-kpi-value">{avg_dept_attrition:.1%}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    driver_table = rank_attrition_drivers(scoped, DRIVER_FEATURE_CANDIDATES)
    if driver_table.empty:
        alert_box("info", "피처별 퇴사율 분석에 필요한 데이터가 부족해요.")
    else:
        with st.container(key="health-driver-card"):
            st.markdown('<div class="reference-label">BURNOUT IMPACT · 이탈률을 가장 많이 올리는 요인</div>', unsafe_allow_html=True)
            hbar_chart(
                list(zip(driver_table["피처"], driver_table["영향도(편차)"] * 100, strict=False)),
                max_value=float(driver_table["영향도(편차)"].max() * 100) or 1.0,
                value_format="{:.1f}%", color="linear-gradient(90deg,#93C5FD,#2563EB)", numbered=True,
            )

    st.radio(
        "부서 필터",
        department_choices,
        horizontal=True,
        format_func=lambda value: value if value == "전체" else translate(value),
        key="stability_department_filter",
    )

    tenure = pd.to_numeric(scoped["Years at Company"], errors="coerce")
    tenure_bins = pd.cut(
        tenure,
        bins=[-1, 1, 3, 5, 10, 20, np.inf],
        labels=["0–1년", "2–3년", "4–5년", "6–10년", "11–20년", "21년 이상"],
    )
    tenure_trend = (
        pd.DataFrame({"근속 구간": tenure_bins, "퇴사 여부": actual_attrition})
        .groupby("근속 구간", observed=True)["퇴사 여부"]
        .mean()
    )
    max_trend = max((float(value) for value in tenure_trend.values), default=1.0) or 1.0
    bars = "".join(
        f'<div class="health-trend-item"><span class="health-trend-value">{float(value):.1%}</span>'
        f'<div class="health-trend-bar" style="height:{max(8, float(value) / max_trend * 82):.1f}px;opacity:{1 if i == len(tenure_trend)-1 else .65}"></div>'
        f'<span class="health-trend-year">{index}</span></div>'
        for i, (index, value) in enumerate(tenure_trend.items())
    )
    st.markdown(
        f'<div class="reference-card health-trend-card"><div class="reference-label">ATTRITION TREND · 근속연차별 퇴사율</div><div class="health-trend-bars">{bars}</div></div>',
        unsafe_allow_html=True,
    )

    with st.expander("상세 구간·부서 분석", expanded=False):
        if not driver_table.empty:
            default_feature = driver_table.iloc[0]["_원본"]
            options = driver_table["_원본"].tolist()
            chosen = st.selectbox("피처 선택", options, index=options.index(default_feature), format_func=lambda f: FEATURE_LABELS.get(f, f), key="stability_driver_select")
            feature_table = attrition_rate_by_feature(scoped, chosen)
            hbar_chart(list(zip(feature_table["구간"], feature_table["퇴사율"] * 100, strict=False)), max_value=100, value_format="{:.1f}%", color="var(--blue)")
        department_table = department_overview(scoped)
        dept_rows = []
        for _, row in department_table.iterrows():
            fields = [("Employees", f"{int(row['직원수']):,}"), ("Attrition Risk", f"{float(row['평균_퇴사위험']):.1%}"), ("Talent Value", f"{float(row['평균_인재가치']):.1f}")]
            if "평균_만족도" in department_table.columns:
                fields.append(("Job Satisfaction", f"{float(row['평균_만족도']):.1f}"))
            dept_rows.append({"title": row["부서"], "fields": fields})
        department_cards(dept_rows)
