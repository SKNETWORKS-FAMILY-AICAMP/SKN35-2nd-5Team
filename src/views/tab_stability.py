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
    department_overview,
    rank_attrition_drivers,
    stability_index,
)
from streamlit_ui import (
    alert_box,
    area_trend_chart,
    department_cards,
    hbar_chart,
    narrative_banner,
    section_heading,
    stat_cards,
)


def render(employees: pd.DataFrame) -> None:
    section_heading(
        "04 · EXECUTIVE HR OVERVIEW",
        "Organization Health",
        "전사 퇴사 위험과 조직별 신호를 하나의 큰 그림으로 확인하고, 선제 대응 대상을 찾아요.",
    )

    actual_attrition = employees["Attrition"].astype(str).eq("Left")
    high_risk = employees["prediction"].ge(0.6)
    key_talent = employees["인재 가치 지수"].ge(75)
    stability = stability_index(employees)
    if stability >= 70:
        stability_tone, status_label = "safe", "Stable"
        narrative_message = "Your organization is showing strong retention signals."
    elif stability >= 50:
        stability_tone, status_label = "warning", "Needs Attention"
        narrative_message = "Some teams show early signs of attrition risk worth watching."
    else:
        stability_tone, status_label = "danger", "At Risk"
        narrative_message = "Multiple teams show elevated attrition risk that needs attention."

    narrative_banner(
        "ORGANIZATION HEALTH",
        f"{stability:.0f}",
        status_label,
        stability_tone,
        narrative_message,
    )

    stat_cards(
        [
            {"label": "Actual Attrition", "value": f"{actual_attrition.mean():.1%}"},
            {"label": "High-Risk Headcount", "value": f"{int(high_risk.sum()):,}"},
            {
                "label": "Key Talent Retention",
                "value": (
                    f"{(1 - employees.loc[key_talent, 'prediction'].mean()):.1%}" if key_talent.any() else "N/A"
                ),
            },
        ]
    )

    st.markdown('<div class="section-divider-thin"></div>', unsafe_allow_html=True)
    st.markdown("**Feature Impact · 이탈률과 연관 높은 주요 피처**")
    driver_table = rank_attrition_drivers(employees, DRIVER_FEATURE_CANDIDATES)
    if driver_table.empty:
        alert_box("info", "피처별 퇴사율 분석에 필요한 데이터가 부족해요.")
    else:
        hbar_chart(
            list(zip(driver_table["피처"], driver_table["영향도(편차)"] * 100, strict=False)),
            max_value=float(driver_table["영향도(편차)"].max() * 100) or 1.0,
            value_format="{:.1f}%",
            color="var(--blue-deep)",
            numbered=True,
        )
        st.caption("숫자가 클수록 값(구간)에 따라 퇴사율 편차가 커서, 이탈에 영향이 큰 피처라는 뜻이에요.")

        st.markdown('<div class="section-spacer-lg"></div>', unsafe_allow_html=True)
        st.markdown("**구간별 퇴사율 확인**")
        default_feature = driver_table.iloc[0]["_원본"]
        options = driver_table["_원본"].tolist()
        chosen = st.selectbox(
            "피처 선택",
            options,
            index=options.index(default_feature),
            format_func=lambda f: FEATURE_LABELS.get(f, f),
            key="stability_driver_select",
        )
        feature_table = attrition_rate_by_feature(employees, chosen)
        hbar_chart(
            list(zip(feature_table["구간"], feature_table["퇴사율"] * 100, strict=False)),
            max_value=100,
            value_format="{:.1f}%",
            color="var(--blue)",
        )
        st.caption(f"{FEATURE_LABELS.get(chosen, chosen)} 값(구간)별 실제 퇴사율")

    st.markdown('<div class="section-divider-thin"></div>', unsafe_allow_html=True)

    st.markdown("**Attrition Trend · 근속연차별 퇴사율**")
    tenure = pd.to_numeric(employees["Years at Company"], errors="coerce")
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
    area_trend_chart(
        [(str(idx), float(val) * 100) for idx, val in tenure_trend.items()],
        value_format="{:.1f}%",
    )

    st.markdown('<div class="section-divider-thin"></div>', unsafe_allow_html=True)

    st.markdown("**Department Stability · 부서별 안정도**")
    department_table = department_overview(employees)
    dept_rows = []
    for _, row in department_table.iterrows():
        fields = [
            ("Employees", f"{int(row['직원수']):,}"),
            ("Attrition Risk", f"{float(row['평균_퇴사위험']):.1%}"),
            ("Talent Value", f"{float(row['평균_인재가치']):.1f}"),
        ]
        if "평균_만족도" in department_table.columns:
            fields.append(("Job Satisfaction", f"{float(row['평균_만족도']):.1f}"))
        dept_rows.append({"title": row["부서"], "fields": fields})
    department_cards(dept_rows)
    st.caption("만족도 점수는 직무 만족도(낮음부터 매우 높음까지)를 0부터 100 사이 값으로 환산한 값이에요.")

    alert_box(
        "info",
        "현재 데이터에는 기준 연도·퇴사일 컬럼이 없어 실제 연도별 추세를 만들 수 없어요. "
        "위 근속연차별 퇴사율을 대체 지표로 확인하세요.",
    )
    alert_box(
        "success",
        "연도별 추세 모니터링을 위해 원천 데이터에 기준 연도 또는 퇴사일 컬럼을 추가하세요.",
        title="운영 제안",
    )
