"""04 · 인사 구조 안정도 탭.

전사 관점에서 "지금 조직이 얼마나 안정적인가"를 보여준다. 이탈률과 연관이 높은 주요
피처를 골라 값(구간)별 퇴사율을 보여주고, 직군별 만족도·인재가치 같은 부가 지표까지
함께 표기해 전체 인사 구조 현황을 한 화면에서 파악할 수 있게 한다.
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
from streamlit_ui import alert_box, hbar_chart, render_table, section_heading, stat_cards


def render(employees: pd.DataFrame) -> None:
    section_heading(
        "04 · EXECUTIVE HR OVERVIEW",
        "전사 인사 구조 안정도",
        "전사 퇴사 위험과 조직별 신호를 모니터링해 선제 대응 대상을 찾아요.",
    )

    actual_attrition = employees["Attrition"].astype(str).eq("Left")
    high_risk = employees["prediction"].ge(0.6)
    key_talent = employees["인재 가치 지수"].ge(75)
    stability = stability_index(employees)
    stability_tone = "safe" if stability >= 70 else "warning" if stability >= 50 else "danger"

    stat_cards(
        [
            {"label": "현재 데이터 퇴사율", "value": f"{actual_attrition.mean():.1%}"},
            {"label": "고위험 인원", "value": f"{int(high_risk.sum()):,}명"},
            {
                "label": "핵심 인재 잔류 가능성",
                "value": (
                    f"{(1 - employees.loc[key_talent, 'prediction'].mean()):.1%}" if key_talent.any() else "N/A"
                ),
            },
            {"label": "전사 안정 지수", "value": f"{stability:.1f} / 100", "tone": stability_tone},
        ]
    )

    st.markdown("**이탈률과 연관 높은 주요 피처**")
    driver_table = rank_attrition_drivers(employees, DRIVER_FEATURE_CANDIDATES)
    if driver_table.empty:
        alert_box("info", "피처별 퇴사율 분석에 필요한 데이터가 부족해요.")
    else:
        hbar_chart(
            list(zip(driver_table["피처"], driver_table["영향도(편차)"] * 100, strict=False)),
            max_value=float(driver_table["영향도(편차)"].max() * 100) or 1.0,
            value_format="{:.1f}%",
            color="var(--blue-deep)",
        )
        st.caption("막대가 길수록 값(구간)에 따라 퇴사율 편차가 커서, 이탈에 영향이 큰 피처라는 뜻이에요.")

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

    st.markdown("**근속연차별 퇴사율**")
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
    hbar_chart(
        [(str(idx), float(val) * 100) for idx, val in tenure_trend.items()],
        max_value=100,
        value_format="{:.1f}%",
    )

    st.markdown('<div class="section-divider-thin"></div>', unsafe_allow_html=True)

    st.markdown("**부서별 안정도 (부가 지표)**")
    department_table = department_overview(employees)
    formats = {"평균_퇴사위험": "{:.1%}", "평균_인재가치": "{:.1f}"}
    bars = {"평균_인재가치": 100.0}
    if "평균_만족도" in department_table.columns:
        formats["평균_만족도"] = "{:.1f}"
        bars["평균_만족도"] = 100.0
    render_table(department_table, formats=formats, bars=bars)
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
