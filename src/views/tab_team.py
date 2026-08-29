"""02 · 팀 구성 탭.

IT 회사는 프로젝트 단위로 팀을 꾸리는 일이 잦다. 예전 버전은 "적합 점수 상위 N명"을
그대로 추천하다 보니, 데이터 특성상 퇴사율이 낮은 시니어 위주로만 추천되는 문제가
있었다. 이번 버전은 반대로, 사용자가 먼저 "어떤 팀을 만들고 싶은지"(부서별·직급별
필요 인원)를 직접 정하고, 그 슬롯 하나하나를 인재 가치·잔류 가능성 기준 최적 인원으로
채우는 방식으로 바꿨다. 추천 결과가 마음에 들지 않으면 한 명을 골라 다른 후보로
교체해볼 수도 있다.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.utils.hr_metrics import (
    LEVEL_KR,
    LEVEL_ORDER,
    TALENT_VALUE_EXPLANATION,
    alternative_candidates,
    department_options,
    fill_team_slots,
    risk_badge_tone,
    team_fit_score,
    team_stability_verdict,
    translate,
)
from streamlit_ui import alert_box, render_table, section_heading, stat_cards


def _slot_key(department: str) -> str:
    return f"team_slot_{department}"


def render(employees: pd.DataFrame) -> None:
    section_heading(
        "02 · TEAM COMPOSITION",
        "프로젝트 팀 구성",
        "필요한 부서·직급별 인원을 먼저 정하면, 슬롯마다 가장 적합한 인원을 채워 드려요.",
    )

    with st.expander("ℹ️ 인재 가치 지수는 어떻게 계산되나요?", expanded=False):
        st.caption(TALENT_VALUE_EXPLANATION)

    all_departments = department_options(employees)
    selected_departments = st.multiselect(
        "팀에 포함할 부서",
        all_departments,
        default=all_departments[:1],
        format_func=lambda d: translate(d),
        key="team_departments",
    )

    if not selected_departments:
        alert_box("info", "팀에 포함할 부서를 하나 이상 선택해 주세요.")
        return

    st.markdown("**부서별 필요 직급 인원**")
    level_counts_by_dept: dict[str, dict[str, int]] = {}
    for department in selected_departments:
        st.markdown(f'<div class="feature-pill" style="margin-bottom:.4rem;">{translate(department)}</div>', unsafe_allow_html=True)
        cols = st.columns(3)
        counts: dict[str, int] = {}
        for col, level in zip(cols, LEVEL_ORDER, strict=False):
            with col:
                counts[level] = st.number_input(
                    LEVEL_KR[level],
                    min_value=0,
                    max_value=20,
                    value=1 if level == "Mid" else 0,
                    step=1,
                    key=f"{_slot_key(department)}_{level}",
                )
        level_counts_by_dept[department] = counts

    total_needed = sum(sum(counts.values()) for counts in level_counts_by_dept.values())

    talent_weight = st.slider(
        "인재 가치 비중",
        0.3,
        0.9,
        0.6,
        0.1,
        key="team_talent_weight",
        help=(
            "슬라이더를 오른쪽으로 옮길수록 인재 가치 지수를 더 많이 반영하고, "
            "왼쪽으로 옮길수록 '잔류 가능성(퇴사 확률이 낮을수록 높은 점수)'을 더 많이 반영해요. "
            "예: 70%로 두면 인재 가치 70% + 잔류 가능성 30%로 팀원을 골라요."
        ),
    )

    if total_needed == 0:
        alert_box("info", "직급별 필요 인원을 1명 이상 입력해 주세요.")
        return

    swap_map: dict = st.session_state.setdefault("team_swap_map", {})

    picks = []
    for department, counts in level_counts_by_dept.items():
        picked = fill_team_slots(employees, department, counts, talent_weight)
        if not picked.empty:
            picks.append(picked)
    if not picks:
        alert_box("warning", "조건에 맞는 인원을 찾지 못했어요. 필요 인원 수를 줄여보세요.")
        return
    recommended_team = pd.concat(picks, ignore_index=True)

    # 사용자가 교체를 선택한 인원이 있으면 반영한다.
    for original_id, replacement_id in list(swap_map.items()):
        if original_id not in recommended_team["Employee ID"].values:
            continue
        replacement_rows = employees.loc[employees["Employee ID"].eq(replacement_id)]
        if replacement_rows.empty:
            continue
        replacement_row = replacement_rows.iloc[0].copy()
        slot_department = recommended_team.loc[
            recommended_team["Employee ID"].eq(original_id), "Job Role"
        ].iloc[0]
        replacement_row["팀 적합 점수"] = float(
            team_fit_score(replacement_row.to_frame().T, talent_weight).iloc[0]
        )
        recommended_team = recommended_team.loc[~recommended_team["Employee ID"].eq(original_id)]
        recommended_team = pd.concat(
            [recommended_team, replacement_row.to_frame().T], ignore_index=True
        )
        del slot_department  # 부서는 원래 슬롯과 동일 부서 후보만 대체 후보로 노출되므로 그대로 둔다.

    mean_risk = float(recommended_team["prediction"].mean())
    verdict_label, verdict_tone, verdict_note = team_stability_verdict(mean_risk)
    high_risk_count = int(recommended_team["prediction"].ge(0.6).sum())

    stat_cards(
        [
            {"label": "구성된 팀원", "value": f"{len(recommended_team)}명 / {total_needed}명"},
            {"label": "평균 인재 가치", "value": f"{recommended_team['인재 가치 지수'].mean():.1f}"},
            {"label": "평균 퇴사 위험", "value": f"{mean_risk:.1%}", "tone": verdict_tone},
            {"label": "고위험 팀원", "value": f"{high_risk_count}명", "tone": "danger" if high_risk_count else "safe"},
        ]
    )
    st.markdown('<div class="section-spacer-lg"></div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="verdict-badge tone-{verdict_tone}">이 팀의 안정도 · {verdict_label}</div>',
        unsafe_allow_html=True,
    )
    alert_box("info" if verdict_tone in ("safe", "info") else verdict_tone, verdict_note)

    display_team = recommended_team.sort_values(["Job Role", "Job Level"]).reset_index(drop=True)
    team_table = display_team[
        ["Employee ID", "Job Role", "Job Level", "인재 가치 지수", "prediction", "팀 적합 점수"]
    ].rename(
        columns={
            "Employee ID": "직원 ID",
            "Job Role": "부서",
            "Job Level": "직급",
            "prediction": "퇴사 예측률",
        }
    )
    team_table["부서"] = team_table["부서"].map(translate)
    team_table["직급"] = team_table["직급"].map(translate)
    render_table(
        team_table,
        formats={"인재 가치 지수": "{:.1f}", "퇴사 예측률": "{:.1%}", "팀 적합 점수": "{:.1f}"},
        badges={"퇴사 예측률": lambda v: (f"{v:.1%}", risk_badge_tone(v))},
        bars={"인재 가치 지수": 100.0, "팀 적합 점수": 100.0},
    )
    st.caption(f"팀 적합 점수 = 인재 가치 {talent_weight:.0%} + 잔류 가능성 {1 - talent_weight:.0%}")

    st.markdown('<div class="section-divider-thin"></div>', unsafe_allow_html=True)
    st.markdown("**팀원 교체해보기**")
    st.caption("추천된 팀원 중 한 명을 골라 같은 부서·직급의 다른 후보로 바꿔볼 수 있어요.")

    swap_target_labels = {
        int(row["Employee ID"]): f"ID {int(row['Employee ID'])} · {translate(row['Job Role'])} · {translate(row['Job Level'])}"
        for _, row in display_team.iterrows()
    }
    swap_cols = st.columns([2, 2, 1])
    with swap_cols[0]:
        target_id = st.selectbox(
            "교체할 팀원",
            list(swap_target_labels.keys()),
            format_func=lambda i: swap_target_labels.get(i, str(i)),
            key="team_swap_target",
        )
    target_row = display_team.loc[display_team["Employee ID"].eq(target_id)].iloc[0]
    exclude_ids = set(recommended_team["Employee ID"].tolist())
    alternatives = alternative_candidates(
        employees, target_row["Job Role"], target_row["Job Level"], talent_weight, exclude_ids
    )
    with swap_cols[1]:
        if alternatives.empty:
            st.selectbox("대체 후보", ["대체 후보 없음"], key="team_swap_candidate_empty", disabled=True)
            replacement_id = None
        else:
            alt_labels = {
                int(row["Employee ID"]): (
                    f"ID {int(row['Employee ID'])} · 적합 {row['팀 적합 점수']:.1f} · 인재가치 {row['인재 가치 지수']:.1f}"
                )
                for _, row in alternatives.iterrows()
            }
            replacement_id = st.selectbox(
                "대체 후보",
                list(alt_labels.keys()),
                format_func=lambda i: alt_labels.get(i, str(i)),
                key="team_swap_candidate",
            )
    with swap_cols[2]:
        st.markdown("<div style='height:1.9rem'></div>", unsafe_allow_html=True)
        if st.button("교체하기", key="team_swap_apply", width="stretch", disabled=replacement_id is None):
            swap_map[target_id] = replacement_id
            st.rerun()

    if swap_map:
        if st.button("교체 초기화", key="team_swap_reset"):
            st.session_state["team_swap_map"] = {}
            st.rerun()
