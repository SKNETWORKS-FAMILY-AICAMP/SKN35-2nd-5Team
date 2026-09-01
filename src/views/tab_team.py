"""02 · 팀 구성 탭.

IT 회사는 프로젝트 단위로 팀을 꾸리는 일이 잦다. 예전 버전은 "적합 점수 상위 N명"을
그대로 추천하다 보니, 데이터 특성상 퇴사율이 낮은 시니어 위주로만 추천되는 문제가
있었다. 이번 버전은 반대로, 사용자가 먼저 "어떤 팀을 만들고 싶은지"(부서별·직급별
필요 인원)를 직접 정하고, 그 슬롯 하나하나를 인재 가치·잔류 가능성 기준 최적 인원으로
채우는 방식으로 바꿨다. 추천 결과가 마음에 들지 않으면 한 명을 골라 다른 후보로
교체해볼 수도 있다.
"""

from __future__ import annotations

import html

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.utils.hr_metrics import (
    LEVEL_KR,
    LEVEL_ORDER,
    TALENT_VALUE_EXPLANATION,
    alternative_candidates,
    department_options,
    fill_team_slots,
    team_fit_score,
    team_stability_verdict,
    translate,
)
from streamlit_ui import alert_box, section_heading


def _slot_key(department: str) -> str:
    return f"team_slot_{department}"


@st.dialog("팀원 교체해보기", width="medium")
def _swap_dialog(
    employees: pd.DataFrame,
    display_team: pd.DataFrame,
    recommended_team: pd.DataFrame,
    talent_weight: float,
    swap_map: dict,
) -> None:
    """추천된 팀원 중 한 명을 같은 부서·직급의 다른 후보로 바꿔보는 모달."""

    st.caption("추천된 팀원 중 한 명을 골라 같은 부서·직급의 다른 후보로 바꿔볼 수 있어요.")

    swap_target_labels = {
        int(row["Employee ID"]): f"ID {int(row['Employee ID'])} · {translate(row['Job Role'])} · {translate(row['Job Level'])}"
        for _, row in display_team.iterrows()
    }
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

    action_cols = st.columns(2)
    with action_cols[0]:
        if st.button(
            "교체하기",
            key="team_swap_apply",
            type="primary",
            width="stretch",
            disabled=replacement_id is None,
        ):
            swap_map[target_id] = replacement_id
            st.rerun()
    with action_cols[1]:
        if st.button("교체 초기화", key="team_swap_reset", width="stretch", disabled=not swap_map):
            st.session_state["team_swap_map"] = {}
            st.rerun()


def render(employees: pd.DataFrame) -> None:
    section_heading(
        "02 · TEAM CONFIGURATION",
        "프로젝트 팀 구성 최적화",
        "프로젝트 요구 기술 스택 충족도와 팀 전체 평균 이탈 위험도를 고려한 AI 팀 추천",
    )

    left, right = st.columns([.95, 2.05], gap="medium")
    with left:
        with st.container(key="team-condition-card"):
            st.markdown('<div class="reference-card-title">프로젝트 조건 <span style="color:#98A2B3">ⓘ</span></div>', unsafe_allow_html=True)
            project_name = st.text_input("프로젝트명", value="2026 차세대 AI 플랫폼 구축", key="team_project_name")
            all_departments = department_options(employees)
            selected_departments = st.multiselect(
                "필요 핵심 부서",
                all_departments,
                default=all_departments[:1],
                format_func=lambda d: translate(d),
                key="team_departments",
            )

    if not selected_departments:
        alert_box("info", "팀에 포함할 부서를 하나 이상 선택해 주세요.")
        return

    level_counts_by_dept: dict[str, dict[str, int]] = {}
    with left:
        with st.container(key="team-condition-levels"):
            st.caption("부서별 필요 직급 인원")
            for department in selected_departments:
                st.markdown(f'<div class="reference-label" style="margin:.35rem 0">{html.escape(translate(department))}</div>', unsafe_allow_html=True)
                cols = st.columns(3)
                counts: dict[str, int] = {}
                for col, level in zip(cols, LEVEL_ORDER, strict=False):
                    with col:
                        counts[level] = st.number_input(
                            LEVEL_KR[level], min_value=0, max_value=20,
                            value=4 if level == "Mid" else 0, step=1,
                            key=f"{_slot_key(department)}_{level}",
                        )
                level_counts_by_dept[department] = counts
            talent_weight = st.slider(
                "인재 가치 반영 비중 (%)", 30, 90, 60, 10,
                key="team_talent_weight_pct",
                help="나머지 비중은 잔류 가능성에 적용됩니다.",
            ) / 100

    total_needed = sum(sum(counts.values()) for counts in level_counts_by_dept.values())

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
    verdict_label, _, verdict_note = team_stability_verdict(mean_risk)
    display_team = recommended_team.sort_values(["Job Role", "Job Level"]).reset_index(drop=True)
    roster_rows = []
    for _, row in display_team.iterrows():
        member_risk = float(row["prediction"])
        member_class = "safe" if member_risk < .15 else "warning" if member_risk < .35 else "danger"
        member_label = "안전" if member_class == "safe" else "보통" if member_class == "warning" else "위험"
        roster_rows.append(
            "<tr>"
            f"<td>Employee #{int(row['Employee ID'])}</td>"
            f"<td class='team-role'>{html.escape(translate(row['Job Role']))} / {html.escape(translate(row['Job Level']))}</td>"
            f"<td>{float(row['인재 가치 지수']):.1f}</td>"
            f"<td><span class='risk-chip {member_class}'>{member_risk:.1%} ({member_label})</span></td>"
            "</tr>"
        )

    avg_talent = float(display_team["인재 가치 지수"].mean())
    avg_fit = float(display_team["팀 적합 점수"].mean())
    with right:
        st.markdown(
            f"""
            <div class="reference-card team-roster-card">
              <div class="team-roster-head"><div class="reference-card-title">추천 팀 구성안</div><div class="reference-card-subtitle">AI 추천 최적 팀 구성 · {html.escape(project_name)}</div></div>
              <table class="team-table"><thead><tr><th>성명</th><th>담당 역할</th><th>인재가치 점수</th><th>퇴사 위험도</th></tr></thead><tbody>{''.join(roster_rows)}</tbody></table>
            </div>
            <div class="reference-kpis">
              <div class="reference-card reference-kpi"><div class="reference-label">팀 평균 퇴사 위험도</div><div class="reference-kpi-value" style="color:#12B76A">{mean_risk:.1%}</div><div class="reference-card-subtitle">{html.escape(verdict_label)} · {html.escape(verdict_note)}</div></div>
              <div class="reference-card reference-kpi"><div class="reference-label">역량 매칭 충족률</div><div class="reference-kpi-value" style="color:#2970FF">{avg_fit:.0f}%</div><div class="reference-card-subtitle">Team Fit 기준</div></div>
              <div class="reference-card reference-kpi"><div class="reference-label">팀 평균 인재가치 점수</div><div class="reference-kpi-value" style="color:#7F56D9">{avg_talent:.1f}점</div><div class="reference-card-subtitle">추천 구성 평균</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        chart_left, chart_right = st.columns(2, gap="medium")
        with chart_left:
            with st.container(key="team-scatter-card"):
                st.markdown('<div class="reference-label">팀원별 인재가치 vs 퇴사 위험도</div>', unsafe_allow_html=True)
                scatter = go.Figure()
                scatter.add_trace(go.Scatter(
                    x=display_team["prediction"] * 100,
                    y=display_team["인재 가치 지수"],
                    mode="markers+text",
                    text=[f"#{int(v)}" for v in display_team["Employee ID"]],
                    textposition="top center",
                    marker={"size": 9, "color": ["#12B76A" if v < .15 else "#F79009" if v < .35 else "#F04438" for v in display_team["prediction"]]},
                ))
                scatter.update_layout(height=220, margin={"l": 28, "r": 8, "t": 16, "b": 28}, showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFFFFF", xaxis={"title":"퇴사위험도", "ticksuffix":"%", "gridcolor":"#EEF2F6"}, yaxis={"title":"인재가치", "gridcolor":"#EEF2F6"})
                st.plotly_chart(scatter, width="stretch", config={"displayModeBar": False})
        with chart_right:
            with st.container(key="team-radar-card"):
                st.markdown('<div class="reference-label">팀 전체 역량 레이더</div>', unsafe_allow_html=True)
                radar_cols = [("학력 점수", "학력"), ("성과 점수", "성과"), ("평판 점수", "회사평판"), ("경력 점수", "근속/경험"), ("리더십 점수", "리더십")]
                categories = [label for col, label in radar_cols if col in display_team]
                values = [float(display_team[col].mean()) for col, _ in radar_cols if col in display_team]
                radar = go.Figure(go.Scatterpolar(r=values + values[:1], theta=categories + categories[:1], fill="toself", line={"color":"#2563EB"}, fillcolor="rgba(37,99,235,.14)"))
                radar.update_layout(height=220, margin={"l":30,"r":30,"t":16,"b":20}, showlegend=False, paper_bgcolor="rgba(0,0,0,0)", polar={"radialaxis":{"range":[0,100],"showticklabels":False,"gridcolor":"#E5EAF0"},"angularaxis":{"gridcolor":"#E5EAF0"}})
                st.plotly_chart(radar, width="stretch", config={"displayModeBar": False})

    with left:
        with st.container(key="team-simulation-card"):
            st.markdown('<div class="reference-card-title">팀원 변경 시뮬레이션</div><div class="reference-card-subtitle">추천된 팀원을 다른 후보로 교체</div>', unsafe_allow_html=True)
            pills = "".join(f'<span class="feature-pill">#{int(v)}</span>' for v in display_team["Employee ID"])
            st.markdown(f'<div class="feature-pills" style="margin:.7rem 0">{pills}</div>', unsafe_allow_html=True)
            with st.container(key="team-swap-fab"):
                swap_fab_clicked = st.button("⇄ 팀원 교체하기", key="team_swap_fab_btn", width="stretch")
        with st.expander("인재 가치 지수 계산 기준", expanded=False):
            st.caption(TALENT_VALUE_EXPLANATION)
    if swap_fab_clicked:
        _swap_dialog(employees, display_team, recommended_team, talent_weight, swap_map)
