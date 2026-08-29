"""03 · 인사 지원 탭 (인사발령 · 승진 · 구조조정).

같은 조건(직군)의 직원들을 인재 가치와 퇴사 예측률이라는 동일한 잣대로 비교한다.
핵심 규칙은 "인재 가치가 비슷하다면 퇴사 예측률이 낮은 쪽에 승진 우선순위를 준다"는
것이며, 반대로 인재 가치가 높은데 만족도가 낮은 직원은 구조조정이 아니라 재배치
검토 후보로 별도 관리한다. 최종 의사결정은 항상 인사 담당자가 내려야 한다.

직원 찾기는 부서 → 직급 → 직원 ID 순으로 좁혀가는 캐스케이딩 선택에, 옆에 직원 ID를
바로 입력해 찾는 칸을 더했다(streamlit_ui.employee_picker 재사용). 여기서 고른(또는
검색한) 직원의 부서가 곧 비교 그룹이 된다. 순위표는 상위 100명을 10명 단위 페이지로
나눠 넘겨보고, 표에서 행을 하나 선택하면 그 직원의 표에 없는 부가 지표를 레이더
차트로 보여준다(행 선택은 진짜 상호작용이 필요해 이 표만 st.dataframe으로 그린다).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.utils.hr_metrics import LEVEL_KR, add_people_decision_scores, translate
from streamlit_ui import alert_box, employee_picker, section_heading, sub_tabs

RADAR_METRICS = [
    ("학력 점수", "학력"),
    ("성과 점수", "성과 평가"),
    ("평판 점수", "회사 평판"),
    ("경력 점수", "총 경력연수"),
    ("리더십 점수", "리더십 기회"),
    ("만족도 점수", "직무 만족도"),
    ("워라밸 점수", "워라밸"),
    ("인정 점수", "직원 인정"),
]

TOP_POOL_SIZE = 100
PAGE_SIZE = 10


def _ranked_table(
    group: pd.DataFrame,
    score_col: str,
    columns: list[str],
    rename: dict[str, str],
    page: int,
    focus_id: int | None,
    key_suffix: str,
) -> tuple[pd.DataFrame, dict | None, int]:
    """score_col 기준 전체 순위를 매기고, 상위 TOP_POOL_SIZE명을 페이지 단위로 보여준다.

    반환값: (화면에 보인 행들, 선택된 행(dict 또는 None), 전체 페이지 수)
    """

    ranked_full = group.sort_values(score_col, ascending=False).reset_index(drop=True)
    ranked_full["순위"] = np.arange(1, len(ranked_full) + 1)
    pool = ranked_full.head(TOP_POOL_SIZE)
    total_pages = max(1, -(-len(pool) // PAGE_SIZE))
    page = max(1, min(page, total_pages))
    start = (page - 1) * PAGE_SIZE
    page_rows = pool.iloc[start : start + PAGE_SIZE].copy()

    focus_row = None
    if focus_id is not None:
        match = ranked_full.loc[ranked_full["Employee ID"].eq(focus_id)]
        if not match.empty:
            focus_row = match.iloc[0]
            if not page_rows["Employee ID"].eq(focus_id).any():
                page_rows = pd.concat([page_rows, match], ignore_index=True)

    table = page_rows[["순위", *columns]].rename(columns=rename)
    if "직급" in table.columns:
        table["직급"] = page_rows["Job Level"].map(lambda v: LEVEL_KR.get(v, v)).values
    for text_col in ("성과 평가", "직무 만족도"):
        if text_col in table.columns:
            table[text_col] = table[text_col].map(translate)

    column_config = {
        "순위": st.column_config.NumberColumn(width="small"),
        "직원 ID": st.column_config.NumberColumn(width="small", format="%d"),
        "직급": st.column_config.TextColumn(width="small"),
        "인재 가치 지수": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f"),
        "퇴사 예측률": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent"),
    }
    score_label = rename.get(score_col, score_col)
    column_config[score_label] = st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f")

    event = st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
        column_config=column_config,
        key=f"hr_actions_grid_{key_suffix}",
    )
    selected_row = None
    if event and event.selection and event.selection.rows:
        selected_row = table.iloc[event.selection.rows[0]].to_dict()

    caption_bits = [f"{page} / {total_pages} 페이지 · 상위 {len(pool)}명 중 표시"]
    if focus_row is not None:
        caption_bits.append(
            f"검색한 직원 ID {int(focus_row['Employee ID'])}의 전체 순위는 "
            f"{int(focus_row['순위'])}위 / {len(ranked_full)}명 중이에요."
        )
    st.caption(" · ".join(caption_bits))

    return page_rows, selected_row, total_pages


def _pagination_controls(key_suffix: str, total_pages: int) -> int:
    """이전/다음 버튼과 페이지 표시가 있는 간단한 페이지네이션. 현재 페이지 번호를 반환한다."""

    page_key = f"hr_actions_page_{key_suffix}"
    st.session_state.setdefault(page_key, 1)
    st.session_state[page_key] = max(1, min(st.session_state[page_key], total_pages))

    cols = st.columns([1, 2, 1])
    with cols[0]:
        if st.button("◀ 이전", key=f"{page_key}_prev", disabled=st.session_state[page_key] <= 1, width="stretch"):
            st.session_state[page_key] -= 1
            st.rerun()
    with cols[1]:
        st.markdown(
            f'<div style="text-align:center; color:var(--muted); font-size:.86rem; padding-top:.5rem;">'
            f"{st.session_state[page_key]} / {total_pages} 페이지</div>",
            unsafe_allow_html=True,
        )
    with cols[2]:
        if st.button(
            "다음 ▶", key=f"{page_key}_next", disabled=st.session_state[page_key] >= total_pages, width="stretch"
        ):
            st.session_state[page_key] += 1
            st.rerun()

    return st.session_state[page_key]


def _radar_chart_section(selected_row: dict | None, group: pd.DataFrame) -> None:
    """그리드에서 선택한 행의, 표에는 없는 부가 지표를 레이더 차트로 보여준다."""

    st.markdown('<div class="section-divider-thin"></div>', unsafe_allow_html=True)
    st.markdown("**직원 상세 지표 (레이더 차트)**")

    if selected_row is None:
        st.caption("위 표에서 행을 하나 선택하면, 표에는 없는 부가 지표를 레이더 차트로 보여줘요.")
        return

    employee_id = int(selected_row["직원 ID"])
    match = group.loc[group["Employee ID"].eq(employee_id)]
    if match.empty:
        st.caption("선택한 직원 정보를 찾지 못했어요.")
        return
    row = match.iloc[0]

    available_metrics = [(col, label) for col, label in RADAR_METRICS if col in row.index]
    if not available_metrics:
        st.caption("부가 지표 데이터가 없어요.")
        return

    st.caption(f"ID {employee_id} · {translate(row['Job Level'])} · 위 표에는 없는 부가 지표예요.")
    categories = [label for _, label in available_metrics]
    values = [float(row[col]) for col, _ in available_metrics]

    figure = go.Figure()
    figure.add_trace(
        go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            line={"color": "#3182F6"},
            fillcolor="rgba(49, 130, 246, 0.25)",
            name=f"ID {employee_id}",
        )
    )
    figure.update_layout(
        polar={"radialaxis": {"visible": True, "range": [0, 100]}},
        showlegend=False,
        margin={"l": 40, "r": 40, "t": 20, "b": 20},
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})


def render(employees: pd.DataFrame) -> None:
    section_heading(
        "03 · PEOPLE DECISIONS",
        "인사발령 · 승진 · 구조조정",
        "인재 가치와 퇴사 위험을 같은 기준으로 계산하되, 최종 판단은 인사 담당자가 수행합니다.",
    )

    st.markdown("**직원 찾기**")
    st.caption("부서 → 직급 → 직원 ID 순으로 좁혀 찾거나, 옆 칸에 직원 ID를 바로 입력해 찾을 수 있어요.")
    focus_id = employee_picker(employees, key_prefix="hr_actions", with_direct_search=True)
    if focus_id is None:
        alert_box("info", "선택한 조건에 해당하는 직원이 없어요. 부서 또는 직급을 바꿔보세요.")
        return
    decision_role = employees.loc[employees["Employee ID"].eq(focus_id), "Job Role"].iloc[0]
    same_condition = add_people_decision_scores(employees[employees["Job Role"].eq(decision_role)].copy())

    st.markdown('<div class="section-spacer-lg"></div>', unsafe_allow_html=True)
    st.markdown(
        f'<span class="badge tone-info">비교 그룹 · {translate(decision_role)}</span>',
        unsafe_allow_html=True,
    )

    active = sub_tabs(
        [
            ("promotion", "승진 우선순위"),
            ("restructuring", "구조조정 검토"),
            ("reassignment", "재배치 · 인사발령 후보"),
        ],
        state_key="hr_actions_tab",
    )
    st.markdown('<div class="section-spacer-lg"></div>', unsafe_allow_html=True)

    page_key_suffix = f"{active}_{decision_role}"
    current_page = st.session_state.get(f"hr_actions_page_{page_key_suffix}", 1)

    if active == "promotion":
        page_rows, selected_row, total_pages = _ranked_table(
            same_condition,
            "승진 우선 점수",
            ["Employee ID", "Job Level", "Performance Rating", "인재 가치 지수", "prediction", "승진 우선 점수"],
            {
                "Employee ID": "직원 ID",
                "Job Level": "직급",
                "Performance Rating": "성과 평가",
                "prediction": "퇴사 예측률",
            },
            current_page,
            focus_id,
            key_suffix=page_key_suffix,
        )
        _pagination_controls(page_key_suffix, total_pages)
        st.caption("승진 우선 점수 = 인재 가치 65% + 잔류 가능성 35%. 동일 조건에서는 퇴사 예측률이 낮은 직원을 우선합니다.")

    elif active == "restructuring":
        page_rows, selected_row, total_pages = _ranked_table(
            same_condition,
            "검토 우선 점수",
            ["Employee ID", "Job Level", "Performance Rating", "인재 가치 지수", "prediction", "검토 우선 점수"],
            {
                "Employee ID": "직원 ID",
                "Job Level": "직급",
                "Performance Rating": "성과 평가",
                "prediction": "퇴사 예측률",
            },
            current_page,
            focus_id,
            key_suffix=page_key_suffix,
        )
        _pagination_controls(page_key_suffix, total_pages)
        alert_box(
            "warning",
            "이 표는 재배치·교육·면담 검토를 위한 보조 정보입니다. 예측값만으로 해고나 불이익을 자동 결정하면 안 됩니다.",
        )

    else:
        if "재배치 신호 점수" not in same_condition.columns:
            alert_box("info", "만족도 데이터가 없어 재배치 신호를 계산할 수 없어요.")
            selected_row = None
        else:
            page_rows, selected_row, total_pages = _ranked_table(
                same_condition,
                "재배치 신호 점수",
                ["Employee ID", "Job Level", "Job Satisfaction", "인재 가치 지수", "prediction", "재배치 신호 점수"],
                {
                    "Employee ID": "직원 ID",
                    "Job Level": "직급",
                    "Job Satisfaction": "직무 만족도",
                    "prediction": "퇴사 예측률",
                },
                current_page,
                focus_id,
                key_suffix=page_key_suffix,
            )
            _pagination_controls(page_key_suffix, total_pages)
            st.caption(
                "재배치 신호 점수 = 인재 가치 50% + 낮은 만족도 30% + 퇴사 예측률 20%. "
                "인재 가치는 높은데 만족도가 낮은 직원은 역할·부서 이동을 우선 검토하세요."
            )

    _radar_chart_section(selected_row, same_condition)
