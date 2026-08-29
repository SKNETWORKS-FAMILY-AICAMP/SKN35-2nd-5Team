"""03 · 인사 지원 탭 (인사발령 · 승진 · 구조조정).

같은 조건(직군)의 직원들을 인재 가치와 퇴사 예측률이라는 동일한 잣대로 비교한다.
핵심 규칙은 "인재 가치가 비슷하다면 퇴사 예측률이 낮은 쪽에 승진 우선순위를 준다"는
것이며, 반대로 인재 가치가 높은데 만족도가 낮은 직원은 구조조정이 아니라 재배치
검토 후보로 별도 관리한다. 최종 의사결정은 항상 인사 담당자가 내려야 한다.

직원 찾기는 부서 → 직급 → 직원 ID를 한 줄에서 좁혀가는 조건 필터에, 직접 ID를
입력하는 칸을 더했다(직접 검색 값이 있으면 그 값이 항상 우선). 직급/직원 ID를
"전체"로 두면 조건에 맞는 모든 직원이 그리드에 나오고, 특정 값으로 좁히면 그
조건에 정확히 일치하는 행만 남는다 — 즉 이 필터가 곧 그리드의 검색 결과다.
그리드 행을 클릭하면(체크박스 없이 셀 선택만으로 동작) 그 직원의 표에는 없는
부가 지표를 레이더 차트 모달로 보여준다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.utils.hr_metrics import (
    LEVEL_KR,
    add_people_decision_scores,
    department_options,
    level_options,
    translate,
)
from streamlit_ui import alert_box, section_heading, sub_tabs

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
ALL_OPTION = "전체"


def _employee_filter(employees: pd.DataFrame, key_prefix: str) -> tuple[str, str | None, int | None]:
    """부서 · 직급 · 직원 ID · 직접 ID 검색을 한 줄에 두고 그리드 필터 조건을 만든다.

    직접 검색에 유효한 ID가 입력되면 부서/직급/직원 ID 드롭다운보다 우선한다.
    반환값: (department, level_filter, id_filter). level_filter/id_filter가
    None이면 "전체"(필터 없음)라는 뜻이다.
    """

    col_dept, col_level, col_id, col_search = st.columns([1.1, 1, 1.1, 1.4])

    dept_options = department_options(employees)
    with col_dept:
        department = st.selectbox("부서", dept_options, format_func=translate, key=f"{key_prefix}_dept")

    lvl_options = [ALL_OPTION, *level_options(employees, department)]
    with col_level:
        level = st.selectbox(
            "직급",
            lvl_options,
            format_func=lambda v: v if v == ALL_OPTION else LEVEL_KR.get(v, v),
            key=f"{key_prefix}_level",
        )
    level_filter = None if level == ALL_OPTION else level

    id_scope = employees[employees["Job Role"].eq(department)]
    if level_filter is not None:
        id_scope = id_scope[id_scope["Job Level"].eq(level_filter)]
    id_options = [ALL_OPTION, *sorted(id_scope["Employee ID"].tolist())]
    with col_id:
        chosen_id = st.selectbox(
            "직원 ID",
            id_options,
            format_func=lambda v: v if v == ALL_OPTION else f"ID {v}",
            key=f"{key_prefix}_id",
        )
    id_filter = None if chosen_id == ALL_OPTION else int(chosen_id)

    with col_search:
        search_text = st.text_input(
            "직원 ID로 바로 찾기", key=f"{key_prefix}_direct_search", placeholder="예: 10345"
        )

    search_id = None
    if search_text.strip():
        try:
            candidate = int(search_text.strip())
        except ValueError:
            candidate = None
        if candidate is not None and employees["Employee ID"].eq(candidate).any():
            search_id = candidate
        else:
            alert_box("warning", "일치하는 직원 ID를 찾지 못했어요. 왼쪽 조건으로 찾아보세요.")

    if search_id is not None:
        department = employees.loc[employees["Employee ID"].eq(search_id), "Job Role"].iloc[0]
        st.caption(f"직원 ID {search_id}를 직접 검색해 찾았어요 · 왼쪽 부서 · 직급 · 직원 ID 조건보다 이 검색이 우선돼요.")
        return department, None, search_id

    return department, level_filter, id_filter


def _ranked_table(
    group: pd.DataFrame,
    score_col: str,
    columns: list[str],
    rename: dict[str, str],
    level_filter: str | None,
    id_filter: int | None,
    page: int,
    key_suffix: str,
) -> tuple[int | None, int]:
    """score_col 기준으로 전체 순위를 매긴 뒤, 필터 조건에 일치하는 행만 페이지 단위로 보여준다.

    반환값: (그리드에서 선택한 직원 ID 또는 None, 전체 페이지 수)
    """

    ranked_full = group.sort_values(score_col, ascending=False).reset_index(drop=True)
    ranked_full["순위"] = np.arange(1, len(ranked_full) + 1)

    filtered = ranked_full
    if level_filter is not None:
        filtered = filtered[filtered["Job Level"].eq(level_filter)]
    if id_filter is not None:
        filtered = filtered[filtered["Employee ID"].eq(id_filter)]

    pool = filtered.head(TOP_POOL_SIZE)
    total_pages = max(1, -(-len(pool) // PAGE_SIZE))
    page = max(1, min(page, total_pages))
    start = (page - 1) * PAGE_SIZE
    page_rows = pool.iloc[start : start + PAGE_SIZE].copy()

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

    # 체크박스 없이 셀을 클릭하는 것만으로 행을 선택하도록 single-cell 모드를 쓴다
    # (single-row/multi-row는 왼쪽에 체크박스 열이 항상 붙는다).
    event = st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="single-cell",
        column_config=column_config,
        key=f"hr_actions_grid_{key_suffix}_{page}",
    )
    selected_employee_id = None
    if event and event.selection and event.selection.cells:
        row_index = event.selection.cells[0][0]
        selected_employee_id = int(table.iloc[row_index]["직원 ID"])

    caption_bits = [f"{page} / {total_pages} 페이지"]
    if id_filter is not None and not pool.empty:
        row_rank = int(pool.iloc[0]["순위"])
        caption_bits.append(f"부서 전체 순위 {row_rank}위 / {len(ranked_full)}명 중")
    else:
        count_text = f"조건에 맞는 {len(filtered)}명 중 표시"
        if len(filtered) > TOP_POOL_SIZE:
            count_text += f" (상위 {TOP_POOL_SIZE}명까지만)"
        caption_bits.append(count_text)
    st.caption(" · ".join(caption_bits))

    return selected_employee_id, total_pages


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


@st.dialog("직원 상세 지표 (레이더 차트)", width="medium")
def _radar_dialog(employee_id: int, group: pd.DataFrame) -> None:
    """그리드에서 선택한 행의, 표에는 없는 부가 지표를 레이더 차트 모달로 보여준다."""

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
    st.caption(
        "부서 → 직급 → 직원 ID 순으로 조건을 좁히거나, 오른쪽 칸에 직원 ID를 바로 입력해 찾을 수 있어요. "
        "직급 · 직원 ID를 '전체'로 두면 조건에 맞는 모든 직원이 아래 표에 나오고, 값을 고르면 그 조건에 "
        "정확히 일치하는 행만 남아요."
    )
    department, level_filter, id_filter = _employee_filter(employees, key_prefix="hr_actions")
    same_condition = add_people_decision_scores(employees[employees["Job Role"].eq(department)].copy())

    st.markdown('<div class="section-spacer-lg"></div>', unsafe_allow_html=True)
    st.markdown(
        f'<span class="badge tone-info">비교 그룹 · {translate(department)}</span>',
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

    key_suffix = f"{active}_{department}_{level_filter}_{id_filter}"
    current_page = st.session_state.get(f"hr_actions_page_{key_suffix}", 1)
    selected_employee_id: int | None = None

    if active == "promotion":
        selected_employee_id, total_pages = _ranked_table(
            same_condition,
            "승진 우선 점수",
            ["Employee ID", "Job Level", "Performance Rating", "인재 가치 지수", "prediction", "승진 우선 점수"],
            {
                "Employee ID": "직원 ID",
                "Job Level": "직급",
                "Performance Rating": "성과 평가",
                "prediction": "퇴사 예측률",
            },
            level_filter,
            id_filter,
            current_page,
            key_suffix=key_suffix,
        )
        _pagination_controls(key_suffix, total_pages)
        st.caption("승진 우선 점수 = 인재 가치 65% + 잔류 가능성 35%. 동일 조건에서는 퇴사 예측률이 낮은 직원을 우선합니다.")

    elif active == "restructuring":
        selected_employee_id, total_pages = _ranked_table(
            same_condition,
            "검토 우선 점수",
            ["Employee ID", "Job Level", "Performance Rating", "인재 가치 지수", "prediction", "검토 우선 점수"],
            {
                "Employee ID": "직원 ID",
                "Job Level": "직급",
                "Performance Rating": "성과 평가",
                "prediction": "퇴사 예측률",
            },
            level_filter,
            id_filter,
            current_page,
            key_suffix=key_suffix,
        )
        _pagination_controls(key_suffix, total_pages)
        alert_box(
            "warning",
            "이 표는 재배치·교육·면담 검토를 위한 보조 정보입니다. 예측값만으로 해고나 불이익을 자동 결정하면 안 됩니다.",
        )

    else:
        if "재배치 신호 점수" not in same_condition.columns:
            alert_box("info", "만족도 데이터가 없어 재배치 신호를 계산할 수 없어요.")
        else:
            selected_employee_id, total_pages = _ranked_table(
                same_condition,
                "재배치 신호 점수",
                ["Employee ID", "Job Level", "Job Satisfaction", "인재 가치 지수", "prediction", "재배치 신호 점수"],
                {
                    "Employee ID": "직원 ID",
                    "Job Level": "직급",
                    "Job Satisfaction": "직무 만족도",
                    "prediction": "퇴사 예측률",
                },
                level_filter,
                id_filter,
                current_page,
                key_suffix=key_suffix,
            )
            _pagination_controls(key_suffix, total_pages)
            st.caption(
                "재배치 신호 점수 = 인재 가치 50% + 낮은 만족도 30% + 퇴사 예측률 20%. "
                "인재 가치는 높은데 만족도가 낮은 직원은 역할·부서 이동을 우선 검토하세요."
            )

    # 그리드에서 새로 선택한 행이 있을 때만 모달을 띄운다. 선택 상태 자체는 다음 재실행에도
    # 남아있으므로, 이미 띄운 적 있는 직원 ID면 다시 열지 않는다(같은 행을 계속 선택 중인 채로
    # 다른 조작을 해도 모달이 매번 다시 뜨지 않도록).
    if selected_employee_id is not None and st.session_state.get("hr_actions_radar_last_shown") != selected_employee_id:
        st.session_state["hr_actions_radar_last_shown"] = selected_employee_id
        _radar_dialog(selected_employee_id, same_condition)
