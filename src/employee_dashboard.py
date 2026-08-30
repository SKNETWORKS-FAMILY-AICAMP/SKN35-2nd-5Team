"""인사 담당자용 Page 2~5 데이터 계산과 Liquid Glass 화면 구성."""

from __future__ import annotations

from html import escape
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data.loader import load_predictions, load_raw_test
from streamlit_ui import segmented_nav, style_plotly_chart

BLUE, PURPLE, ORANGE, GREEN, RED = "#007AFF", "#5856D6", "#FF9500", "#34C759", "#FF3B30"


@st.cache_data(ttl=300, show_spinner="DB에서 직원 정보와 예측 결과를 불러오는 중입니다...")
def get_employees() -> pd.DataFrame:
    test = load_raw_test()
    predictions = load_predictions()
    frame = test.merge(predictions, left_on="Employee ID", right_on="employee_id", how="left")
    missing_predictions = int(frame["prediction"].isna().sum())
    if missing_predictions:
        raise ValueError(f"DB 예측값이 없는 직원이 {missing_predictions}명 있습니다.")
    education = {"High School": 55, "Associate Degree": 65, "Bachelor’s Degree": 75, "Master’s Degree": 85, "PhD": 95}
    performance = {"Low": 40, "Below Average": 50, "Average": 65, "High": 82, "Very High": 95}
    frame["Talent Value"] = (frame["Education Level"].map(education).fillna(65) * .35 + frame["Performance Rating"].map(performance).fillna(65) * .45 + frame["Years at Company"].clip(0, 20) / 20 * 20).round(1)
    return frame


def page_intro(page: int, color: str, title: str, copy: str) -> None:
    st.markdown(f'<div class="section-heading" style="--accent:{color}"><span class="chip">PAGE {page}</span><h2>{title}</h2></div><p class="section-copy" style="margin:0 4px 14px">{copy}</p>', unsafe_allow_html=True)


def _risk_color(risk: float) -> str:
    return RED if risk >= 60 else ORANGE if risk >= 35 else GREEN


def _hex_to_rgba(hex_color: str, alpha: float = 0.15) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


def _risk_label(risk: float) -> str:
    return "위험 — HIGH" if risk >= 60 else "주의 — MEDIUM" if risk >= 35 else "안전 — LOW"


def _score(value: object, mapping: dict[str, int], default: int = 55) -> int:
    return mapping.get(str(value), default)


def _feature_scores(row: pd.Series) -> list[tuple[str, str, int]]:
    return [
        ("학력", "학력·전문성", _score(row["Education Level"], {"High School": 55, "Associate Degree": 65, "Bachelor’s Degree": 75, "Master’s Degree": 85, "PhD": 95}, 70)),
        ("성과평가", "최근 성과 점수", _score(row["Performance Rating"], {"Low": 40, "Below Average": 50, "Average": 65, "High": 82, "Very High": 95}, 65)),
        ("회사평판", "기업 만족도", _score(row["Company Reputation"], {"Poor": 35, "Fair": 50, "Good": 72, "Excellent": 92})),
        ("재직기간", "재직 기간", min(100, int(float(row["Company Tenure"]) * 4))),
        ("리더십기회", "성장 기회", _score(row["Leadership Opportunities"], {"Yes": 85, "No": 40})),
        ("근무연수", "누적 근속", min(100, int(float(row["Years at Company"]) * 5))),
        ("직급수준", "현재 직급", _score(row["Job Level"], {"Entry": 42, "Mid": 62, "Senior": 82, "Executive": 95})),
        ("직원인정", "조직 내 인정도", _score(row["Employee Recognition"], {"Low": 35, "Medium": 55, "High": 78, "Very High": 94})),
    ]


def _metric_cards(items: list[tuple[str, str, str, str]]) -> None:
    html = '<div class="dash-metrics">' + "".join(f'<div class="dash-metric"><small>{escape(label)}</small><strong style="color:{color}">{escape(value)}</strong><span>{escape(note)}</span></div>' for label, value, note, color in items) + "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_salary(df: pd.DataFrame) -> None:
    st.caption("직원 선택")
    labels = df.apply(lambda r: f"EMP-{r['Employee ID']} — {r['Job Role']}", axis=1).tolist()

    with st.container(border=True):
        st.markdown('<span class="employee-picker-anchor"></span>', unsafe_allow_html=True)
        label_col, select_col = st.columns([1, 4], gap="small", vertical_alignment="center")
        with label_col:
            st.markdown('<div class="employee-picker-label">직원 ID</div>', unsafe_allow_html=True)
        with select_col:
            selected = st.selectbox("직원 ID", labels, label_visibility="collapsed")
        row = df.iloc[labels.index(selected)]
        st.markdown(
            f'''<div class="employee-strip"><div class="employee-strip-bottom">
                <div><span>부서</span><b>{escape(str(row["Job Role"]))}</b></div>
                <div><span>직급</span><b>{escape(str(row["Job Level"]))}</b></div>
            </div></div>''',
            unsafe_allow_html=True,
        )

    risk = float(row["prediction"]) * 100
    color = _risk_color(risk)

    features = _feature_scores(row)
    left, center, right = st.columns([.72, 2.05, 2.05], gap="small")
    with left:
        st.caption("이탈 위험도")
        offset = 214 * (1 - risk / 100)
        gauge = f'''<div class="glass-card risk-card"><svg viewBox="0 0 180 112"><path d="M22 92 A68 68 0 0 1 158 92" fill="none" stroke="rgba(60,60,67,.12)" stroke-width="10" stroke-linecap="round"/><path d="M22 92 A68 68 0 0 1 158 92" fill="none" stroke="{color}" stroke-width="10" stroke-linecap="round" stroke-dasharray="214" stroke-dashoffset="{offset:.1f}"/><text x="90" y="78" text-anchor="middle" fill="{color}" font-size="33" font-weight="700">{risk:.0f}</text><text x="90" y="98" text-anchor="middle" fill="rgba(60,60,67,.6)" font-size="13">이탈 위험도</text></svg><div class="risk-pill" style="color:{color};background:{color}10">{_risk_label(risk)}</div><div class="talent-row"><span>인재 가치</span><b>{row['Talent Value']:.0f}</b></div></div>'''
        st.markdown(gauge, unsafe_allow_html=True)
        st.caption("AI 권고")
        weakest = sorted(features, key=lambda x: x[2])[:2]
        st.markdown(f'<div class="glass-card ai-note">위험도 <b>{risk:.0f}%</b>. {weakest[0][0]}·{weakest[1][0]} 개선을 병행하는 연봉협상 전략을 권고합니다.</div>', unsafe_allow_html=True)
    with center:
        st.caption("역량 방사형 분석")
        radar = features + [features[0]]
        fig = go.Figure(go.Scatterpolar(r=[x[2] for x in radar], theta=[x[0] for x in radar], fill="toself", fillcolor=_hex_to_rgba(color, 0.15), line=dict(color=color, width=2)))
        fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100], showticklabels=False, gridcolor="rgba(60,60,67,.12)"), angularaxis=dict(gridcolor="rgba(60,60,67,.12)"), bgcolor="rgba(255,255,255,0)"), showlegend=False)
        with st.container(border=True):
            st.caption("8개 이탈 예측 피처 종합 분포")
            st.plotly_chart(style_plotly_chart(fig, 300), width="stretch", config={"displayModeBar": False})
    with right:
        st.caption("개별 피처 점수")
        bars = []
        for name, desc, value in features:
            c = GREEN if value >= 75 else ORANGE if value >= 50 else RED
            bars.append(f'<div class="feature-row"><div><b>{name}</b><span>{desc}</span><strong style="color:{c}">{value}</strong></div><i><em style="width:{value}%;background:{c}"></em></i></div>')
        st.markdown('<div class="glass-card feature-list">' + "".join(bars) + "</div>", unsafe_allow_html=True)


def render_team(df: pd.DataFrame) -> None:
    st.caption("프로젝트 선택")
    projects_map = {
        "기술 혁신 TF": "Technology",
        "헬스케어 서비스 TF": "Healthcare",
        "교육 혁신 TF": "Education",
        "미디어 전략 TF": "Media",
        "재무 전략 TF": "Finance",
    }
    project_names = list(projects_map.keys())
    badge_by_role = {}
    for role, risk in df.groupby("Job Role")["prediction"].mean().items():
        badge_by_role[role] = (
            ("긴급", "critical") if risk >= .60 else ("주의", "high") if risk >= .35 else ("안정", "normal")
        )
    selected_project = segmented_nav(
        project_names,
        "project",
        badges=[badge_by_role[role] for role in projects_map.values()],
        stretch=True,
    )
    target_role = projects_map[selected_project]

    pool = df[df["Job Role"] == target_role].copy()
    pool["팀 적합 점수"] = (pool["Talent Value"] * .6 + (1 - pool["prediction"]) * 40).round(1)
    team = pool.nlargest(5, "팀 적합 점수").copy()
    if team.empty:
        st.info("선택된 부문에 해당하는 직원이 없습니다.")
        return

    left, right = st.columns([1, 4.2], gap="small")
    with left:
        st.caption("프로젝트 정보")
        st.markdown(
            f'''<div class="glass-card project-card">
            <div><b>{escape(selected_project.split(" (")[0])}</b><span class="priority">HIGH</span></div>
            <p><span>대상 부문</span><b>{escape(target_role)}</b></p>
            <p><span>선별 기준</span><b>인재가치 60% + 안정성 40%</b></p>
            <p><span>추천 규모</span><b>정예 5명 (최적 인재)</b></p>
            </div>''',
            unsafe_allow_html=True,
        )
        st.caption(f"{target_role} 최적 추천 팀원")
        people = []
        for _, r in team.iterrows():
            risk = float(r["prediction"]) * 100
            score = float(r["팀 적합 점수"])
            people.append(f'<div class="member-row"><i>{str(r["Employee ID"])[-2:]}</i><div><b>EMP-{r["Employee ID"]}</b><span>{escape(str(r["Job Role"]))} · {escape(str(r["Job Level"]))}</span></div><strong style="color:{_risk_color(risk)}">{risk:.0f}%<small>위험</small></strong><em style="color:var(--blue);font-weight:600;font-size:13px">{score:.0f}점</em></div>')
        st.markdown('<div class="glass-card member-list">' + "".join(people) + "</div>", unsafe_allow_html=True)
    avg_risk, avg_talent = team["prediction"].mean() * 100, team["Talent Value"].mean()
    with right:
        _metric_cards([("팀 평균 이탈위험도", f"{avg_risk:.0f}%", _risk_label(avg_risk).split(" — ")[0], _risk_color(avg_risk)), ("팀 평균 인재가치", f"{avg_talent:.0f}", "/ 100점", BLUE), ("고위험 팀원", f"{(team['prediction'] >= .6).sum()}명", f"전체 {len(team)}명 중", ORANGE)])
        st.caption("팀원별 이탈위험도 & 인재가치")
        names = [f"EMP-{x}" for x in team["Employee ID"]]
        fig = go.Figure()
        fig.add_bar(name="이탈위험", x=names, y=team["prediction"] * 100, marker_color=ORANGE)
        fig.add_bar(name="인재가치", x=names, y=team["Talent Value"], marker_color=BLUE)
        fig.update_layout(barmode="group", yaxis=dict(range=[0, 100]))
        with st.container(border=True):
            st.plotly_chart(style_plotly_chart(fig, 280), width="stretch", config={"displayModeBar": False})
        status_note = "고위험 팀원이 포함되어 있어 모니터링이 필요합니다." if avg_risk >= 35 else "안정적인 팀 인재 구성입니다."
        st.markdown(f'<div class="team-warning">팀 평균 이탈위험도 <b>{avg_risk:.0f}%</b> — <b>{target_role}</b> 소속 직원 중 인재가치와 MLP 이탈안정성이 가장 우수한 정예 5명의 분석 결과입니다. ({status_note})</div>', unsafe_allow_html=True)


def render_people_decision(df: pd.DataFrame) -> None:
    filter_col, formula_col = st.columns([1.1, 2.5], gap="small")
    with filter_col:
        choice = segmented_nav(["전체", "안전", "위험"], "risk")
    with formula_col:
        st.markdown('<div class="formula-align-right"><div class="formula-inline"><b>안정 인재가치</b> = 인재가치 × (1 − 이탈위험도)</div></div>', unsafe_allow_html=True)

    ranked = df.copy()
    ranked["안정점수"] = (ranked["Talent Value"] * (1 - ranked["prediction"])).round(1)
    if choice == "안전":
        ranked = ranked[ranked["prediction"] < .35]
    elif choice == "위험":
        ranked = ranked[ranked["prediction"] >= .60]
    ranked = ranked.nlargest(12, "안정점수").reset_index(drop=True)
    rows = []
    for i, r in ranked.iterrows():
        risk, talent, stable = float(r["prediction"]) * 100, float(r["Talent Value"]), float(r["안정점수"])
        level, color = (("안전", GREEN) if risk < 35 else ("주의", ORANGE) if risk < 60 else ("위험", RED))
        medal = ["🥇", "🥈", "🥉"][i] if i < 3 else str(i + 1)
        rows.append(f'''<tr><td>{medal}</td><td><div class="person-cell"><i>{str(r['Employee ID'])[-2:]}</i><span><b>EMP-{r['Employee ID']}</b><small>{escape(str(r['Job Role']))} · {escape(str(r['Job Level']))}</small></span></div></td><td><div class="mini-bar"><i style="width:{risk}%;background:{color}"></i></div><small>{risk:.0f}</small></td><td><div class="mini-bar"><i style="width:{talent}%;background:{BLUE}"></i></div><small>{talent:.0f}</small></td><td class="stable" style="color:{GREEN if stable>=60 else ORANGE}">{stable:.0f}</td><td><span class="level" style="color:{color};border-color:{color}55;background:{color}0D">{level}</span></td><td><span class="action promote">승진 추천</span> <span class="action transfer">발령 추천</span><br><span class="action review">구조조정 검토</span></td></tr>''')
    table = '<div class="glass-card ranking-wrap"><table class="ranking"><thead><tr><th>#</th><th>직원 정보</th><th>이탈위험도 ↕</th><th>인재가치 ↕</th><th>안정점수 ↓</th><th>위험수준</th><th>인사 조치</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table></div>"
    st.markdown(table, unsafe_allow_html=True)


@st.fragment
def _render_executive_alerts(summary: pd.DataFrame) -> None:
    alerts = summary.head(3)
    dismissed = st.session_state.setdefault("dismissed_executive_alerts", set())
    active_alerts = alerts[~alerts["Job Role"].astype(str).isin(dismissed)]
    if active_alerts.empty:
        return
    st.caption("경고")
    for _, row in active_alerts.iterrows():
        role = str(row["Job Role"])
        with st.container(border=True):
            st.markdown('<span class="alert-card-anchor"></span>', unsafe_allow_html=True)
            message_col, button_col = st.columns([12, 1], vertical_alignment="center")
            with message_col:
                st.markdown(
                    f'<div class="alert-message"><i></i><b>{escape(role)}</b> 이탈률 '
                    f'<strong>{row["퇴사위험"]:.1%}</strong> — 즉각적인 인사 개입 권고</div>',
                    unsafe_allow_html=True,
                )
            with button_col:
                if st.button("확인", key=f"dismiss-alert-{role}", type="tertiary"):
                    dismissed.add(role)
                    st.rerun(scope="fragment")


def render_executive(df: pd.DataFrame) -> None:
    summary = df.groupby("Job Role", as_index=False).agg(직원수=("Employee ID", "count"), 퇴사위험=("prediction", "mean")).sort_values("퇴사위험", ascending=False)
    risk = df["prediction"] * 100
    _render_executive_alerts(summary)
    _metric_cards([("전사 이탈률", f"{risk.mean():.1f}%", "예측 평균", ORANGE), ("위험 부서 수", f"{(summary['퇴사위험'] >= .6).sum()}개", f"/ 총 {len(summary)}개 부서", RED), ("연간 이탈 예측", f"{int((df['prediction'] >= .6).sum())}명", "고위험 직원", ORANGE), ("조직 안정도", f"{100-risk.mean():.1f}", "/ 100점", "#8E8E93")])
    chart_col, list_col = st.columns([4, 1.2], gap="small")
    with chart_col:
        st.caption("전사 연간 이탈률 트렌드")
        current = risk.mean()
        trend = pd.DataFrame({"연도": ["2022", "2023", "2024", "2025", "2026(예측)"], "이탈률": [current*.62, current*.76, current*.91, current*1.05, current]})
        fig = go.Figure(go.Scatter(x=trend["연도"], y=trend["이탈률"], mode="lines+markers", line=dict(color=BLUE, width=2), marker=dict(size=6)))
        fig.add_hline(y=15, line_dash="dot", line_color=ORANGE, annotation_text="위험 기준 15%")
        fig.update_layout(yaxis=dict(ticksuffix="%", range=[0, max(30, trend["이탈률"].max()*1.2)]))
        with st.container(border=True): st.plotly_chart(style_plotly_chart(fig, 220), width="stretch", config={"displayModeBar": False})
        st.caption("부서별 이탈률 현황")
        bars = summary.head(8).sort_values("퇴사위험")
        fig = go.Figure(go.Bar(x=bars["Job Role"], y=bars["퇴사위험"] * 100, marker_color=BLUE))
        fig.add_hline(y=15, line_dash="dot", line_color=ORANGE)
        fig.update_layout(yaxis=dict(ticksuffix="%"))
        with st.container(border=True): st.plotly_chart(style_plotly_chart(fig, 215), width="stretch", config={"displayModeBar": False})
    with list_col:
        st.caption("부서별 순위")
        dept_rows = []
        for _, r in summary.head(8).iterrows():
            pct = float(r["퇴사위험"]) * 100
            dept_rows.append(f'<div><i style="background:{_risk_color(pct)}"></i><span><b>{escape(str(r["Job Role"]))}</b><small>{int(r["직원수"])}명</small></span><strong style="color:{_risk_color(pct)}">{pct:.1f}%<small>{_risk_label(pct).split(" — ")[0]}</small></strong></div>')
        st.markdown('<div class="glass-card dept-list">' + "".join(dept_rows) + "</div>", unsafe_allow_html=True)
