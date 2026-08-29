"""인사 담당자용 Page 2~5 데이터 계산과 Liquid Glass 화면 구성."""

from __future__ import annotations

from html import escape
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data.loader import load_predictions, load_raw_test
from streamlit_ui import style_plotly_chart

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
    labels = df.apply(lambda r: f"EMP-{r['Employee ID']} — {r['Job Role']}", axis=1).tolist()
    selected = st.selectbox("직원 선택", labels, label_visibility="visible")
    row = df.iloc[labels.index(selected)]
    risk = float(row["prediction"]) * 100
    color = _risk_color(risk)
    st.markdown(f'<div class="employee-strip"><div><span>부서</span><b>{escape(str(row["Job Role"]))}</b></div><div><span>직급</span><b>{escape(str(row["Job Level"]))}</b></div></div>', unsafe_allow_html=True)

    features = _feature_scores(row)
    left, center, right = st.columns([.72, 2.05, 2.05], gap="small")
    with left:
        st.caption("이탈 위험도")
        offset = 214 * (1 - risk / 100)
        gauge = f'''<div class="glass-card risk-card"><svg viewBox="0 0 180 112"><path d="M22 92 A68 68 0 0 1 158 92" fill="none" stroke="rgba(60,60,67,.12)" stroke-width="10" stroke-linecap="round"/><path d="M22 92 A68 68 0 0 1 158 92" fill="none" stroke="{color}" stroke-width="10" stroke-linecap="round" stroke-dasharray="214" stroke-dashoffset="{offset:.1f}"/><text x="90" y="78" text-anchor="middle" fill="{color}" font-size="30" font-weight="700">{risk:.0f}</text><text x="90" y="98" text-anchor="middle" fill="rgba(60,60,67,.6)" font-size="10">이탈 위험도</text></svg><div class="risk-pill" style="color:{color};background:{color}10">{_risk_label(risk)}</div><div class="talent-row"><span>인재 가치</span><b>{row['Talent Value']:.0f}</b></div></div>'''
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
    roles = sorted(df["Job Role"].dropna().unique())
    project_names = ["AI 추천 엔진 고도화", "고객 데이터 플랫폼 구축", "보안 인프라 강화", "차세대 UX 리뉴얼"]
    project = st.segmented_control("프로젝트 선택", project_names, default=project_names[0])
    project_index = project_names.index(project or project_names[0])
    role_window = [roles[(project_index + i) % len(roles)] for i in range(min(3, len(roles)))]
    pool = df[df["Job Role"].isin(role_window)].copy()
    pool["팀 적합 점수"] = (pool["Talent Value"] * .6 + (1 - pool["prediction"]) * 40).round(1)
    pool = pool.nlargest(20, "팀 적합 점수")
    label_map = {f"EMP-{r['Employee ID']} · {r['Job Role']}": r["Employee ID"] for _, r in pool.iterrows()}
    choices = list(label_map)
    left, right = st.columns([1, 4.2], gap="small")
    with left:
        st.caption("프로젝트 정보")
        priority = ["HIGH", "MEDIUM", "CRITICAL", "MEDIUM"][project_index]
        st.markdown(f'<div class="glass-card project-card"><div><b>{project}</b><span class="priority">{priority}</span></div><p><span>마감일</span><b>2026-09-{30-project_index*3:02d}</b></p><p><span>팀 구성</span><b>최대 7명</b></p></div>', unsafe_allow_html=True)
        st.caption("현재 팀원")
        chosen = st.multiselect("팀원 교체", choices, default=choices[:5], max_selections=7, label_visibility="collapsed")
        team = pool[pool["Employee ID"].isin([label_map[x] for x in chosen])].copy()
        if team.empty:
            st.info("한 명 이상의 팀원을 선택해 주세요.")
            return
        people = []
        for _, r in team.iterrows():
            risk = float(r["prediction"]) * 100
            people.append(f'<div class="member-row"><i>{str(r["Employee ID"])[-2:]}</i><div><b>EMP-{r["Employee ID"]}</b><span>{escape(str(r["Job Role"]))}</span></div><strong style="color:{_risk_color(risk)}">{risk:.0f}%<small>위험</small></strong><em>교체</em></div>')
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
        st.markdown(f'<div class="team-warning">팀 평균 이탈위험도 <b>{avg_risk:.0f}%</b> — 고위험 팀원이 다수 포함되어 있습니다. 위 팀원 선택에서 구성을 최적화하세요.</div>', unsafe_allow_html=True)


def render_people_decision(df: pd.DataFrame) -> None:
    choice = st.segmented_control("위험 필터", ["전체", "안전", "위험"], default="전체")
    ranked = df.copy()
    ranked["안정점수"] = (ranked["Talent Value"] * (1 - ranked["prediction"])).round(1)
    if choice == "안전":
        ranked = ranked[ranked["prediction"] < .35]
    elif choice == "위험":
        ranked = ranked[ranked["prediction"] >= .60]
    ranked = ranked.nlargest(12, "안정점수").reset_index(drop=True)
    st.markdown('<div class="formula"><b>안정 인재가치</b> = 인재가치 × (1 − 이탈위험도)</div>', unsafe_allow_html=True)
    rows = []
    for i, r in ranked.iterrows():
        risk, talent, stable = float(r["prediction"]) * 100, float(r["Talent Value"]), float(r["안정점수"])
        level, color = (("안전", GREEN) if risk < 35 else ("주의", ORANGE) if risk < 60 else ("위험", RED))
        medal = ["🥇", "🥈", "🥉"][i] if i < 3 else str(i + 1)
        rows.append(f'''<tr><td>{medal}</td><td><div class="person-cell"><i>{str(r['Employee ID'])[-2:]}</i><span><b>EMP-{r['Employee ID']}</b><small>{escape(str(r['Job Role']))} · {escape(str(r['Job Level']))}</small></span></div></td><td><div class="mini-bar"><i style="width:{risk}%;background:{color}"></i></div><small>{risk:.0f}</small></td><td><div class="mini-bar"><i style="width:{talent}%;background:{BLUE}"></i></div><small>{talent:.0f}</small></td><td class="stable" style="color:{GREEN if stable>=60 else ORANGE}">{stable:.0f}</td><td><span class="level" style="color:{color};border-color:{color}55;background:{color}0D">{level}</span></td><td><span class="action">승진 추천</span> <span class="action">발령 추천</span><br><span class="action">구조조정 검토</span></td></tr>''')
    table = '<div class="glass-card ranking-wrap"><table class="ranking"><thead><tr><th>#</th><th>직원 정보</th><th>이탈위험도 ↕</th><th>인재가치 ↕</th><th>안정점수 ↓</th><th>위험수준</th><th>인사 조치</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table></div>"
    st.markdown(table, unsafe_allow_html=True)


def render_executive(df: pd.DataFrame) -> None:
    summary = df.groupby("Job Role", as_index=False).agg(직원수=("Employee ID", "count"), 퇴사위험=("prediction", "mean")).sort_values("퇴사위험", ascending=False)
    risk = df["prediction"] * 100
    alerts = summary.head(3)
    st.caption("경고")
    alert_html = '<div class="glass-card alert-list">' + "".join(f'<div><i></i><b>{escape(str(r["Job Role"]))}</b> 이탈률 <strong>{r["퇴사위험"]:.1%}</strong> — 즉각적인 인사 개입 권고 <span>확인</span></div>' for _, r in alerts.iterrows()) + "</div>"
    st.markdown(alert_html, unsafe_allow_html=True)
    _metric_cards([("전사 이탈률", f"{risk.mean():.1f}%", "예측 평균", ORANGE), ("위험 부서 수", f"{(summary['퇴사위험'] >= .6).sum()}개", f"/ 총 {len(summary)}개 부서", RED), ("연간 이탈 예측", f"{int((df['prediction'] >= .6).sum())}명", "고위험 직원", ORANGE), ("조직 안정도", f"{100-risk.mean():.1f}", "/ 100점", "#8E8E93")])
    chart_col, list_col = st.columns([5.2, .75], gap="small")
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
