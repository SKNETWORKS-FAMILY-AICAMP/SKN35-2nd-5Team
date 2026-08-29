"""인사 담당자용 Page 2~5 데이터 계산과 화면 구성."""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data.loader import load_raw_test, load_raw_train
from src.data.prediction import create_employee_predictions, load_prediction_model
from streamlit_ui import COLORS, style_plotly_chart

MODEL_PATH = Path("artifacts/ml/best_ml_model.joblib")
CYAN, VIOLET, AMBER, EMERALD = "#21d4ee", "#9b7df4", "#ffa20a", "#38d0a0"


@st.cache_data(show_spinner="DB에서 직원 데이터를 불러오는 중입니다...")
def get_employees() -> pd.DataFrame:
    train, test = load_raw_train(), load_raw_test()
    model = load_prediction_model(MODEL_PATH)
    predictions = create_employee_predictions(test, train, model)
    frame = test.merge(predictions, left_on="Employee ID", right_on="employee_id", how="left")
    frame["prediction"] = frame["prediction"].fillna(0.5)
    education = {"High School": 55, "Associate Degree": 65, "Bachelor’s Degree": 75, "Master’s Degree": 85, "PhD": 95}
    performance = {"Low": 40, "Below Average": 50, "Average": 65, "High": 82, "Very High": 95}
    frame["Talent Value"] = (frame["Education Level"].map(education).fillna(65) * 0.35 + frame["Performance Rating"].map(performance).fillna(65) * 0.45 + frame["Years at Company"].clip(0, 20) / 20 * 20).round(1)
    return frame


def page_intro(page: int, color: str, title: str, copy: str) -> None:
    st.markdown(f'<div class="section-heading" style="--accent:{color}"><span class="chip">PAGE {page}</span><h2>{title}</h2></div><p class="section-copy" style="margin:0 8px 28px">{copy}</p>', unsafe_allow_html=True)


def render_salary(df: pd.DataFrame) -> None:
    page_intro(2, CYAN, "개별 ID 시뮬레이션", "LightGBM이 계산한 퇴사 확률과 사내 인재 가치 지수를 협상 판단에 활용합니다.")
    labels = df.apply(lambda row: f"EMP-{row['Employee ID']} · {row['Job Role']} · {row['Job Level']}", axis=1)
    selected = st.selectbox("직원 선택", labels, index=0)
    row = df.iloc[labels.tolist().index(selected)]
    left, right = st.columns(2)
    inputs = pd.DataFrame({"입력 변수": ["월 소득", "워크라이프 밸런스", "직무 만족도", "승진 횟수", "성별", "재직 기간"], "현재 값": [f"${row['Monthly Income']:,.0f}", row["Work-Life Balance"], row["Job Satisfaction"], int(row["Number of Promotions"]), row["Gender"], f"{row['Years at Company']:.0f}년"]})
    talent = pd.DataFrame({"평가 항목": ["학력", "성과 평가", "직급", "회사 경력"], "점수": [row["Education Level"], row["Performance Rating"], row["Job Level"], f"{row['Company Tenure']:.0f}년"]})
    with left:
        with st.container(border=True):
            st.markdown('<div class="panel-title">ATTRITION PREDICTION — INPUT VARIABLES</div>', unsafe_allow_html=True)
            st.dataframe(inputs, hide_index=True, use_container_width=True)
        with st.container(border=True):
            st.markdown('<div class="panel-title" style="color:#9b7df4">TALENT VALUE — INPUT VARIABLES</div>', unsafe_allow_html=True)
            st.dataframe(talent, hide_index=True, use_container_width=True)
    risk = float(row["prediction"]) * 100
    with right:
        with st.container(border=True):
            note = "고위험 — 즉각 개입 필요" if risk >= 60 else "관찰 및 정기 면담"
            st.markdown(f'<div class="panel-title">ATTRITION RISK</div><div class="big-number" style="color:{COLORS["rose"]}">{risk:.1f}%</div><div class="center-note" style="color:{COLORS["rose"]}">⚠ {note}</div>', unsafe_allow_html=True)
            st.progress(risk / 100)
        with st.container(border=True):
            st.markdown(f'<div class="panel-title" style="color:#9b7df4">TALENT VALUE INDEX</div><div class="big-number" style="color:#9b7df4">{row["Talent Value"]:.0f}</div><div class="center-note" style="color:#9b7df4">⭐ 성장 인재</div>', unsafe_allow_html=True)
            st.progress(float(row["Talent Value"]) / 100)


def render_team(df: pd.DataFrame) -> None:
    page_intro(3, VIOLET, "팀원 변경 시뮬레이션", "후보 직무와 팀원을 바꾸면서 평균 퇴사 위험과 팀 적합도를 확인합니다.")
    roles = sorted(df["Job Role"].dropna().unique())
    picked_roles = st.multiselect("후보 직무", roles, default=roles[:3])
    pool = df[df["Job Role"].isin(picked_roles)].copy()
    pool["팀 적합 점수"] = (pool["Talent Value"] * 0.6 + (1 - pool["prediction"]) * 40).round(1)
    pool = pool.nlargest(20, "팀 적합 점수")
    labels = pool.apply(lambda row: f"EMP-{row['Employee ID']} · {row['Job Role']}", axis=1)
    chosen = st.multiselect("팀원 선택", labels, default=labels.head(5).tolist())
    team = pool[labels.isin(chosen)].copy()
    if team.empty:
        st.info("한 명 이상의 팀원을 선택해 주세요.")
        return
    values = [f"{len(team)}명", f"{team['팀 적합 점수'].mean():.0f}", f"{team['prediction'].mean()*100:.1f}%", f"{team['Talent Value'].mean():.1f}"]
    labels_text = ["선택된 팀", "팀 안정도", "평균 퇴사 위험", "평균 인재 가치"]
    st.markdown('<div class="metric-strip">' + "".join(f'<div><small>{label}</small><strong>{value}</strong></div>' for label, value in zip(labels_text, values, strict=True)) + "</div>", unsafe_allow_html=True)
    table = team[["Employee ID", "Job Role", "Job Level", "prediction", "Talent Value", "팀 적합 점수"]].rename(columns={"Employee ID": "직원", "Job Role": "직무", "Job Level": "직급", "prediction": "퇴사 위험", "Talent Value": "인재 가치 지수"})
    st.dataframe(table.style.format({"퇴사 위험": "{:.1%}", "인재 가치 지수": "{:.1f}", "팀 적합 점수": "{:.1f}"}).background_gradient(subset=["팀 적합 점수"], cmap="Purples"), hide_index=True, use_container_width=True)


def render_people_decision(df: pd.DataFrame) -> None:
    page_intro(4, AMBER, "승진·발령 우선순위", "인재 가치와 잔류 가능성을 함께 반영한 복합 점수로 검토 순위를 제공합니다.")
    role = st.selectbox("동일 조건 비교 그룹", sorted(df["Job Role"].dropna().unique()))
    ranked = df[df["Job Role"] == role].copy()
    ranked["복합 점수"] = (ranked["Talent Value"] - ranked["prediction"] * 30).round(1)
    ranked = ranked.nlargest(15, "복합 점수").reset_index(drop=True)
    ranked["순위"] = ranked.index + 1
    ranked["판정"] = ["승진 우선" if i < 5 else "관찰" if i < 10 else "구조조정 검토" for i in ranked.index]
    table = ranked[["순위", "Employee ID", "Job Level", "Performance Rating", "Talent Value", "prediction", "복합 점수", "판정"]].rename(columns={"Employee ID": "직원", "Job Level": "직급", "Performance Rating": "성과 평가", "Talent Value": "인재 가치 지수", "prediction": "퇴사 위험"})
    st.dataframe(table.style.format({"인재 가치 지수": "{:.1f}", "퇴사 위험": "{:.1%}", "복합 점수": "{:.1f}"}), hide_index=True, use_container_width=True, height=390)
    for col, label in zip(st.columns(3), ["승진 우선 대상", "관찰 대상", "구조조정 검토"], strict=True):
        col.metric(label, f"{min(5, len(ranked))}명")


def render_executive(df: pd.DataFrame) -> None:
    page_intro(5, EMERALD, "부서별 모니터링", "고위험 인원과 직무별 평균 퇴사 위험을 확인해 선제 대응 대상을 찾습니다.")
    risk = df["prediction"] * 100
    values = [f"{risk.mean():.1f}%", f"{(risk >= 60).sum():,}명", f"{df['Years at Company'].mean():.1f}년", f"{100-risk.mean():.1f}"]
    labels_text = ["예측 평균 퇴사율", "고위험 직원", "평균 재직 기간", "전사 안정 지수"]
    st.markdown('<div class="metric-strip">' + "".join(f'<div><small>{label}</small><strong>{value}</strong></div>' for label, value in zip(labels_text, values, strict=True)) + "</div>", unsafe_allow_html=True)
    summary = df.groupby("Job Role", as_index=False).agg(직원수=("Employee ID", "count"), 퇴사위험=("prediction", "mean")).sort_values("퇴사위험", ascending=False)
    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.markdown('<div class="panel-title">DEPARTMENT RISK</div><h3>직무별 평균 퇴사 위험도</h3>', unsafe_allow_html=True)
            fig = go.Figure(go.Bar(x=summary["퇴사위험"] * 100, y=summary["Job Role"], orientation="h", marker_color=AMBER))
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(style_plotly_chart(fig, 270), use_container_width=True)
    with right:
        with st.container(border=True):
            st.markdown('<div class="panel-title" style="color:#ffa20a">RISK DISTRIBUTION</div><h3>위험 구간별 직원 분포</h3>', unsafe_allow_html=True)
            bands = pd.cut(risk, [-1, 35, 60, 101], labels=["저위험 (0~35%)", "중위험 (35~60%)", "고위험 (60%+)"]).value_counts()
            fig = go.Figure(go.Pie(labels=bands.index, values=bands.values, hole=.55, marker_colors=[EMERALD, AMBER, COLORS["rose"]]))
            st.plotly_chart(style_plotly_chart(fig, 270), use_container_width=True)
    st.markdown('<div class="panel-title" style="color:#ff6b86;margin-top:22px">⚠ IMMEDIATE INTERVENTION</div><h3>즉각 개입 권고 직무</h3>', unsafe_allow_html=True)
    alert = summary.head(8).copy()
    alert["퇴사위험"] *= 100
    alert["조치"] = "모니터링 강화"
    st.dataframe(alert.style.format({"퇴사위험": "{:.1f}%"}), hide_index=True, use_container_width=True)
