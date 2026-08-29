"""인사팀 Page 2~5에서 공유하는 DB·LightGBM 데이터와 화면 렌더러."""

# ruff: noqa: E501

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data.loader import load_raw_test, load_raw_train
from src.data.prediction import create_employee_predictions, load_prediction_model
from streamlit_ui import style_plotly_chart

CYAN, VIOLET, AMBER, EMERALD, ROSE = "#22D3EE", "#A78BFA", "#F59E0B", "#34D399", "#FB7185"
MODEL_PATH = Path("artifacts/ml/best_ml_model.joblib")


@st.cache_data(ttl=300)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_raw_train(), load_raw_test()


@st.cache_resource
def load_model(path: str, modified_time: float):
    del modified_time
    return load_prediction_model(path)


@st.cache_data(ttl=300)
def predict_employees(test: pd.DataFrame, train: pd.DataFrame, path: str, modified_time: float) -> pd.DataFrame:
    return create_employee_predictions(test, train, load_model(path, modified_time))


def score_map(series: pd.Series, mapping: dict[str, float], fallback: float = 50) -> pd.Series:
    return series.astype(str).map(mapping).fillna(fallback).astype(float)


def add_talent_score(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    education = score_map(result["Education Level"], {"High School": 40, "Associate Degree": 55, "Bachelor’s Degree": 72, "Bachelor's Degree": 72, "Master’s Degree": 86, "Master's Degree": 86, "PhD": 100})
    performance = score_map(result["Performance Rating"], {"Low": 35, "Below Average": 45, "Average": 62, "High": 82, "Excellent": 100})
    reputation = score_map(result["Company Reputation"], {"Poor": 35, "Fair": 55, "Good": 78, "Excellent": 100})
    tenure = pd.to_numeric(result["Company Tenure"], errors="coerce").fillna(0).clip(0, 30) / 30 * 100
    leadership = score_map(result["Leadership Opportunities"], {"No": 45, "Yes": 100})
    result["인재 가치 지수"] = pd.concat([education, performance, reputation, tenure, leadership], axis=1).mean(axis=1).round(1)
    result["학력 점수"], result["성과 점수"], result["평판 점수"], result["경력 점수"], result["리더십 점수"] = education, performance, reputation, tenure, leadership
    return result


def get_employees() -> pd.DataFrame:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("최종 예측 모델이 없습니다: artifacts/ml/best_ml_model.joblib")
    train, test = load_data()
    predictions = predict_employees(test, train, str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
    return add_talent_score(test.merge(predictions, left_on="Employee ID", right_on="employee_id", how="inner", validate="one_to_one"))


def page_intro(number: int, label: str, title: str, description: str, color: str) -> None:
    st.markdown(f'<div class="section-heading"><span class="section-chip" style="color:{color};border-color:{color}55;background:{color}15">PAGE {number}</span><h2>{title}</h2></div><div class="muted" style="margin-bottom:1.6rem">{description}</div>', unsafe_allow_html=True)


def render_salary(employees: pd.DataFrame) -> None:
    selected_id = st.selectbox("직원 선택", employees["Employee ID"], format_func=lambda value: f"EMP-{value} · {employees.loc[employees['Employee ID'].eq(value), 'Job Role'].iloc[0]} · {employees.loc[employees['Employee ID'].eq(value), 'Job Level'].iloc[0]}")
    employee = employees.loc[employees["Employee ID"].eq(selected_id)].iloc[0]
    risk, talent = float(employee["prediction"]), float(employee["인재 가치 지수"])
    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.markdown('<div class="panel-label">ATTRITION PREDICTION — INPUT VARIABLES</div>', unsafe_allow_html=True)
            inputs = pd.DataFrame({"입력 변수": ["월 소득", "워크-라이프 밸런스", "직무 만족도", "승진 횟수", "성별", "재직 기간"], "현재 값": [f"${float(employee['Monthly Income']):,.0f}", str(employee["Work-Life Balance"]), str(employee["Job Satisfaction"]), str(employee["Number of Promotions"]), str(employee["Gender"]), f"{employee['Years at Company']}년"]})
            st.dataframe(inputs, width="stretch", hide_index=True)
            st.markdown('<div class="panel-label" style="color:#A78BFA;margin-top:1.4rem">TALENT VALUE — INPUT VARIABLES</div>', unsafe_allow_html=True)
            talent_inputs = pd.DataFrame({"평가 항목": ["학력", "성과 평가", "회사 평판", "전사 기여 연수", "리더십 기회"], "점수": [employee["학력 점수"], employee["성과 점수"], employee["평판 점수"], employee["경력 점수"], employee["리더십 점수"]]})
            st.dataframe(talent_inputs.style.format({"점수": "{:.1f}"}), width="stretch", hide_index=True)
    with right:
        risk_color = ROSE if risk >= .6 else AMBER if risk >= .35 else EMERALD
        risk_text = "⚠️ 고위험 — 즉각 개입 필요" if risk >= .6 else "⚡ 중위험 — 모니터링 권고" if risk >= .35 else "✅ 저위험 — 양호"
        with st.container(border=True):
            st.markdown('<div class="panel-label">ATTRITION RISK</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="text-align:center;color:{risk_color};font:800 4.8rem JetBrains Mono;margin:1rem 0">{risk:.1%}</div><div style="text-align:center;color:{risk_color}">{risk_text}</div>', unsafe_allow_html=True)
            st.progress(risk)
        with st.container(border=True):
            st.markdown('<div class="panel-label" style="color:#A78BFA">TALENT VALUE INDEX</div>', unsafe_allow_html=True)
            talent_text = "🌟 핵심 인재" if talent >= 75 else "⭐ 성장 인재" if talent >= 50 else "📌 관찰 인재"
            st.markdown(f'<div style="text-align:center;color:{VIOLET};font:800 4.8rem JetBrains Mono;margin:1rem 0">{talent:.0f}</div><div style="text-align:center;color:{VIOLET}">{talent_text}</div>', unsafe_allow_html=True)
            st.progress(talent / 100)
    if risk >= .6 and talent >= 70:
        advice = "핵심 인재이나 이탈 위험이 높습니다. 즉각적인 보상 조정 및 경력 개발 면담을 권고합니다."
    elif risk >= .35:
        advice = "중간 위험 구간입니다. 정기 면담으로 직무 만족도와 워라밸 원인을 확인하세요."
    else:
        advice = "안정적인 상태입니다. 현재 처우를 유지하되 정기 모니터링을 지속하세요."
    st.markdown(f'<div class="decision-note"><b>💡 인사 담당자 권고</b><br>{advice}</div>', unsafe_allow_html=True)


def render_team(employees: pd.DataFrame) -> None:
    roles = st.multiselect("후보 직무", sorted(employees["Job Role"].dropna().unique()), default=sorted(employees["Job Role"].dropna().unique())[:3])
    pool = employees[employees["Job Role"].isin(roles)].copy()
    pool["팀 적합 점수"] = (pool["인재 가치 지수"] * .6 + (1 - pool["prediction"]) * 40).round(1)
    candidates = pool.nlargest(12, "팀 적합 점수")
    selected = st.multiselect("팀원 선택", candidates["Employee ID"], default=candidates.head(5)["Employee ID"].tolist(), format_func=lambda value: f"EMP-{value} · {candidates.loc[candidates['Employee ID'].eq(value), 'Job Role'].iloc[0]}")
    team = candidates[candidates["Employee ID"].isin(selected)]
    if team.empty:
        st.info("팀원을 한 명 이상 선택해 주세요.")
        return
    with st.container(key="stat-bar-team"):
        kpis = st.columns(4)
        kpis[0].metric("선택된 팀", f"{len(team)}명")
        kpis[1].metric("팀 안정도", f"{(1-team['prediction'].mean())*100:.0f}")
        kpis[2].metric("평균 퇴사 위험", f"{team['prediction'].mean():.1%}")
        kpis[3].metric("평균 인재 가치", f"{team['인재 가치 지수'].mean():.1f}")
    display = team[["Employee ID", "Job Role", "Job Level", "prediction", "인재 가치 지수", "팀 적합 점수"]].rename(columns={"Employee ID": "직원", "Job Role": "직무", "Job Level": "직급", "prediction": "퇴사 위험"})
    st.dataframe(display.style.format({"퇴사 위험": "{:.1%}", "인재 가치 지수": "{:.1f}", "팀 적합 점수": "{:.1f}"}).background_gradient(subset=["팀 적합 점수"], cmap="Purples"), width="stretch", hide_index=True)


def render_people_decision(employees: pd.DataFrame) -> None:
    role = st.selectbox("동일 조건 비교 그룹", sorted(employees["Job Role"].dropna().unique()), key="people_role")
    group = employees[employees["Job Role"].eq(role)].copy()
    group["복합 점수"] = (group["인재 가치 지수"] - group["prediction"] * 30).round(1)
    group = group.sort_values("복합 점수", ascending=False).head(15)
    group.insert(0, "순위", np.arange(1, len(group) + 1))
    third = max(1, len(group) // 3)
    group["판정"] = ["승진 우선" if index < third else "구조조정 검토" if index >= len(group) - third else "관찰" for index in range(len(group))]
    display = group[["순위", "Employee ID", "Job Level", "Performance Rating", "인재 가치 지수", "prediction", "복합 점수", "판정"]].rename(columns={"Employee ID": "직원", "Job Level": "직급", "Performance Rating": "성과 평가", "prediction": "퇴사 위험"})
    st.dataframe(display.style.format({"퇴사 위험": "{:.1%}", "인재 가치 지수": "{:.1f}", "복합 점수": "{:.1f}"}), width="stretch", hide_index=True)
    summary = st.columns(3)
    summary[0].metric("승진 우선 대상", f"{(group['판정']=='승진 우선').sum()}명")
    summary[1].metric("관찰 대상", f"{(group['판정']=='관찰').sum()}명")
    summary[2].metric("구조조정 검토", f"{(group['판정']=='구조조정 검토').sum()}명")
    st.warning("이 결과는 면담·재배치·교육 검토를 위한 보조 정보이며 자동 불이익 결정에 사용할 수 없습니다.")


def render_executive(employees: pd.DataFrame) -> None:
    high = employees["prediction"].ge(.6)
    tenure = pd.to_numeric(employees["Years at Company"], errors="coerce")
    with st.container(key="stat-bar-overview"):
        kpis = st.columns(4)
        kpis[0].metric("예측 평균 퇴사율", f"{employees['prediction'].mean():.1%}")
        kpis[1].metric("고위험 직원", f"{high.sum():,}명", f"전체의 {high.mean():.1%}", delta_color="inverse")
        kpis[2].metric("평균 재직기간", f"{tenure.mean():.1f}년")
        kpis[3].metric("전사 안정 지수", f"{(1-employees['prediction'].mean())*100:.1f}")
    role_risk = employees.groupby("Job Role", as_index=False).agg(직원수=("Employee ID", "size"), 퇴사위험=("prediction", "mean")).sort_values("퇴사위험", ascending=False)
    risk_band = pd.cut(employees["prediction"], [-.01, .35, .6, 1], labels=["저위험 (0–35%)", "중위험 (35–60%)", "고위험 (60%+)"]).value_counts(sort=False)
    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.markdown('<div class="panel-label">DEPARTMENT RISK</div><div class="panel-title">직무별 평균 퇴사 위험도</div>', unsafe_allow_html=True)
            colors = [ROSE if value >= .6 else AMBER if value >= .35 else EMERALD for value in role_risk["퇴사위험"]]
            fig = go.Figure(go.Bar(x=role_risk["퇴사위험"], y=role_risk["Job Role"], orientation="h", marker_color=colors))
            style_plotly_chart(fig, 300).update_xaxes(tickformat=".0%")
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    with right:
        with st.container(border=True):
            st.markdown('<div class="panel-label" style="color:#F59E0B">RISK DISTRIBUTION</div><div class="panel-title">위험 구간별 직원 분포</div>', unsafe_allow_html=True)
            fig = go.Figure(go.Pie(labels=risk_band.index.astype(str), values=risk_band.values, hole=.58, marker_colors=[EMERALD, AMBER, ROSE]))
            style_plotly_chart(fig, 300)
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.markdown('<div class="panel-label" style="color:#FB7185;margin-top:1.5rem">⚠ IMMEDIATE INTERVENTION</div><div class="panel-title">즉각 개입 권고 직무</div>', unsafe_allow_html=True)
    alert = role_risk[role_risk["퇴사위험"].ge(.35)].copy()
    alert["조치"] = np.where(alert["퇴사위험"].ge(.6), "즉각 개입 필요", "모니터링 강화")
    st.dataframe(alert.style.format({"퇴사위험": "{:.1%}"}), width="stretch", hide_index=True)
