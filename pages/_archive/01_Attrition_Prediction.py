from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.data.loader import load_raw_test, load_raw_train
from src.data.prediction import create_employee_predictions, load_prediction_model
from streamlit_ui import apply_page_style, page_header, top_navigation

st.set_page_config(page_title="퇴사 예측 · EXITWISE", layout="wide")

MODEL_PATH = Path("artifacts/ml/best_ml_model.joblib")
RISK_FEATURES = [
    "Monthly Income",
    "Work-Life Balance",
    "Job Satisfaction",
    "Number of Promotions",
    "Gender",
    "Years at Company",
]
VALUE_FEATURES = [
    "Education Level",
    "Performance Rating",
    "Company Reputation",
    "Company Tenure",
    "Leadership Opportunities",
]

FEATURE_LABELS = {
    "Monthly Income": "월 소득",
    "Work-Life Balance": "일과 삶의 균형",
    "Job Satisfaction": "직무 만족도",
    "Number of Promotions": "승진 횟수",
    "Gender": "성별",
    "Years at Company": "현 직장 근속연수",
    "Education Level": "학력",
    "Performance Rating": "성과 평가",
    "Company Reputation": "회사 평판",
    "Company Tenure": "총 경력연수",
    "Leadership Opportunities": "리더십 기회",
}


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_raw_train(), load_raw_test()


@st.cache_resource
def load_model(path: str, modified_time: float):
    del modified_time
    return load_prediction_model(path)


@st.cache_data
def predict_all_employees(
    raw_test: pd.DataFrame,
    raw_train: pd.DataFrame,
    model_path: str,
    modified_time: float,
) -> pd.DataFrame:
    model = load_model(model_path, modified_time)
    return create_employee_predictions(raw_test, raw_train, model)


def normalized_map(series: pd.Series, scores: dict[str, float], fallback: float = 50.0) -> pd.Series:
    return series.astype(str).map(scores).fillna(fallback).astype(float)


def add_talent_value(frame: pd.DataFrame) -> pd.DataFrame:
    """요청된 5개 항목을 동일 가중치로 환산해 0~100 인재 가치 지수를 만든다."""

    result = frame.copy()
    result["학력 점수"] = normalized_map(
        result["Education Level"],
        {
            "High School": 40,
            "Associate Degree": 55,
            "Bachelor’s Degree": 72,
            "Bachelor's Degree": 72,
            "Master’s Degree": 86,
            "Master's Degree": 86,
            "PhD": 100,
        },
    )
    result["성과 점수"] = normalized_map(
        result["Performance Rating"],
        {"Low": 35, "Below Average": 45, "Average": 62, "High": 82, "Excellent": 100},
    )
    result["평판 점수"] = normalized_map(
        result["Company Reputation"],
        {"Poor": 35, "Fair": 55, "Good": 78, "Excellent": 100},
    )
    tenure = pd.to_numeric(result["Company Tenure"], errors="coerce").fillna(0)
    result["경력 점수"] = (tenure.clip(lower=0, upper=30) / 30 * 100).round(1)
    result["리더십 점수"] = normalized_map(
        result["Leadership Opportunities"], {"No": 45, "Yes": 100}
    )
    score_columns = ["학력 점수", "성과 점수", "평판 점수", "경력 점수", "리더십 점수"]
    result["인재 가치 지수"] = result[score_columns].mean(axis=1).round(1)
    return result


def section_header(section_id: str, number: str, title: str, description: str) -> None:
    st.markdown(
        f"""
        <div id="{section_id}" class="scroll-section">
            <div class="section-kicker">{number}</div>
            <div class="section-title">{title}</div>
            <div class="section-desc">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def feature_pills(features: list[str]) -> None:
    pills = "".join(
        f'<span class="feature-pill">{FEATURE_LABELS.get(feature, feature)}</span>'
        for feature in features
    )
    st.markdown(f'<div class="feature-pills">{pills}</div>', unsafe_allow_html=True)


apply_page_style()
top_navigation("attrition")
page_header(
    "INTERNAL PEOPLE INTELLIGENCE",
    "퇴사 예측",
    "사내 인사팀의 보상·팀 구성·인사발령·조직 안정성 판단을 하나의 흐름에서 지원합니다.",
)

st.markdown(
    """
    <nav class="section-jump">
        <a href="#salary">01 연봉 협상</a>
        <a href="#team">02 팀 구성</a>
        <a href="#decision">03 인사발령</a>
        <a href="#overview">04 전사 안정도</a>
    </nav>
    """,
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists():
    st.warning("최종 예측 모델이 없습니다. artifacts/ml/best_ml_model.joblib을 먼저 생성해 주세요.")
    st.stop()

try:
    train_data, test_data = load_data()
    predictions = predict_all_employees(
        test_data,
        train_data,
        str(MODEL_PATH),
        MODEL_PATH.stat().st_mtime,
    )
    employees = test_data.merge(
        predictions,
        left_on="Employee ID",
        right_on="employee_id",
        how="inner",
        validate="one_to_one",
    )
    employees = add_talent_value(employees)
except Exception as exc:
    st.error(f"직원 데이터 또는 예측 모델을 불러오지 못했습니다: {exc}")
    st.stop()


# 01. Salary negotiation
section_header(
    "salary",
    "01 · COMPENSATION INTELLIGENCE",
    "연봉 협상 지원",
    "직원 ID별 퇴사 확률과 인재 가치 지수를 함께 확인합니다.",
)

employee_ids = employees["Employee ID"].tolist()
selected_id = st.selectbox(
    "직원 선택",
    employee_ids,
    format_func=lambda employee_id: (
        f"ID {employee_id} · "
        f"{employees.loc[employees['Employee ID'].eq(employee_id), 'Job Role'].iloc[0]} · "
        f"{employees.loc[employees['Employee ID'].eq(employee_id), 'Job Level'].iloc[0]}"
    ),
)
employee = employees.loc[employees["Employee ID"].eq(selected_id)].iloc[0]
risk = float(employee["prediction"])
talent = float(employee["인재 가치 지수"])
risk_label = "높음" if risk >= 0.6 else "보통" if risk >= 0.35 else "낮음"

with st.container(key="stat-bar-salary"):
    metrics = st.columns(4)
    metrics[0].metric("부서 · 직급", f"{employee['Job Role']} · {employee['Job Level']}")
    metrics[1].metric("현재 월 소득", f"${float(employee['Monthly Income']):,.0f}")
    metrics[2].metric("퇴사 예측률", f"{risk:.1%}", risk_label, delta_color="inverse")
    metrics[3].metric("인재 가치 지수", f"{talent:.1f} / 100")

left, right = st.columns(2)
with left:
    st.subheader("퇴사 예측 활용 항목")
    feature_pills(RISK_FEATURES)
    risk_display = pd.DataFrame(
        {
            "항목": [FEATURE_LABELS[feature] for feature in RISK_FEATURES],
            "현재 값": [str(employee[feature]) for feature in RISK_FEATURES],
        }
    )
    st.dataframe(risk_display, width="stretch", hide_index=True)
with right:
    st.subheader("인재 가치 구성")
    feature_pills(VALUE_FEATURES)
    talent_breakdown = pd.DataFrame(
        {
            "항목": ["학력", "성과 평가", "회사 평판", "총 경력연수", "리더십 기회"],
            "점수": [
                employee["학력 점수"],
                employee["성과 점수"],
                employee["평판 점수"],
                employee["경력 점수"],
                employee["리더십 점수"],
            ],
        }
    ).set_index("항목")
    st.bar_chart(talent_breakdown, horizontal=True, height=250)

if risk >= 0.6 and talent >= 70:
    salary_message = "퇴사 위험과 인재 가치가 모두 높습니다. 보상 수준 및 성장 경로 면담을 우선 검토하세요."
elif risk >= 0.6:
    salary_message = "퇴사 위험이 높습니다. 보상 외 업무 만족도와 워라밸 요인을 함께 확인하세요."
else:
    salary_message = "현재 잔류 가능성이 비교적 안정적입니다. 성과와 역할 확장 가능성을 중심으로 협상하세요."
st.markdown(f'<div class="decision-note"><b>협상 제안</b> · {salary_message}</div>', unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# 02. Team composition
section_header(
    "team",
    "02 · TEAM COMPOSITION",
    "프로젝트 팀 구성",
    "낮은 퇴사 위험과 높은 인재 가치를 함께 반영해 안정적인 후보 조합을 제안합니다.",
)

team_controls = st.columns([2, 1])
with team_controls[0]:
    selected_roles = st.multiselect(
        "후보 부서",
        sorted(employees["Job Role"].dropna().unique()),
        default=sorted(employees["Job Role"].dropna().unique())[:3],
    )
with team_controls[1]:
    team_size = st.slider("추천 인원", 3, 10, 5)

candidate_pool = employees[employees["Job Role"].isin(selected_roles)].copy()
candidate_pool["팀 적합 점수"] = (
    candidate_pool["인재 가치 지수"] * 0.6
    + (1 - candidate_pool["prediction"]) * 100 * 0.4
).round(1)
recommended_team = candidate_pool.nlargest(team_size, "팀 적합 점수")

if recommended_team.empty:
    st.info("후보 부서를 하나 이상 선택해 주세요.")
else:
    with st.container(key="stat-bar-team"):
        team_metrics = st.columns(4)
        team_metrics[0].metric("추천 팀원", f"{len(recommended_team)}명")
        team_metrics[1].metric("평균 인재 가치", f"{recommended_team['인재 가치 지수'].mean():.1f}")
        team_metrics[2].metric("평균 퇴사 위험", f"{recommended_team['prediction'].mean():.1%}")
        team_metrics[3].metric("팀 안정성", f"{(1 - recommended_team['prediction'].mean()):.1%}")
    team_table = recommended_team[
        ["Employee ID", "Job Role", "Job Level", "인재 가치 지수", "prediction", "팀 적합 점수"]
    ].rename(
        columns={
            "Employee ID": "직원 ID",
            "Job Role": "부서",
            "Job Level": "직급",
            "prediction": "퇴사 예측률",
        }
    )
    st.dataframe(
        team_table.style.format({"퇴사 예측률": "{:.1%}", "인재 가치 지수": "{:.1f}", "팀 적합 점수": "{:.1f}"}),
        width="stretch",
        hide_index=True,
    )
    st.caption("팀 적합 점수 = 인재 가치 60% + 잔류 가능성 40%")

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# 03. Promotion and restructuring support
section_header(
    "decision",
    "03 · PEOPLE DECISIONS",
    "인사발령 · 승진 · 구조조정",
    "인재 가치와 퇴사 위험을 같은 기준으로 계산하되, 최종 판단은 인사 검토자가 수행합니다.",
)

decision_role = st.selectbox(
    "비교할 동일 조건 그룹",
    sorted(employees["Job Role"].dropna().unique()),
    key="decision_role",
)
same_condition = employees[employees["Job Role"].eq(decision_role)].copy()
same_condition["승진 우선 점수"] = (
    same_condition["인재 가치 지수"] * 0.65
    + (1 - same_condition["prediction"]) * 100 * 0.35
).round(1)
same_condition["검토 우선 점수"] = (
    (100 - same_condition["인재 가치 지수"]) * 0.55
    + same_condition["prediction"] * 100 * 0.45
).round(1)

promotion_tab, restructuring_tab = st.tabs(["승진 우선순위", "구조조정 검토"])
with promotion_tab:
    promotion = same_condition.nlargest(10, "승진 우선 점수")[
        ["Employee ID", "Job Level", "Performance Rating", "인재 가치 지수", "prediction", "승진 우선 점수"]
    ].rename(
        columns={
            "Employee ID": "직원 ID",
            "Job Level": "직급",
            "Performance Rating": "성과 평가",
            "prediction": "퇴사 예측률",
        }
    )
    promotion.insert(0, "순위", np.arange(1, len(promotion) + 1))
    st.dataframe(
        promotion.style.format({"퇴사 예측률": "{:.1%}", "인재 가치 지수": "{:.1f}", "승진 우선 점수": "{:.1f}"}),
        width="stretch",
        hide_index=True,
    )
    st.caption("승진 우선 점수 = 인재 가치 65% + 잔류 가능성 35%. 동일 조건에서는 퇴사 예측률이 낮은 직원을 우선합니다.")
with restructuring_tab:
    restructuring = same_condition.nlargest(10, "검토 우선 점수")[
        ["Employee ID", "Job Level", "Performance Rating", "인재 가치 지수", "prediction", "검토 우선 점수"]
    ].rename(
        columns={
            "Employee ID": "직원 ID",
            "Job Level": "직급",
            "Performance Rating": "성과 평가",
            "prediction": "퇴사 예측률",
        }
    )
    restructuring.insert(0, "검토 순위", np.arange(1, len(restructuring) + 1))
    st.dataframe(
        restructuring.style.format({"퇴사 예측률": "{:.1%}", "인재 가치 지수": "{:.1f}", "검토 우선 점수": "{:.1f}"}),
        width="stretch",
        hide_index=True,
    )
    st.warning("이 표는 재배치·교육·면담 검토를 위한 보조 정보입니다. 예측값만으로 해고나 불이익을 자동 결정하면 안 됩니다.")

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# 04. Executive overview
section_header(
    "overview",
    "04 · EXECUTIVE HR OVERVIEW",
    "전사 인사 구조 안정도",
    "전사 퇴사 위험과 조직별 신호를 모니터링해 선제 대응 대상을 찾습니다.",
)

actual_attrition = employees["Attrition"].astype(str).eq("Left")
high_risk = employees["prediction"].ge(0.6)
key_talent = employees["인재 가치 지수"].ge(75)
stability = (1 - employees["prediction"].mean()) * 100

with st.container(key="stat-bar-overview"):
    overview_metrics = st.columns(4)
    overview_metrics[0].metric("현재 데이터 퇴사율", f"{actual_attrition.mean():.1%}")
    overview_metrics[1].metric("고위험 인원", f"{int(high_risk.sum()):,}명")
    overview_metrics[2].metric("핵심 인재 잔류 가능성", f"{(1 - employees.loc[key_talent, 'prediction'].mean()):.1%}")
    overview_metrics[3].metric("전사 안정 지수", f"{stability:.1f} / 100")

st.info("현재 데이터에는 기준 연도·퇴사일 컬럼이 없어 실제 연도별 추세를 만들 수 없습니다. 아래에는 근속연차 구간별 퇴사율을 대체 지표로 표시합니다.")

overview_left, overview_right = st.columns([2, 1])
with overview_left:
    st.subheader("근속연차별 퇴사율")
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
        .rename("퇴사율")
    )
    st.line_chart(tenure_trend, height=310)
with overview_right:
    st.subheader("부서별 위험도")
    department_risk = (
        employees.groupby("Job Role", as_index=False)
        .agg(
            직원수=("Employee ID", "size"),
            평균_퇴사위험=("prediction", "mean"),
            평균_인재가치=("인재 가치 지수", "mean"),
        )
        .sort_values("평균_퇴사위험", ascending=False)
        .rename(columns={"Job Role": "부서"})
    )
    st.dataframe(
        department_risk.style.format({"평균_퇴사위험": "{:.1%}", "평균_인재가치": "{:.1f}"}),
        width="stretch",
        hide_index=True,
    )

st.markdown(
    '<div class="decision-note"><b>운영 제안</b> · 연도별 추세 모니터링을 위해 원천 데이터에 기준 연도 또는 퇴사일 컬럼을 추가하세요.</div>',
    unsafe_allow_html=True,
)
