from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.data.loader import load_raw_test, load_raw_train
from src.data.prediction import (
    attrition_probability,
    load_prediction_model,
    prepare_model_input,
)
from streamlit_ui import apply_page_style, page_header, top_navigation
from wheel_picker import wheel_picker_component

st.set_page_config(page_title="퇴사 위험 시나리오", page_icon="✨", layout="wide")

MODEL_PATH = Path("artifacts/ml/best_ml_model.joblib")
ACTIONABLE_FEATURES = [
    "Job Role",
    "Monthly Income",
    "Work-Life Balance",
    "Job Satisfaction",
    "Number of Promotions",
    "Overtime",
    "Distance from Home",
    "Job Level",
    "Company Size",
    "Remote Work",
    "Leadership Opportunities",
    "Innovation Opportunities",
    "Company Reputation",
    "Employee Recognition",
]
FEATURE_LABELS = {
    "Job Role": "직무",
    "Monthly Income": "월 소득",
    "Work-Life Balance": "일과 삶의 균형",
    "Job Satisfaction": "직무 만족도",
    "Number of Promotions": "승진 횟수",
    "Overtime": "초과 근무",
    "Distance from Home": "출퇴근 거리",
    "Job Level": "직급",
    "Company Size": "회사 규모",
    "Remote Work": "재택근무",
    "Leadership Opportunities": "리더십 기회",
    "Innovation Opportunities": "혁신 기회",
    "Company Reputation": "회사 평판",
    "Employee Recognition": "직원 인정",
}
VALUE_LABELS = {
    "Yes": "예",
    "No": "아니요",
    "Low": "낮음",
    "Medium": "보통",
    "High": "높음",
    "Very High": "매우 높음",
    "Small": "소규모",
    "Large": "대규모",
    "Excellent": "매우 좋음",
    "Good": "좋음",
    "Fair": "보통",
    "Poor": "나쁨",
    "Education": "교육",
    "Media": "미디어",
    "Healthcare": "의료",
    "Technology": "기술",
    "Finance": "금융",
}


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_raw_train(), load_raw_test()


@st.cache_resource
def load_model(path: str, modified_time: float):
    del modified_time
    return load_prediction_model(path)


def wheel_options(reference: pd.Series, current: int | float) -> list[int | float]:
    """긴 숫자 목록은 휠이 가볍게 움직이도록 고르게 줄인다."""

    values = sorted(reference.unique().tolist())
    if len(values) > 121:
        interval = ceil(len(values) / 120)
        values = values[::interval]
    values.append(current)
    return sorted(set(values))


def wheel_picker(
    label: str,
    options: list[int | float],
    current: int | float,
    key: str,
) -> int | float:
    labels = [f"{value:,}" if isinstance(value, int) else f"{value:,.1f}" for value in options]
    selected = wheel_picker_component(
        label=label,
        options=options,
        labels=labels,
        value=current,
        default=current,
        key=key,
    )
    return type(current)(selected)


apply_page_style()
top_navigation("attrition")
page_header(
    "조건 변화 모의실험",
    "퇴사 위험 시나리오 ✨",
    "직원의 업무 조건을 조금씩 조정해 모델이 예상하는 퇴사 확률 변화를 살펴봐요.",
)


if not MODEL_PATH.exists():
    st.warning("최고 모델이 아직 없어요. 먼저 머신러닝 학습 또는 튜닝을 실행해 주세요.")
    st.stop()

try:
    train_data, test_data = load_data()
    model = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
except Exception as exc:
    st.error(f"모델 또는 데이터를 불러오지 못했어요: {exc}")
    st.stop()

feature_names = list(getattr(model, "feature_names_in_", []))
if not feature_names:
    st.error("저장된 모델에서 입력 피처 정보를 찾지 못했어요.")
    st.stop()

selector_area, threshold_area = st.columns([2, 1])
with selector_area:
    employee_index = st.selectbox(
        "살펴볼 테스트 직원 선택",
        options=range(len(test_data)),
        format_func=lambda index: (
            f"행 {index:,} · 직원 번호 {test_data.iloc[index]['Employee ID']}"
        ),
        help="목록을 스크롤하거나 행 번호·직원 ID를 입력해 검색할 수 있어요.",
    )
with threshold_area:
    threshold = st.slider("고위험 기준", 0.10, 0.90, 0.50, 0.05)

source_row = test_data.iloc[employee_index].copy()
employee_id = source_row.get("Employee ID", employee_index)
st.caption(f"선택한 직원 번호 · {employee_id}")

editable_features = [feature for feature in ACTIONABLE_FEATURES if feature in test_data.columns]
adjusted_values: dict[str, object] = {}

st.subheader("바꿔볼 업무 조건")
with st.form("scenario_form"):
    widget_columns = st.columns(2)
    for index, feature in enumerate(editable_features):
        reference = train_data[feature].dropna()
        current = source_row[feature]
        with widget_columns[index % 2]:
            if pd.api.types.is_numeric_dtype(reference):
                if pd.api.types.is_integer_dtype(reference):
                    current_value = int(current)
                    adjusted_values[feature] = wheel_picker(
                        FEATURE_LABELS.get(feature, feature),
                        wheel_options(reference.astype(int), current_value),
                        current_value,
                        key=f"scenario_{employee_index}_{feature}",
                    )
                else:
                    current_value = float(current)
                    adjusted_values[feature] = wheel_picker(
                        FEATURE_LABELS.get(feature, feature),
                        wheel_options(reference.astype(float), current_value),
                        current_value,
                        key=f"scenario_{employee_index}_{feature}",
                    )
            else:
                options = sorted(reference.astype(str).unique().tolist())
                current_text = str(current)
                adjusted_values[feature] = st.selectbox(
                    FEATURE_LABELS.get(feature, feature),
                    options,
                    index=options.index(current_text),
                    format_func=lambda value: VALUE_LABELS.get(value, value),
                    key=f"scenario_{employee_index}_{feature}",
                )

    submitted = st.form_submit_button("이 조건으로 비교하기", type="primary", width="stretch")

if submitted:
    adjusted_row = source_row.copy()
    for feature, value in adjusted_values.items():
        if pd.api.types.is_integer_dtype(train_data[feature]):
            value = int(value)
        adjusted_row[feature] = value

    try:
        baseline_input = prepare_model_input(
            source_row.to_frame().T,
            train_data,
            feature_names,
        )

        adjusted_input = prepare_model_input(
            adjusted_row.to_frame().T,
            train_data,
            feature_names,
        )
        before_probability = attrition_probability(model, baseline_input)
        after_probability = attrition_probability(model, adjusted_input)
    except Exception as exc:
        st.error(f"예측 중 오류가 발생했어요: {exc}")
        st.stop()

    probability_delta = after_probability - before_probability
    before_label = "고위험" if before_probability >= threshold else "저위험"
    after_label = "고위험" if after_probability >= threshold else "저위험"

    st.subheader("비교 결과")
    result_metrics = st.columns(3)
    result_metrics[0].metric("조정 전", f"{before_probability:.1%}", before_label)
    result_metrics[1].metric(
        "조정 후",
        f"{after_probability:.1%}",
        delta=f"{probability_delta:+.1%}",
        delta_color="inverse",
    )
    result_metrics[2].metric("현재 판정", after_label)

    result = pd.DataFrame(
        {
            "시나리오": ["조정 전", "조정 후"],
            "퇴사 확률": [before_probability, after_probability],
        }
    )
    st.bar_chart(result.set_index("시나리오"))

    changes = [
        {
            "입력 항목": FEATURE_LABELS.get(feature, feature),
            "조정 전": VALUE_LABELS.get(str(source_row[feature]), source_row[feature]),
            "조정 후": VALUE_LABELS.get(str(adjusted_row[feature]), adjusted_row[feature]),
        }
        for feature in editable_features
        if str(source_row[feature]) != str(adjusted_row[feature])
        and not (
            isinstance(source_row[feature], (float, np.floating))
            and isinstance(adjusted_row[feature], (float, np.floating))
            and np.isclose(source_row[feature], adjusted_row[feature])
        )
    ]
    with st.expander("내가 바꾼 조건 보기", expanded=bool(changes)):
        if changes:
            st.dataframe(pd.DataFrame(changes), width="stretch", hide_index=True)
        else:
            st.info("바뀐 조건이 없어요. 값을 하나 이상 조정한 뒤 다시 비교해 보세요.")

    st.warning("이 결과를 개인에 대한 자동 평가나 불이익 부과에 사용하면 안 돼요.")
