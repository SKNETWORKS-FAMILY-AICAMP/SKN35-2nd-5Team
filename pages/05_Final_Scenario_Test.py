from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.data.loader import load_raw_test, load_raw_train
from src.data.preprocess import preprocess_pipeline
from streamlit_ui import apply_page_style, home_button, page_header

st.set_page_config(page_title="퇴사 위험 시나리오", page_icon="✨", layout="wide")

MODEL_PATH = Path("artifacts/models/best_ml_model.joblib")
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


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_raw_train(), load_raw_test()


@st.cache_resource
def load_model(path: str, modified_time: float):
    del modified_time
    return joblib.load(path)


def prepare_model_input(
    raw_frame: pd.DataFrame,
    raw_train: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    """원본 입력을 학습 때와 같은 40개 전처리 피처로 변환한다."""

    processed = preprocess_pipeline(raw_frame.copy(), reference=raw_train.copy())
    processed = processed.drop(columns=["Attrition", "Unnamed: 0"], errors="ignore")
    missing = [feature for feature in feature_names if feature not in processed.columns]
    if missing:
        raise ValueError("전처리 후 누락된 피처: " + ", ".join(missing))
    return processed.reindex(columns=feature_names)


def attrition_probability(model, frame: pd.DataFrame) -> float:
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("모델 클래스에 퇴사 라벨 1이 없습니다.")
    return float(model.predict_proba(frame)[0, classes.index(1)])


apply_page_style()
home_button()
page_header(
    "WHAT-IF SCENARIO",
    "퇴사 위험 시나리오 ✨",
    "직원의 업무 조건을 조금씩 조정해 모델이 예상하는 퇴사 확률 변화를 살펴봐요.",
)
st.info(
    "예측 변화는 인과관계가 아닌 가상 비교예요. "
    "실제 인사 결정은 반드시 사람의 검토를 거쳐야 합니다."
)

if not MODEL_PATH.exists():
    st.warning("최고 모델이 아직 없어요. 먼저 ML 학습 또는 튜닝을 실행해 주세요.")
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
    employee_index = int(
        st.number_input(
            "살펴볼 테스트 직원의 행 번호",
            min_value=0,
            max_value=len(test_data) - 1,
            value=0,
            step=1,
        )
    )
with threshold_area:
    threshold = st.slider("고위험 기준", 0.10, 0.90, 0.50, 0.05)

source_row = test_data.iloc[employee_index].copy()
employee_id = source_row.get("Employee ID", employee_index)
st.caption(f"선택한 직원 ID · {employee_id}")

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
                adjusted_values[feature] = st.number_input(
                    feature,
                    min_value=float(reference.min()),
                    max_value=float(reference.max()),
                    value=float(current),
                    step=1.0,
                    key=f"scenario_{employee_index}_{feature}",
                )
            else:
                options = sorted(reference.astype(str).unique().tolist())
                current_text = str(current)
                adjusted_values[feature] = st.selectbox(
                    feature,
                    options,
                    index=options.index(current_text),
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
        {"피처": feature, "조정 전": source_row[feature], "조정 후": adjusted_row[feature]}
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
