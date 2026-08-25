from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.data.loader import load_raw_test, load_raw_train


MODEL_PATH = Path("artifacts/models/best_ml_model.joblib")


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_raw_train(), load_raw_test()


@st.cache_resource
def load_model(path: str, modified_time: float):
    del modified_time
    return joblib.load(path)


def attrition_probability(model, frame: pd.DataFrame) -> float:
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("모델 클래스에 퇴사 라벨 1이 없습니다.")
    positive_index = classes.index(1)
    return float(model.predict_proba(frame)[0, positive_index])


st.title("5. 해결책 시나리오 최종 테스트")
st.caption("테스트 직원의 피처를 조정하고, 조정 전·후 예상 퇴사 위험을 비교합니다.")
st.info(
    "이 화면은 저장된 최고 ML 모델을 이용한 가상 시나리오입니다. "
    "피처 변경이 실제 퇴사율을 인과적으로 바꾼다는 의미는 아닙니다."
)

if not MODEL_PATH.exists():
    st.warning("최고 ML 모델이 없습니다: artifacts/models/best_ml_model.joblib")
    st.stop()

try:
    train_data, test_data = load_data()
    model = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
except Exception as exc:
    st.error(f"모델 또는 데이터를 불러오지 못했습니다: {exc}")
    st.stop()

feature_names = list(getattr(model, "feature_names_in_", []))
if not feature_names:
    feature_names = [
        column for column in train_data.columns if column not in {"Employee ID", "Attrition"}
    ]

missing_features = [column for column in feature_names if column not in test_data.columns]
if missing_features:
    st.error("테스트 데이터에 모델 입력 피처가 없습니다: " + ", ".join(missing_features))
    st.stop()

selector_area, threshold_area = st.columns([2, 1])
with selector_area:
    employee_index = st.number_input(
        "테스트 직원 행 번호",
        min_value=0,
        max_value=len(test_data) - 1,
        value=0,
        step=1,
    )
with threshold_area:
    threshold = st.slider(
        "퇴사 위험 판정 기준",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
    )

employee_index = int(employee_index)
source_row = test_data.iloc[employee_index]
employee_id = source_row.get("Employee ID", employee_index)
st.caption(f"선택 직원 ID: {employee_id}")

baseline = source_row[feature_names].to_frame().T.copy()
adjusted_values: dict[str, object] = {}

st.subheader("피처 조정")
with st.form("scenario_form"):
    widget_columns = st.columns(3)
    for index, feature in enumerate(feature_names):
        reference = train_data[feature].dropna()
        current = source_row[feature]
        with widget_columns[index % 3]:
            if pd.api.types.is_numeric_dtype(reference):
                if pd.isna(current):
                    current = reference.median()
                minimum = reference.min()
                maximum = reference.max()
                if pd.api.types.is_integer_dtype(reference):
                    adjusted_values[feature] = st.number_input(
                        feature,
                        min_value=int(minimum),
                        max_value=int(maximum),
                        value=int(current),
                        step=1,
                        key=f"scenario_{employee_index}_{feature}",
                    )
                else:
                    adjusted_values[feature] = st.number_input(
                        feature,
                        min_value=float(minimum),
                        max_value=float(maximum),
                        value=float(current),
                        key=f"scenario_{employee_index}_{feature}",
                    )
            else:
                options = sorted(reference.astype(str).unique().tolist())
                current_text = str(current)
                if current_text not in options:
                    options.insert(0, current_text)
                adjusted_values[feature] = st.selectbox(
                    feature,
                    options,
                    index=options.index(current_text),
                    key=f"scenario_{employee_index}_{feature}",
                )

    submitted = st.form_submit_button(
        "조정 결과 테스트", type="primary", width="stretch"
    )

if submitted:
    adjusted = pd.DataFrame([adjusted_values], columns=feature_names)
    try:
        before_probability = attrition_probability(model, baseline)
        after_probability = attrition_probability(model, adjusted)
    except Exception as exc:
        st.error(f"예측 중 오류가 발생했습니다: {exc}")
        st.stop()

    before_label = "고위험" if before_probability >= threshold else "저위험"
    after_label = "고위험" if after_probability >= threshold else "저위험"
    probability_delta = after_probability - before_probability

    result = pd.DataFrame(
        [
            {"시나리오": "조정 전", "퇴사 확률": before_probability, "판정": before_label},
            {"시나리오": "조정 후", "퇴사 확률": after_probability, "판정": after_label},
        ]
    )

    st.subheader("최종 테스트 결과표")
    result_metrics = st.columns(3)
    result_metrics[0].metric("조정 전 퇴사 확률", f"{before_probability:.1%}")
    result_metrics[1].metric(
        "조정 후 퇴사 확률",
        f"{after_probability:.1%}",
        delta=f"{probability_delta:+.1%}",
        delta_color="inverse",
    )
    result_metrics[2].metric("조정 후 판정", after_label)
    st.dataframe(
        result.style.format({"퇴사 확률": "{:.1%}"}),
        width="stretch",
        hide_index=True,
    )
    st.bar_chart(result.set_index("시나리오")["퇴사 확률"])

    changed_rows = []
    for feature in feature_names:
        before = baseline.iloc[0][feature]
        after = adjusted.iloc[0][feature]
        if (pd.isna(before) and pd.isna(after)) or str(before) == str(after):
            continue
        if isinstance(before, (float, np.floating)) and isinstance(
            after, (float, np.floating)
        ):
            if np.isclose(before, after):
                continue
        changed_rows.append({"피처": feature, "조정 전": before, "조정 후": after})

    st.subheader("적용한 변경사항")
    if changed_rows:
        st.dataframe(pd.DataFrame(changed_rows), width="stretch", hide_index=True)
    else:
        st.info("변경된 피처가 없습니다.")

    st.warning(
        "이 결과는 의사결정 참고용 시뮬레이션입니다. 개인에 대한 자동 인사 결정이나 "
        "불이익 부과에 사용하지 마세요."
    )
