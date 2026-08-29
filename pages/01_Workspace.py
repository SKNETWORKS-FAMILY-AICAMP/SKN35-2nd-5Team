"""인사팀 / 기술개발팀(관리자) 공용 워크스페이스.

접속 유형(main.py에서 선택)에 따라 같은 화면을 공유하되, 관리자에게만 모델 성능
평가 탭을 추가로 보여준다. 탭 전환은 Streamlit 사이드바 대신, 라운드 처리된 버튼을
가로로 늘어놓은 세그먼트 버튼바로 구현했다.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.data.loader import load_raw_test, load_raw_train
from src.data.prediction import create_employee_predictions, load_prediction_model
from src.utils.hr_metrics import add_talent_value
from src.views import tab_hr_actions, tab_models, tab_salary, tab_stability, tab_team
from streamlit_ui import alert_box, apply_page_style, page_header, stat_cards, top_navigation

st.set_page_config(page_title="HR Workspace · STAYON", layout="wide")
apply_page_style()

role = st.session_state.get("role")
if role not in ("hr", "admin"):
    top_navigation()
    page_header(
        "ACCESS REQUIRED",
        "접속 유형을 먼저 선택해 주세요",
        "인사팀 또는 기술개발팀(관리자) 중 하나로 접속해야 워크스페이스를 볼 수 있어요.",
    )
    alert_box("info", "처음 화면에서 접속 유형을 선택하면 바로 이어서 볼 수 있어요.")
    if st.button("처음으로 돌아가기", type="primary"):
        st.switch_page("main.py")
    st.stop()

top_navigation(role)

ROLE_LABELS = {"hr": "인사팀", "admin": "기술개발팀 · 관리자"}
page_header(
    "INTERNAL PEOPLE INTELLIGENCE",
    "HR Workspace",
    f"{ROLE_LABELS[role]} 모드로 접속했어요. 아래 버튼으로 원하는 업무 화면을 바로 전환할 수 있어요.",
)

MODEL_PATH = Path("artifacts/ml/best_ml_model.joblib")
if not MODEL_PATH.exists():
    alert_box(
        "danger",
        "최종 예측 모델이 없습니다. artifacts/ml/best_ml_model.joblib을 먼저 생성해 주세요.",
    )
    st.stop()


@st.cache_data
def load_data():
    return load_raw_train(), load_raw_test()


@st.cache_resource
def load_model(path: str, modified_time: float):
    del modified_time
    return load_prediction_model(path)


@st.cache_data
def predict_all_employees(all_raw, raw_train, model_path: str, modified_time: float):
    # train.csv(약 59,598명) + test.csv(약 14,900명)를 합쳐 회사 전체 인원을 기준으로
    # 예측·통계를 낸다. 두 파일의 Employee ID는 서로 겹치지 않는 것을 확인했다.
    model = load_model(model_path, modified_time)
    return create_employee_predictions(all_raw, raw_train, model)


try:
    train_data, test_data = load_data()
    all_raw = pd.concat([train_data, test_data], ignore_index=True)
    model = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
    predictions = predict_all_employees(
        all_raw, train_data, str(MODEL_PATH), MODEL_PATH.stat().st_mtime
    )
    employees = all_raw.merge(
        predictions,
        left_on="Employee ID",
        right_on="employee_id",
        how="inner",
        validate="one_to_one",
    )
    employees = add_talent_value(employees)
except Exception as exc:  # noqa: BLE001
    alert_box("danger", f"직원 데이터 또는 예측 모델을 불러오지 못했습니다: {exc}")
    st.stop()

attrition_rate = employees["Attrition"].astype(str).eq("Left").mean()
stat_cards(
    [
        {"label": "분석 대상 인원", "value": f"{len(employees):,}명"},
        {"label": "실제 퇴사율", "value": f"{attrition_rate:.1%}"},
        {"label": "평균 인재 가치", "value": f"{employees['인재 가치 지수'].mean():.1f} / 100"},
        {"label": "평균 퇴사 예측률", "value": f"{employees['prediction'].mean():.1%}"},
    ]
)

TAB_OPTIONS = [
    ("salary", "💰 연봉 협상"),
    ("team", "🧩 팀 구성"),
    ("actions", "🗂️ 인사 지원"),
    ("stability", "📊 인사 구조 안정도"),
]
if role == "admin":
    TAB_OPTIONS.append(("models", "🤖 모델 성능 평가"))

st.session_state.setdefault("workspace_tab", "salary")
if st.session_state["workspace_tab"] not in {key for key, _ in TAB_OPTIONS}:
    st.session_state["workspace_tab"] = "salary"

with st.container(key="tabbar-workspace"):
    tab_columns = st.columns(len(TAB_OPTIONS))
    for column, (key, label) in zip(tab_columns, TAB_OPTIONS, strict=False):
        with column:
            is_active = st.session_state["workspace_tab"] == key
            if st.button(
                label,
                key=f"workspace_tab_btn_{key}",
                type="primary" if is_active else "secondary",
                width="stretch",
            ):
                st.session_state["workspace_tab"] = key
                st.rerun()

st.markdown('<div class="section-divider-thin"></div>', unsafe_allow_html=True)

active_tab = st.session_state["workspace_tab"]
if active_tab == "salary":
    tab_salary.render(employees, train_data, model)
elif active_tab == "team":
    tab_team.render(employees)
elif active_tab == "actions":
    tab_hr_actions.render(employees)
elif active_tab == "stability":
    tab_stability.render(employees)
elif active_tab == "models" and role == "admin":
    tab_models.render()
