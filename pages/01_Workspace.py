"""인사팀 / 기술개발팀(관리자) 공용 워크스페이스.

접속 유형(main.py에서 선택)에 따라 같은 화면을 공유하되, 관리자에게만 모델 성능
평가 탭을 추가로 보여준다. 탭 전환은 Streamlit 사이드바 대신, Liquid Glass 세그먼트
버튼바로 구현했다.
"""

import pandas as pd
import streamlit as st

from src.data.loader import load_raw_test, load_raw_train
from src.data.prediction import create_employee_predictions, load_hr_prediction_model
from src.utils.hr_metrics import add_talent_value
from src.utils.paths import (
    MLP_BEST_PARAMS_PATH,
    MLP_METADATA_PATH,
    MLP_MODEL_PATH,
    MLP_PREPROCESSOR_PATH,
    MLP_THRESHOLD_PATH,
)
from src.views import tab_hr_actions, tab_models, tab_salary, tab_stability, tab_team
from streamlit_ui import (
    alert_box,
    apply_page_style,
    empty_state,
    page_header,
    stat_cards,
    top_navigation,
    workspace_navigation,
)

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
    empty_state(
        "Unauthorized",
        "처음 화면에서 접속 유형을 선택하면 바로 이어서 볼 수 있어요.",
        icon="🔒",
    )
    st.markdown('<div class="section-spacer-lg"></div>', unsafe_allow_html=True)
    if st.button("처음으로 돌아가기", type="primary"):
        st.switch_page("main.py")
    st.stop()

workspace_navigation(role)

ROLE_LABELS = {"hr": "인사팀", "admin": "기술개발팀 · 관리자"}
with st.container(key="workspace-page-header"):
    st.markdown(
        '<h1 class="workspace-page-title">HR Workspace</h1>'
        f'<p class="workspace-page-desc">{ROLE_LABELS[role]} 모드로 접속했어요. '
        '아래 버튼으로 원하는 업무화면을 바로 전환할 수 있어요.</p>',
        unsafe_allow_html=True,
    )

MODEL_PATHS = (
    MLP_MODEL_PATH,
    MLP_PREPROCESSOR_PATH,
    MLP_BEST_PARAMS_PATH,
    MLP_THRESHOLD_PATH,
    MLP_METADATA_PATH,
)
missing_model_paths = [str(path) for path in MODEL_PATHS if not path.exists()]
if missing_model_paths:
    alert_box(
        "danger",
        "HR 예측용 MLP 산출물이 없습니다: " + ", ".join(missing_model_paths),
    )
    st.stop()


@st.cache_data
def load_data():
    return load_raw_train(), load_raw_test()


@st.cache_resource
def load_model(artifact_signature: tuple[tuple[str, float], ...]):
    del artifact_signature
    return load_hr_prediction_model()


@st.cache_data
def predict_all_employees(all_raw, raw_train, artifact_signature):
    # train.csv(약 59,598명) + test.csv(약 14,900명)를 합쳐 회사 전체 인원을 기준으로
    # 예측·통계를 낸다. 두 파일의 Employee ID는 서로 겹치지 않는 것을 확인했다.
    model = load_model(artifact_signature)
    return create_employee_predictions(all_raw, raw_train, model)


try:
    train_data, test_data = load_data()
    all_raw = pd.concat([train_data, test_data], ignore_index=True)
    artifact_signature = tuple((str(path), path.stat().st_mtime) for path in MODEL_PATHS)
    model = load_model(artifact_signature)
    predictions = predict_all_employees(all_raw, train_data, artifact_signature)
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
        {"label": "Employees Analyzed", "value": f"{len(employees):,}"},
        {"label": "Actual Attrition", "value": f"{attrition_rate:.1%}"},
        {"label": "Avg. Talent Value", "value": f"{employees['인재 가치 지수'].mean():.1f}"},
        {"label": "Predicted Attrition Risk", "value": f"{employees['prediction'].mean():.1%}"},
    ]
)

TAB_OPTIONS = [
    ("salary", "◎  연봉협상"),
    ("team", "♙  팀 구성"),
    ("actions", "▣  인사 지원"),
    ("stability", "◇  안정도"),
]
if role == "admin":
    TAB_OPTIONS.append(("models", "⊞  모델"))

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
