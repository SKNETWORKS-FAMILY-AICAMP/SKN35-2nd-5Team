import streamlit as st

from src.utils.streamlit_ui import render_not_ready

st.set_page_config(page_title="06. DL Training", page_icon="🧠", layout="wide")
render_not_ready(
    "🧠 06. 딥러닝 모델 학습",
    "딥러닝(MLP) 모델은 아직 검증 및 화면 구성이 완료되지 않았습니다.",
)
