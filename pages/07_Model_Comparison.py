import streamlit as st

from src.utils.streamlit_ui import render_not_ready

st.set_page_config(page_title="07. Model Comparison", page_icon="🏆", layout="wide")
render_not_ready(
    "🏆 07. 전체 모델 종합 비교",
    "DL을 포함한 전체 모델 비교는 준비 중입니다. ML 4종 비교는 05 페이지에서 확인할 수 있습니다.",
)
