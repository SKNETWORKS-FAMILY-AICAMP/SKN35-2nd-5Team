import streamlit as st

from src.utils.streamlit_ui import render_not_ready

st.set_page_config(page_title="08. SHAP Analysis", page_icon="🔍", layout="wide")
render_not_ready(
    "🔍 08. 모델 해석 및 SHAP 분석",
    "모델 설명과 SHAP 시각화는 검증을 마친 뒤 공개할 예정입니다.",
)
