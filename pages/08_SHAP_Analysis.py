"""SHAP and fallback permutation importance page."""

import joblib
import streamlit as st

from src.load_data.loader import load_train_data
from src.ml.explainability import permutation_feature_importance, shap_feature_importance
from src.utils.paths import BEST_ML_MODEL_PATH

st.set_page_config(page_title="08 SHAP Analysis", page_icon="🧭", layout="wide")
st.title("08. 모델 해석 및 중요 피처 분석")

if not BEST_ML_MODEL_PATH.exists():
    st.warning("최고 ML 모델이 없습니다. 먼저 02 ML Training을 실행하세요.")
    st.stop()

sample_size = st.slider("해석 표본 수", min_value=100, max_value=2000, value=500, step=100)
if st.button("중요 피처 계산", type="primary"):
    model = joblib.load(BEST_ML_MODEL_PATH)
    data = load_train_data()
    with st.spinner("SHAP 값을 계산하고 있습니다..."):
        try:
            importance = shap_feature_importance(model, data, max_samples=sample_size)
            method = "TreeSHAP"
        except Exception as exc:
            st.warning(f"TreeSHAP을 적용할 수 없어 순열 중요도로 대체합니다: {exc}")
            importance = permutation_feature_importance(model, data, max_samples=sample_size)
            method = "Permutation importance"
    st.success(f"해석 방식: {method}")
    top = importance.head(20)
    st.bar_chart(top.set_index("feature")["importance_mean"])
    st.dataframe(top, use_container_width=True, hide_index=True)
    st.caption(
        "중요도는 인과관계를 의미하지 않으며, HR 의사결정 전에 "
        "편향과 업무 맥락을 별도로 검토해야 합니다."
    )
