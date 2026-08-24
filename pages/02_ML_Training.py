import streamlit as st

from src.load_data.loader import load_train_data
from src.ml.trainer import train_ml_models
from src.utils.constants import MODEL_NAMES

st.set_page_config(page_title="02 ML Training", page_icon="🌳", layout="wide")
st.title("02. ML 모델 학습 및 평가")
st.caption("튜닝 전 기준선입니다. 모든 모델은 동일한 계층화 홀드아웃으로 비교합니다.")

selected = st.multiselect("학습 모델", options=list(MODEL_NAMES), default=list(MODEL_NAMES))
if st.button("선택 모델 학습", type="primary", disabled=not selected):
    with st.spinner("모델을 학습하고 있습니다..."):
        try:
            _, leaderboard, unavailable = train_ml_models(
                load_train_data(), selected=selected, save_artifacts=True
            )
        except Exception as exc:
            st.exception(exc)
        else:
            st.success(f"최고 모델: {leaderboard.iloc[0]['model']}")
            metric_columns = [
                "model", "roc_auc", "average_precision", "f1", "precision",
                "recall", "accuracy", "train_seconds",
            ]
            st.dataframe(
                leaderboard[metric_columns].style.format(precision=4),
                use_container_width=True,
                hide_index=True,
            )
            if unavailable:
                st.warning(f"설치되지 않아 제외된 모델: {', '.join(unavailable)}")
            st.info("모델과 리더보드는 artifacts/models 및 artifacts/reports에 저장했습니다.")
