"""MLP training page."""

import streamlit as st

from src.dl.mlp import train_mlp
from src.load_data.loader import load_train_data

st.set_page_config(page_title="03 DL Training", page_icon="🧠", layout="wide")
st.title("03. 딥러닝 MLP 학습 및 평가")
st.caption("사전학습 가중치 없이 현재 train.csv에서 처음부터 학습하는 MLP 기준선입니다.")

hidden_text = st.text_input("은닉층 크기 (쉼표 구분)", value="64, 32")
max_iter = st.slider("최대 epoch", min_value=20, max_value=300, value=100, step=10)

if st.button("MLP 학습", type="primary"):
    try:
        hidden_layers = tuple(
            int(value.strip()) for value in hidden_text.split(",") if value.strip()
        )
        if not hidden_layers or any(value <= 0 for value in hidden_layers):
            raise ValueError("은닉층은 1개 이상의 양의 정수로 입력하세요.")
        with st.spinner("MLP를 학습하고 있습니다..."):
            _, metrics = train_mlp(
                load_train_data(), hidden_layer_sizes=hidden_layers, max_iter=max_iter
            )
    except Exception as exc:
        st.exception(exc)
    else:
        st.success(f"학습 완료: {metrics['epochs']} epochs")
        excluded = {"tn", "fp", "fn", "tp", "model"}
        display = {key: value for key, value in metrics.items() if key not in excluded}
        st.json(display)
        st.write(
            f"Confusion matrix — TN {metrics['tn']}, FP {metrics['fp']}, "
            f"FN {metrics['fn']}, TP {metrics['tp']}"
        )
