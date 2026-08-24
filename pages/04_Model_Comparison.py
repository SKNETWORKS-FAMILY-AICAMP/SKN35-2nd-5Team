"""Model benchmark and curve comparison page."""

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.load_data.loader import load_train_data
from src.ml.evaluation import classification_curves, evaluate_classifier
from src.ml.trainer import make_train_valid_split
from src.utils.paths import DL_MODEL_PATH, ML_LEADERBOARD_PATH

st.set_page_config(page_title="04 Model Comparison", page_icon="🏁", layout="wide")
st.title("04. 모델 종합 벤치마크")

if not ML_LEADERBOARD_PATH.exists():
    st.warning("저장된 ML 결과가 없습니다. 먼저 02 ML Training을 실행하세요.")
    st.stop()

leaderboard = pd.read_csv(ML_LEADERBOARD_PATH)
_, x_valid, _, y_valid = make_train_valid_split(load_train_data())
model_paths = dict(zip(leaderboard["model"], leaderboard["artifact_path"]))
if DL_MODEL_PATH.exists():
    model_paths["mlp"] = str(DL_MODEL_PATH)

evaluated = {}
for name, path in model_paths.items():
    model = joblib.load(path)
    evaluated[name] = {
        "metrics": evaluate_classifier(model, x_valid, y_valid),
        "curves": classification_curves(model, x_valid, y_valid),
    }

ranking = pd.DataFrame(
    [{"model": name, **result["metrics"]} for name, result in evaluated.items()]
).sort_values(["roc_auc", "f1"], ascending=False, ignore_index=True)
st.dataframe(ranking.style.format(precision=4), use_container_width=True, hide_index=True)

roc_axis, pr_axis = plt.subplots(1, 2, figsize=(13, 5))
for name, result in evaluated.items():
    curves = result["curves"]
    score = result["metrics"]["roc_auc"]
    roc_axis.plot(curves["fpr"], curves["tpr"], label=f"{name} ({score:.3f})")
    pr_axis.plot(curves["recall"], curves["precision"], label=name)
roc_axis.plot([0, 1], [0, 1], linestyle="--", color="grey")
roc_axis.set(title="ROC Curve", xlabel="False Positive Rate", ylabel="True Positive Rate")
pr_axis.set(title="Precision-Recall Curve", xlabel="Recall", ylabel="Precision")
roc_axis.legend()
pr_axis.legend()
st.pyplot(roc_axis.figure)
plt.close(roc_axis.figure)

