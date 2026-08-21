"""Machine-learning training and four-model performance comparison page."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import RESULTS_DIR
from src.utils.streamlit_ui import apply_korean_font_css, configure_matplotlib_korean


st.set_page_config(page_title="05. ML 성능 비교", page_icon="🤖", layout="wide")

MODEL_ORDER = ["decision_tree", "random_forest", "xgboost", "lightgbm"]
METRICS_PATH = RESULTS_DIR / "ml_metrics.json"


@st.cache_data(show_spinner=False)
def load_ml_metrics(path: str, modified_ns: int):
    del modified_ns
    metrics_path = Path(path)
    try:
        with metrics_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return None, "ML 평가 결과가 아직 없습니다."
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"ML 평가 결과를 읽지 못했습니다: {exc}"
    if not isinstance(data, dict):
        return None, "ML 평가 결과의 JSON 구조가 올바르지 않습니다."
    return data, None


def draw_confusion_matrix(matrix, title: str):
    values = np.asarray(matrix, dtype=int)
    figure, axis = plt.subplots(figsize=(4, 4))
    axis.imshow(values, cmap="Blues")
    threshold = values.max() / 2 if values.size else 0
    for row in range(2):
        for column in range(2):
            axis.text(
                column, row, f"{values[row, column]:,}", ha="center", va="center",
                color="white" if values[row, column] > threshold else "black",
            )
    axis.set_xticks([0, 1], labels=["Retained", "Churned"])
    axis.set_yticks([0, 1], labels=["Retained", "Churned"])
    axis.set(xlabel="Predicted", ylabel="Actual", title=title)
    figure.tight_layout()
    return figure


def render_curves(metrics_data) -> None:
    left, right = st.columns(2)
    with left:
        st.subheader("📈 ROC Curve")
        figure, axis = plt.subplots(figsize=(7, 5))
        for model_id in MODEL_ORDER:
            metrics = metrics_data.get(model_id, {})
            curve = metrics.get("roc_curve")
            if curve:
                axis.plot(
                    curve["fpr"], curve["tpr"],
                    label=f"{metrics.get('model_name', model_id)} (AUC={metrics.get('roc_auc', 0):.3f})",
                )
        axis.plot([0, 1], [0, 1], "k--", alpha=0.5)
        axis.set(xlabel="False Positive Rate", ylabel="True Positive Rate")
        axis.legend(loc="lower right")
        figure.tight_layout()
        st.pyplot(figure, width="stretch")
        plt.close(figure)
    with right:
        st.subheader("📉 Precision-Recall Curve")
        figure, axis = plt.subplots(figsize=(7, 5))
        for model_id in MODEL_ORDER:
            metrics = metrics_data.get(model_id, {})
            curve = metrics.get("pr_curve")
            if curve:
                axis.plot(
                    curve["recall"], curve["precision"],
                    label=f"{metrics.get('model_name', model_id)} (AUC={metrics.get('pr_auc', 0):.3f})",
                )
        axis.set(xlabel="Recall", ylabel="Precision")
        axis.legend(loc="lower left")
        figure.tight_layout()
        st.pyplot(figure, width="stretch")
        plt.close(figure)


def main() -> None:
    apply_korean_font_css()
    configure_matplotlib_korean()
    st.title("🤖 05. 머신러닝 4종 성능 비교")
    st.markdown("Decision Tree, Random Forest, XGBoost, LightGBM의 검증 결과를 비교합니다.")

    if st.button("🚀 ML 4종 모델 다시 학습", type="primary"):
        try:
            with st.spinner("4개 머신러닝 모델을 학습하고 있습니다..."):
                from src.ml.trainer import train_and_save_all_ml_models

                train_and_save_all_ml_models()
            st.cache_data.clear()
            st.success("ML 모델 학습과 결과 저장이 완료됐습니다.")
            st.rerun()
        except Exception as exc:
            st.error(f"모델 학습 중 오류가 발생했습니다: {exc}")

    modified_ns = METRICS_PATH.stat().st_mtime_ns if METRICS_PATH.exists() else 0
    metrics_data, error = load_ml_metrics(str(METRICS_PATH), modified_ns)
    if error or not metrics_data:
        st.info(error or "표시할 ML 결과가 없습니다.")
        return

    rows = []
    for model_id in MODEL_ORDER:
        metrics = metrics_data.get(model_id)
        if not isinstance(metrics, dict):
            continue
        rows.append(
            {
                "Model": metrics.get("model_name", model_id),
                "Train Accuracy": metrics.get("train_accuracy"),
                "Test Accuracy": metrics.get("test_accuracy"),
                "Precision": metrics.get("precision"),
                "Recall": metrics.get("recall"),
                "F1-Score": metrics.get("f1_score"),
                "ROC-AUC": metrics.get("roc_auc"),
                "PR-AUC": metrics.get("pr_auc"),
                "CV Macro-F1": metrics.get("best_cv_f1_macro"),
                "Train Time (s)": metrics.get("train_time_sec"),
            }
        )
    if not rows:
        st.warning("지원하는 ML 모델 결과가 없습니다.")
        return

    st.subheader("🏆 모델별 성능 지표")
    comparison = pd.DataFrame(rows).set_index("Model")
    comparison = comparison.apply(pd.to_numeric, errors="coerce")
    st.dataframe(
        comparison.style.format("{:.4f}", na_rep="-"),
        width="stretch",
    )

    decision_tree = metrics_data.get("decision_tree")
    if isinstance(decision_tree, dict):
        gap = decision_tree["train_accuracy"] - decision_tree["test_accuracy"]
        st.warning(
            "**Decision Tree 과적합 점검** — "
            f"Train/Test 정확도 차이는 {gap * 100:.2f}%p입니다."
        )

    random_forest = metrics_data.get("random_forest")
    if isinstance(random_forest, dict) and random_forest.get("best_params"):
        cv_score = random_forest.get("best_cv_f1_macro")
        if cv_score is not None:
            st.success(
                f"**Random Forest Optuna 튜닝** — "
                f"Stratified CV Macro-F1: **{float(cv_score):.4f}**"
            )
        with st.expander("Random Forest 최적 하이퍼파라미터"):
            st.json(random_forest["best_params"])

    st.divider()
    st.subheader("🟦 Confusion Matrix")
    columns = st.columns(4)
    for index, model_id in enumerate(MODEL_ORDER):
        metrics = metrics_data.get(model_id, {})
        with columns[index]:
            matrix = metrics.get("confusion_matrix")
            if not matrix:
                st.info(f"{model_id} 결과 없음")
                continue
            figure = draw_confusion_matrix(
                matrix, metrics.get("model_name", model_id)
            )
            st.pyplot(figure, width="stretch")
            plt.close(figure)

    st.divider()
    render_curves(metrics_data)


if __name__ == "__main__":
    main()
