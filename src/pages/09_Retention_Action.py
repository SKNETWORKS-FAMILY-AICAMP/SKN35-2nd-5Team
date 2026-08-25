import joblib
import streamlit as st

from src.analysis.retention import score_retention_risk
from src.load_data.loader import load_test_data
from src.utils.paths import BEST_ML_MODEL_PATH

st.set_page_config(page_title="09 Retention Action", page_icon="🤝", layout="wide")
st.title("09. 맞춤형 리텐션 액션")
st.caption("모델 점수는 지원 도구이며, 자동 인사결정이나 불이익 부과에 사용하면 안 됩니다.")

if not BEST_ML_MODEL_PATH.exists():
    st.warning("최고 ML 모델이 없습니다. 먼저 02 ML Training을 실행하세요.")
    st.stop()

threshold = st.slider("고위험 기준 확률", min_value=0.30, max_value=0.90, value=0.50, step=0.05)
top_n = st.slider("표시 인원", min_value=10, max_value=500, value=100, step=10)

if st.button("리텐션 대상 후보 산출", type="primary"):
    model = joblib.load(BEST_ML_MODEL_PATH)
    scored = score_retention_risk(model, load_test_data(), threshold=threshold)
    high_risk = scored[scored["attrition_probability"] >= threshold]
    columns = st.columns(3)
    columns[0].metric("전체 대상", f"{len(scored):,}")
    columns[1].metric("고위험 후보", f"{len(high_risk):,}")
    columns[2].metric("고위험 비율", f"{len(high_risk) / len(scored):.1%}")
    st.dataframe(scored.head(top_n), use_container_width=True, hide_index=True)
    st.download_button(
        "전체 결과 CSV 다운로드",
        data=scored.to_csv(index=False).encode("utf-8-sig"),
        file_name="retention_risk_scores.csv",
        mime="text/csv",
    )
    st.warning(
        "제안 액션은 관찰 변수 기반의 검토 출발점이며, "
        "차별적·징벌적 조치에 사용하지 마세요."
    )
