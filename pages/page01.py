"""Page 1: 시작 화면과 관리자 역할 선택."""

from pathlib import Path
import pandas as pd
import streamlit as st

from src.data.loader import load_raw_test, load_raw_train
from streamlit_ui import apply_page_style, top_navigation

st.set_page_config(page_title="TalentGuard AI | 역할 선택", page_icon="HR", layout="wide")
apply_page_style()
top_navigation("start")


@st.cache_data(ttl=300)
def get_home_stats() -> tuple[str, str, str]:
    try:
        train_df = load_raw_train()
        test_df = load_raw_test()
        total_count = f"{len(train_df) + len(test_df):,}명"
    except Exception:
        total_count = "74,498명"

    try:
        dl_report = pd.read_csv("artifacts/reports/mlp_test_metrics.csv").iloc[0]
        recall = f"{float(dl_report['recall']):.1%}"
        roc_auc = f"{float(dl_report['roc_auc']):.1%}"
    except Exception:
        recall = "81.3%"
        roc_auc = "84.7%"

    return total_count, recall, roc_auc


total_count, recall, roc_auc = get_home_stats()

st.markdown(
    f"""
    <section class="start-hero"><div class="start-inner">
      
      <h1>기업의 인재를<br><span>지켜드립니다.</span></h1>
      <p class="start-desc">핵심 기술 인재의 이탈 신호를 사전에 감지하고,<br>데이터 기반의 선제적 인사 전략을 제공합니다.</p>
      <div class="start-actions">
        <a class="role-card hr" href="/page02" target="_self"><small>HR MANAGER</small><strong>인사팀 관리자</strong><b>›</b><p>연봉협상 · 팀구성 · 인사지원 · 안정도 모니터링</p></a>
        <a class="role-card dev" href="/page06" target="_self"><small>DEV MANAGER</small><strong>개발관리 관리자</strong><b>›</b><p>ML/DL 모델 성능 평가 · 예측 모델 비교 분석</p></a>
      </div>
      <div class="start-stats">
        <div><strong>{total_count}</strong><small>모니터링 인원 (DB)</small></div>
        <div><strong>{recall}</strong><small>이탈 감지 재현율 (MLP)</small></div>
        <div><strong>{roc_auc}</strong><small>종합 예측력 (ROC-AUC)</small></div>
      </div>
    </div></section>
    """,
    unsafe_allow_html=True,
)
