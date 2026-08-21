"""Shared Streamlit UI helpers, including cross-platform Korean font setup."""

from __future__ import annotations

from functools import lru_cache

import matplotlib as mpl
from matplotlib import font_manager
import streamlit as st


KOREAN_FONT_CANDIDATES = (
    "Noto Sans CJK KR",
    "NanumGothic",
    "Malgun Gothic",
    "Noto Sans KR",
    "Apple SD Gothic Neo",
    "AppleGothic",
)


def apply_korean_font_css() -> None:
    """Prefer Korean-capable fonts already installed in the user's browser."""
    st.markdown(
        """
        <style>
        html, body, [class*="st-"], [data-testid="stAppViewContainer"] {
            font-family: Pretendard, "Noto Sans KR", "Malgun Gothic",
                         "Apple SD Gothic Neo", sans-serif;
        }
        code, pre, kbd, samp {
            font-family: "D2Coding", "Noto Sans Mono CJK KR", Consolas, monospace;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@lru_cache(maxsize=1)
def configure_matplotlib_korean() -> str | None:
    """Select an installed Korean font for server-rendered Matplotlib figures."""
    installed = {font.name for font in font_manager.fontManager.ttflist}
    selected = next((name for name in KOREAN_FONT_CANDIDATES if name in installed), None)
    if selected:
        mpl.rcParams["font.family"] = selected
    mpl.rcParams["axes.unicode_minus"] = False
    return selected


def render_not_ready(title: str, description: str) -> None:
    """Render a safe placeholder without importing unfinished feature code."""
    apply_korean_font_css()
    st.title(title)
    st.info("🚧 아직 준비되지 않았습니다.")
    st.write(description)
    st.markdown(
        "현재 공개 범위는 **01 EDA · 02 이탈 정의 · 03 피처/상관관계 · "
        "04 고객 군집 · 05 머신러닝 4종 성능 비교**입니다."
    )
