"""Streamlit 화면에서 공통으로 사용하는 스타일과 작은 UI 도우미."""

import streamlit as st


def apply_page_style() -> None:
    """모든 페이지에 토스 스타일의 플랫 대시보드 레이아웃을 적용한다."""

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;700;800&family=Inter:wght@400;500;600&display=swap');

        :root {
            --ink: #191F28;
            --muted: #4E5968;
            --faint: #8B95A1;
            --blue: #3182F6;
            --blue-deep: #1B64DA;
            --blue-soft: #EAF2FF;
            --surface: #FFFFFF;
            --surface-alt: #F7F8FA;
            --line: #EDEFF2;
        }

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        .stApp {
            background: var(--surface-alt);
            color: var(--ink);
        }
        .block-container {
            width: min(92vw, 1560px);
            max-width: 1560px;
            padding: 3.6rem 1.5rem 3rem;
        }
        [data-testid="stSidebar"] {
            background: var(--surface);
            border-right: 1px solid var(--line);
        }
        header[data-testid="stHeader"] {
            background: var(--surface-alt);
        }
        header[data-testid="stHeader"]::before {
            content: "";
            position: absolute;
            inset: 0;
            border-bottom: 1px solid var(--line);
        }

        /* Typography */
        h1, h2, h3 {
            font-family: 'Plus Jakarta Sans', sans-serif;
            letter-spacing: -0.02em;
            color: var(--ink);
            font-weight: 800;
        }
        h1 { font-size: 1.9rem !important; }
        h3 {
            font-size: 1.02rem !important;
            font-weight: 700 !important;
            margin-top: 2.1rem !important;
            padding-bottom: .6rem;
            border-bottom: 1px solid var(--line);
        }

        /* Page header: left-aligned, rule-based, no filled card */
        .page-head {
            padding: 0 0 1.35rem;
            margin-bottom: 1.7rem;
            border-bottom: 1px solid var(--line);
        }
        .page-head-eyebrow {
            display: inline-block;
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-size: .74rem;
            font-weight: 700;
            letter-spacing: .09em;
            text-transform: uppercase;
            color: var(--blue-deep);
            padding-left: .8rem;
            border-left: 3px solid var(--blue);
            margin-bottom: .7rem;
        }
        .page-head h1 { margin: .05rem 0 .55rem; }
        .page-head .muted { font-size: .98rem; max-width: 640px; }

        /* Home link: plain text, no button chrome */
        .home-link {
            display: inline-flex;
            align-items: center;
            gap: .3rem;
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-weight: 700;
            font-size: .88rem;
            color: var(--muted);
            text-decoration: none !important;
            padding: .15rem 0;
            transition: color .15s ease, gap .15s ease;
        }
        .home-link:hover { color: var(--blue-deep); gap: .5rem; }
        .st-key-home-nav { margin-top: .2rem; margin-bottom: 1rem; overflow: visible; }
        .st-key-home-nav [data-testid="stMarkdownContainer"] { padding: 0; overflow: visible; }

        /* Metrics: plain figures, no boxed card by default */
        div[data-testid="stMetric"] {
            background: transparent;
            border: none;
            border-radius: 0;
            padding: 0;
            box-shadow: none;
        }
        div[data-testid="stMetricLabel"] {
            color: var(--muted);
            font-size: .82rem;
        }
        div[data-testid="stMetricValue"] {
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-variant-numeric: tabular-nums;
            color: var(--ink);
        }

        /* Stat strip: wrap st.columns(...) metrics in st.container(key="stat-bar") */
        .st-key-stat-bar [data-testid="stHorizontalBlock"] {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 1.1rem .3rem;
        }
        .st-key-stat-bar [data-testid="stHorizontalBlock"] > div {
            padding: 0 1.3rem;
        }
        .st-key-stat-bar [data-testid="stHorizontalBlock"] > div:not(:last-child) {
            border-right: 1px solid var(--line);
        }

        div[data-testid="stAlert"] {
            border-radius: 12px;
            border: 1px solid var(--line);
        }
        div[data-testid="stDataFrame"] {
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid var(--line);
        }
        div[data-testid="stSlider"] {
            padding: .55rem .8rem .15rem;
            border: 1px solid var(--line);
            border-radius: 12px;
            background: var(--surface);
        }
        div[data-testid="stSlider"] [data-baseweb="slider"] > div { border-radius: 999px; }

        /* Buttons */
        .stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
            border-radius: 10px;
            border: 1px solid var(--line);
            background: var(--surface);
            color: var(--ink);
            min-height: 2.6rem;
            font-weight: 700;
            font-family: 'Plus Jakarta Sans', sans-serif;
            box-shadow: none;
            transition: color .15s ease, border-color .15s ease, background .15s ease;
        }
        .stButton > button:hover,
        .stDownloadButton > button:hover,
        .stFormSubmitButton > button:hover {
            color: #FFFFFF;
            border-color: var(--blue);
            background: var(--blue);
        }
        .stButton > button[kind="primary"],
        .stFormSubmitButton > button[kind="primary"] {
            background: var(--blue);
            border-color: var(--blue);
            color: #FFFFFF;
        }
        .stButton > button[kind="primary"]:hover,
        .stFormSubmitButton > button[kind="primary"]:hover {
            background: var(--blue-deep);
            border-color: var(--blue-deep);
        }

        /* Home nav: flat row list instead of icon-grid cards */
        .nav-list {
            display: flex;
            flex-direction: column;
            border-top: 1px solid var(--line);
            margin-top: .3rem;
        }
        .nav-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1.5rem;
            padding: 1.35rem .4rem;
            border-bottom: 1px solid var(--line);
            text-decoration: none !important;
            color: var(--ink);
            transition: background .15s ease, padding-left .15s ease;
        }
        .nav-row:hover { background: var(--blue-soft); padding-left: .9rem; }
        .nav-row-index {
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-size: .82rem;
            font-weight: 700;
            color: var(--faint);
            width: 1.6rem;
            flex-shrink: 0;
        }
        .nav-row-text { display: flex; flex-direction: column; gap: .28rem; flex-grow: 1; }
        .nav-row-title {
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-weight: 800;
            font-size: 1.08rem;
            letter-spacing: -.01em;
        }
        .nav-row-desc { font-size: .88rem; color: var(--muted); }
        .nav-row-arrow {
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-size: 1.2rem;
            color: var(--faint);
            transition: transform .15s ease, color .15s ease;
        }
        .nav-row:hover .nav-row-arrow { color: var(--blue); transform: translateX(4px); }

        @media (max-width: 560px) {
            .block-container { width: 100%; padding-left: 1rem; padding-right: 1rem; }
            .nav-row { padding: 1.1rem .2rem; }
        }
        .muted { color: var(--muted); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(tag: str, title: str, description: str) -> None:
    """페이지 상단 제목을 좌측 정렬 룰 기반 헤더로 표시한다."""

    st.markdown(
        f"""
        <div class="page-head">
            <span class="page-head-eyebrow">{tag}</span>
            <h1>{title}</h1>
            <div class="muted">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def home_button() -> None:
    """하위 페이지 왼쪽 위에 메인 화면 이동 링크를 표시한다."""

    with st.container(key="home-nav"):
        st.markdown(
            '<a class="home-link" href="/" target="_self">&larr; 홈</a>',
            unsafe_allow_html=True,
        )
