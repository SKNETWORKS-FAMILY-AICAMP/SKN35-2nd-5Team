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
            padding: 1.2rem 1.5rem 3rem;
        }
        [data-testid="stSidebar"] {
            display: none;
        }
        header[data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stDecoration"] {
            display: none !important;
        }

        /* Top navigation */
        .top-nav {
            position: relative;
            display: grid;
            grid-template-columns: 1fr auto 1fr;
            align-items: center;
            min-height: 72px;
            margin: -1.2rem calc(50% - 50vw) 2.2rem;
            padding: 0 max(3rem, calc((100vw - 1560px) / 2));
            background: rgba(255, 255, 255, .88);
            border-bottom: 1px solid var(--line);
            backdrop-filter: blur(22px);
        }
        .st-key-top-navigation {
            position: sticky;
            top: 0;
            z-index: 999;
            overflow: visible !important;
        }
        .st-key-top-navigation [data-testid="stMarkdownContainer"] {
            overflow: visible !important;
        }
        .top-nav-brand {
            color: var(--ink) !important;
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-size: 1.38rem;
            font-weight: 800;
            letter-spacing: .055em;
            text-decoration: none !important;
        }
        .top-nav-links { display: flex; align-items: center; gap: .35rem; }
        .top-nav-link {
            color: var(--muted) !important;
            padding: .68rem 1rem;
            border-radius: 10px;
            font-size: .88rem;
            font-weight: 600;
            text-decoration: none !important;
            transition: color .15s ease, background .15s ease;
        }
        .top-nav-link:hover, .top-nav-link.active {
            color: var(--ink) !important;
            background: #F2F3F5;
        }
        .top-nav-status { justify-self: end; color: var(--faint); font-size: .76rem; }
        .top-nav-status::before {
            content: '';
            display: inline-block;
            width: 7px;
            height: 7px;
            margin-right: .42rem;
            border-radius: 50%;
            background: #30B957;
        }

        /* Long-form attrition workspace */
        .section-jump {
            position: sticky;
            top: 78px;
            z-index: 20;
            display: flex;
            gap: .45rem;
            overflow-x: auto;
            margin: 0 0 2rem;
            padding: .55rem;
            border: 1px solid var(--line);
            border-radius: 14px;
            background: rgba(255, 255, 255, .9);
            backdrop-filter: blur(18px);
        }
        .section-jump a {
            flex: 1;
            min-width: max-content;
            padding: .72rem .85rem;
            border-radius: 9px;
            color: var(--muted) !important;
            font-size: .82rem;
            font-weight: 700;
            text-align: center;
            text-decoration: none !important;
        }
        .section-jump a:hover { color: var(--blue-deep) !important; background: var(--blue-soft); }
        .scroll-section { scroll-margin-top: 155px; padding-top: .45rem; }
        .section-kicker {
            margin-bottom: .45rem;
            color: var(--blue-deep);
            font-size: .72rem;
            font-weight: 800;
            letter-spacing: .08em;
        }
        .section-title {
            margin: 0 0 .4rem;
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-size: 1.55rem;
            font-weight: 800;
            letter-spacing: -.03em;
        }
        .section-desc { margin-bottom: 1.3rem; color: var(--muted); font-size: .92rem; }
        .section-divider { height: 1px; margin: 4rem 0; background: var(--line); }
        .feature-pills { display: flex; flex-wrap: wrap; gap: .45rem; margin: .8rem 0 1.2rem; }
        .feature-pill {
            padding: .42rem .68rem;
            border-radius: 999px;
            background: #F2F3F5;
            color: var(--muted);
            font-size: .75rem;
            font-weight: 600;
        }
        .decision-note {
            margin: 1rem 0;
            padding: 1rem 1.1rem;
            border-radius: 12px;
            background: var(--blue-soft);
            color: var(--blue-deep);
            font-size: .86rem;
            line-height: 1.55;
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
        [class*="st-key-stat-bar"] [data-testid="stHorizontalBlock"] {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 1.1rem .3rem;
        }
        [class*="st-key-stat-bar"] [data-testid="stHorizontalBlock"] > div {
            padding: 0 1.3rem;
        }
        [class*="st-key-stat-bar"] [data-testid="stHorizontalBlock"] > div:not(:last-child) {
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
            .top-nav {
                grid-template-columns: 1fr;
                gap: .6rem;
                margin-left: -1rem;
                margin-right: -1rem;
                padding: .8rem 1rem;
            }
            .top-nav-brand, .top-nav-status { display: none; }
            .top-nav-links { justify-content: center; overflow-x: auto; }
            .top-nav-link { padding: .6rem .75rem; white-space: nowrap; }
            .section-jump { top: 67px; }
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


def top_navigation(active: str) -> None:
    """사이드바 대신 모든 화면에서 사용하는 상단 내비게이션을 표시한다."""

    links = [
        ("main", "/", "메인 (EDA)"),
        ("models", "/ML_Comparison", "모델 분석"),
        ("attrition", "/Attrition_Prediction", "퇴사 예측"),
    ]
    link_html = "".join(
        (
            f'<a class="top-nav-link{" active" if key == active else ""}" '
            f'href="{href}" target="_self">{label}</a>'
        )
        for key, href, label in links
    )
    with st.container(key="top-navigation"):
        st.markdown(
            f"""
            <div class="top-nav">
                <a class="top-nav-brand" href="/" target="_self">EXITWISE</a>
                <div class="top-nav-links">{link_html}</div>
                <div class="top-nav-status">내부 서버 정상</div>
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
