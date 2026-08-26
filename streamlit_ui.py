"""Streamlit 화면에서 공통으로 사용하는 스타일과 작은 UI 도우미."""

import streamlit as st


def apply_page_style() -> None:
    """모든 페이지에 부드러운 카드형 스타일을 적용한다."""

    st.markdown(
        """
        <style>
        :root {
            --ink: #263238;
            --muted: #65736f;
            --mint: #62a88f;
            --mint-soft: #eaf6f1;
            --peach-soft: #fff3ea;
            --line: #dce9e3;
        }
        .stApp {
            background: linear-gradient(145deg, #fbfdfc 0%, #f4faf7 55%, #fffaf6 100%);
            color: var(--ink);
        }
        .block-container {
            width: min(94vw, 1720px);
            max-width: 1720px;
            padding: 2.2rem 1.5rem 3rem;
        }
        [data-testid="stSidebar"] {
            background: #f0f8f4;
            border-right: 1px solid var(--line);
        }
        h1, h2, h3 { letter-spacing: -0.035em; color: var(--ink); }
        h1 { font-size: 2.25rem !important; }
        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.86);
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 1rem 1.1rem;
            box-shadow: 0 8px 24px rgba(52, 92, 77, 0.07);
        }
        div[data-testid="stMetricLabel"] { color: var(--muted); }
        div[data-testid="stAlert"] { border-radius: 16px; }
        div[data-testid="stDataFrame"] { border-radius: 16px; overflow: hidden; }
        .stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
            border-radius: 999px;
            border: 1px solid #88bda9;
            min-height: 2.7rem;
            font-weight: 700;
        }
        .nav-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 1.15rem;
            margin: .4rem 0 1rem;
        }
        .nav-card {
            display: flex;
            min-height: 150px;
            align-items: flex-start;
            justify-content: center;
            flex-direction: column;
            padding: 1.2rem 1.35rem;
            border-radius: 22px;
            border: 1px solid var(--line);
            background: rgba(255,255,255,.88);
            color: var(--ink);
            text-decoration: none !important;
            box-shadow: 0 8px 24px rgba(52, 92, 77, 0.06);
            transition: transform .18s ease, box-shadow .18s ease,
                        border-color .18s ease, background .18s ease;
        }
        .nav-card:hover {
            transform: translateY(-5px) scale(1.025);
            border-color: #56a486;
            background: linear-gradient(145deg, #ffffff 0%, #eaf7f1 100%);
            box-shadow: 0 18px 38px rgba(52, 92, 77, 0.17);
            color: #2f755d;
        }
        .nav-card-icon { font-size: 1.65rem; margin-bottom: .55rem; }
        .nav-card-title {
            font-size: 1.45rem;
            line-height: 1.2;
            font-weight: 850;
            letter-spacing: -.03em;
            color: var(--ink);
            margin-bottom: .7rem;
        }
        .nav-card-description {
            font-size: .96rem;
            line-height: 1.55;
            color: var(--muted);
        }
        @media (max-width: 900px) {
            .nav-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }
        @media (max-width: 560px) {
            .nav-grid { grid-template-columns: 1fr; }
            .block-container {
                width: 100%;
                padding-left: 1rem;
                padding-right: 1rem;
            }
        }
        .home-link {
            display: inline-flex;
            align-items: center;
            gap: .35rem;
            width: fit-content;
            min-height: 2.35rem;
            padding: .45rem .9rem;
            border-radius: 999px;
            border: 1px solid var(--line);
            background: rgba(255,255,255,.9);
            color: #39745e;
            font-weight: 800;
            text-decoration: none !important;
            transition: transform .15s ease, background .15s ease;
        }
        .home-link:hover {
            transform: translateX(-3px);
            background: var(--mint-soft);
            border-color: #82bca6;
            color: #2f755d;
        }
        .soft-card {
            padding: 1.15rem 1.25rem;
            border-radius: 20px;
            border: 1px solid var(--line);
            background: rgba(255,255,255,.82);
            box-shadow: 0 8px 24px rgba(52, 92, 77, 0.06);
            margin: .6rem 0 1rem;
        }
        .eyebrow {
            display: inline-block;
            padding: .35rem .75rem;
            border-radius: 999px;
            background: var(--mint-soft);
            color: #39745e;
            font-size: .82rem;
            font-weight: 800;
            margin-bottom: .55rem;
        }
        .muted { color: var(--muted); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(tag: str, title: str, description: str) -> None:
    """페이지 상단 제목을 같은 톤으로 표시한다."""

    st.markdown(
        f"""
        <div class="soft-card">
            <span class="eyebrow">{tag}</span>
            <h1 style="margin:.1rem 0 .4rem;">{title}</h1>
            <div class="muted">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def home_button() -> None:
    """하위 페이지 왼쪽 위에 메인 화면 이동 버튼을 표시한다."""

    with st.container(key="home-nav"):
        st.markdown(
            '<a class="home-link" href="/" target="_self">🏠 HOME</a>',
            unsafe_allow_html=True,
        )
