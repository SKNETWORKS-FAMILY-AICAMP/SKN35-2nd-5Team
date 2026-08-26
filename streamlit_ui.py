"""Streamlit 화면에서 공통으로 사용하는 스타일과 작은 UI 도우미."""

import streamlit as st


def apply_page_style() -> None:
    """모든 페이지에 부드러운 카드형 스타일을 적용한다."""

    st.markdown(
        """
        <style>
        :root {
            --ink: #302b38;
            --muted: #756d78;
            --rose: #b83f63;
            --rose-deep: #7b2944;
            --rose-soft: #f2bfd0;
            --blue: #4779bd;
            --blue-soft: #c4d9f4;
            --ivory: #fcf9f8;
            --line: #e8dfe4;
        }
        .stApp {
            background:
                radial-gradient(circle at 5% 4%, rgba(201, 55, 96, .28), transparent 35%),
                radial-gradient(circle at 96% 8%, rgba(54, 105, 177, .30), transparent 38%),
                linear-gradient(145deg, #fbeef2 0%, #f8eff4 44%, #e6effa 100%);
            color: var(--ink);
        }
        .block-container {
            width: min(94vw, 1720px);
            max-width: 1720px;
            padding: 2.2rem 1.5rem 3rem;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #fbf5f7, #f1f4f9);
            border-right: 1px solid var(--line);
        }
        h1, h2, h3 { letter-spacing: -0.035em; color: var(--ink); }
        h1 { font-size: 2.25rem !important; }
        div[data-testid="stMetric"] {
            background: linear-gradient(145deg, #fff6f9, #efc6d3);
            border: 1px solid #dda6b7;
            border-radius: 22px;
            padding: 1rem 1.1rem;
            box-shadow: 0 12px 32px rgba(80, 58, 76, .08);
            transition: transform .25s ease, box-shadow .25s ease;
        }
        div[data-testid="stHorizontalBlock"] > div:nth-child(even) div[data-testid="stMetric"] {
            background: linear-gradient(145deg, #f5f9ff, #c3d8f2);
            border-color: #97b5dc;
        }
        div[data-testid="stMetric"]:hover {
            transform: translateY(-3px);
            border-color: transparent;
            background:
                linear-gradient(145deg, #fff4f8, #edc3d1 48%, #c5daf3) padding-box,
                linear-gradient(110deg, #cf3e65, #8161ab, #397dcc, #cf3e65) border-box;
            background-size: 100% 100%, 280% 100%;
            box-shadow: 0 16px 36px rgba(82, 67, 120, .20);
            animation: gradientFlow 4.5s ease infinite;
        }
        div[data-testid="stMetricLabel"] { color: var(--muted); }
        div[data-testid="stAlert"] { border-radius: 16px; }
        div[data-testid="stDataFrame"] { border-radius: 16px; overflow: hidden; }
        div[data-testid="stSlider"] {
            padding: .55rem .8rem .15rem;
            border: 1px solid var(--line);
            border-radius: 16px;
            background: rgba(255,255,255,.76);
        }
        div[data-testid="stSlider"] [data-baseweb="slider"] > div {
            border-radius: 999px;
        }
        .stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
            border-radius: 999px;
            border: 1px solid #c796a4;
            min-height: 2.7rem;
            font-weight: 700;
            box-shadow: 0 6px 16px rgba(117, 71, 91, .08);
            transition: color .2s ease, border-color .2s ease,
                        box-shadow .2s ease, transform .2s ease;
        }
        .stButton > button:hover,
        .stDownloadButton > button:hover,
        .stFormSubmitButton > button:hover {
            color: white;
            border-color: transparent;
            background: linear-gradient(110deg, #bd4568, #806aa7, #477fc4, #bd4568);
            background-size: 260% 100%;
            box-shadow: 0 10px 26px rgba(92, 65, 133, .25);
            transform: translateY(-2px);
            animation: gradientFlow 4s ease infinite;
        }
        .nav-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 1.15rem;
            margin: .4rem 0 1rem;
        }
        .nav-card {
            position: relative;
            isolation: isolate;
            overflow: hidden;
            display: flex;
            min-height: 150px;
            align-items: flex-start;
            justify-content: center;
            flex-direction: column;
            padding: 1.2rem 1.35rem;
            border-radius: 22px;
            border: 1px solid var(--line);
            background: rgba(255,255,255,.9);
            color: var(--ink);
            text-decoration: none !important;
            box-shadow: 0 12px 30px rgba(74, 58, 78, .07);
            transition: transform .18s ease, box-shadow .18s ease,
                        border-color .18s ease, background .18s ease;
        }
        .nav-card:nth-child(odd) {
            background: linear-gradient(145deg, #fff5f8 0%, #f0bfd0 100%);
            border-color: #dc9eb2;
        }
        .nav-card:nth-child(even) {
            background: linear-gradient(145deg, #f6faff 0%, #c4d9f3 100%);
            border-color: #99b8df;
        }
        .nav-card > * { position: relative; z-index: 2; }
        .nav-card::before {
            content: "";
            position: absolute;
            z-index: 1;
            width: 155%;
            height: 105px;
            left: -28%;
            bottom: -120px;
            border-radius: 45% 55% 42% 58%;
            background: linear-gradient(
                95deg,
                rgba(210, 54, 98, .58),
                rgba(123, 92, 177, .46),
                rgba(53, 116, 201, .56)
            );
            filter: blur(7px);
            opacity: 0;
            transition: bottom .5s ease, opacity .4s ease;
        }
        .nav-card:hover {
            transform: translateY(-5px) scale(1.025);
            border: 2px solid transparent;
            background:
                linear-gradient(145deg, #fff6f9, #efd0dc 48%, #ccdef4) padding-box,
                linear-gradient(110deg, #d23b65, #8562ad, #397dcc, #d23b65) border-box;
            background-size: 100% 100%, 280% 100%;
            box-shadow: 0 20px 44px rgba(103, 67, 87, .16);
            color: var(--rose-deep);
            animation: gradientFlow 4.5s ease infinite;
        }
        .nav-card:hover::before {
            bottom: -38px;
            opacity: .30;
            animation: waveFloat 3.8s ease-in-out infinite alternate;
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
            background: rgba(255,255,255,.92);
            color: var(--rose-deep);
            font-weight: 800;
            text-decoration: none !important;
            transition: transform .15s ease, background .15s ease;
        }
        .st-key-home-nav {
            margin-top: .6rem;
            margin-bottom: .35rem;
            overflow: visible;
        }
        .st-key-home-nav [data-testid="stMarkdownContainer"] {
            padding: .75rem .2rem .25rem;
            overflow: visible;
        }
        .home-link:hover {
            transform: translateX(-3px);
            background: linear-gradient(135deg, var(--rose-soft), var(--blue-soft));
            border-color: #c58b9b;
            color: #743d50;
        }
        .soft-card {
            padding: 1.15rem 1.25rem;
            border-radius: 20px;
            border: 1px solid #d49aad;
            background: linear-gradient(
                140deg,
                #fff5f8,
                #f0c7d5 48%,
                #cbdff5
            );
            background-size: 180% 180%;
            box-shadow: 0 14px 38px rgba(81, 61, 82, .08);
            margin: .6rem 0 1rem;
            animation: headerFlow 10s ease-in-out infinite;
        }
        .eyebrow {
            display: inline-block;
            padding: .35rem .75rem;
            border-radius: 999px;
            background: linear-gradient(120deg, #edb2c4, #bcd4f1);
            color: var(--rose-deep);
            font-size: .82rem;
            font-weight: 800;
            margin-bottom: .55rem;
        }
        .muted { color: var(--muted); }
        @keyframes gradientFlow {
            0% { background-position: 0% 50%, 0% 50%; }
            50% { background-position: 100% 50%, 100% 50%; }
            100% { background-position: 0% 50%, 0% 50%; }
        }
        @keyframes waveFloat {
            0% { transform: translateX(-4%) rotate(-2deg) scaleY(.90); }
            100% { transform: translateX(5%) rotate(2deg) scaleY(1.08); }
        }
        @keyframes headerFlow {
            0%, 100% { background-position: 0% 45%; }
            50% { background-position: 100% 55%; }
        }
        @media (prefers-reduced-motion: reduce) {
            .nav-card, .nav-card::before, .soft-card,
            .stButton > button, .stDownloadButton > button,
            .stFormSubmitButton > button { animation: none !important; }
        }
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
            '<a class="home-link" href="/" target="_self">🏠 홈</a>',
            unsafe_allow_html=True,
        )
