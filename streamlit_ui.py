"""Streamlit 화면에서 공통으로 사용하는 스타일과 작은 UI 도우미.

이 프로젝트는 화면 대부분을 Streamlit 기본 위젯 대신 직접 만든 HTML/CSS 컴포넌트로
그린다. 클릭·선택처럼 실제 상호작용이 필요한 부분(버튼, 셀렉트박스, 슬라이더, 폼)만
Streamlit 위젯을 쓰고, 통계 카드·표·배지·안내 박스·막대 그래프처럼 값만 보여주는
요소는 모두 이 파일의 함수들이 만드는 순수 HTML/CSS 조각으로 그린다.
"""

import html as _html
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pandas as pd
import streamlit as st


def _esc(value: Any) -> str:
    return _html.escape(str(value))


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
        .section-divider-thin { height: 1px; margin: 2.3rem 0; background: var(--line); }
        .section-spacer-lg { height: 1.6rem; }
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
            padding: .95rem 1.1rem .55rem;
            border: 1px solid var(--line);
            border-radius: 12px;
            background: var(--surface);
            overflow: visible;
        }
        /* 슬라이더 트랙/썸/틱바가 카드 테두리 밖으로 나가지 않도록 여유 폭을 준다.
           react-aria 기반 슬라이더(stSliderThumbValue/stSliderTickBar)는 값 라벨을
           트랙 양 끝에 절대 위치로 붙이기 때문에, 좌우 안쪽 여백이 없으면 라벨
           일부가 카드 밖으로 잘려 보인다. */
        div[data-testid="stSlider"] [data-rac] {
            overflow: visible;
        }
        div[data-testid="stSlider"] [data-testid="stSliderThumbValue"] {
            z-index: 5;
            white-space: nowrap;
        }
        div[data-testid="stSlider"] [data-testid="stSliderTickBar"] {
            display: flex;
            justify-content: space-between;
            padding: 0 .1rem;
            margin-top: .3rem;
        }
        div[data-testid="stSlider"] [data-testid="stSliderTickBar"] p {
            white-space: nowrap;
        }

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

        /* ---- Tone tokens (badge / alert / stat hint 공용) ---- */
        .tone-safe    { --tone-color: #0F9D58; --tone-bg: #E7F7EE; }
        .tone-info    { --tone-color: #1B64DA; --tone-bg: #EAF2FF; }
        .tone-warning { --tone-color: #B7791F; --tone-bg: #FFF6E1; }
        .tone-danger  { --tone-color: #D64545; --tone-bg: #FDECEC; }
        .tone-neutral { --tone-color: #4E5968; --tone-bg: #F2F3F5; }

        /* ---- Custom stat card grid (st.metric 대체) ---- */
        .stat-card-grid {
            display: flex;
            flex-wrap: wrap;
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 1.15rem .3rem;
            margin: .2rem 0 1.2rem;
        }
        .stat-card { flex: 1 1 0; min-width: 150px; padding: 0 1.3rem; }
        .stat-card:not(:last-child) { border-right: 1px solid var(--line); }
        .stat-card-label { color: var(--muted); font-size: .8rem; margin-bottom: .4rem; }
        .stat-card-value {
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-size: 1.55rem;
            font-weight: 800;
            color: var(--ink);
            font-variant-numeric: tabular-nums;
            line-height: 1.2;
        }
        .stat-card-hint { display: inline-block; margin-top: .4rem; font-size: .78rem; font-weight: 700; color: var(--tone-color); }
        @media (max-width: 760px) {
            .stat-card-grid { flex-direction: column; }
            .stat-card { padding: .7rem 1rem; }
            .stat-card:not(:last-child) { border-right: none; border-bottom: 1px solid var(--line); }
        }

        /* ---- Badge / pill ---- */
        .badge {
            display: inline-flex;
            align-items: center;
            padding: .32rem .72rem;
            border-radius: 999px;
            font-size: .78rem;
            font-weight: 700;
            color: var(--tone-color);
            background: var(--tone-bg);
            white-space: nowrap;
        }
        .verdict-badge {
            display: inline-block;
            padding: .55rem 1.05rem;
            border-radius: 10px;
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-weight: 800;
            font-size: .95rem;
            color: var(--tone-color);
            background: var(--tone-bg);
            margin-bottom: .6rem;
        }
        .role-pill {
            display: inline-flex;
            align-items: center;
            gap: .4rem;
            padding: .35rem .85rem;
            border-radius: 999px;
            background: var(--surface-alt);
            border: 1px solid var(--line);
            font-size: .78rem;
            font-weight: 700;
            color: var(--muted);
        }

        /* ---- Alert box (st.info/warning/error 대체) ---- */
        .alert-box {
            display: flex;
            gap: .7rem;
            align-items: flex-start;
            padding: 1rem 1.15rem;
            border-radius: 12px;
            background: var(--tone-bg);
            border-left: 3px solid var(--tone-color);
            margin: .8rem 0;
        }
        .alert-icon { font-size: 1.05rem; line-height: 1.4; }
        .alert-title { font-weight: 800; color: var(--ink); margin-bottom: .2rem; }
        .alert-body { color: var(--muted); font-size: .88rem; line-height: 1.55; }
        .alert-body b, .alert-body strong { color: var(--ink); }

        /* ---- Custom table (st.dataframe 대체) ---- */
        .table-wrap {
            width: 100%;
            max-width: 100%;
            overflow-x: auto;
            border: 1px solid var(--line);
            border-radius: 12px;
            margin: .6rem 0 1rem;
        }
        table.pretty-table { width: 100%; border-collapse: collapse; font-size: .87rem; }
        table.pretty-table thead th {
            position: sticky;
            top: 0;
            text-align: left;
            padding: .8rem 1rem;
            background: var(--surface-alt);
            color: var(--muted);
            font-weight: 700;
            font-size: .74rem;
            text-transform: uppercase;
            letter-spacing: .04em;
            border-bottom: 1px solid var(--line);
            white-space: nowrap;
        }
        table.pretty-table tbody td {
            padding: .72rem 1rem;
            border-bottom: 1px solid var(--line);
            color: var(--ink);
            white-space: nowrap;
        }
        table.pretty-table tbody tr:last-child td { border-bottom: none; }
        table.pretty-table tbody tr:hover { background: var(--blue-soft); }
        table.pretty-table tbody tr:nth-child(even) { background: rgba(247, 248, 250, .55); }
        table.pretty-table tbody tr:nth-child(even):hover { background: var(--blue-soft); }

        /* ---- Inline bar cell (표 안의 점수 막대) ---- */
        .bar-cell { display: flex; align-items: center; gap: .6rem; min-width: 130px; }
        .bar-cell-track { flex: 1; height: 8px; border-radius: 999px; background: var(--surface-alt); overflow: hidden; }
        .bar-cell-fill { height: 100%; border-radius: 999px; background: var(--blue); }
        .bar-cell-text { font-variant-numeric: tabular-nums; font-size: .82rem; color: var(--muted); min-width: 46px; text-align: right; }

        /* ---- Horizontal bar chart (st.bar_chart/line_chart 대체) ---- */
        .hbar-chart { display: flex; flex-direction: column; gap: .6rem; margin: .5rem 0 1rem; }
        .hbar-row { display: grid; grid-template-columns: 130px 1fr 64px; align-items: center; gap: .8rem; }
        .hbar-label { font-size: .84rem; color: var(--muted); }
        .hbar-track { height: 11px; border-radius: 999px; background: var(--surface-alt); overflow: hidden; }
        .hbar-fill { height: 100%; border-radius: 999px; transition: width .25s ease; }
        .hbar-value { font-size: .84rem; font-weight: 700; color: var(--ink); text-align: right; font-variant-numeric: tabular-nums; }
        @media (max-width: 560px) {
            .hbar-row { grid-template-columns: 92px 1fr 52px; }
        }

        /* ---- Landing hero & role cards ---- */
        .hero-wrap { text-align: center; padding: 3rem 0 2.4rem; }
        .hero-eyebrow {
            display: inline-block;
            padding: .42rem .95rem;
            border-radius: 999px;
            background: var(--blue-soft);
            color: var(--blue-deep);
            font-size: .76rem;
            font-weight: 800;
            letter-spacing: .07em;
            margin-bottom: 1.1rem;
        }
        .hero-title {
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-size: 2.35rem;
            font-weight: 800;
            letter-spacing: -.03em;
            margin: 0 0 .85rem;
        }
        .hero-desc {
            color: var(--muted);
            font-size: 1.02rem;
            max-width: 620px;
            margin: 0 auto;
            line-height: 1.65;
            text-align: center;
        }
        .role-card {
            display: flex;
            flex-direction: column;
            height: 100%;
            min-height: 336px;
            padding: 2.1rem 1.9rem 1.6rem;
            border: 1px solid var(--line);
            border-radius: 20px;
            background: var(--surface);
            transition: box-shadow .18s ease, transform .18s ease, border-color .18s ease;
        }
        /* role-card의 실제 높이(두 카드 중 더 큰 쪽)는 JS(equalize_role_cards, main.py)가
           렌더 후 측정해서 min-height를 직접 맞춘다. Streamlit이 컬럼 내부를 여러
           단계의 flex 래퍼(stElementContainer/stLayoutWrapper 등)로 감싸다 보니,
           순수 CSS의 flex-grow/align-items: stretch만으로는 중간 래퍼 중 하나가
           block 레이아웃이라 높이가 전달되지 않아 두 버튼 위치가 어긋나는 문제가
           있었다. min-height: 336px는 JS가 아직 실행되기 전 첫 페인트에 쓰이는
           기본값이다. */
        div[class*="st-key-role-panel-"] .role-card {
            margin-bottom: 1rem;
        }
        div[class*="st-key-role-panel-"]:hover .role-card {
            transform: translateY(-3px);
            box-shadow: 0 16px 34px rgba(25, 31, 40, .09);
            border-color: var(--blue);
        }
        .role-card-icon {
            width: 3rem;
            height: 3rem;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
            border-radius: 14px;
            background: var(--blue-soft);
            margin-bottom: 1.1rem;
        }
        .role-card-title { font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.32rem; font-weight: 800; margin-bottom: .55rem; }
        .role-card-desc { color: var(--muted); font-size: .92rem; margin-bottom: 1.15rem; line-height: 1.6; }
        .role-card-points { list-style: none; margin: auto 0 .3rem; padding: 0; display: flex; flex-direction: column; gap: .5rem; }
        .role-card-points li { font-size: .86rem; color: var(--muted); padding-left: 1.05rem; position: relative; }
        .role-card-points li::before { content: '—'; position: absolute; left: 0; color: var(--blue); font-weight: 700; }

        /* ---- Segmented tab bar (버튼 기반, st.tabs 대체) ---- */
        div[class*="st-key-tabbar-"] div[data-testid="stHorizontalBlock"] {
            gap: .55rem;
        }
        div[class*="st-key-tabbar-"] .stButton > button {
            border-radius: 12px;
        }
        div[class*="st-key-subtabbar-"] .stButton > button {
            border-radius: 10px;
            min-height: 2.3rem;
            font-size: .86rem;
        }

        /* ---- 연봉 협상 탭: 좌하단 플로팅 시뮬레이션 버튼 ---- */
        .st-key-salary-sim-fab {
            position: fixed;
            right: 1.6rem;
            bottom: 1.6rem;
            z-index: 998;
            width: auto;
        }
        .st-key-salary-sim-fab .stButton > button {
            border-radius: 999px;
            padding: .85rem 1.3rem;
            background: var(--blue);
            border-color: var(--blue);
            color: #FFFFFF;
            box-shadow: 0 10px 26px rgba(49, 130, 246, .35);
            font-size: .86rem;
        }
        .st-key-salary-sim-fab .stButton > button:hover {
            background: var(--blue-deep);
            border-color: var(--blue-deep);
        }

        /* ---- Native input polish (선택박스/멀티셀렉트/슬라이더를 카드 톤에 맞춤) ---- */
        div[data-testid="stSelectbox"] > div > div,
        div[data-baseweb="select"] > div {
            border-radius: 10px !important;
            border-color: var(--line) !important;
        }
        div[data-baseweb="tag"] { border-radius: 8px !important; background: var(--blue-soft) !important; }
        div[data-baseweb="tag"] span { color: var(--blue-deep) !important; }
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


def top_navigation(role: str | None = None) -> None:
    """사이드바 대신 모든 화면에서 사용하는 상단 내비게이션을 표시한다.

    role이 주어지면 오른쪽에 현재 접속 유형 배지를 보여준다. 이 프로젝트는 페이지가
    랜딩(main.py)과 워크스페이스(pages/01_Workspace.py) 둘뿐이라, 상단 내비게이션은
    브랜드 링크와 상태 표시만 담당하고 실제 기능 전환은 워크스페이스 안의 버튼형
    탭이 맡는다.
    """

    role_label = {"hr": "인사팀 모드", "admin": "관리자 모드"}.get(role or "", "")
    role_html = f'<span class="role-pill">👤 {role_label}</span>' if role_label else ""
    with st.container(key="top-navigation"):
        st.markdown(
            f"""
            <div class="top-nav">
                <a class="top-nav-brand" href="/" target="_self">STAYON</a>
                <div class="top-nav-links"></div>
                <div class="top-nav-status">{role_html}<span style="margin-left:.6rem;">내부 서버 정상</span></div>
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


def stat_cards(items: Sequence[Mapping[str, Any]]) -> None:
    """st.metric 대신 쓰는 카드형 통계 스트립.

    items 각 원소: {"label": str, "value": str, "hint": str(optional), "tone": str(optional)}
    """

    cards = "".join(
        (
            '<div class="stat-card">'
            f'<div class="stat-card-label">{_esc(item["label"])}</div>'
            f'<div class="stat-card-value">{item["value"]}</div>'
            + (
                f'<div class="stat-card-hint tone-{item.get("tone", "info")}">{_esc(item["hint"])}</div>'
                if item.get("hint")
                else ""
            )
            + "</div>"
        )
        for item in items
    )
    st.markdown(f'<div class="stat-card-grid">{cards}</div>', unsafe_allow_html=True)


def badge(text: str, tone: str = "info") -> str:
    """어디서나 재사용하는 배지 HTML 조각을 반환한다."""

    return f'<span class="badge tone-{tone}">{_esc(text)}</span>'


def alert_box(kind: str, message: str, title: str | None = None) -> None:
    """st.info/st.warning/st.error/st.success 대신 쓰는 커스텀 안내 박스.

    kind: "info" | "warning" | "danger" | "success"
    """

    icons = {"info": "ℹ️", "warning": "⚠️", "danger": "⛔", "success": "✅"}
    tone = {"info": "info", "warning": "warning", "danger": "danger", "success": "safe"}.get(
        kind, "info"
    )
    title_html = f'<div class="alert-title">{_esc(title)}</div>' if title else ""
    st.markdown(
        f'<div class="alert-box tone-{tone}">'
        f'<div class="alert-icon">{icons.get(kind, "ℹ️")}</div>'
        f'<div class="alert-body">{title_html}<div>{message}</div></div>'
        "</div>",
        unsafe_allow_html=True,
    )


def render_table(
    frame: pd.DataFrame,
    formats: Mapping[str, str] | None = None,
    badges: Mapping[str, Callable[[Any], tuple[str, str]]] | None = None,
    bars: Mapping[str, float] | None = None,
    widths: Mapping[str, str] | None = None,
) -> None:
    """DataFrame을 카드 톤에 맞춘 순수 HTML 표로 그린다 (st.dataframe 대체).

    - formats: {열이름: "{:.1f}" 같은 포맷 문자열}
    - badges: {열이름: value -> (표시 라벨, tone)}  → 배지로 렌더링
    - bars: {열이름: 최댓값}  → 인라인 진행 막대로 렌더링
    - widths: {열이름: "40%"/"180px" 같은 폭}  → 지정하면 <colgroup>으로 폭을 고정해서,
      행마다 값 길이가 달라져도(예: 부서/직급을 바꿀 때) 컬럼 폭이 흔들리지 않게 한다.
    """

    formats = formats or {}
    badges = badges or {}
    bars = bars or {}
    widths = widths or {}

    colgroup = ""
    table_style = ""
    if widths:
        colgroup = (
            "<colgroup>"
            + "".join(
                f'<col style="width:{_esc(widths[col])}">' if col in widths else "<col>"
                for col in frame.columns
            )
            + "</colgroup>"
        )
        table_style = ' style="table-layout:fixed;"'

    headers = "".join(f"<th>{_esc(col)}</th>" for col in frame.columns)
    body_rows: list[str] = []
    for _, row in frame.iterrows():
        cells = []
        for col in frame.columns:
            value = row[col]
            if col in badges:
                label, tone = badges[col](value)
                cells.append(f'<td><span class="badge tone-{tone}">{_esc(label)}</span></td>')
            elif col in bars:
                max_value = bars[col] or 1
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    numeric = 0.0
                pct = max(0.0, min(100.0, (numeric / max_value) * 100))
                text = formats[col].format(value) if col in formats else str(value)
                cells.append(
                    '<td><div class="bar-cell">'
                    f'<div class="bar-cell-track"><div class="bar-cell-fill" style="width:{pct:.1f}%"></div></div>'
                    f'<span class="bar-cell-text">{_esc(text)}</span>'
                    "</div></td>"
                )
            else:
                text = formats[col].format(value) if col in formats else value
                cells.append(f"<td>{_esc(text)}</td>")
        body_rows.append(f"<tr>{''.join(cells)}</tr>")

    table_html = (
        f'<div class="table-wrap"><table class="pretty-table"{table_style}>{colgroup}'
        f"<thead><tr>{headers}</tr></thead><tbody>{''.join(body_rows)}</tbody></table></div>"
    )
    st.markdown(table_html, unsafe_allow_html=True)


def hbar_chart(
    items: Sequence[tuple[str, float]],
    max_value: float | None = None,
    value_format: str = "{:.1f}",
    color: str = "var(--blue)",
    min_height: str | None = None,
) -> None:
    """st.bar_chart/st.line_chart 대신 쓰는 가로 막대 그래프 (단일 계열).

    min_height를 주면 막대 개수가 적어도(예: 5개) 옆 카드/표와 높이를 맞출 수 있게,
    전체 높이를 그 값만큼 확보하고 막대들을 위아래로 고르게 펼쳐(justify-content:
    space-between) 배치한다. 지정하지 않으면 기존처럼 내용 높이만큼만 차지한다.
    """

    values = [float(v) for _, v in items]
    scale = max_value if max_value is not None else (max(values) if values else 1) or 1
    rows = "".join(
        (
            '<div class="hbar-row">'
            f'<div class="hbar-label">{_esc(label)}</div>'
            '<div class="hbar-track">'
            f'<div class="hbar-fill" style="width:{max(0.0, min(100.0, (value / scale) * 100)):.1f}%; background:{color};"></div>'
            "</div>"
            f'<div class="hbar-value">{value_format.format(value)}</div>'
            "</div>"
        )
        for label, value in items
    )
    style = (
        f' style="min-height:{_esc(min_height)}; justify-content:space-between;"'
        if min_height
        else ""
    )
    st.markdown(f'<div class="hbar-chart"{style}>{rows}</div>', unsafe_allow_html=True)


def section_heading(kicker: str, title: str, description: str) -> None:
    """탭 콘텐츠 상단에 쓰는 좌측 정렬 섹션 제목."""

    st.markdown(
        f"""
        <div class="section-kicker">{_esc(kicker)}</div>
        <div class="section-title">{_esc(title)}</div>
        <div class="section-desc">{_esc(description)}</div>
        """,
        unsafe_allow_html=True,
    )


def feature_pills(features: Sequence[str], labels: Mapping[str, str]) -> None:
    """피처 이름 목록을 알약형 태그 줄로 표시한다."""

    pills = "".join(f'<span class="feature-pill">{_esc(labels.get(f, f))}</span>' for f in features)
    st.markdown(f'<div class="feature-pills">{pills}</div>', unsafe_allow_html=True)


def employee_picker(
    employees: pd.DataFrame,
    key_prefix: str,
    with_direct_search: bool = False,
) -> int | None:
    """부서 → 직급 → ID 순으로 좁혀가는 3단 드롭다운 직원 선택기.

    with_direct_search=True면 위에 직원 ID 직접 검색창을 추가로 보여주고, 값을
    입력해 일치하는 직원이 있으면 그 ID를 우선 반환한다(캐스케이딩 선택 무시).
    """

    from src.utils.hr_metrics import LEVEL_KR, department_options, level_options, translate

    if with_direct_search:
        search_text = st.text_input(
            "직원 ID로 바로 찾기",
            key=f"{key_prefix}_direct_search",
            placeholder="예: 10345",
        )
        if search_text.strip():
            try:
                search_id = int(search_text.strip())
            except ValueError:
                search_id = None
            if search_id is not None and employees["Employee ID"].eq(search_id).any():
                st.markdown(
                    f'<span class="badge tone-safe">ID {search_id} 검색됨</span>',
                    unsafe_allow_html=True,
                )
                return search_id
            alert_box("warning", "일치하는 직원 ID를 찾지 못했어요. 아래 조건으로 찾아보세요.")
        st.markdown('<div class="section-divider-thin"></div>', unsafe_allow_html=True)

    dept_options = department_options(employees)
    col1, col2, col3 = st.columns(3)
    with col1:
        department = st.selectbox(
            "부서",
            dept_options,
            format_func=lambda d: translate(d),
            key=f"{key_prefix}_dept",
        )
    lvl_options = level_options(employees, department)
    with col2:
        level = st.selectbox(
            "직급",
            lvl_options,
            format_func=lambda level_value: LEVEL_KR.get(level_value, level_value),
            key=f"{key_prefix}_level",
        )
    subset = employees[employees["Job Role"].eq(department) & employees["Job Level"].eq(level)]
    ids = subset["Employee ID"].tolist()
    if not ids:
        with col3:
            st.selectbox("직원 ID", ["해당 없음"], key=f"{key_prefix}_id_empty", disabled=True)
        return None
    id_labels = dict(
        zip(subset["Employee ID"], "ID " + subset["Employee ID"].astype(str), strict=False)
    )
    with col3:
        selected_id = st.selectbox(
            "직원 ID",
            ids,
            format_func=lambda employee_id: id_labels.get(employee_id, str(employee_id)),
            key=f"{key_prefix}_id",
        )
    return selected_id


def sub_tabs(options: Sequence[tuple[str, str]], state_key: str, default: str | None = None) -> str:
    """st.tabs 대신 쓰는 세그먼트 버튼형 하위 탭. 선택된 키를 반환한다."""

    if state_key not in st.session_state:
        st.session_state[state_key] = default or options[0][0]

    with st.container(key=f"subtabbar-{state_key}"):
        columns = st.columns(len(options))
        for column, (key, label) in zip(columns, options, strict=False):
            with column:
                is_active = st.session_state[state_key] == key
                if st.button(
                    label,
                    key=f"{state_key}_{key}",
                    type="primary" if is_active else "secondary",
                    width="stretch",
                ):
                    st.session_state[state_key] = key
                    st.rerun()

    return st.session_state[state_key]
