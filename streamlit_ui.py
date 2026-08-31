"""Streamlit 화면에서 공통으로 사용하는 스타일과 UI 컴포넌트 (STAYON Design System).

이 프로젝트는 화면 대부분을 Streamlit 기본 위젯 대신 직접 만든 HTML/CSS 컴포넌트로
그린다. 클릭·선택처럼 실제 상호작용이 필요한 부분(버튼, 셀렉트박스, 슬라이더, 폼)만
Streamlit 위젯을 쓰고, 통계·표·배지·안내 박스·차트·랭킹 리스트처럼 값만 보여주는
요소는 모두 이 파일의 함수들이 만드는 순수 HTML/CSS 조각으로 그린다.

디자인 언어는 "Apple Store + Vision Pro + Enterprise SaaS"를 목표로 한다: 넓은
여백, 큰 타이포그래피, Liquid Glass(반투명 + blur) 레이어, 알약형 컨트롤, 절제된
색상, 에디토리얼한 데이터 시각화를 핵심 원칙으로 삼는다. 모든 화면은 이 파일이
정의하는 하나의 디자인 시스템(색·타이포·간격·radius·shadow·glass·컴포넌트)을
공유한다.
"""

import html as _html
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pandas as pd
import streamlit as st


def _esc(value: Any) -> str:
    return _html.escape(str(value))


def apply_page_style() -> None:
    """모든 페이지에 STAYON 디자인 시스템(Liquid Glass · Apple Store 톤)을 적용한다."""

    st.markdown(
        """
        <style>
        :root {
            /* ---- Neutral base ---- */
            --ink: #1D1D1F;
            --muted: #6E6E73;
            --faint: #86868B;
            --surface: #FFFFFF;
            --surface-alt: #F5F5F7;
            --surface-sunken: #EDEDF0;
            --line: #D2D2D7;
            --line-soft: #E8E8ED;

            /* ---- Brand accent ---- */
            --blue: #0071E3;
            --blue-deep: #0058C7;
            --blue-soft: rgba(0, 113, 227, .08);

            /* ---- Liquid glass ---- */
            --glass-bg: rgba(255, 255, 255, .62);
            --glass-bg-strong: rgba(255, 255, 255, .78);
            --glass-border: rgba(255, 255, 255, .55);
            --glass-blur: 28px;

            /* ---- Radius scale ---- */
            --r-sm: 12px;
            --r-md: 20px;
            --r-lg: 28px;
            --r-xl: 36px;
            --r-pill: 999px;

            /* ---- Shadow scale ---- */
            --shadow-sm: 0 1px 3px rgba(0, 0, 0, .05);
            --shadow-md: 0 10px 28px rgba(0, 0, 0, .07);
            --shadow-lg: 0 26px 64px rgba(0, 0, 0, .14);
            --shadow-glow: 0 10px 30px rgba(0, 113, 227, .28);
        }

        html, body, [class*="css"] {
            font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'SF Pro Text', 'Helvetica Neue', Arial, sans-serif;
            -webkit-font-smoothing: antialiased;
        }
        .stApp {
            background:
                radial-gradient(1100px 620px at 12% -8%, rgba(0, 113, 227, .07), transparent 60%),
                radial-gradient(900px 560px at 108% 4%, rgba(94, 92, 230, .05), transparent 55%),
                var(--surface-alt);
            color: var(--ink);
        }
        .block-container {
            width: min(90vw, 1440px);
            max-width: 1440px;
            padding: 1.6rem 1.5rem 6rem;
        }
        [data-testid="stSidebar"] { display: none; }
        header[data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stDecoration"] {
            display: none !important;
        }

        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after { animation-duration: .001ms !important; transition-duration: .001ms !important; }
        }
        @keyframes stayon-rise {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .stayon-rise { animation: stayon-rise .5s cubic-bezier(.2,.8,.2,1) both; }

        /* =========================================================
           GLOBAL FLOATING NAVIGATION (Glass Navigation)
           ========================================================= */
        .top-nav {
            position: relative;
            display: grid;
            grid-template-columns: 1fr auto 1fr;
            align-items: center;
            min-height: 60px;
            margin: -1.6rem calc(50% - 50vw) 3.2rem;
            padding: 0 max(3rem, calc((100vw - 1440px) / 2));
            background: var(--glass-bg-strong);
            border-bottom: 1px solid rgba(210, 210, 215, .55);
            backdrop-filter: saturate(190%) blur(var(--glass-blur));
            -webkit-backdrop-filter: saturate(190%) blur(var(--glass-blur));
            box-shadow: 0 1px 0 rgba(255, 255, 255, .6) inset, 0 12px 30px rgba(0, 0, 0, .04);
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
            font-size: 1.02rem;
            font-weight: 700;
            letter-spacing: -.01em;
            text-decoration: none !important;
        }
        .top-nav-links { display: flex; align-items: center; gap: .3rem; }
        .top-nav-link {
            color: var(--muted) !important;
            padding: .55rem .9rem;
            border-radius: var(--r-pill);
            font-size: .78rem;
            font-weight: 600;
            text-decoration: none !important;
            transition: color .18s ease, background .18s ease;
        }
        .top-nav-link:hover, .top-nav-link.active {
            color: var(--ink) !important;
            background: rgba(0, 0, 0, .045);
        }
        .top-nav-status {
            justify-self: end;
            display: flex;
            align-items: center;
            gap: .6rem;
            color: var(--faint);
            font-size: .72rem;
            font-weight: 600;
            letter-spacing: .02em;
            text-transform: uppercase;
        }
        .top-nav-status .status-dot {
            display: inline-flex;
            align-items: center;
            gap: .42rem;
        }
        .top-nav-status .status-dot::before {
            content: '';
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: #29CC5F;
            box-shadow: 0 0 0 3px rgba(41, 204, 95, .18);
        }

        /* =========================================================
           TYPOGRAPHY SYSTEM
           Display > H1 > H2 > H3 > Body > Caption
           ========================================================= */
        h1, h2, h3 { letter-spacing: -.02em; color: var(--ink); font-weight: 700; }
        h1 { font-size: clamp(1.9rem, 2.2vw + 1rem, 2.5rem) !important; letter-spacing: -.03em !important; }
        h3 {
            font-size: 1.02rem !important;
            font-weight: 600 !important;
            margin-top: 2.3rem !important;
            padding-bottom: .65rem;
            border-bottom: 1px solid var(--line-soft);
        }
        .stayon-display {
            font-size: clamp(2.6rem, 3.6vw + 1rem, 4.4rem);
            font-weight: 700;
            letter-spacing: -.045em;
            line-height: 1.04;
            color: var(--ink);
            margin: 0;
        }
        .stayon-caption {
            font-size: .74rem;
            font-weight: 700;
            letter-spacing: .1em;
            text-transform: uppercase;
            color: var(--muted);
        }
        .muted { color: var(--muted); }

        /* =========================================================
           SECTION HEADING (에디토리얼 섹션 타이틀)
           ========================================================= */
        .section-jump {
            position: sticky;
            top: 66px;
            z-index: 20;
            display: flex;
            gap: .4rem;
            overflow-x: auto;
            margin: 0 0 2.4rem;
            padding: .5rem;
            border: 1px solid var(--glass-border);
            border-radius: var(--r-pill);
            background: var(--glass-bg-strong);
            backdrop-filter: blur(var(--glass-blur));
            -webkit-backdrop-filter: blur(var(--glass-blur));
        }
        .section-jump a {
            flex: 1;
            min-width: max-content;
            padding: .65rem .9rem;
            border-radius: var(--r-pill);
            color: var(--muted) !important;
            font-size: .8rem;
            font-weight: 600;
            text-align: center;
            text-decoration: none !important;
        }
        .section-jump a:hover { color: var(--blue) !important; background: var(--blue-soft); }
        .scroll-section { scroll-margin-top: 140px; padding-top: .45rem; }
        .section-kicker {
            display: flex;
            align-items: center;
            gap: .6rem;
            margin-bottom: .7rem;
            color: var(--blue);
            font-size: .72rem;
            font-weight: 700;
            letter-spacing: .1em;
            text-transform: uppercase;
        }
        .section-kicker::before {
            content: '';
            width: 22px;
            height: 2px;
            border-radius: 2px;
            background: var(--blue);
        }
        .section-title {
            margin: 0 0 .6rem;
            font-size: clamp(1.7rem, 1.4vw + 1.2rem, 2.35rem);
            font-weight: 700;
            letter-spacing: -.03em;
            line-height: 1.12;
        }
        .section-desc { margin-bottom: 2rem; color: var(--muted); font-size: 1rem; max-width: 640px; line-height: 1.6; }
        .section-divider { height: 1px; margin: 4.5rem 0; background: linear-gradient(90deg, transparent, var(--line), transparent); }
        .section-divider-thin { height: 1px; margin: 2.6rem 0; background: var(--line-soft); }
        .section-spacer-lg { height: 1.8rem; }
        .feature-pills { display: flex; flex-wrap: wrap; gap: .5rem; margin: .9rem 0 1.3rem; }
        .feature-pill {
            padding: .4rem .85rem;
            border-radius: var(--r-pill);
            background: var(--surface-alt);
            border: 1px solid var(--line-soft);
            color: var(--muted);
            font-size: .76rem;
            font-weight: 600;
        }
        .decision-note {
            margin: 1rem 0;
            padding: 1.1rem 1.3rem;
            border-radius: var(--r-md);
            background: var(--blue-soft);
            color: var(--blue-deep);
            font-size: .9rem;
            line-height: 1.6;
        }

        /* =========================================================
           PAGE HEADER
           ========================================================= */
        .page-head { padding: 0 0 1.6rem; margin-bottom: 2.2rem; border-bottom: 1px solid var(--line-soft); }
        .page-head-eyebrow {
            display: inline-block;
            font-size: .74rem;
            font-weight: 700;
            letter-spacing: .1em;
            text-transform: uppercase;
            color: var(--blue);
            margin-bottom: .8rem;
        }
        .page-head h1 { margin: .05rem 0 .65rem; }
        .page-head .muted { font-size: 1.05rem; max-width: 660px; line-height: 1.55; }

        .home-link {
            display: inline-flex;
            align-items: center;
            gap: .3rem;
            font-weight: 600;
            font-size: .86rem;
            color: var(--blue);
            text-decoration: none !important;
            padding: .15rem 0;
            transition: gap .18s ease, color .18s ease;
        }
        .home-link:hover { color: var(--blue-deep); gap: .5rem; }
        .st-key-home-nav { margin-top: .2rem; margin-bottom: 1rem; overflow: visible; }
        .st-key-home-nav [data-testid="stMarkdownContainer"] { padding: 0; overflow: visible; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    _apply_page_style_part2()


def _apply_page_style_part2() -> None:
    st.markdown(
        """
        <style>
        /* =========================================================
           METRICS / KPI (거대 숫자 중심)
           ========================================================= */
        div[data-testid="stMetric"] { background: transparent; border: none; border-radius: 0; padding: 0; box-shadow: none; }
        div[data-testid="stMetricLabel"] { color: var(--muted); font-size: .82rem; }
        div[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; font-weight: 700; color: var(--ink); }

        [class*="st-key-stat-bar"] [data-testid="stHorizontalBlock"] {
            background: var(--glass-bg-strong);
            border: 1px solid var(--glass-border);
            border-radius: var(--r-lg);
            padding: 1.8rem .4rem;
            backdrop-filter: blur(var(--glass-blur));
            -webkit-backdrop-filter: blur(var(--glass-blur));
            box-shadow: var(--shadow-md);
        }
        [class*="st-key-stat-bar"] [data-testid="stHorizontalBlock"] > div { padding: 0 1.6rem; }
        [class*="st-key-stat-bar"] [data-testid="stHorizontalBlock"] > div:not(:last-child) { border-right: 1px solid var(--line-soft); }

        div[data-testid="stAlert"] { border-radius: var(--r-md); border: 1px solid var(--line-soft); }
        div[data-testid="stDataFrame"] {
            border-radius: var(--r-md);
            overflow: hidden;
            border: 1px solid var(--line-soft);
            box-shadow: var(--shadow-sm);
        }
        div[data-testid="stDataFrame"] [data-testid="stElementToolbar"] { opacity: .5; }
        div[data-testid="stSlider"] {
            padding: 1.05rem 1.2rem .65rem;
            border: 1px solid var(--line-soft);
            border-radius: var(--r-md);
            background: var(--surface);
            overflow: visible;
        }
        div[data-testid="stSlider"] [data-rac] { overflow: visible; }
        div[data-testid="stSlider"] [data-testid="stSliderThumbValue"] { z-index: 5; white-space: nowrap; }
        div[data-testid="stSlider"] [data-testid="stSliderTickBar"] { display: flex; justify-content: space-between; padding: 0 .1rem; margin-top: .3rem; }
        div[data-testid="stSlider"] [data-testid="stSliderTickBar"] p { white-space: nowrap; }

        /* =========================================================
           BUTTONS — pill / capsule CTA
           ========================================================= */
        .stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
            border-radius: var(--r-pill);
            border: 1px solid transparent;
            background: var(--surface-alt);
            color: var(--ink);
            min-height: 2.8rem;
            font-weight: 600;
            letter-spacing: -.01em;
            box-shadow: none;
            transition: color .18s ease, border-color .18s ease, background .18s ease, transform .12s ease;
        }
        .stButton > button:hover, .stDownloadButton > button:hover, .stFormSubmitButton > button:hover {
            color: var(--ink);
            border-color: transparent;
            background: var(--surface-sunken);
        }
        .stButton > button:active, .stFormSubmitButton > button:active { transform: scale(.98); }
        .stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {
            background: var(--blue);
            border-color: var(--blue);
            color: #FFFFFF;
            box-shadow: 0 8px 20px rgba(0, 113, 227, .25);
        }
        .stButton > button[kind="primary"]:hover, .stFormSubmitButton > button[kind="primary"]:hover {
            background: var(--blue-deep);
            border-color: var(--blue-deep);
        }

        .nav-list { display: flex; flex-direction: column; border-top: 1px solid var(--line-soft); margin-top: .3rem; }
        .nav-row {
            display: flex; align-items: center; justify-content: space-between; gap: 1.5rem;
            padding: 1.5rem .5rem; border-bottom: 1px solid var(--line-soft);
            text-decoration: none !important; color: var(--ink);
            transition: background .18s ease, padding-left .18s ease;
        }
        .nav-row:hover { background: var(--surface-alt); padding-left: 1rem; }
        .nav-row-index { font-size: .8rem; font-weight: 700; color: var(--faint); width: 1.6rem; flex-shrink: 0; }
        .nav-row-text { display: flex; flex-direction: column; gap: .3rem; flex-grow: 1; }
        .nav-row-title { font-weight: 700; font-size: 1.12rem; letter-spacing: -.01em; }
        .nav-row-desc { font-size: .9rem; color: var(--muted); }
        .nav-row-arrow { font-size: 1.2rem; color: var(--faint); transition: transform .18s ease, color .18s ease; }
        .nav-row:hover .nav-row-arrow { color: var(--blue); transform: translateX(4px); }

        @media (max-width: 560px) {
            .block-container { width: 100%; padding-left: 1rem; padding-right: 1rem; }
            .nav-row { padding: 1.15rem .2rem; }
            .top-nav { grid-template-columns: 1fr; gap: .6rem; margin-left: -1rem; margin-right: -1rem; padding: .8rem 1rem; }
            .top-nav-brand, .top-nav-status { display: none; }
            .top-nav-links { justify-content: center; overflow-x: auto; }
            .top-nav-link { padding: .6rem .75rem; white-space: nowrap; }
            .section-jump { top: 48px; }
            .stayon-display { font-size: 2.4rem; }
        }
        @media (min-width: 1920px) { .block-container { max-width: 1640px; } }
        @media (min-width: 2400px) { .block-container { max-width: 1840px; } }

        /* ---- Tone tokens (Apple 시스템 컬러) ---- */
        .tone-safe    { --tone-color: #1E8A3E; --tone-bg: #E7F8EC; }
        .tone-info    { --tone-color: #0071E3; --tone-bg: rgba(0, 113, 227, .08); }
        .tone-warning { --tone-color: #B25900; --tone-bg: #FFF2E0; }
        .tone-danger  { --tone-color: #D70015; --tone-bg: #FFEBEA; }
        .tone-neutral { --tone-color: #6E6E73; --tone-bg: #F5F5F7; }

        /* =========================================================
           STAT CARD GRID — 큰 숫자 KPI
           ========================================================= */
        .stat-card-grid {
            display: flex; flex-wrap: wrap;
            background: var(--glass-bg-strong);
            border: 1px solid var(--glass-border);
            border-radius: var(--r-lg);
            padding: 1.9rem .4rem;
            margin: .2rem 0 1.6rem;
            backdrop-filter: blur(var(--glass-blur));
            -webkit-backdrop-filter: blur(var(--glass-blur));
            box-shadow: var(--shadow-md);
        }
        .stat-card { flex: 1 1 0; min-width: 160px; padding: 0 1.6rem; }
        .stat-card:not(:last-child) { border-right: 1px solid var(--line-soft); }
        .stat-card-label { color: var(--muted); font-size: .78rem; font-weight: 600; letter-spacing: .02em; margin-bottom: .5rem; }
        .stat-card-value {
            font-size: clamp(1.9rem, 1.4vw + 1.3rem, 2.6rem);
            font-weight: 700;
            color: var(--ink);
            font-variant-numeric: tabular-nums;
            letter-spacing: -.03em;
            line-height: 1.1;
        }
        .stat-card-hint { display: inline-block; margin-top: .55rem; font-size: .78rem; font-weight: 700; color: var(--tone-color); }
        @media (max-width: 760px) {
            .stat-card-grid { flex-direction: column; }
            .stat-card { padding: .8rem 1.1rem; }
            .stat-card:not(:last-child) { border-right: none; border-bottom: 1px solid var(--line-soft); }
        }

        /* ---- Badge / pill ---- */
        .badge {
            display: inline-flex; align-items: center; padding: .36rem .8rem;
            border-radius: var(--r-pill); font-size: .78rem; font-weight: 700;
            color: var(--tone-color); background: var(--tone-bg); white-space: nowrap;
        }
        .verdict-badge {
            display: inline-block; padding: .65rem 1.15rem; border-radius: var(--r-md);
            font-weight: 700; font-size: .98rem; color: var(--tone-color); background: var(--tone-bg);
            margin-bottom: .7rem;
        }
        .role-pill {
            display: inline-flex; align-items: center; gap: .4rem; padding: .38rem .95rem;
            border-radius: var(--r-pill); background: var(--surface-alt); border: 1px solid var(--line-soft);
            font-size: .78rem; font-weight: 700; color: var(--muted);
        }

        /* ---- Alert / notification (Glass tint) ---- */
        .alert-box {
            display: flex; gap: .75rem; align-items: flex-start;
            padding: 1.1rem 1.25rem; border-radius: var(--r-md);
            background: var(--tone-bg); border: 1px solid rgba(0,0,0,.03);
            border-left: 3px solid var(--tone-color); margin: .9rem 0;
        }
        .alert-icon { font-size: 1.05rem; line-height: 1.4; }
        .alert-title { font-weight: 700; color: var(--ink); margin-bottom: .25rem; }
        .alert-body { color: var(--muted); font-size: .9rem; line-height: 1.6; }
        .alert-body b, .alert-body strong { color: var(--ink); }

        /* ---- Empty / error state panel ---- */
        .empty-state {
            display: flex; flex-direction: column; align-items: center; text-align: center; gap: .6rem;
            padding: 3.2rem 2rem; border-radius: var(--r-lg);
            background: var(--surface-alt); border: 1px dashed var(--line);
            color: var(--muted);
        }
        .empty-state-icon { font-size: 1.8rem; opacity: .55; }
        .empty-state-title { font-weight: 700; color: var(--ink); font-size: 1.05rem; }
        .empty-state-desc { font-size: .88rem; max-width: 420px; line-height: 1.55; }

        /* ---- Table (에디토리얼 데이터 리스트) ---- */
        .table-wrap { width: 100%; max-width: 100%; overflow-x: auto; border: 1px solid var(--line-soft); border-radius: var(--r-md); margin: .7rem 0 1.2rem; }
        table.pretty-table { width: 100%; border-collapse: collapse; font-size: .89rem; }
        table.pretty-table thead th {
            position: sticky; top: 0; text-align: left; padding: .95rem 1.1rem;
            background: var(--surface-alt); color: var(--muted); font-weight: 700;
            font-size: .7rem; text-transform: uppercase; letter-spacing: .05em;
            border-bottom: 1px solid var(--line-soft); white-space: nowrap;
        }
        table.pretty-table tbody td { padding: .85rem 1.1rem; border-bottom: 1px solid var(--line-soft); color: var(--ink); white-space: nowrap; }
        table.pretty-table tbody tr:last-child td { border-bottom: none; }
        table.pretty-table tbody tr:hover { background: var(--blue-soft); }
        table.pretty-table tbody tr:nth-child(even) { background: rgba(245, 245, 247, .55); }
        table.pretty-table tbody tr:nth-child(even):hover { background: var(--blue-soft); }

        .bar-cell { display: flex; align-items: center; gap: .6rem; min-width: 130px; }
        .bar-cell-track { flex: 1; height: 6px; border-radius: var(--r-pill); background: var(--surface-sunken); overflow: hidden; }
        .bar-cell-fill { height: 100%; border-radius: var(--r-pill); background: var(--blue); }
        .bar-cell-text { font-variant-numeric: tabular-nums; font-size: .82rem; color: var(--muted); min-width: 46px; text-align: right; }

        /* ---- Editorial horizontal chart (numbered ranking option) ---- */
        .hbar-chart { display: flex; flex-direction: column; gap: .7rem; margin: .5rem 0 1.1rem; }
        .hbar-row { display: grid; grid-template-columns: 130px 1fr 64px; align-items: center; gap: .9rem; }
        .hbar-row.numbered { grid-template-columns: 30px 150px 1fr 76px; }
        .hbar-rank { font-size: .82rem; font-weight: 700; color: var(--faint); font-variant-numeric: tabular-nums; }
        .hbar-label { font-size: .86rem; color: var(--muted); font-weight: 500; }
        .hbar-row.numbered .hbar-label { color: var(--ink); font-weight: 600; }
        .hbar-track { height: 8px; border-radius: var(--r-pill); background: var(--surface-sunken); overflow: hidden; }
        .hbar-fill { height: 100%; border-radius: var(--r-pill); transition: width .4s cubic-bezier(.2,.8,.2,1); }
        .hbar-value { font-size: .86rem; font-weight: 700; color: var(--ink); text-align: right; font-variant-numeric: tabular-nums; }
        .hbar-row.numbered .hbar-value { font-size: .98rem; font-weight: 700; }
        @media (max-width: 560px) {
            .hbar-row { grid-template-columns: 92px 1fr 52px; }
            .hbar-row.numbered { grid-template-columns: 22px 100px 1fr 60px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    _apply_page_style_part3()


def _apply_page_style_part3() -> None:
    st.markdown(
        """
        <style>
        /* =========================================================
           LANDING HERO & ROLE SELECTION (Apple product page)
           ========================================================= */
        .hero-wrap { text-align: center !important; padding: 5.5rem 0 3.6rem; width: 100%; }
        .hero-eyebrow {
            display: inline-block; padding: .5rem 1.1rem; border-radius: var(--r-pill);
            background: var(--glass-bg-strong); border: 1px solid var(--glass-border);
            backdrop-filter: blur(var(--glass-blur)); -webkit-backdrop-filter: blur(var(--glass-blur));
            color: var(--blue); font-size: .76rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase;
            margin-bottom: 1.6rem;
        }
        .hero-title {
            font-size: clamp(2.6rem, 4vw + 1rem, 4.6rem);
            font-weight: 700; letter-spacing: -.045em; line-height: 1.05;
            margin: 0 0 1.2rem;
            background: linear-gradient(180deg, #1D1D1F 0%, #3A3A3D 100%);
            -webkit-background-clip: text; background-clip: text; color: transparent;
        }
        .hero-desc { color: var(--muted); font-size: 1.18rem; max-width: 660px; margin: 0 auto !important; line-height: 1.6; text-align: center !important; width: 100%; }

        .role-card {
            position: relative;
            display: flex; flex-direction: column; height: 100%; min-height: 372px;
            padding: 2.6rem 2.2rem 1.9rem;
            border: 1px solid var(--glass-border);
            border-radius: var(--r-xl);
            background: var(--glass-bg-strong);
            backdrop-filter: blur(var(--glass-blur)); -webkit-backdrop-filter: blur(var(--glass-blur));
            box-shadow: var(--shadow-md);
            overflow: hidden;
            transition: box-shadow .3s cubic-bezier(.2,.8,.2,1), transform .3s cubic-bezier(.2,.8,.2,1), border-color .3s ease;
        }
        .role-card::before {
            content: '';
            position: absolute; inset: -40% -20% auto auto; width: 60%; height: 140%;
            background: radial-gradient(circle, rgba(0, 113, 227, .12), transparent 70%);
            opacity: 0; transition: opacity .35s ease;
            pointer-events: none;
        }
        /* 역할 카드는 전체 영역이 하나의 버튼처럼 동작한다. 실제 Streamlit 버튼은
           투명 오버레이로 남겨 페이지 전환과 키보드 접근성을 그대로 유지한다. */
        div[class*="st-key-role-panel-"] {
            position: relative;
            cursor: pointer;
            border-radius: var(--r-xl);
            gap: 0 !important;
        }
        div[class*="st-key-role-panel-"] .role-card { margin-bottom: 0; }
        div[class*="st-key-role-panel-"]:hover .role-card {
            transform: translateY(-6px) scale(1.008);
            box-shadow: var(--shadow-lg);
            border-color: rgba(0, 113, 227, .35);
        }
        div[class*="st-key-role-panel-"]:active .role-card {
            transform: translateY(-2px) scale(.998);
            transition-duration: .1s;
        }
        div[class*="st-key-role-panel-"]:focus-within .role-card {
            border-color: rgba(0, 113, 227, .5);
            box-shadow: 0 0 0 4px rgba(0, 113, 227, .1), var(--shadow-lg);
        }
        div[class*="st-key-role-panel-"]:hover .role-card::before { opacity: 1; }
        div[class*="st-key-role-panel-"] div[class*="st-key-role_"][class*="_btn"] {
            position: absolute;
            inset: 0 0 -1rem;
            z-index: 5;
            margin: 0;
        }
        div[class*="st-key-role-panel-"] div[class*="st-key-role_"][class*="_btn"] .stButton,
        div[class*="st-key-role-panel-"] div[class*="st-key-role_"][class*="_btn"] .stButton > button {
            position: absolute;
            inset: 0;
            width: 100%;
            height: 100%;
            min-height: 100%;
            margin: 0;
        }
        div[class*="st-key-role-panel-"] div[class*="st-key-role_"][class*="_btn"] .stButton > button {
            border: 0 !important;
            border-radius: var(--r-xl) !important;
            background: transparent !important;
            box-shadow: none !important;
            color: transparent !important;
            cursor: pointer;
            opacity: 0;
        }
        .role-card-title { font-size: 1.5rem; font-weight: 700; letter-spacing: -.02em; margin-bottom: .4rem; }
        .role-card-subtitle { font-size: .82rem; font-weight: 700; letter-spacing: .05em; text-transform: uppercase; color: var(--blue); margin-bottom: 1rem; }
        .role-card-desc { color: var(--muted); font-size: .96rem; margin-bottom: 1.3rem; line-height: 1.6; }
        .role-card-points { list-style: none; margin: auto 0 .3rem; padding: 0; display: flex; flex-direction: column; gap: .6rem; }
        .role-card-points li { font-size: .89rem; color: var(--muted); padding-left: 1.15rem; position: relative; }
        .role-card-points li::before { content: '—'; position: absolute; left: 0; color: var(--blue); font-weight: 700; }

        /* =========================================================
           SEGMENTED / FLOATING NAV (워크스페이스 탭)
           ========================================================= */
        div[class*="st-key-tabbar-"] {
            padding: .45rem;
            border-radius: var(--r-pill);
            background: var(--glass-bg-strong);
            border: 1px solid var(--glass-border);
            backdrop-filter: blur(var(--glass-blur)); -webkit-backdrop-filter: blur(var(--glass-blur));
            box-shadow: var(--shadow-sm);
        }
        div[class*="st-key-tabbar-"] div[data-testid="stHorizontalBlock"] { gap: .35rem; }
        div[class*="st-key-tabbar-"] .stButton > button {
            border-radius: var(--r-pill);
            min-height: 2.9rem;
        }
        div[class*="st-key-tabbar-"] .stButton > button[kind="secondary"] { background: transparent; }
        div[class*="st-key-tabbar-"] .stButton > button[kind="secondary"]:hover { background: rgba(0,0,0,.045); }
        div[class*="st-key-subtabbar-"] { padding: .35rem; border-radius: var(--r-pill); background: var(--surface-alt); }
        div[class*="st-key-subtabbar-"] .stButton > button {
            border-radius: var(--r-pill); min-height: 2.4rem; font-size: .86rem;
        }
        div[class*="st-key-subtabbar-"] .stButton > button[kind="secondary"] { background: transparent; }
        div[class*="st-key-subtabbar-"] .stButton > button[kind="secondary"]:hover { background: rgba(0,0,0,.05); }

        /* Employee search / filter bar → glass */
        .st-key-employee-picker {
            padding: 1.1rem 1.2rem;
            border-radius: var(--r-lg);
            background: var(--glass-bg-strong);
            border: 1px solid var(--glass-border);
            backdrop-filter: blur(var(--glass-blur)); -webkit-backdrop-filter: blur(var(--glass-blur));
            box-shadow: var(--shadow-sm);
            margin-bottom: .4rem;
        }

        /* =========================================================
           FLOATING ACTION BUTTON
           ========================================================= */
        div[class*="st-key-"][class*="-fab"] { position: fixed; right: 1.9rem; bottom: 1.9rem; z-index: 998; width: auto; }
        div[class*="st-key-"][class*="-fab"] .stButton > button {
            border-radius: var(--r-pill);
            padding: 1rem 1.6rem;
            background: var(--blue); border-color: var(--blue); color: #FFFFFF;
            box-shadow: var(--shadow-glow), 0 2px 0 rgba(255,255,255,.25) inset;
            font-size: .92rem; font-weight: 700;
            backdrop-filter: blur(6px);
        }
        div[class*="st-key-"][class*="-fab"] .stButton > button:hover { background: var(--blue-deep); border-color: var(--blue-deep); transform: translateY(-2px); }

        /* =========================================================
           GLASS MODAL (st.dialog)
           ========================================================= */
        div[data-testid="stDialog"] div[role="dialog"] {
            border-radius: var(--r-xl) !important;
            border: 1px solid var(--glass-border) !important;
            background: var(--glass-bg-strong) !important;
            backdrop-filter: blur(34px) saturate(180%) !important;
            -webkit-backdrop-filter: blur(34px) saturate(180%) !important;
            box-shadow: var(--shadow-lg) !important;
        }
        div[data-testid="stDialog"] [data-testid="stVerticalBlock"] { gap: .6rem; }

        /* ---- Native input polish ---- */
        div[data-testid="stSelectbox"] > div > div,
        div[data-testid="stMultiSelect"] > div > div,
        div[data-baseweb="select"] > div,
        div[data-testid="stTextInput"] > div > div,
        div[data-testid="stTextInput"] input,
        div[data-testid="stNumberInput"] > div,
        div[data-testid="stNumberInput"] input {
            border-radius: var(--r-sm) !important; border-color: var(--line) !important; background: var(--surface) !important;
        }
        div[data-testid="stTextInput"] input, div[data-testid="stNumberInput"] input { color: var(--ink) !important; }
        div[data-baseweb="tag"] { border-radius: 8px !important; background: var(--blue-soft) !important; }
        div[data-baseweb="tag"] span { color: var(--blue) !important; }

        /* =========================================================
           NEW HERO COMPONENTS
           ========================================================= */
        .glass-panel {
            border-radius: var(--r-xl);
            background: var(--glass-bg-strong);
            border: 1px solid var(--glass-border);
            backdrop-filter: blur(var(--glass-blur)); -webkit-backdrop-filter: blur(var(--glass-blur));
            box-shadow: var(--shadow-md);
            padding: 2.4rem 2.6rem;
            margin: .3rem 0 1.8rem;
        }

        .narrative-banner { text-align: center; padding: 3.4rem 2rem; }
        .narrative-banner .stayon-caption { display: block; margin-bottom: 1.1rem; }
        .narrative-banner .giant-value { font-size: clamp(3.4rem, 4vw + 1.4rem, 5.6rem); font-weight: 700; letter-spacing: -.04em; line-height: 1; color: var(--ink); }
        .narrative-banner .narrative-status { display: inline-block; margin: 1rem 0 1.3rem; }
        .narrative-banner .narrative-message { max-width: 560px; margin: 0 auto; color: var(--muted); font-size: 1.05rem; line-height: 1.6; }

        .giant-stat-value { font-size: clamp(2.4rem, 2.6vw + 1.2rem, 3.6rem); font-weight: 700; letter-spacing: -.035em; color: var(--ink); line-height: 1.05; font-variant-numeric: tabular-nums; }
        .giant-stat-unit { font-size: 1.1rem; font-weight: 600; color: var(--muted); margin-left: .25rem; }
        .giant-stat-label { margin-top: .5rem; font-size: .82rem; font-weight: 600; color: var(--muted); }

        /* ---- Employee hero split panel ---- */
        .employee-hero { padding: 2.6rem 2.8rem; }
        .employee-hero-grid { display: grid; grid-template-columns: 1.1fr 1fr; gap: 2.4rem; align-items: center; }
        @media (max-width: 900px) { .employee-hero-grid { grid-template-columns: 1fr; } }
        .employee-hero-id { font-size: clamp(1.7rem, 1.4vw + 1.1rem, 2.2rem); font-weight: 700; letter-spacing: -.02em; margin: .5rem 0 .6rem; }
        .employee-hero-meta { display: flex; gap: .5rem; flex-wrap: wrap; }
        .employee-hero-risk { text-align: right; }
        @media (max-width: 900px) { .employee-hero-risk { text-align: left; } }
        .employee-hero-risk-value { font-size: clamp(3rem, 3vw + 1.4rem, 4.4rem); font-weight: 700; letter-spacing: -.04em; line-height: 1; font-variant-numeric: tabular-nums; }
        .employee-hero-divider { height: 1px; background: var(--line-soft); margin: 2rem 0 1.6rem; }
        .talent-value-row { display: flex; align-items: baseline; justify-content: space-between; margin-bottom: .7rem; }
        .talent-value-row .stayon-caption { margin: 0; }
        .talent-value-score { font-size: 1.6rem; font-weight: 700; color: var(--ink); font-variant-numeric: tabular-nums; }
        .talent-value-score .of100 { font-size: .95rem; font-weight: 600; color: var(--muted); }
        .talent-value-track { height: 10px; border-radius: var(--r-pill); background: var(--surface-sunken); overflow: hidden; }
        .talent-value-fill { height: 100%; border-radius: var(--r-pill); background: linear-gradient(90deg, var(--blue), #5E5CE6); }

        /* ---- Ranking list ---- */
        .ranking-list { display: flex; flex-direction: column; gap: .7rem; margin: .6rem 0 1.2rem; }
        .ranking-row {
            display: grid; grid-template-columns: 52px 1.4fr repeat(var(--metric-count, 3), minmax(90px, .8fr));
            align-items: center; gap: 1.2rem;
            padding: 1.1rem 1.4rem; border-radius: var(--r-lg);
            background: var(--surface); border: 1px solid var(--line-soft);
            transition: box-shadow .2s ease, transform .2s ease, border-color .2s ease;
        }
        .ranking-row:hover { box-shadow: var(--shadow-md); transform: translateY(-2px); border-color: rgba(0,113,227,.25); }
        .ranking-row.highlight { background: linear-gradient(180deg, var(--blue-soft), var(--surface)); border-color: rgba(0,113,227,.3); }
        .ranking-rank { font-size: 1.15rem; font-weight: 700; color: var(--faint); font-variant-numeric: tabular-nums; }
        .ranking-row.highlight .ranking-rank { color: var(--blue); }
        .ranking-title { font-weight: 700; font-size: .98rem; color: var(--ink); margin-bottom: .25rem; }
        .ranking-subtitle { font-size: .8rem; color: var(--muted); }
        .ranking-metric { text-align: center; }
        .ranking-metric-label { font-size: .68rem; font-weight: 700; letter-spacing: .05em; text-transform: uppercase; color: var(--faint); margin-bottom: .35rem; }
        .ranking-metric-value { font-size: 1rem; font-weight: 700; color: var(--ink); font-variant-numeric: tabular-nums; }
        .ranking-metric-bar-track { height: 5px; border-radius: var(--r-pill); background: var(--surface-sunken); margin-top: .4rem; overflow: hidden; }
        .ranking-metric-bar-fill { height: 100%; border-radius: var(--r-pill); background: var(--blue); }
        .ranking-ring-wrap { display: flex; flex-direction: column; align-items: center; gap: .3rem; }
        @media (max-width: 900px) {
            .ranking-row { grid-template-columns: 40px 1fr; row-gap: .8rem; }
            .ranking-metric { text-align: left; grid-column: span 1; }
        }

        /* ---- Area / trend chart (Apple Health 느낌) ---- */
        .area-chart-wrap { margin: .8rem 0 1.2rem; }
        .area-chart-labels { display: flex; justify-content: space-between; margin-top: .5rem; }
        .area-chart-label { font-size: .76rem; color: var(--muted); text-align: center; flex: 1; }
        .area-chart-label .val { display: block; font-weight: 700; color: var(--ink); font-size: .82rem; margin-top: .15rem; }

        /* ---- Model intelligence cards ---- */
        .model-card-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; margin: .6rem 0 1.4rem; }
        .model-card {
            position: relative; padding: 1.6rem 1.5rem; border-radius: var(--r-lg);
            background: var(--surface); border: 1px solid var(--line-soft);
            transition: box-shadow .2s ease, transform .2s ease, border-color .2s ease;
        }
        .model-card:hover { box-shadow: var(--shadow-md); transform: translateY(-3px); }
        .model-card.best { background: linear-gradient(165deg, var(--blue-soft), var(--surface) 60%); border-color: rgba(0,113,227,.35); box-shadow: var(--shadow-md); }
        .model-card-name { font-size: .78rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); margin-bottom: 1rem; }
        .model-card.best .model-card-name { color: var(--blue-deep); }
        .model-card-metrics { display: flex; flex-direction: column; gap: .7rem; }
        .model-card-metric-row { display: flex; align-items: baseline; justify-content: space-between; }
        .model-card-metric-label { font-size: .78rem; color: var(--muted); }
        .model-card-metric-value { font-size: 1.15rem; font-weight: 700; color: var(--ink); font-variant-numeric: tabular-nums; }
        .model-card-best-badge {
            position: absolute; top: -.6rem; right: 1.2rem;
            padding: .3rem .7rem; border-radius: var(--r-pill);
            background: var(--blue); color: #fff; font-size: .66rem; font-weight: 800; letter-spacing: .06em;
            box-shadow: var(--shadow-sm);
        }

        /* ---- Versus hero (ML vs DL) ---- */
        .versus-hero { display: grid; grid-template-columns: 1fr auto 1fr; align-items: center; gap: 1.6rem; padding: 2.4rem 1.5rem; text-align: center; }
        @media (max-width: 700px) { .versus-hero { grid-template-columns: 1fr; } }
        .versus-side .giant-stat-value { font-size: clamp(2.2rem, 2.4vw + 1rem, 3.1rem); }
        .versus-divider { font-size: .82rem; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; color: var(--faint); }
        .versus-side.winner .giant-stat-value { color: var(--blue); }

        /* ---- Department cards ---- */
        .dept-card-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; margin: .6rem 0 1.2rem; }
        .dept-card { padding: 1.5rem 1.5rem 1.6rem; border-radius: var(--r-lg); background: var(--surface); border: 1px solid var(--line-soft); transition: box-shadow .2s ease, transform .2s ease; }
        .dept-card:hover { box-shadow: var(--shadow-md); transform: translateY(-3px); }
        .dept-card-title { font-weight: 700; font-size: 1rem; margin-bottom: 1rem; }
        .dept-card-row { display: flex; align-items: baseline; justify-content: space-between; padding: .4rem 0; border-top: 1px solid var(--line-soft); }
        .dept-card-row:first-of-type { border-top: none; }
        .dept-card-row-label { font-size: .8rem; color: var(--muted); }
        .dept-card-row-value { font-size: .92rem; font-weight: 700; color: var(--ink); font-variant-numeric: tabular-nums; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    _apply_ios_reference_overrides()


def _apply_ios_reference_overrides() -> None:
    """Figma iOS 레퍼런스의 밀도·색·고정 내비게이션을 최종 스타일로 덮어쓴다."""

    st.markdown(
        """
        <style>
        :root {
            --ink: #111827;
            --muted: #64748B;
            --faint: #94A3B8;
            --surface: #FFFFFF;
            --surface-alt: #F8FAFC;
            --surface-sunken: #F1F5F9;
            --line: #DCE2EA;
            --line-soft: #EDF1F5;
            --blue: #2563EB;
            --blue-deep: #1D4ED8;
            --blue-soft: rgba(37, 99, 235, .08);
            --glass-bg: rgba(255, 255, 255, .60);
            --glass-bg-strong: rgba(255, 255, 255, .86);
            --glass-border: rgba(255, 255, 255, .82);
            --glass-blur: 32px;
            --r-sm: 12px;
            --r-md: 16px;
            --r-lg: 20px;
            --r-xl: 24px;
            --shadow-sm: 0 1px 3px rgba(15, 23, 42, .07);
            --shadow-md: 0 8px 24px rgba(15, 23, 42, .07);
            --shadow-lg: 0 20px 52px rgba(15, 23, 42, .13);
            --shadow-glow: 0 8px 24px rgba(37, 99, 235, .28);
        }

        html, body, [class*="css"] {
            font-family: Inter, -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", Arial, sans-serif;
        }
        .stApp { background: #F8FAFC; }
        body:has(.hero-wrap) .stApp {
            background:
                radial-gradient(ellipse at 18% 22%, rgba(147, 197, 253, .28), transparent 52%),
                radial-gradient(ellipse at 80% 75%, rgba(196, 181, 253, .18), transparent 50%),
                linear-gradient(155deg, #EEF3FC 0%, #E7EDF9 38%, #F3F7FD 100%);
        }
        .block-container {
            width: min(calc(100vw - 32px), 1024px);
            max-width: 1024px;
            padding: 1.25rem 0 8rem;
        }

        /* iOS compact workspace header */
        .st-key-workspace-navigation {
            position: sticky;
            top: 0;
            z-index: 999;
            margin: -1.25rem calc(50% - 50vw) 1.65rem;
            padding: .48rem max(1rem, calc((100vw - 1024px) / 2));
            background: rgba(255, 255, 255, .82);
            border-bottom: 1px solid #EEF1F5;
            backdrop-filter: blur(24px) saturate(180%);
            -webkit-backdrop-filter: blur(24px) saturate(180%);
        }
        .st-key-workspace-navigation [data-testid="stHorizontalBlock"] {
            align-items: center;
            gap: .35rem;
        }
        .workspace-brand {
            font-size: .82rem;
            font-weight: 800;
            letter-spacing: .12em;
            color: var(--ink);
        }
        .workspace-status {
            display: flex;
            justify-content: flex-end;
            align-items: center;
            gap: .45rem;
            color: var(--faint);
            font-size: .68rem;
            font-weight: 500;
            white-space: nowrap;
        }
        .workspace-status::before {
            content: "";
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: #22C55E;
            box-shadow: 0 0 7px rgba(34, 197, 94, .65);
        }
        .st-key-workspace-navigation .stButton > button {
            min-height: 2rem;
            padding: .3rem .72rem;
            border: 0;
            border-radius: 999px;
            box-shadow: none;
            font-size: .7rem;
            letter-spacing: .03em;
        }
        .st-key-workspace-navigation .stButton > button[kind="secondary"] {
            background: rgba(120, 120, 128, .10);
            color: #6B7280;
        }
        .st-key-workspace-navigation .stButton > button[kind="primary"] {
            background: #FFFFFF;
            color: var(--blue);
            box-shadow: 0 1px 6px rgba(0, 0, 0, .13);
        }
        .st-key-workspace-navigation div[class*="st-key-workspace-exit"] button {
            background: transparent !important;
            color: #94A3B8 !important;
            white-space: nowrap;
        }

        /* Page and section hierarchy */
        .page-head { margin: .35rem 0 1.2rem; }
        .page-head-eyebrow, .section-kicker {
            color: var(--blue);
            font-size: .7rem;
            font-weight: 800;
            letter-spacing: .12em;
        }
        .page-head h1 {
            margin: .25rem 0 .25rem !important;
            font-size: clamp(1.9rem, 3vw, 2.35rem) !important;
            line-height: 1.08;
        }
        .page-head .muted, .section-desc { color: #7C8799; font-size: .82rem; }
        .section-kicker { margin-bottom: .35rem; }
        .section-kicker::before { width: 22px; height: 1px; background: var(--blue); }
        .section-title { font-size: clamp(1.45rem, 2.4vw, 1.8rem); margin-bottom: .3rem; }

        /* KPI strip */
        .stat-card-grid {
            margin: 0 0 1.35rem;
            border: 1px solid var(--line-soft);
            border-radius: 16px;
            background: #FFFFFF;
            box-shadow: var(--shadow-sm);
        }
        .stat-card { min-height: 80px; padding: .9rem 1.25rem; }
        .stat-card-label { color: #94A3B8; font-size: .68rem; font-weight: 500; }
        .stat-card-value { color: var(--ink); font-size: clamp(1.45rem, 2.4vw, 1.85rem); }

        /* Fixed iOS liquid-glass tab bar */
        div[class*="st-key-tabbar-"] {
            position: fixed;
            left: 50%;
            bottom: 20px;
            z-index: 2147483000 !important;
            isolation: isolate;
            width: min(calc(100vw - 32px), 768px);
            transform: translateX(-50%);
            padding: 6px;
            border: .5px solid rgba(255, 255, 255, .72);
            border-radius: 22px;
            background: rgba(255, 255, 255, .76);
            backdrop-filter: blur(36px) saturate(190%);
            -webkit-backdrop-filter: blur(36px) saturate(190%);
            box-shadow: 0 8px 32px rgba(15, 23, 42, .12), inset 0 1px 0 #FFFFFF;
        }
        div[class*="st-key-tabbar-"] [data-testid="stHorizontalBlock"] {
            gap: 0;
            flex-wrap: nowrap !important;
            position: relative;
            z-index: 2;
        }
        div[class*="st-key-tabbar-"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
            flex: 1 1 0 !important;
            width: auto !important;
            min-width: 0 !important;
        }
        div[class*="st-key-tabbar-"] .stButton > button {
            min-height: 3.65rem;
            border: 0;
            border-radius: 16px;
            font-size: .72rem;
            font-weight: 650;
            line-height: 1.15;
            white-space: nowrap;
        }
        div[class*="st-key-tabbar-"] .stButton > button[kind="primary"] {
            color: var(--blue);
            background: transparent;
            box-shadow: none;
        }
        div[class*="st-key-tabbar-"] .stButton > button[kind="secondary"] {
            color: #8E8E93;
            background: transparent;
        }
        div[class*="st-key-tabbar-"]::after {
            content:"";
            position:absolute;
            top:6px;
            bottom:6px;
            left:6px;
            z-index:1;
            width:calc((100% - 12px) / var(--tab-count, 4));
            border-radius:16px;
            background:rgba(255,255,255,.97);
            box-shadow:0 2px 13px rgba(15,23,42,.13), inset 0 1px 0 #FFF;
            transform:translateX(calc(var(--tab-index, 0) * 100%));
            transition:transform .42s cubic-bezier(.34,1.32,.64,1), width .3s ease;
            will-change:transform;
            pointer-events:none;
        }
        div[class*="st-key-tabbar-"]:has(.workspace-tab-state.tabs-4) { --tab-count:4; }
        div[class*="st-key-tabbar-"]:has(.workspace-tab-state.tabs-5) { --tab-count:5; }
        div[class*="st-key-tabbar-"]:has(.workspace-tab-state.is-salary) { --tab-index:0; }
        div[class*="st-key-tabbar-"]:has(.workspace-tab-state.is-team) { --tab-index:1; }
        div[class*="st-key-tabbar-"]:has(.workspace-tab-state.is-actions) { --tab-index:2; }
        div[class*="st-key-tabbar-"]:has(.workspace-tab-state.is-stability) { --tab-index:3; }
        div[class*="st-key-tabbar-"]:has(.workspace-tab-state.is-models) { --tab-index:4; }
        div[class*="st-key-tabbar-"] .stButton > button {
            transition:color .25s ease, transform .18s ease !important;
        }
        div[class*="st-key-tabbar-"] .stButton > button:active { transform:scale(.96); }
        div[data-baseweb="popover"], div[data-baseweb="menu"] {
            z-index: 2147482000 !important;
        }

        /* Compact iOS cards and controls */
        .glass-panel, .role-card, .dept-card, .model-card, .table-wrap,
        div[data-testid="stPlotlyChart"] {
            border-color: var(--line-soft);
            background: #FFFFFF;
            box-shadow: var(--shadow-sm);
        }
        .glass-panel { border-radius: 18px; padding: 1.35rem 1.5rem; margin-bottom: 1rem; }
        .model-card-grid { grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: .7rem; }
        .model-card { border-radius: 16px; padding: 1rem; }
        .model-card-name { margin-bottom: .7rem; font-size: .68rem; }
        .model-card-metrics { gap: .38rem; }
        .model-card-metric-label { font-size: .7rem; }
        .model-card-metric-value { font-size: .82rem; }
        .table-wrap { border-radius: 16px; }
        table.pretty-table { font-size: .78rem; }
        table.pretty-table thead th { padding: .72rem .8rem; font-size: .64rem; }
        table.pretty-table tbody td { padding: .68rem .8rem; }
        div[data-testid="stSelectbox"] > div > div,
        div[data-testid="stMultiSelect"] > div > div,
        div[data-baseweb="select"] > div,
        div[data-testid="stTextInput"] input,
        div[data-testid="stNumberInput"] > div {
            border-color: #DCE2EA !important;
            border-radius: 12px !important;
            background: #FFFFFF !important;
        }
        .st-key-employee-picker {
            padding: 0;
            border: 0;
            border-radius: 0;
            background: transparent;
            box-shadow: none;
            backdrop-filter: none;
        }
        div[class*="st-key-subtabbar-"] {
            padding: 3px;
            border-radius: 13px;
            background: #F1F3F6;
        }
        div[class*="st-key-subtabbar-"] .stButton > button {
            min-height: 2.55rem;
            border: 0;
            border-radius: 10px;
        }
        div[class*="st-key-subtabbar-"] .stButton > button[kind="primary"] {
            background: var(--blue);
            color: #FFFFFF;
            box-shadow: 0 2px 5px rgba(37, 99, 235, .22);
        }

        /* Landing */
        .hero-wrap { padding: 4.5rem 0 2.7rem; }
        .hero-eyebrow {
            margin-bottom: 1.5rem;
            color: #6B7280;
            background: rgba(255, 255, 255, .58);
            border-color: rgba(255, 255, 255, .8);
            font-size: .67rem;
        }
        .hero-title { font-size: clamp(2.4rem, 5vw, 3.35rem); color: var(--ink); }
        .hero-desc { max-width: 600px; color: #6B7280; font-size: .92rem; }
        .role-card {
            height: auto;
            min-height: 390px;
            padding: 2rem 2rem 1.5rem;
            border-radius: 24px;
            background: rgba(255, 255, 255, .58);
            box-shadow: 0 20px 60px rgba(15, 23, 42, .07), inset 0 1px 0 #FFFFFF;
        }
        .role-card-title { font-size: 1.45rem; }
        .role-card-desc, .role-card-points li { font-size: .82rem; }

        /* Screenshot-faithful compact workspace shell */
        .stApp:has(.st-key-workspace-navigation) .block-container {
            width: min(calc(100vw - 26px), 1024px);
            padding-top: 0;
            padding-bottom: 7.5rem;
        }
        .stApp:has(.st-key-workspace-navigation) .block-container > [data-testid="stVerticalBlock"] {
            gap: 1rem !important;
        }
        .stApp:has(.st-key-workspace-navigation) .block-container > [data-testid="stVerticalBlock"]
        > [data-testid="stElementContainer"]:has(style) {
            display: none !important;
        }
        .st-key-workspace-navigation {
            min-height: 42px;
            left: auto;
            width: 100vw !important;
            max-width: none !important;
            margin: 0 0 .75rem calc(50% - 50vw);
            padding: 4px max(13px, calc((100vw - 1024px) / 2));
            box-sizing: border-box;
            transform: none;
        }
        .st-key-workspace-navigation > div,
        .st-key-workspace-navigation [data-testid="stVerticalBlock"] { gap: 0 !important; }
        .st-key-workspace-navigation [data-testid="stHorizontalBlock"] { min-height: 32px; }
        .st-key-workspace-navigation [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
            min-width: 0 !important;
            width: auto !important;
        }
        .st-key-workspace-navigation [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child {
            flex: 1 1 auto !important;
        }
        .st-key-workspace-navigation [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(2),
        .st-key-workspace-navigation [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(3) {
            flex: 0 0 42px !important;
        }
        .workspace-brand {
            display:inline-flex; align-items:center; min-height:1.7rem;
            color:#111827 !important; font-size:.7rem; font-weight:800;
            letter-spacing:.13em; text-decoration:none !important;
        }
        .workspace-brand:hover { color:#2563EB !important; }
        .st-key-workspace-navigation div[class*="st-key-workspace_role_hr"],
        .st-key-workspace-navigation div[class*="st-key-workspace_role_admin"] {
            position: absolute !important;
            top: 7px;
            z-index: 3;
            width: 42px !important;
            min-width: 42px !important;
        }
        .st-key-workspace-navigation div[class*="st-key-workspace_role_hr"] {
            right: calc(max(13px, calc((100vw - 1024px) / 2)) + 47px);
        }
        .st-key-workspace-navigation div[class*="st-key-workspace_role_admin"] {
            right: max(13px, calc((100vw - 1024px) / 2));
        }
        .st-key-workspace-navigation div[class*="st-key-workspace_role_hr"] .stButton,
        .st-key-workspace-navigation div[class*="st-key-workspace_role_admin"] .stButton { width:42px !important; }
        .workspace-role-state, .workspace-tab-state { display:none !important; }
        div[class*="st-key-tabbar-"] [data-testid="stElementContainer"]:has(.workspace-tab-state) {
            display:none !important;
        }
        .st-key-workspace-navigation::before,
        .st-key-workspace-navigation::after {
            content:"";
            position:absolute;
            pointer-events:none;
            border-radius:999px;
        }
        .st-key-workspace-navigation::before {
            top:5px;
            right:max(13px, calc((100vw - 1024px) / 2));
            z-index:1;
            width:89px;
            height:31px;
            border:.5px solid rgba(0,0,0,.06);
            background:rgba(120,120,128,.12);
        }
        .st-key-workspace-navigation::after {
            top:7px;
            right:calc(max(13px, calc((100vw - 1024px) / 2)) + 47px);
            z-index:2;
            width:42px;
            height:27px;
            background:rgba(255,255,255,.98);
            box-shadow:0 1px 7px rgba(15,23,42,.14), inset 0 .5px 0 #FFF;
            transform:translateX(0);
            transition:transform .38s cubic-bezier(.34,1.35,.64,1), box-shadow .25s ease;
            will-change:transform;
        }
        .st-key-workspace-navigation:has(.workspace-role-state.is-admin)::after {
            transform:translateX(47px);
        }
        .workspace-status { font-size: .58rem; gap: .35rem; }
        .workspace-status::before { width: 6px; height: 6px; }
        .st-key-workspace-navigation .stButton > button {
            min-height: 1.7rem;
            height: 1.7rem;
            padding: .15rem .55rem;
            font-size: .58rem;
            min-width: 0;
            white-space: nowrap;
            background:transparent !important;
            border-color:transparent !important;
            box-shadow:none !important;
            transition:color .25s ease, transform .18s ease !important;
        }
        .st-key-workspace-navigation .stButton > button[kind="primary"] { color:#2563EB !important; }
        .st-key-workspace-navigation .stButton > button[kind="secondary"] { color:#7C8493 !important; }
        .st-key-workspace-navigation .stButton > button:active { transform:scale(.93); }
        .st-key-workspace-page-header { margin: 1.35rem 0 1.75rem; }
        .workspace-page-title {
            margin: 0 0 .7rem !important;
            color: #182033;
            font-size: 1.45rem !important;
            font-weight: 750 !important;
            letter-spacing: -.025em;
            line-height: 1.15;
        }
        .workspace-page-desc { margin: 0; color: #667085; font-size: .7rem; }
        .stat-card-grid { flex-wrap:nowrap; margin: 0 0 2rem; padding: 0; border-radius: 14px; }
        .stat-card { min-height: 66px; padding: .72rem 1rem; }
        .stat-card-label { margin-bottom: .24rem; font-size: .61rem; }
        .stat-card-value { font-size: 1.33rem; line-height: 1.05; }

        /* Exact tab composition used by the supplied iOS screens */
        .section-kicker { margin-bottom: .58rem; font-size: .64rem; letter-spacing: .09em; }
        .section-kicker::before { width: 20px; }
        .section-title { margin-bottom: .55rem; font-size: 1.27rem; line-height: 1.16; }
        .section-desc { max-width: 600px; font-size: .69rem; line-height: 1.42; }
        .section-desc { margin-bottom: 1.5rem; }
        .reference-card {
            overflow: hidden;
            border: 1px solid #E7EAF0;
            border-radius: 14px;
            background: #FFFFFF;
            box-shadow: 0 1px 3px rgba(15, 23, 42, .10);
        }
        .reference-card-title { color: #111827; font-size: .78rem; font-weight: 750; }
        .reference-card-subtitle { color: #98A2B3; font-size: .61rem; }
        .reference-label { color: #98A2B3; font-size: .58rem; font-weight: 600; letter-spacing: .02em; }
        .reference-value { color: #182033; font-size: .73rem; font-weight: 650; }
        .reference-grid-2 { display: grid; grid-template-columns: minmax(220px, .9fr) minmax(0, 2.1fr); gap: 1.2rem; }
        .reference-kpis {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1rem;
        }
        .reference-kpi { padding: .85rem 1rem; }
        .reference-kpi-value { margin-top: .25rem; font-size: 1.25rem; font-weight: 800; line-height: 1; }

        /* Salary intelligence panel */
        .salary-intel { margin-top: 1.45rem; }
        .st-key-salary-intelligence-card {
            overflow: hidden; margin-top: 1.45rem; border: 1px solid #E7EAF0; border-radius: 14px;
            background: #FFFFFF; box-shadow: 0 1px 3px rgba(15, 23, 42, .10);
        }
        .st-key-employee-picker > div,
        .st-key-employee-picker [data-testid="stVerticalBlock"] { gap: .65rem !important; }
        .st-key-employee-picker .section-divider-thin { display:none; }
        .st-key-salary-intelligence-card > div,
        .st-key-salary-intelligence-card [data-testid="stVerticalBlock"] { gap: 0 !important; }
        .salary-intel-head {
            display: flex; align-items: center; justify-content: space-between;
            min-height: 41px; padding: .65rem .95rem; border-bottom: 1px solid #F0F2F5;
        }
        .salary-intel-employee { display: flex; align-items: center; gap: .65rem; }
        .salary-intel-id { color: #182033; font-size: .84rem; font-weight: 800; }
        .salary-intel-meta { display: flex; gap: 1rem; color: #98A2B3; font-size: .61rem; }
        .salary-intel-meta b { margin-left: .25rem; color: #344054; }
        .salary-intel-body { display: grid; grid-template-columns: 160px minmax(250px, 1fr) 194px; min-height: 300px; }
        .salary-risk-column, .salary-radar-column, .salary-feature-column { padding: .85rem 1rem; }
        .salary-risk-column, .salary-radar-column { border-right: 1px solid #F0F2F5; }
        .salary-risk-column { display: flex; flex-direction: column; }
        .salary-gauge { display: flex; justify-content: center; margin: .25rem 0 0; }
        .salary-talent-row { display: flex; justify-content: space-between; margin-top: .15rem; font-size: .65rem; }
        .salary-ai-note { margin-top: .9rem; padding: .75rem; border-radius: 11px; background: #F8FAFC; color: #475467; font-size: .63rem; line-height: 1.55; }
        .salary-ai-note b { display: block; margin-bottom: .35rem; color: #98A2B3; font-size: .57rem; }
        .salary-radar-wrap { height: 238px; display: grid; place-items: center; }
        .salary-feature-column { display: flex; flex-direction: column; gap: .58rem; }
        .salary-feature-name { display: flex; align-items: baseline; justify-content: space-between; gap: .4rem; font-size: .61rem; }
        .salary-feature-name b { color: #344054; }
        .salary-feature-name small { margin-left: .25rem; color: #98A2B3; font-size: .54rem; }
        .salary-feature-track { height: 4px; margin-top: .22rem; border-radius: 999px; background: #F0F2F5; }
        .salary-feature-fill { height: 100%; border-radius: inherit; }
        .salary-intel-action { padding: .65rem 1rem .8rem; border-top: 1px solid #F0F2F5; }
        .st-key-salary-sim-fab { padding: .65rem 1rem .8rem; border-top: 1px solid #F0F2F5; }
        .salary-intel-action .stButton > button { min-height: 2.25rem; }

        /* Team composition */
        .team-left-stack { display: flex; flex-direction: column; gap: .8rem; }
        .team-condition-card { padding: 1rem; }
        .team-condition-card .reference-card-title { margin-bottom: .8rem; }
        .st-key-team-condition-card,
        .st-key-team-condition-levels,
        .st-key-team-simulation-card,
        .st-key-team-scatter-card,
        .st-key-team-radar-card {
            padding: .9rem 1rem; border: 1px solid #E7EAF0; border-radius: 14px;
            background: #FFFFFF; box-shadow: 0 1px 3px rgba(15,23,42,.10);
        }
        .st-key-team-condition-card { border-bottom: 0; border-radius: 14px 14px 0 0; padding-bottom: .35rem; }
        .st-key-team-condition-levels { margin-top: -1px; border-radius: 0 0 14px 14px; padding-top: .2rem; }
        .st-key-team-condition-card [data-testid="stVerticalBlock"],
        .st-key-team-condition-levels [data-testid="stVerticalBlock"] { gap: .55rem; }
        .st-key-team-simulation-card { margin-top: 1.4rem; }
        .st-key-team-scatter-card, .st-key-team-radar-card {
            min-height: 255px;
            margin-top: 1.35rem;
            padding-bottom: .2rem;
        }
        .st-key-team-scatter-card [data-testid="stPlotlyChart"],
        .st-key-team-radar-card [data-testid="stPlotlyChart"] { border: 0; box-shadow: none; }
        .team-roster-card { margin-bottom: 1.3rem; }
        .team-roster-head { padding: .85rem 1rem; border-bottom: 1px solid #F2F4F7; }
        .team-table { width: 100%; border-collapse: collapse; font-size: .67rem; }
        .team-table th { padding: .62rem 1rem; color: #98A2B3; font-weight: 500; text-align: left; }
        .team-table td { padding: .72rem 1rem; border-top: 1px solid #F2F4F7; color: #344054; }
        .team-table td:first-child { color: #182033; font-weight: 700; }
        .team-role { color: #155EEF !important; }
        .risk-chip { display: inline-block; padding: .15rem .45rem; border-radius: 999px; font-size: .58rem; font-weight: 700; }
        .risk-chip.safe { color: #079455; background: #ECFDF3; }
        .risk-chip.warning { color: #DC6803; background: #FFFAEB; }
        .risk-chip.danger { color: #D92D20; background: #FEF3F2; }
        .team-chart-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.25rem; margin-top: 1.3rem; }
        .team-chart-card { min-height: 240px; padding: .85rem 1rem; }

        /* Actions and stability */
        .st-key-actions-filter-row { margin-top: 1.4rem; margin-bottom:1.1rem; }
        .st-key-actions-filter-row [data-testid="stHorizontalBlock"] { align-items: end; }
        .st-key-actions-filter-row [data-testid="stVerticalBlock"] { gap: .15rem; }
        .st-key-subtabbar-hr_actions_tab { margin-top:.55rem; padding:0; border-bottom:1px solid #EAECF0; border-radius:0; background:transparent; }
        .st-key-subtabbar-hr_actions_tab .stButton > button { min-height:2.25rem; border-radius:9px 9px 0 0; font-size:.68rem; }
        .st-key-subtabbar-hr_actions_tab .stButton > button[kind="primary"] { background:#2563EB; color:#FFF; }
        .st-key-actions-table .stDataFrame { border: 1px solid #E7EAF0; border-radius: 14px; overflow: hidden; }
        .st-key-actions-table [data-testid="stDataFrame"] { font-size: .68rem; }
        [data-testid="stDataFrame"] { border:1px solid #E7EAF0; border-radius:14px; overflow:hidden; box-shadow:0 1px 3px rgba(15,23,42,.08); }
        .health-score-card { padding: 1.1rem; text-align: center; }
        .health-score-label { color: #98A2B3; font-size: .59rem; font-weight: 600; text-transform: uppercase; }
        .health-score-value { margin: .2rem 0; color: #111827; font-size: 3.65rem; font-weight: 800; line-height: 1; }
        .health-score-note { margin-top: .55rem; color: #98A2B3; font-size: .56rem; }
        .health-driver-card { margin-top: .75rem; padding: .95rem 1rem; }
        .health-score-card + .reference-kpis { margin-top: 1.25rem !important; }
        .st-key-health-driver-card { margin-top: 1.35rem; padding: 1.2rem 1.25rem; border: 1px solid #E7EAF0; border-radius: 14px; background:#FFF; box-shadow:0 1px 3px rgba(15,23,42,.10); }
        .st-key-health-driver-card [data-testid="stVerticalBlock"] { gap: .2rem; }
        .health-driver-card .hbar-chart { padding: .65rem 0 0; border: 0; box-shadow: none; }
        .st-key-health-driver-card .hbar-chart { padding: .5rem 0 0; border: 0; box-shadow: none; }
        div[data-testid="stRadio"] { margin: .55rem 0 .2rem; }
        div[data-testid="stRadio"] > label { color:#98A2B3; font-size:.6rem; }
        div[data-testid="stRadio"] [role="radiogroup"] { gap:.35rem; }
        div[data-testid="stRadio"] [role="radiogroup"] label { padding:.18rem .55rem; border-radius:999px; background:#F2F4F7; font-size:.6rem; }
        .health-dept-filter { display: flex; align-items: center; gap: .45rem; margin: .75rem 0; }
        .health-trend-card { padding: .9rem 1rem 1rem; }
        .health-trend-bars { height: 105px; display: flex; align-items: end; gap: .55rem; margin-top: .7rem; }
        .health-trend-item { flex: 1; display: flex; height: 100%; flex-direction: column; justify-content: end; align-items: center; gap: .25rem; }
        .health-trend-value, .health-trend-year { color: #667085; font-size: .58rem; }
        .health-trend-bar { width: 100%; min-height: 8px; border-radius: 6px 6px 0 0; background: linear-gradient(180deg,#60A5FA,#2563EB); }

        div.st-key-salary-sim-fab, div.st-key-team-swap-fab { position: static !important; inset: auto !important; width: auto !important; }
        div.st-key-salary-sim-fab .stButton > button, div.st-key-team-swap-fab .stButton > button {
            min-height: 2.15rem; border: 1px solid #B9D2FF; border-radius: 9px;
            padding: .35rem .75rem; background: #FFFFFF !important; border-color:#B9D2FF !important;
            color: #155EEF !important; box-shadow: none !important; font-size: .65rem;
        }

        @media (max-width: 760px) {
            .block-container { width: min(calc(100vw - 20px), 1024px); }
            .st-key-workspace-navigation { padding-inline: 10px; }
            .workspace-status { display: none; }
            .st-key-workspace-navigation [data-testid="stHorizontalBlock"] { flex-wrap:nowrap !important; }
            .stat-card-grid { display: grid; grid-template-columns: 1fr 1fr; }
            .stat-card { min-height: 70px; }
            .stat-card:nth-child(2) { border-right: 0; }
            .stat-card:nth-child(-n+2) { border-bottom: 1px solid var(--line-soft); }
            div[class*="st-key-tabbar-"] { bottom: 10px; width: calc(100vw - 20px); }
            div[class*="st-key-tabbar-"] .stButton > button { padding-inline: .25rem; font-size: .64rem; }
            .role-card { min-height: 0; }
            .reference-grid-2, .salary-intel-body, .team-chart-grid { grid-template-columns: 1fr; }
            .salary-risk-column, .salary-radar-column { border-right: 0; border-bottom: 1px solid #F0F2F5; }
            .reference-kpis { grid-template-columns: 1fr; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(tag: str, title: str, description: str) -> None:
    """페이지 상단 제목을 좌측 정렬 룰 기반 헤더로 표시한다."""

    st.markdown(
        f"""
        <div class="page-head stayon-rise">
            <span class="page-head-eyebrow">{tag}</span>
            <h1>{title}</h1>
            <div class="muted">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def top_navigation(role: str | None = None) -> None:
    """사이드바 대신 모든 화면에서 사용하는 상단 Glass Navigation을 표시한다.

    role이 주어지면 오른쪽에 현재 접속 유형 배지를 보여준다. 이 프로젝트는 페이지가
    랜딩(main.py)과 워크스페이스(pages/01_Workspace.py) 둘뿐이라, 상단 내비게이션은
    브랜드 링크와 상태 표시만 담당하고 실제 기능 전환은 워크스페이스 안의 세그먼트형
    탭이 맡는다.
    """

    role_label = {"hr": "HR TEAM", "admin": "ADMIN"}.get(role or "", "")
    role_html = f'<span class="role-pill">{role_label}</span>' if role_label else ""
    with st.container(key="top-navigation"):
        st.markdown(
            f"""
            <div class="top-nav">
                <a class="top-nav-brand" href="/" target="_self">STAYON</a>
                <div class="top-nav-links"></div>
                <div class="top-nav-status">{role_html}<span class="status-dot">System Operational</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def workspace_navigation(role: str) -> None:
    """워크스페이스 전용 헤더: 홈 로고와 HR/DV 역할 전환만 제공한다."""

    with st.container(key="workspace-navigation"):
        st.markdown(
            '<a class="workspace-brand" href="/" target="_self" aria-label="홈으로 이동">STAYON</a>'
            f'<span class="workspace-role-state is-{role}"></span>',
            unsafe_allow_html=True,
        )
        if st.button(
            "HR",
            key="workspace_role_hr",
            type="primary" if role == "hr" else "secondary",
        ):
            st.session_state["role"] = "hr"
            if st.session_state.get("workspace_tab") == "models":
                st.session_state["workspace_tab"] = "salary"
            st.rerun()
        if st.button(
            "DV",
            key="workspace_role_admin",
            type="primary" if role == "admin" else "secondary",
        ):
            st.session_state["role"] = "admin"
            st.session_state["workspace_tab"] = "models"
            st.rerun()


def home_button() -> None:
    """하위 페이지 왼쪽 위에 메인 화면 이동 링크를 표시한다."""

    with st.container(key="home-nav"):
        st.markdown(
            '<a class="home-link" href="/" target="_self">&larr; Home</a>',
            unsafe_allow_html=True,
        )


def stat_cards(items: Sequence[Mapping[str, Any]]) -> None:
    """st.metric 대신 쓰는 KPI 스트립. 숫자를 압도적으로 크게 보여준다.

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
    st.markdown(f'<div class="stat-card-grid stayon-rise">{cards}</div>', unsafe_allow_html=True)


def badge(text: str, tone: str = "info") -> str:
    """어디서나 재사용하는 배지 HTML 조각을 반환한다."""

    return f'<span class="badge tone-{tone}">{_esc(text)}</span>'


def alert_box(kind: str, message: str, title: str | None = None) -> None:
    """st.info/st.warning/st.error/st.success 대신 쓰는 Glass 안내 박스.

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


def empty_state(title: str, message: str, icon: str = "○") -> None:
    """검색 결과 없음 / 접근 불가 등, 정제된 빈 상태 패널."""

    st.markdown(
        f"""
        <div class="empty-state">
            <div class="empty-state-icon">{_esc(icon)}</div>
            <div class="empty-state-title">{_esc(title)}</div>
            <div class="empty-state-desc">{_esc(message)}</div>
        </div>
        """,
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
    numbered: bool = False,
) -> None:
    """st.bar_chart/st.line_chart 대신 쓰는 가로 막대 그래프 (단일 계열).

    numbered=True면 "01 · 라벨 · 값"처럼 순위형 에디토리얼 리스트로 그린다
    (인사 구조 안정도 탭의 "이탈률과 연관 높은 주요 피처" 랭킹 등에 사용).
    min_height를 주면 막대 개수가 적어도(예: 5개) 옆 카드/표와 높이를 맞출 수 있게,
    전체 높이를 그 값만큼 확보하고 막대들을 위아래로 고르게 펼쳐(justify-content:
    space-between) 배치한다. 지정하지 않으면 기존처럼 내용 높이만큼만 차지한다.
    """

    values = [float(v) for _, v in items]
    scale = max_value if max_value is not None else (max(values) if values else 1) or 1
    row_class = "hbar-row numbered" if numbered else "hbar-row"
    rows = "".join(
        (
            f'<div class="{row_class}">'
            + (f'<div class="hbar-rank">{index + 1:02d}</div>' if numbered else "")
            + f'<div class="hbar-label">{_esc(label)}</div>'
            '<div class="hbar-track">'
            f'<div class="hbar-fill" style="width:{max(0.0, min(100.0, (value / scale) * 100)):.1f}%; background:{color};"></div>'
            "</div>"
            f'<div class="hbar-value">{value_format.format(value)}</div>'
            "</div>"
        )
        for index, (label, value) in enumerate(items)
    )
    style = (
        f' style="min-height:{_esc(min_height)}; justify-content:space-between;"'
        if min_height
        else ""
    )
    st.markdown(f'<div class="hbar-chart"{style}>{rows}</div>', unsafe_allow_html=True)


def section_heading(kicker: str, title: str, description: str) -> None:
    """탭 콘텐츠 상단에 쓰는 좌측 정렬 에디토리얼 섹션 제목."""

    st.markdown(
        f"""
        <div class="stayon-rise">
            <div class="section-kicker">{_esc(kicker)}</div>
            <div class="section-title">{title}</div>
            <div class="section-desc">{description}</div>
        </div>
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
    """부서 → 직급 → ID 순으로 좁혀가는 Apple 스타일 Glass 필터 바.

    with_direct_search=True면 위에 직원 ID 직접 검색창을 추가로 보여주고, 값을
    입력해 일치하는 직원이 있으면 그 ID를 우선 반환한다(캐스케이딩 선택 무시).
    """

    from src.utils.hr_metrics import LEVEL_KR, department_options, level_options, translate

    with st.container(key="employee-picker"):
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
    """st.tabs 대신 쓰는 Liquid Glass 세그먼트 버튼형 하위 탭. 선택된 키를 반환한다."""

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


def giant_stat(value: str, label: str, unit: str = "", tone: str = "ink") -> None:
    """숫자 하나를 압도적으로 크게 보여주는 최소 단위 컴포넌트.

    tone: "ink"(기본) | "safe" | "info" | "warning" | "danger" — 값 색상을 톤에 맞춘다.
    """

    color = "var(--ink)" if tone == "ink" else "var(--tone-color)"
    tone_wrap = "" if tone == "ink" else f'tone-{tone}'
    unit_html = f'<span class="giant-stat-unit">{_esc(unit)}</span>' if unit else ""
    st.markdown(
        f"""
        <div class="{tone_wrap}">
            <div class="giant-stat-value" style="color:{color};">{value}{unit_html}</div>
            <div class="giant-stat-label">{_esc(label)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def narrative_banner(
    eyebrow: str,
    value: str,
    status: str,
    status_tone: str,
    message: str,
) -> None:
    """조직 건강도처럼, 화면 전체 상태를 하나의 큰 시각적 서사로 보여주는 배너.

    예: ORGANIZATION HEALTH · 84 · Stable · "Your organization is showing strong
    retention signals."
    """

    st.markdown(
        f"""
        <div class="glass-panel narrative-banner stayon-rise">
            <span class="stayon-caption">{_esc(eyebrow)}</span>
            <div class="giant-value">{_esc(value)}</div>
            <div class="narrative-status">{badge(status, status_tone)}</div>
            <div class="narrative-message">{_esc(message)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def employee_hero(
    employee_id: int,
    department: str,
    level: str,
    risk_pct: float,
    risk_label_text: str,
    risk_tone: str,
    talent_score: float,
    extra_badges: Sequence[str] | None = None,
) -> None:
    """연봉 협상 탭 상단의 "Executive Intelligence" 직원 요약 패널.

    왼쪽에는 직원 기본 정보, 오른쪽에는 퇴사 위험 확률을 거대한 숫자로, 아래에는
    인재 가치 지수를 막대와 함께 보여준다.
    """

    risk_color = "var(--tone-color)"
    st.markdown(
        f"""
        <div class="glass-panel employee-hero stayon-rise tone-{risk_tone}">
            <div class="employee-hero-grid">
                <div>
                    <span class="stayon-caption">Employee</span>
                    <div class="employee-hero-id">Employee #{employee_id}</div>
                    <div class="employee-hero-meta">
                        {badge(translate_label(department), "neutral")}
                        {badge(translate_label(level), "neutral")}
                        {"".join(badge(b, "neutral") for b in (extra_badges or []))}
                    </div>
                </div>
                <div class="employee-hero-risk">
                    <span class="stayon-caption">Attrition Risk</span>
                    <div class="employee-hero-risk-value" style="color:{risk_color};">{risk_pct:.0%}</div>
                    <div style="margin-top:.5rem;">{badge(risk_label_text, risk_tone)}</div>
                </div>
            </div>
            <div class="employee-hero-divider"></div>
            <div class="talent-value-row">
                <span class="stayon-caption">Talent Value</span>
                <span class="talent-value-score">{talent_score:.0f}<span class="of100"> /100</span></span>
            </div>
            <div class="talent-value-track"><div class="talent-value-fill" style="width:{max(0.0, min(100.0, talent_score)):.1f}%;"></div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def translate_label(value: str) -> str:
    """employee_hero 등에서 부서/직급 원본 값을 한글 라벨로 바꾼다(순환 import 방지용 지연 import)."""

    from src.utils.hr_metrics import translate

    return translate(value)


def ring_svg(value: float, max_value: float = 100.0, size: int = 52, stroke: int = 5, color: str = "var(--blue)") -> str:
    """Apple Watch 활동 링 스타일의 작은 진행률 링을 SVG 문자열로 반환한다."""

    radius = (size - stroke) / 2
    circumference = 2 * math.pi * radius
    pct = max(0.0, min(1.0, (value / max_value) if max_value else 0.0))
    dash = circumference * pct
    center = size / 2
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" style="transform:rotate(-90deg);">'
        f'<circle cx="{center}" cy="{center}" r="{radius}" fill="none" stroke="var(--surface-sunken)" stroke-width="{stroke}"></circle>'
        f'<circle cx="{center}" cy="{center}" r="{radius}" fill="none" stroke="{color}" stroke-width="{stroke}" '
        f'stroke-linecap="round" stroke-dasharray="{dash:.2f} {circumference:.2f}"></circle>'
        "</svg>"
    )


def ranking_list(rows: Sequence[Mapping[str, Any]]) -> None:
    """팀 추천 · 직원 랭킹처럼, 순위가 있는 데이터를 카드형 랭킹 리스트로 그린다.

    rows 각 원소: {
        "rank": int, "title": str, "subtitle": str, "highlight": bool(optional),
        "metrics": [
            {"label": str, "value": str, "kind": "bar"|"ring"|"text", "pct": float(0~100, bar/ring일 때)},
            ...
        ],
    }
    """

    body_rows = []
    for row in rows:
        metrics = row.get("metrics", [])
        metric_cells = []
        for metric in metrics:
            kind = metric.get("kind", "text")
            if kind == "ring":
                ring_html = ring_svg(metric.get("pct", 0.0), 100.0, size=48, stroke=5)
                metric_cells.append(
                    '<div class="ranking-metric">'
                    f'<div class="ranking-metric-label">{_esc(metric["label"])}</div>'
                    f'<div class="ranking-ring-wrap">{ring_html}'
                    f'<div class="ranking-metric-value">{_esc(metric["value"])}</div></div>'
                    "</div>"
                )
            elif kind == "bar":
                pct = max(0.0, min(100.0, metric.get("pct", 0.0)))
                metric_cells.append(
                    '<div class="ranking-metric">'
                    f'<div class="ranking-metric-label">{_esc(metric["label"])}</div>'
                    f'<div class="ranking-metric-value">{_esc(metric["value"])}</div>'
                    f'<div class="ranking-metric-bar-track"><div class="ranking-metric-bar-fill" style="width:{pct:.1f}%;"></div></div>'
                    "</div>"
                )
            else:
                metric_cells.append(
                    '<div class="ranking-metric">'
                    f'<div class="ranking-metric-label">{_esc(metric["label"])}</div>'
                    f'<div class="ranking-metric-value">{_esc(metric["value"])}</div>'
                    "</div>"
                )
        highlight_class = " highlight" if row.get("highlight") else ""
        body_rows.append(
            f'<div class="ranking-row{highlight_class}" style="--metric-count:{max(1, len(metrics))};">'
            f'<div class="ranking-rank">{int(row["rank"]):02d}</div>'
            '<div>'
            f'<div class="ranking-title">{_esc(row["title"])}</div>'
            f'<div class="ranking-subtitle">{_esc(row["subtitle"])}</div>'
            "</div>"
            + "".join(metric_cells)
            + "</div>"
        )
    st.markdown(f'<div class="ranking-list">{"".join(body_rows)}</div>', unsafe_allow_html=True)


def area_trend_chart(
    items: Sequence[tuple[str, float]],
    value_format: str = "{:.1f}%",
    color: str = "#0071E3",
    height: int = 160,
) -> None:
    """근속연차별 퇴사율처럼, 구간형 추세를 Apple Health 스타일 부드러운 영역 차트로 그린다."""

    values = [float(v) for _, v in items]
    labels = [str(label) for label, _ in items]
    if not values:
        return
    vmax = max(values) or 1.0
    vmin = min(0.0, min(values))
    span = (vmax - vmin) or 1.0
    width = 640
    n = len(values)
    step = width / max(1, n - 1) if n > 1 else 0
    points = []
    for index, value in enumerate(values):
        x = index * step if n > 1 else width / 2
        y = height - ((value - vmin) / span) * (height - 20) - 10
        points.append((x, y))

    path_d = " ".join(f"{'M' if i == 0 else 'L'} {x:.1f} {y:.1f}" for i, (x, y) in enumerate(points))
    area_d = path_d + f" L {points[-1][0]:.1f} {height} L {points[0][0]:.1f} {height} Z"
    dots = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{color}" stroke="white" stroke-width="2"></circle>' for x, y in points)
    gradient_id = "areaGrad"

    svg = (
        f'<svg width="100%" viewBox="0 0 {width} {height}" preserveAspectRatio="none" style="display:block;">'
        f'<defs><linearGradient id="{gradient_id}" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0%" stop-color="{color}" stop-opacity="0.32"></stop>'
        f'<stop offset="100%" stop-color="{color}" stop-opacity="0"></stop>'
        "</linearGradient></defs>"
        f'<path d="{area_d}" fill="url(#{gradient_id})" stroke="none"></path>'
        f'<path d="{path_d}" fill="none" stroke="{color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"></path>'
        f"{dots}"
        "</svg>"
    )
    labels_html = "".join(
        f'<div class="area-chart-label">{_esc(label)}<span class="val">{value_format.format(value)}</span></div>'
        for label, value in zip(labels, values, strict=False)
    )
    st.markdown(
        f'<div class="area-chart-wrap">{svg}<div class="area-chart-labels">{labels_html}</div></div>',
        unsafe_allow_html=True,
    )


def model_card_grid(models: Sequence[Mapping[str, Any]]) -> None:
    """ML 모델들을 compact card 그리드로 비교한다.

    models 각 원소: {"name": str, "metrics": [(label, value_str), ...], "best": bool(optional)}
    """

    cards = []
    for model in models:
        best = model.get("best", False)
        best_badge = '<div class="model-card-best-badge">BEST MODEL</div>' if best else ""
        rows = "".join(
            '<div class="model-card-metric-row">'
            f'<span class="model-card-metric-label">{_esc(label)}</span>'
            f'<span class="model-card-metric-value">{_esc(value)}</span>'
            "</div>"
            for label, value in model.get("metrics", [])
        )
        card_class = "model-card best" if best else "model-card"
        cards.append(
            f'<div class="{card_class}">{best_badge}'
            f'<div class="model-card-name">{_esc(model["name"])}</div>'
            f'<div class="model-card-metrics">{rows}</div>'
            "</div>"
        )
    st.markdown(f'<div class="model-card-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def versus_hero(left_label: str, left_value: str, right_label: str, right_value: str, winner: str | None = None) -> None:
    """두 값(ML vs DL 등)을 거대한 숫자로 나란히 비교하는 히어로 블록."""

    left_class = "versus-side winner" if winner == left_label else "versus-side"
    right_class = "versus-side winner" if winner == right_label else "versus-side"
    st.markdown(
        f"""
        <div class="glass-panel">
            <div class="versus-hero">
                <div class="{left_class}">
                    <div class="giant-stat-value">{_esc(left_value)}</div>
                    <div class="giant-stat-label">{_esc(left_label)}</div>
                </div>
                <div class="versus-divider">VS</div>
                <div class="{right_class}">
                    <div class="giant-stat-value">{_esc(right_value)}</div>
                    <div class="giant-stat-label">{_esc(right_label)}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def department_cards(rows: Sequence[Mapping[str, Any]]) -> None:
    """부서별 안정도를 표 대신 카드 그리드로 보여준다.

    rows 각 원소: {"title": str, "fields": [(label, value_str), ...]}
    """

    cards = []
    for row in rows:
        fields_html = "".join(
            '<div class="dept-card-row">'
            f'<span class="dept-card-row-label">{_esc(label)}</span>'
            f'<span class="dept-card-row-value">{_esc(value)}</span>'
            "</div>"
            for label, value in row["fields"]
        )
        cards.append(
            f'<div class="dept-card"><div class="dept-card-title">{_esc(row["title"])}</div>{fields_html}</div>'
        )
    st.markdown(f'<div class="dept-card-grid">{"".join(cards)}</div>', unsafe_allow_html=True)
