"""Streamlit 화면에서 공통으로 사용하는 스타일과 작은 UI 도우미."""

# ruff: noqa: E501

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

        /* TalentGuard dark intelligence theme */
        :root {
            --ink: #F8FAFC; --muted: #94A3B8; --faint: #64748B;
            --blue: #22D3EE; --blue-deep: #0EA5E9;
            --blue-soft: rgba(34, 211, 238, .10);
            --surface: #0D1530; --surface-alt: #070C1A;
            --line: rgba(148, 163, 184, .14);
        }
        .stApp {
            background: radial-gradient(circle at 12% 0%, rgba(34,211,238,.08), transparent 30rem),
                        radial-gradient(circle at 88% 20%, rgba(167,139,250,.07), transparent 34rem),
                        var(--surface-alt);
        }
        .top-nav { background: rgba(7,12,26,.90); border-color: rgba(34,211,238,.12); }
        .top-nav-brand { color: #FFF !important; letter-spacing: -.01em; }
        .top-nav-brand span { color: #22D3EE; }
        .top-nav-brand::before {
            content:'HR'; display:inline-grid; place-items:center; width:32px; height:32px;
            margin-right:.65rem; border-radius:9px; background:linear-gradient(135deg,#22D3EE,#0EA5E9);
            color:#070C1A; font-size:.72rem; font-weight:900; vertical-align:middle;
        }
        .top-nav-link:hover, .top-nav-link.active {
            color: #22D3EE !important; background: rgba(34,211,238,.10);
            border: 1px solid rgba(34,211,238,.25);
        }
        .page-head-eyebrow, .section-kicker { color: #22D3EE; }
        .page-head-eyebrow { border-left-color: #22D3EE; }
        .page-head, h3 { border-color: var(--line); }
        [class*="st-key-stat-bar"] [data-testid="stHorizontalBlock"] {
            background: rgba(13,21,48,.82); border-color: var(--line);
            box-shadow: 0 18px 45px rgba(0,0,0,.16);
        }
        [class*="st-key-stat-bar"] [data-testid="stHorizontalBlock"] > div:not(:last-child) { border-color: var(--line); }
        div[data-testid="stDataFrame"], div[data-testid="stSlider"] { background: rgba(13,21,48,.75); border-color: var(--line); }
        .stButton > button, .stDownloadButton > button, .stFormSubmitButton > button { background: rgba(13,21,48,.9); border-color: var(--line); color: #E2E8F0; }
        .section-jump { background: rgba(7,12,26,.88); border-color: var(--line); }
        .feature-pill { background: rgba(148,163,184,.09); color: #94A3B8; }
        .decision-note { background: rgba(34,211,238,.08); color: #67E8F9; }
        .nav-row:hover { background: rgba(34,211,238,.06); }
        .hero {
            min-height: 525px; display: flex; align-items: center; justify-content: center;
            text-align: center; margin: -2.2rem calc(50% - 50vw) 0; padding: 5rem 1.5rem 4rem;
            position: relative; overflow: hidden;
            background: linear-gradient(135deg,rgba(7,12,26,.96),rgba(7,12,26,.80),rgba(7,12,26,.94)),
                        repeating-linear-gradient(0deg,transparent,transparent 59px,rgba(34,211,238,.07) 60px),
                        repeating-linear-gradient(90deg,transparent,transparent 59px,rgba(34,211,238,.07) 60px);
            border-bottom: 1px solid rgba(34,211,238,.10);
        }
        .hero-inner { max-width: 850px; position: relative; z-index: 1; }
        .hero-badge { display:inline-flex; align-items:center; gap:.55rem; padding:.45rem .9rem; border-radius:999px; border:1px solid rgba(34,211,238,.3); background:rgba(34,211,238,.08); color:#22D3EE; font-size:.7rem; font-weight:700; letter-spacing:.13em; }
        .hero-badge::before { content:''; width:6px; height:6px; border-radius:50%; background:#22D3EE; box-shadow:0 0 12px #22D3EE; }
        .hero h1 { margin:1.8rem 0 1.1rem; color:#FFF; font-size:clamp(2.8rem,6vw,4.8rem) !important; line-height:1.08; }
        .hero-gradient { background:linear-gradient(90deg,#22D3EE,#A78BFA); -webkit-background-clip:text; color:transparent; }
        .hero p { max-width:680px; margin:0 auto; color:#94A3B8; font-size:1.05rem; line-height:1.8; }
        .module-section { padding:5rem 0 2rem; text-align:center; }
        .module-eyebrow { color:#22D3EE; font-size:.72rem; font-weight:800; letter-spacing:.14em; }
        .module-section h2 { margin:.8rem 0 .6rem; font-size:2.2rem; }
        .module-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:1rem; margin:2.4rem 0 3.5rem; text-align:left; }
        .module-card { display:block; min-height:235px; padding:1.7rem; border-radius:18px; background:rgba(13,21,48,.78); border:1px solid var(--line); text-decoration:none !important; transition:transform .2s,border-color .2s,box-shadow .2s; }
        .module-card:hover { transform:translateY(-5px); border-color:var(--accent); box-shadow:0 22px 45px rgba(0,0,0,.22); }
        .module-icon { width:48px; height:48px; display:grid; place-items:center; border-radius:12px; background:rgba(34,211,238,.10); color:var(--accent); font-size:1.35rem; }
        .module-label { margin-top:1.3rem; color:var(--accent); font-size:.68rem; font-weight:800; letter-spacing:.12em; }
        .module-title { margin:.45rem 0 .55rem; color:#FFF; font-family:'Plus Jakarta Sans',sans-serif; font-size:1.08rem; font-weight:800; }
        .module-desc { color:#64748B; font-size:.82rem; line-height:1.65; }
        .dashboard-card {
            padding: 1.5rem; margin-bottom: 1.25rem; border-radius: 18px;
            background: rgba(13,21,48,.80); border: 1px solid var(--line);
        }
        .panel-label { color:#22D3EE; font-size:.68rem; font-weight:800; letter-spacing:.14em; text-transform:uppercase; }
        .panel-title { margin:.45rem 0 1.2rem; color:#FFF; font-family:'Plus Jakarta Sans',sans-serif; font-size:1.28rem; font-weight:800; }
        .section-heading { display:flex; align-items:center; gap:.75rem; margin:2.6rem 0 1.2rem; }
        .section-chip { padding:.3rem .7rem; border-radius:999px; background:rgba(34,211,238,.10); border:1px solid rgba(34,211,238,.28); color:#22D3EE; font-size:.7rem; font-weight:800; letter-spacing:.1em; }
        .section-chip.violet { background:rgba(167,139,250,.10); border-color:rgba(167,139,250,.28); color:#A78BFA; }
        .section-heading h2 { margin:0; font-size:1.55rem; }
        .flow-divider { display:flex; align-items:center; gap:1rem; margin:3.5rem 0; color:#475569; font-size:.65rem; letter-spacing:.13em; }
        .flow-divider::before, .flow-divider::after { content:''; height:1px; flex:1; background:var(--line); }
        .champion-card { padding:1.6rem; border-radius:18px; background:rgba(13,21,48,.82); border:1px solid var(--accent); min-height:270px; }
        .champion-tag { display:inline-block; padding:.28rem .55rem; border-radius:6px; background:color-mix(in srgb,var(--accent) 13%,transparent); color:var(--accent); font-size:.66rem; font-weight:800; letter-spacing:.1em; }
        .champion-name { margin:.9rem 0 1rem; color:var(--accent); font-family:'JetBrains Mono',monospace; font-size:1.9rem; font-weight:800; }
        .champion-metrics { display:grid; grid-template-columns:repeat(3,1fr); gap:.55rem; margin-bottom:1rem; }
        .champion-metric { padding:.65rem .3rem; border-radius:8px; background:rgba(255,255,255,.025); text-align:center; }
        .champion-metric b { display:block; color:var(--accent); font-family:'JetBrains Mono',monospace; }
        .champion-metric span, .champion-list { color:#64748B; font-size:.72rem; }
        .champion-list { line-height:1.9; }
        .insight-grid { display:grid; grid-template-columns:repeat(2,1fr); gap:.75rem; }
        .insight-card { padding:1rem; border-radius:10px; background:rgba(34,211,238,.035); border:1px solid rgba(34,211,238,.08); color:#94A3B8; font-size:.82rem; line-height:1.55; }
        .decision-box { padding:1.8rem; border-radius:18px; background:linear-gradient(135deg,rgba(34,211,238,.06),rgba(167,139,250,.06)); border:1px solid rgba(34,211,238,.20); }
        .role-grid { display:grid; grid-template-columns:repeat(2,1fr); gap:1.25rem; margin-top:2rem; }
        .role-card { display:block; padding:2rem; min-height:230px; border-radius:18px; background:rgba(13,21,48,.80); border:1px solid var(--line); text-decoration:none !important; transition:transform .2s,border-color .2s,box-shadow .2s; }
        .role-card:hover { transform:translateY(-5px); border-color:var(--accent); box-shadow:0 22px 45px rgba(0,0,0,.25); }
        .role-code { color:var(--accent); font-size:.7rem; font-weight:800; letter-spacing:.12em; }
        .role-card h2 { margin:1rem 0 .5rem; color:#FFF; font-size:1.7rem; }
        .role-card p { color:#64748B; line-height:1.7; min-height:55px; }
        .role-link { margin-top:1.2rem; color:var(--accent); font-size:.86rem; font-weight:700; }
        .start-hero {
            min-height:calc(100vh - 72px); margin:-2.2rem calc(50% - 50vw) -3rem;
            display:flex; align-items:center; justify-content:center; text-align:center;
            padding:5rem 1.5rem; position:relative; overflow:hidden;
            background-image:linear-gradient(135deg,rgba(5,10,24,.94),rgba(5,12,28,.72),rgba(5,10,24,.92)),url('https://images.unsplash.com/photo-1541746972996-4e0b0f43e02a?w=1920&h=1080&fit=crop&auto=format');
            background-size:cover; background-position:center;
        }
        .start-hero::after { content:''; position:absolute; inset:0; background:repeating-linear-gradient(90deg,transparent,transparent 59px,rgba(34,211,238,.035) 60px); pointer-events:none; }
        .start-hero-inner { width:min(900px,100%); position:relative; z-index:1; }
        .start-badge { display:inline-flex; align-items:center; gap:.55rem; padding:.45rem .9rem; border-radius:999px; border:1px solid rgba(34,211,238,.35); background:rgba(34,211,238,.09); color:#22D3EE; font:700 .7rem 'JetBrains Mono',monospace; letter-spacing:.12em; }
        .start-badge::before { content:''; width:6px; height:6px; border-radius:50%; background:#22D3EE; box-shadow:0 0 10px #22D3EE; }
        .start-hero h1 { margin:1.7rem 0 1.1rem; color:#FFF; font-size:clamp(3.2rem,7vw,5.5rem) !important; line-height:1.06; }
        .start-gradient { background:linear-gradient(90deg,#22D3EE,#A78BFA); -webkit-background-clip:text; color:transparent; }
        .start-description { color:#94A3B8; font-size:1.02rem; line-height:1.8; }
        .start-actions { display:flex; justify-content:center; gap:1rem; margin-top:2.2rem; }
        .start-action { min-width:230px; padding:1rem 1.3rem; border-radius:13px; background:color-mix(in srgb,var(--accent) 10%,rgba(7,12,26,.65)); border:1px solid color-mix(in srgb,var(--accent) 42%,transparent); color:var(--accent) !important; text-decoration:none !important; text-align:left; transition:transform .2s,background .2s,box-shadow .2s; }
        .start-action:hover { transform:translateY(-4px); background:color-mix(in srgb,var(--accent) 18%,rgba(7,12,26,.72)); box-shadow:0 16px 35px rgba(0,0,0,.28); }
        .start-action-code { font:700 .64rem 'JetBrains Mono',monospace; letter-spacing:.12em; opacity:.75; }
        .start-action-title { display:flex; justify-content:space-between; margin-top:.35rem; font-size:1rem; font-weight:800; }
        div[data-testid="stTabs"] button { color:#64748B; border-radius:10px; padding:.75rem 1rem; }
        div[data-testid="stTabs"] button[aria-selected="true"] { color:#22D3EE; background:rgba(34,211,238,.10); }
        div[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background-color:#22D3EE; }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] { color:#22D3EE; font-family:'JetBrains Mono',monospace; }
        @media (max-width:850px) { .insight-grid,.role-grid { grid-template-columns:1fr; } .champion-metrics { grid-template-columns:1fr; } .start-actions { flex-direction:column; align-items:center; } .start-action { width:min(100%,360px); } }
        @media (max-width:850px) { .module-grid { grid-template-columns:1fr; } .hero { min-height:460px; } }
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

    if active in {"salary", "team", "people", "stability"}:
        links = [
            ("start", "/page01", "역할 선택"),
            ("salary", "/page02", "연봉 협상"),
            ("team", "/page03", "팀 구성"),
            ("people", "/page04", "인사 지원"),
            ("stability", "/page05", "구조 안정도"),
        ]
    elif active == "models":
        links = [("start", "/page01", "역할 선택"), ("models", "/page06", "ML / DL 성능평가")]
    else:
        links = [("salary", "/page02", "인사팀 담당자"), ("models", "/page06", "개발 관리자")]
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
                <a class="top-nav-brand" href="/" target="_self">TalentGuard<span>AI</span></a>
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


def style_plotly_chart(figure, height: int = 300):
    """첨부 시안과 같은 다크 Plotly 차트 스타일을 적용한다."""

    figure.update_layout(
        height=height,
        margin={"l": 18, "r": 18, "t": 25, "b": 18},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#94A3B8", "family": "Inter"},
        legend={"orientation": "h", "y": 1.12, "x": 0},
        hoverlabel={"bgcolor": "#0D1530", "bordercolor": "#22D3EE"},
    )
    figure.update_xaxes(gridcolor="rgba(148,163,184,.08)", zeroline=False)
    figure.update_yaxes(gridcolor="rgba(148,163,184,.08)", zeroline=False)
    return figure
