"""TalentGuard AI Streamlit 공통 UI."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

COLORS = {"bg": "#07101f", "panel": "#0b1428", "line": "#25314b", "text": "#f6f8ff", "muted": "#91a0bc", "cyan": "#21d4ee", "violet": "#9b7df4", "amber": "#ffa20a", "green": "#38d0a0", "rose": "#ff6b86"}


def apply_page_style() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@500;600&family=Noto+Sans+KR:wght@400;500;600;700;800&display=swap');
:root{color-scheme:dark}html,body,[class*="css"]{font-family:'Noto Sans KR',sans-serif}body,.stApp{background:linear-gradient(110deg,#06101d,#090d20);color:#f6f8ff}[data-testid="stHeader"],[data-testid="stSidebar"],#MainMenu,footer{display:none!important}.stAppViewContainer>.main{overflow:visible}.block-container{max-width:none;padding:0 34px 48px!important}
.tg-nav{height:64px;margin:0 -34px 28px;padding:0 34px;display:grid;grid-template-columns:1fr auto 1fr;align-items:center;border-bottom:1px solid #18314a;background:rgba(5,12,26,.96);position:sticky;top:0;z-index:999}.tg-brand{color:#f7f9ff!important;font-size:19px;font-weight:800;text-decoration:none!important;display:flex;align-items:center;gap:10px}.tg-brand:before{content:'HR';width:30px;height:30px;border-radius:9px;display:grid;place-items:center;background:#20bde4;color:#06101d;font:700 10px 'IBM Plex Mono'}.tg-brand span{color:#27cde9}.tg-links{display:flex;gap:10px;align-items:center}.tg-links a{color:#8996af!important;text-decoration:none!important;font-size:13px;font-weight:700;padding:11px 15px;border:1px solid transparent;border-radius:9px}.tg-links a.active{color:#22d4ef!important;background:#09283a;border-color:#126078}.tg-health{justify-self:end;color:#68758c;font-size:11px}.tg-health:before{content:'';display:inline-block;width:6px;height:6px;border-radius:50%;background:#22d76d;margin-right:6px}
.page-header{padding:4px 8px 25px;border-bottom:1px solid #202a3f;margin-bottom:52px}.eyebrow{color:#20d4ef;font:600 11px 'IBM Plex Mono';letter-spacing:.04em;border-left:2px solid #20d4ef;padding-left:11px;margin-bottom:20px}.page-header h1{font-size:28px;margin:0 0 24px;color:#fff}.page-header p,.section-copy{color:#91a0bc;font-size:14px}.section-heading{display:flex;align-items:center;gap:12px;margin:0 8px 25px}.section-heading h2{font-size:22px;margin:0;color:#fff}.chip{padding:5px 10px;border-radius:999px;color:var(--accent,#20d4ef);border:1px solid color-mix(in srgb,var(--accent,#20d4ef) 45%,transparent);background:color-mix(in srgb,var(--accent,#20d4ef) 10%,transparent);font:700 10px 'IBM Plex Mono'}.panel-title{color:#20d4ef;font:700 10px 'IBM Plex Mono';letter-spacing:.08em;margin-bottom:8px}.big-number{font:500 clamp(44px,5vw,72px) 'IBM Plex Mono';text-align:center;margin:22px 0 8px}.center-note{text-align:center;font-weight:600;margin-bottom:14px}
.metric-strip{display:grid;grid-template-columns:repeat(4,1fr);background:#0c1730;border:1px solid #243252;border-radius:14px;padding:18px 6px;margin:14px 0 16px}.metric-strip>div{padding:0 20px;border-right:1px solid #25314b}.metric-strip>div:last-child{border:0}.metric-strip small{display:block;color:#a4b0c5;font-size:11px}.metric-strip strong{color:#22d5f2;font:500 30px 'IBM Plex Mono'}.tg-card,[data-testid="stVerticalBlockBorderWrapper"]{border-color:#35405a!important;border-radius:8px!important;background:rgba(6,13,28,.48)}[data-testid="stDataFrame"]{border:1px solid #2b3857;border-radius:10px;overflow:hidden}[data-testid="stDataFrame"] *{font-size:12px}[data-baseweb="select"]>div,[data-baseweb="input"]>div{background:#0e1935!important;border-color:#243354!important;color:#fff!important}label,.stMarkdown p{color:#a8b7cf}div[data-testid="stMetric"]{background:#0d1831;border:1px solid #243252;padding:16px 18px;border-radius:12px}div[data-testid="stMetricValue"]{color:#22d5f2;font-family:'IBM Plex Mono'}
.champ-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}.champ{border:1px solid var(--accent);border-radius:13px;padding:18px;background:#0c1630}.champ .tag{color:var(--accent);font:700 9px 'IBM Plex Mono';background:color-mix(in srgb,var(--accent) 12%,transparent);display:inline-block;padding:5px 8px;border-radius:4px}.champ h3{color:var(--accent);font:500 21px 'IBM Plex Mono'}.champ-stats{display:grid;grid-template-columns:repeat(3,1fr);gap:6px}.champ-stats div{text-align:center;background:#111b35;padding:12px 5px;border-radius:6px;color:var(--accent);font:600 12px 'IBM Plex Mono'}.champ-stats small{display:block;color:#75839e;font:400 8px 'Noto Sans KR';margin-top:7px}.champ ul{list-style:none;padding:4px 0 0;color:#8e9bb5;font-size:10px;line-height:1.9}.decision{border:1px solid #164e65;background:#091b2d;border-radius:13px;padding:18px;margin-top:10px}.decision h3{margin:0 0 15px;font-size:16px}.decision-row{display:grid;grid-template-columns:1fr 1fr;gap:8px}.decision-row span{background:#0c2237;border:1px solid #163958;padding:12px;border-radius:7px;color:#81b8d7;font-size:10px}
.start-hero{min-height:calc(100vh - 64px);margin:-28px -34px -48px;display:flex;align-items:center;justify-content:center;text-align:center;position:relative;background:linear-gradient(90deg,rgba(4,12,26,.93),rgba(5,14,30,.74),rgba(5,12,25,.91)),url('https://images.unsplash.com/photo-1541746972996-4e0b0f43e02a?w=1920&h=1080&fit=crop&auto=format') center/cover}.start-inner{width:min(760px,90vw);position:relative;z-index:2}.start-badge{display:inline-block;color:#22d4ef;border:1px solid #17738a;background:#083046aa;border-radius:999px;padding:8px 16px;font:600 11px 'IBM Plex Mono';letter-spacing:.08em}.start-hero h1{color:#fff;font-size:clamp(48px,6vw,78px);line-height:1.15;margin:36px 0 28px;letter-spacing:-.05em}.start-hero h1 span{background:linear-gradient(90deg,#25cbe5,#9a7af4);-webkit-background-clip:text;color:transparent}.start-desc{font-size:17px;line-height:1.8;color:#a9b5ca}.start-actions{display:grid;grid-template-columns:1fr 1fr;gap:16px;width:min(540px,90%);margin:35px auto 0;text-align:left}.role-card{padding:18px 22px;border-radius:14px;text-decoration:none!important}.role-card.hr{border:1px solid #1683a0;background:#062b3ddd;color:#23d5ef!important}.role-card.dev{border:1px solid #6c59a4;background:#211b42dd;color:#ac8cff!important}.role-card small{display:block;font:600 10px 'IBM Plex Mono';letter-spacing:.12em;margin-bottom:7px}.role-card strong{font-size:18px}.role-card b{float:right;font-size:21px}hr{border-color:#202a3f!important}
@media(max-width:800px){.tg-nav{grid-template-columns:1fr auto;margin:0 -16px 20px;padding:0 16px}.tg-health{display:none}.tg-links{gap:0}.tg-links a{padding:9px 7px;font-size:10px}.metric-strip,.champ-grid,.decision-row,.start-actions{grid-template-columns:1fr}.metric-strip>div{border-right:0;border-bottom:1px solid #25314b;padding:10px 18px}.start-hero h1{font-size:48px}.block-container{padding:0 16px 30px!important}.start-hero{margin:-20px -16px -30px}}
</style>
        """,
        unsafe_allow_html=True,
    )


def top_navigation(active: str) -> None:
    if active in {"salary", "team", "people", "stability"}:
        links = [("start", "/page01", "역할 선택"), ("salary", "/page02", "연봉 협상"), ("team", "/page03", "팀 구성"), ("people", "/page04", "인사 지원"), ("stability", "/page05", "구조 안정도")]
    elif active == "models":
        links = [("start", "/page01", "역할 선택"), ("models", "/page06", "ML / DL 성능평가")]
    else:
        links = [("salary", "/page02", "인사팀 담당자"), ("models", "/page06", "개발 관리자")]
    items = "".join(f'<a class="{"active" if key == active else ""}" href="{url}" target="_self">{label}</a>' for key, url, label in links)
    st.markdown(f'<nav class="tg-nav"><a class="tg-brand" href="/page01" target="_self">TalentGuard<span>AI</span></a><div class="tg-links">{items}</div><div class="tg-health">내부 서버 정상</div></nav>', unsafe_allow_html=True)


def page_header(kicker: str, title: str, description: str) -> None:
    st.markdown(f'<header class="page-header"><div class="eyebrow">{kicker}</div><h1>{title}</h1><p>{description}</p></header>', unsafe_allow_html=True)


def style_plotly_chart(fig: go.Figure, height: int = 320) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=35, r=20, t=35, b=35), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#dbe4f5", family="Noto Sans KR"), legend=dict(orientation="h", y=1.12))
    fig.update_xaxes(gridcolor="#152038", zerolinecolor="#27334d")
    fig.update_yaxes(gridcolor="#152038", zerolinecolor="#27334d")
    return fig
