"""TalentShield Streamlit 공통 Liquid Glass UI."""

from __future__ import annotations

from html import escape

import plotly.graph_objects as go
import streamlit as st

COLORS = {"bg": "#F2F2F7", "panel": "rgba(255,255,255,.5)", "line": "rgba(60,60,67,.18)", "text": "#000000", "muted": "rgba(60,60,67,.6)", "cyan": "#007AFF", "violet": "#5856D6", "amber": "#FF9500", "green": "#34C759", "rose": "#FF3B30"}


def apply_page_style() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;600;700&display=swap');
:root{color-scheme:light;--blue:#007AFF;--purple:#5856D6;--green:#34C759;--orange:#FF9500;--red:#FF3B30;--secondary:rgba(60,60,67,.60);--separator:rgba(60,60,67,.18);--glass:rgba(255,255,255,.50)}
html{font-size:19px}html,body,[class*="css"]{font-family:-apple-system,BlinkMacSystemFont,'SF Pro Text','Noto Sans KR','Helvetica Neue',Arial,sans-serif}body,.stApp{background:#F2F2F7;color:#000}.stApp:before{content:'';position:fixed;inset:0;pointer-events:none;background:radial-gradient(circle at 12% 5%,rgba(0,122,255,.22),transparent 30%),radial-gradient(circle at 96% 38%,rgba(88,86,214,.18),transparent 29%),radial-gradient(circle at 22% 92%,rgba(52,199,89,.14),transparent 28%),radial-gradient(circle at 2% 65%,rgba(255,149,0,.11),transparent 23%);filter:blur(28px);z-index:0}
[data-testid="stHeader"],[data-testid="stSidebar"],#MainMenu,footer{display:none!important}.stAppViewContainer>.main{overflow:visible}.block-container{max-width:1180px;padding:0 24px 122px!important;position:relative;z-index:1}
.tg-nav{height:44px;margin:0 -24px 20px;padding:0 16px;display:grid;grid-template-columns:1fr auto 1fr;align-items:center;position:sticky;top:0;z-index:999;background:rgba(255,255,255,.48);backdrop-filter:blur(24px) saturate(180%);-webkit-backdrop-filter:blur(24px) saturate(180%);border-bottom:.5px solid rgba(255,255,255,.75);box-shadow:inset 0 -1px 0 rgba(60,60,67,.12),0 1px 0 rgba(255,255,255,.65)}
.tg-home{justify-self:start;color:var(--blue)!important;text-decoration:none!important;font-size:19px;min-width:82px}.tg-home:before{content:'‹';font-size:28px;line-height:0;vertical-align:-2px;margin-right:4px}.tg-title{font-size:18px;font-weight:600;letter-spacing:-.02em;color:#000}.tg-brand{font-weight:700}.tg-brand span{color:var(--blue)}.tg-role{justify-self:end;display:flex;align-items:center;gap:6px;min-width:82px;justify-content:flex-end;color:var(--blue);font-size:13px;font-weight:700}.tg-role:before{content:'✓';display:grid;place-items:center;width:22px;height:22px;border-radius:6px;background:var(--blue);color:#fff}.tg-role b{background:rgba(0,122,255,.10);padding:2px 6px;border-radius:5px}.tg-health{justify-self:end;color:var(--secondary);font-size:14px}.tg-health:before{content:'';display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--green);margin-right:6px}
.tg-tabs{position:fixed;left:50%;bottom:18px;transform:translateX(-50%);z-index:1000;width:min(620px,calc(100vw - 32px));display:flex;gap:2px;padding:6px;border:.5px solid rgba(255,255,255,.78);border-radius:26px;background:rgba(255,255,255,.52);backdrop-filter:blur(26px) saturate(190%);-webkit-backdrop-filter:blur(26px) saturate(190%);box-shadow:inset 0 1px 0 rgba(255,255,255,.96),0 10px 34px rgba(31,38,61,.16)}.tg-tabs a{flex:1;display:flex;flex-direction:column;align-items:center;gap:3px;padding:8px 4px 7px;border-radius:20px;text-decoration:none!important;color:rgba(60,60,67,.52)!important;font-size:13px;white-space:nowrap;transition:.2s}.tg-tabs a .tab-icon{font-size:21px;line-height:21px}.tg-tabs a.active{color:var(--blue)!important;background:rgba(255,255,255,.72);box-shadow:inset 0 1px 0 rgba(255,255,255,.95),0 1px 7px rgba(0,0,0,.09);font-weight:600}.tg-tabs.dev{max-width:270px}.tg-tabs a:hover{background:rgba(255,255,255,.38)}
.page-header{padding:18px 4px 20px;margin-bottom:18px}.eyebrow{display:inline-block;color:var(--blue);font-size:14px;font-weight:700;letter-spacing:.055em;padding:5px 10px;border-radius:999px;background:rgba(0,122,255,.08);margin-bottom:12px}.page-header h1{font-size:clamp(31px,4vw,37px);line-height:1.08;letter-spacing:-.04em;margin:0 0 7px;color:#000}.page-header p,.section-copy{color:var(--secondary);font-size:16px;margin:0}.section-heading{display:flex;align-items:center;gap:10px;margin:4px 4px 8px}.section-heading h2{font-size:22px;letter-spacing:-.025em;margin:0;color:#000}.chip{padding:4px 8px;border-radius:7px;color:var(--accent,var(--blue));background:color-mix(in srgb,var(--accent,var(--blue)) 10%,transparent);font-size:13px;font-weight:700}.panel-title{color:var(--blue);font-size:13px;font-weight:700;letter-spacing:.07em;margin-bottom:8px}.big-number{font:700 clamp(47px,5vw,71px) -apple-system,BlinkMacSystemFont,sans-serif;text-align:center;letter-spacing:-.045em;margin:20px 0 4px}.center-note{text-align:center;font-weight:500;margin-bottom:14px;font-size:16px}
.metric-strip{display:grid;grid-template-columns:repeat(4,1fr);background:var(--glass);border:.5px solid rgba(255,255,255,.75);border-radius:16px;padding:18px 6px;margin:16px 0;backdrop-filter:blur(24px) saturate(180%);box-shadow:inset 0 1px 0 rgba(255,255,255,.92),0 8px 26px rgba(31,38,61,.08)}.metric-strip>div{padding:0 20px;border-right:.5px solid var(--separator)}.metric-strip>div:last-child{border:0}.metric-strip small{display:block;color:var(--secondary);font-size:14px;margin-bottom:4px}.metric-strip strong{color:#000;font-size:30px;line-height:1;font-weight:700;letter-spacing:-.035em}
[data-testid="stVerticalBlockBorderWrapper"]{border:.5px solid rgba(255,255,255,.78)!important;border-radius:16px!important;background:rgba(255,255,255,.50)!important;backdrop-filter:blur(24px) saturate(180%);box-shadow:inset 0 1px 0 rgba(255,255,255,.92),0 8px 25px rgba(31,38,61,.07)}[data-testid="stDataFrame"]{border:.5px solid var(--separator);border-radius:10px;overflow:hidden;background:rgba(255,255,255,.55)}[data-testid="stDataFrame"] *{font-size:15px}[data-baseweb="select"]>div,[data-baseweb="input"]>div,[data-baseweb="base-input"]{background:rgba(255,255,255,.65)!important;border-color:rgba(60,60,67,.20)!important;color:#000!important;border-radius:10px!important}label,.stMarkdown p,.stCaptionContainer{color:var(--secondary)!important}.stMarkdown h3{color:#000}div[data-testid="stMetric"]{background:rgba(255,255,255,.52);border:.5px solid rgba(255,255,255,.76);padding:16px 18px;border-radius:14px;box-shadow:inset 0 1px 0 rgba(255,255,255,.9),0 5px 18px rgba(31,38,61,.06)}div[data-testid="stMetricValue"]{color:#000;font-weight:700}[data-testid="stProgress"]>div>div>div>div{background:linear-gradient(90deg,var(--blue),var(--purple))!important}
.champ-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.champ{border:.5px solid rgba(255,255,255,.78);border-radius:16px;padding:20px;background:rgba(255,255,255,.50);backdrop-filter:blur(24px);box-shadow:inset 0 1px 0 rgba(255,255,255,.92),0 8px 24px rgba(31,38,61,.07)}.champ .tag{color:var(--accent);font-size:12px;font-weight:700;background:color-mix(in srgb,var(--accent) 10%,transparent);display:inline-block;padding:5px 8px;border-radius:6px}.champ h3{color:var(--accent);font-size:24px;margin:14px 0}.champ-stats{display:grid;grid-template-columns:repeat(3,1fr);gap:6px}.champ-stats div{text-align:center;background:rgba(118,118,128,.08);padding:12px 5px;border-radius:9px;color:var(--accent);font-weight:700;font-size:16px}.champ-stats small{display:block;color:var(--secondary);font-size:12px;font-weight:400;margin-top:6px}.champ ul{list-style:none;padding:6px 0 0;color:var(--secondary);font-size:14px;line-height:1.9}.decision{border:.5px solid rgba(0,122,255,.20);background:rgba(0,122,255,.07);border-radius:16px;padding:19px;margin-top:12px}.decision h3{margin:0 0 14px;font-size:19px;color:#000}.decision-row{display:grid;grid-template-columns:1fr 1fr;gap:8px}.decision-row span{background:rgba(255,255,255,.48);padding:12px;border-radius:9px;color:rgba(0,80,170,.82);font-size:14px}
.start-hero{min-height:calc(100vh - 64px);margin:-20px -24px -122px;display:flex;align-items:center;justify-content:center;text-align:center;position:relative;overflow:hidden}.start-hero:before,.start-hero:after{content:'';position:absolute;border-radius:50%;filter:blur(60px)}.start-hero:before{width:520px;height:520px;top:2%;left:8%;background:radial-gradient(circle,rgba(0,122,255,.16),transparent 68%)}.start-hero:after{width:500px;height:500px;right:5%;bottom:5%;background:radial-gradient(circle,rgba(88,86,214,.13),transparent 68%)}.start-inner{width:min(660px,92vw);position:relative;z-index:2;padding:52px 0 84px}.start-badge{display:inline-flex;color:var(--secondary);border:.5px solid rgba(255,255,255,.76);background:rgba(255,255,255,.42);backdrop-filter:blur(24px);border-radius:999px;padding:7px 14px;font-size:14px;box-shadow:inset 0 1px 0 rgba(255,255,255,.95)}.start-badge::first-letter{color:var(--green)}.start-hero h1{color:#000;font-size:clamp(45px,6vw,64px);line-height:1.08;margin:34px 0 18px;letter-spacing:-.055em}.start-hero h1 span{color:var(--blue)}.start-desc{font-size:18px!important;line-height:1.65;color:var(--secondary)!important}.start-actions{display:grid;grid-template-columns:1fr 1fr;gap:12px;width:min(540px,92%);margin:38px auto 0;text-align:left}.role-card{padding:23px 25px;border:.5px solid rgba(255,255,255,.78);border-radius:16px;text-decoration:none!important;background:rgba(255,255,255,.42);backdrop-filter:blur(24px) saturate(180%);box-shadow:inset 0 1px 0 rgba(255,255,255,.95),0 8px 24px rgba(31,38,61,.07);transition:.2s}.role-card:hover{transform:translateY(-2px);background:rgba(255,255,255,.65)}.role-card.hr{color:var(--blue)!important}.role-card.dev{color:var(--purple)!important}.role-card small{display:block;font-size:13px;font-weight:700;letter-spacing:.09em;margin-bottom:6px}.role-card strong{font-size:20px;color:#000}.role-card b{float:right;font-size:25px}.role-card p{font-size:14px!important;line-height:1.5;margin:10px 0 0;color:var(--secondary)!important}.start-stats{display:grid;grid-template-columns:repeat(3,1fr);width:min(540px,92%);margin:26px auto 0;border:.5px solid rgba(255,255,255,.78);border-radius:16px;background:rgba(255,255,255,.40);backdrop-filter:blur(24px);box-shadow:inset 0 1px 0 rgba(255,255,255,.92);overflow:hidden}.start-stats div{padding:17px 10px;border-right:.5px solid var(--separator)}.start-stats div:last-child{border:0}.start-stats strong{display:block;color:#000;font-size:23px}.start-stats small{color:var(--secondary);font-size:13px}hr{border-color:var(--separator)!important}
@media(max-width:800px){.block-container{padding:0 16px 112px!important}.tg-nav{margin:0 -16px 16px}.tg-title{font-size:16px}.metric-strip,.champ-grid,.decision-row,.start-actions{grid-template-columns:1fr}.metric-strip>div{border-right:0;border-bottom:.5px solid var(--separator);padding:11px 18px}.start-hero{margin:-16px -16px -112px}.start-hero h1{font-size:45px}.start-stats{grid-template-columns:1fr}.start-stats div{border-right:0;border-bottom:.5px solid var(--separator)}.tg-tabs{bottom:10px}.tg-tabs a{font-size:12px}.tg-home,.tg-role{min-width:65px}}
</style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
<style>
.glass-card{background:rgba(255,255,255,.52);border:.5px solid rgba(255,255,255,.82);border-radius:12px;backdrop-filter:blur(24px) saturate(180%);box-shadow:inset 0 1px 0 rgba(255,255,255,.92),0 7px 22px rgba(31,38,61,.055)}
.employee-strip{margin:0;border-top:.5px solid var(--separator);overflow:hidden}
.employee-strip-top{display:flex;align-items:center;justify-content:space-between;padding:4px 16px;border-bottom:.5px solid var(--separator)}
.employee-strip-top span{font-size:15px;font-weight:600;color:#000}
.employee-strip-top .select-wrap{min-width:240px}
.employee-strip-top [data-testid="stSelectbox"]{margin-bottom:0!important}
.employee-strip-top [data-baseweb="select"]>div{background:transparent!important;border:none!important;box-shadow:none!important;padding-right:0!important}
.employee-strip-top [data-baseweb="select"] *{color:var(--blue)!important;font-weight:600!important;font-size:16px!important;text-align:right}
.employee-strip-bottom{display:grid;grid-template-columns:1fr 1fr}
.employee-strip-bottom>div{display:flex;justify-content:space-between;align-items:center;padding:10px 16px;font-size:15px}
.employee-strip-bottom>div:first-child{border-right:.5px solid var(--separator)}
.employee-strip-bottom span{color:var(--secondary)}
.employee-strip-bottom b{color:#000;font-weight:500}
.risk-card{padding:18px 10px 0;text-align:center}.risk-card svg{width:100%;max-width:190px}.risk-pill{padding:10px;border-radius:9px;font-size:15px;font-weight:700}.talent-row{display:flex;justify-content:space-between;margin:18px -10px 0;padding:14px 16px;border-top:.5px solid var(--separator);font-size:15px}.talent-row b{color:var(--blue);font-size:18px}.ai-note{padding:15px;font-size:15px;line-height:1.65;color:var(--secondary)}.ai-note b{color:var(--red)}
.feature-list{padding:0 14px}.feature-row{padding:10px 0;border-bottom:.5px solid var(--separator)}.feature-row:last-child{border:0}.feature-row>div{display:flex;align-items:baseline;gap:6px;font-size:15px}.feature-row b{font-weight:500}.feature-row span{font-size:13px;color:var(--secondary);flex:1}.feature-row strong{font-size:15px}.feature-row i,.mini-bar{display:block;height:3px;border-radius:4px;background:rgba(60,60,67,.11);overflow:hidden;margin-top:7px}.feature-row em,.mini-bar i{display:block;height:100%;border-radius:4px}
.dash-metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:12px}.dash-metric{background:rgba(255,255,255,.52);border:.5px solid rgba(255,255,255,.82);border-radius:12px;padding:18px;min-height:104px}.dash-metric small{display:block;color:var(--secondary);font-size:13px}.dash-metric strong{display:block;font-size:31px;line-height:1;margin:10px 0 5px;letter-spacing:-.04em}.dash-metric span{display:block;color:var(--secondary);font-size:13px}
.project-card{overflow:hidden}.project-card>div,.project-card p{display:flex;justify-content:space-between;align-items:center;padding:13px 15px;margin:0;border-bottom:.5px solid var(--separator);font-size:14px}.project-card p:last-child{border:0}.project-card>div>b{font-size:15px}.project-card p span{color:var(--secondary)}.project-card p b{font-weight:500}.priority{padding:3px 6px;border:1px solid rgba(255,149,0,.45);border-radius:5px;color:var(--orange);font-size:12px}.member-list{overflow:hidden}.member-row{display:grid;grid-template-columns:36px minmax(100px,1fr) auto auto;gap:11px;align-items:center;padding:12px;border-bottom:.5px solid var(--separator)}.member-row:last-child{border:0}.member-row>i,.person-cell>i{display:grid;place-items:center;width:30px;height:30px;border-radius:9px;background:rgba(255,149,0,.10);color:var(--orange);font-style:normal;font-size:13px;font-weight:700}.member-row>i{width:36px;height:36px;border-radius:10px;font-size:14px}.member-row div b,.member-row div span{display:block}.member-row div b{font-size:16px;line-height:1.25}.member-row div span{font-size:14px;line-height:1.35;color:var(--secondary);margin-top:3px}.member-row strong{font-size:15px;line-height:1.2;text-align:right}.member-row strong small{display:block;color:var(--secondary);font-size:12px;font-weight:400;margin-top:2px}.member-row>em{font-style:normal;color:var(--blue);font-size:15px!important}.team-warning{padding:14px;border:1px solid rgba(255,149,0,.28);border-radius:10px;background:rgba(255,149,0,.04);font-size:14px;color:var(--secondary)}.team-warning b{color:var(--orange)}
.formula{display:inline-block;margin:0 0 14px;padding:9px 13px;border:.5px solid rgba(0,122,255,.18);border-radius:10px;background:rgba(255,255,255,.42);color:var(--secondary);font-size:14px}.formula b{color:var(--blue)}.formula-bar{display:flex;align-items:center;gap:14px;margin-bottom:12px}.formula-inline{display:inline-flex;align-items:center;height:36px;padding:0 16px;border:.5px solid rgba(0,122,255,.16);border-radius:12px;background:rgba(255,255,255,.50);backdrop-filter:blur(20px);color:var(--secondary);font-size:15px;box-shadow:inset 0 1px 0 rgba(255,255,255,.90)}.formula-inline b{color:var(--blue);margin-right:4px}.ranking-wrap{overflow:auto}.ranking{width:100%;border-collapse:collapse;font-size:13px}.ranking th{text-align:left;color:var(--secondary);font-weight:600;padding:10px 12px;border-bottom:.5px solid rgba(60,60,67,.28);white-space:nowrap}.ranking td{padding:9px 12px;border-bottom:.5px solid var(--separator);vertical-align:middle}.ranking tr:last-child td{border:0}.person-cell{display:flex;align-items:center;gap:10px;min-width:150px}.person-cell>i{background:rgba(52,199,89,.08);color:var(--green)}.person-cell b,.person-cell small{display:block}.person-cell b{font-size:14px}.person-cell small{color:var(--secondary);margin-top:2px}.ranking .mini-bar{width:100%;min-width:86px}.ranking td>small{color:var(--secondary)}.ranking .stable{font-size:18px;font-weight:700}.level{display:inline-block;padding:3px 7px;border:1px solid;border-radius:999px}.action{display:inline-flex;align-items:center;padding:5px 8px;margin:2px 1px;border:1px solid transparent;border-radius:7px;font-size:14px;font-weight:600;line-height:1.15;white-space:nowrap}.action.promote{color:#0066D6;background:rgba(0,122,255,.11);border-color:rgba(0,122,255,.22)}.action.transfer{color:#4B49B9;background:rgba(88,86,214,.10);border-color:rgba(88,86,214,.20)}.action.review{color:#C9342B;background:rgba(255,59,48,.09);border-color:rgba(255,59,48,.20)}
.alert-card{display:flex;align-items:center;justify-content:space-between;padding:8px 14px;margin-bottom:6px;background:rgba(255,255,255,.52);border:.5px solid rgba(255,255,255,.82);border-radius:10px;backdrop-filter:blur(24px) saturate(180%);-webkit-backdrop-filter:blur(24px) saturate(180%);box-shadow:inset 0 1px 0 rgba(255,255,255,.92),0 4px 12px rgba(31,38,61,.04)}
.alert-card:last-child{margin-bottom:10px}
.alert-message{display:flex;align-items:center;gap:8px;color:#000;font-size:14px;line-height:1.4}
.alert-message i{width:7px;height:7px;flex:0 0 7px;border-radius:50%;background:var(--red)}
.alert-message strong{color:var(--red)}
.alert-dismiss{color:var(--secondary)!important;font-size:13px!important;font-weight:500!important;text-decoration:none!important;padding:3px 8px;border-radius:6px;transition:.15s;white-space:nowrap;line-height:1.2}
.alert-dismiss:hover{color:var(--blue)!important;background:rgba(0,122,255,.08)}
.alert-card-anchor{display:none}
div[data-testid="stElementContainer"]:has(.alert-card-anchor){display:none!important;margin:0!important;padding:0!important;height:0!important}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.alert-card-anchor){padding:0!important;overflow:hidden;border-color:rgba(255,255,255,.82)!important;border-radius:10px!important;background:rgba(255,255,255,.52)!important;box-shadow:inset 0 1px 0 rgba(255,255,255,.92),0 4px 12px rgba(31,38,61,.04)!important;margin-bottom:6px!important}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.alert-card-anchor)>div{gap:0!important;padding:0!important}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.alert-card-anchor) [data-testid="stVerticalBlock"]{gap:0!important;padding:0!important}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.alert-card-anchor) [data-testid="stElementContainer"]{margin:0!important;padding:0!important}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.alert-card-anchor) [data-testid="stHorizontalBlock"]{align-items:center;min-height:36px;padding:4px 12px;gap:8px!important}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.alert-card-anchor) [data-testid="stHorizontalBlock"]>div:last-child [data-testid="stButton"]{display:flex;justify-content:flex-end}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.alert-card-anchor) button{width:auto!important;min-width:0!important;min-height:24px!important;height:24px!important;padding:0 6px!important;border:0!important;background:transparent!important;color:var(--secondary)!important;font-size:13px!important;font-weight:500!important;line-height:24px!important;box-shadow:none!important}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.alert-card-anchor) button:hover{color:var(--blue)!important;background:rgba(0,122,255,.07)!important}
.dept-list{overflow:hidden}.dept-list>div{display:grid;grid-template-columns:8px 1fr auto;gap:9px;align-items:center;padding:12px 13px;border-bottom:.5px solid var(--separator)}.dept-list>div:last-child{border:0}.dept-list>div>i{width:7px;height:7px;border-radius:50%}.dept-list span b,.dept-list span small,.dept-list strong small{display:block}.dept-list span b{font-size:14px}.dept-list span small{font-size:12px;color:var(--secondary);margin-top:2px}.dept-list strong{font-size:14px;text-align:right}.dept-list strong small{font-size:11px;font-weight:500;margin-top:2px}
.perf-table{overflow:auto;margin-bottom:14px}.perf-table table{width:100%;border-collapse:collapse;font-size:13px}.perf-table th{padding:10px 12px;text-align:left;color:var(--secondary);font-weight:500;border-bottom:.5px solid rgba(60,60,67,.27)}.perf-table td{padding:11px 12px;border-bottom:.5px solid var(--separator)}.perf-table tr:last-child td{border:0}.perf-table .best-cell{color:var(--green);font-weight:700}.model-type{padding:3px 6px;border-radius:5px;background:rgba(0,122,255,.09);color:var(--blue);font-size:11px;font-weight:600}.model-type.dl{background:rgba(88,86,214,.10);color:var(--purple)}.model-type.ml{background:rgba(0,122,255,.09);color:var(--blue)}.champion{overflow:hidden}.champion>small{display:block;padding:13px 15px 0;font-size:12px;font-weight:700}.champion h3{margin:7px 15px 13px;color:#000}.champion>div{display:grid;grid-template-columns:repeat(3,1fr);border-top:.5px solid var(--separator)}.champion>div>div{padding:11px 15px;border-right:.5px solid var(--separator)}.champion>div>div:last-child{border:0}.champion div small,.champion div b{display:block}.champion div small{font-size:12px;color:var(--secondary)}.champion div b{font-size:15px;margin-top:4px}.adopted{overflow:hidden;margin-top:14px}.adopted-head{display:flex;align-items:center;padding:17px}.adopted-head>i{display:grid;place-items:center;width:40px;height:40px;border-radius:11px;background:rgba(88,86,214,.09);color:var(--purple);font-size:27px;font-style:normal}.adopted-head>span{margin-left:13px}.adopted-head>span small{color:var(--purple);font-size:12px}.adopted-head h3{margin:5px 0 0}.adopted-head h3 em{font-style:normal;font-size:11px;color:var(--purple);background:rgba(88,86,214,.08);padding:3px 6px;border-radius:4px;margin-left:7px}.adopted-head>div{margin-left:auto;display:flex;gap:25px}.adopted-head>div>div{text-align:right}.adopted-head>div b,.adopted-head>div small{display:block}.adopted-head>div b{color:var(--purple);font-size:19px}.adopted-head>div small{font-size:11px;color:var(--secondary)}.reason-grid{display:grid;grid-template-columns:repeat(4,1fr);border-top:.5px solid var(--separator)}.reason-grid>div{padding:15px;border-right:.5px solid var(--separator)}.reason-grid>div:last-child{border:0}.reason-grid i{display:inline-block;width:6px;height:6px;border-radius:50%;margin-right:7px}.reason-grid b{font-size:12px}.reason-grid strong,.reason-grid span{display:block}.reason-grid strong{font-size:16px;margin:8px 0}.reason-grid span{font-size:11px;color:var(--secondary);line-height:1.5}.adopted-bars{padding:5px 14px}.confusion{padding-top:12px;overflow:hidden}.confusion>small{display:block;padding:0 14px 10px;color:var(--secondary)}.confusion>div{display:grid;grid-template-columns:1fr 1fr;border-top:.5px solid var(--separator)}.confusion span{padding:17px;border-right:.5px solid var(--separator);border-bottom:.5px solid var(--separator)}.confusion span:nth-child(even){border-right:0}.confusion span b,.confusion span strong,.confusion span small{display:block}.confusion span b,.confusion span small{font-size:11px}.confusion span strong{font-size:23px;margin:4px 0}.confusion span small{color:var(--secondary)}.confusion .tp{color:var(--green)}.confusion .fp{color:var(--orange)}.confusion .fn{color:var(--red)}.confusion .tn{color:var(--blue)}
.segmented-nav{display:inline-flex;align-items:stretch;gap:2px;padding:3px;border-radius:10px;background:rgba(118,118,128,.14);backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);box-shadow:inset 0 .5px .5px rgba(0,0,0,.04);max-width:100%;min-height:36px;overflow-x:auto}
.segmented-nav.stretch{display:flex;width:50%}.segmented-nav.stretch a{flex:1}
.segmented-nav a{display:flex;align-items:center;justify-content:center;gap:7px;min-height:30px;padding:5px 14px;border-radius:8px;color:rgba(60,60,67,.60)!important;text-decoration:none!important;font-size:15px;font-weight:500;line-height:1;white-space:nowrap;transition:background .18s,color .18s,box-shadow .18s}
.segmented-nav a:hover{background:rgba(255,255,255,.35);color:#000!important}.segmented-nav a.active{background:#fff;color:#000!important;font-weight:700;box-shadow:0 1px 3px rgba(0,0,0,.12),0 .5px .5px rgba(0,0,0,.04)}
.segmented-nav .nav-badge{padding:3px 5px;border:1px solid currentColor;border-radius:4px;font-size:12px;font-weight:700;line-height:1}.segmented-nav .nav-badge.high{color:var(--orange);background:rgba(255,149,0,.06)}.segmented-nav .nav-badge.critical{color:var(--red);background:rgba(255,59,48,.06)}.segmented-nav .nav-badge.normal{color:var(--blue);background:rgba(0,122,255,.06)}
div[data-testid="stElementContainer"]:has(.segmented-nav){margin-bottom:12px}
.employee-picker-anchor{display:none}.employee-picker-label{display:flex;align-items:center;height:36px;color:#000;font-size:15px;font-weight:500;padding-left:2px}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.employee-picker-anchor){padding:0!important;border-radius:12px!important;overflow:hidden}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.employee-picker-anchor)>div{gap:0!important;padding:0!important}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.employee-picker-anchor) [data-testid="stHorizontalBlock"]{align-items:center;padding:4px 14px!important;gap:8px!important}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.employee-picker-anchor) [data-testid="stSelectbox"]{margin:0!important}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.employee-picker-anchor) [data-testid="stSelectbox"]>div>div{min-height:34px!important;background:transparent!important;border:0!important;box-shadow:none!important}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.employee-picker-anchor) [data-baseweb="select"] *{color:var(--blue)!important;font-size:15px!important;font-weight:600!important;text-align:right}
[data-testid="stSelectbox"]>div>div{background:rgba(255,255,255,.58)!important;border:.5px solid rgba(255,255,255,.82)!important;border-radius:14px!important;backdrop-filter:blur(24px) saturate(180%)!important;box-shadow:inset 0 1px 0 rgba(255,255,255,.95),0 4px 14px rgba(31,38,61,.05)!important}
.js-plotly-plot .plotly .modebar{display:none!important}
/* Readability pass: keep the layout density while lifting small supporting text. */
[data-testid="stCaptionContainer"] p{font-size:16px!important;line-height:1.45!important}
.page-header p,.section-copy{font-size:17px!important}.segmented-nav a{font-size:16px}
.project-card>div,.project-card p,.team-warning,.ai-note{font-size:15px}.project-card>div>b{font-size:16px}
.dash-metric small,.dash-metric span{font-size:14px}.feature-row>div{font-size:16px}.feature-row span{font-size:14px}.feature-row strong{font-size:16px}
.ranking,.perf-table table{font-size:14px}.ranking th,.perf-table th{font-size:14px}.person-cell b{font-size:15px}.person-cell small,.ranking td>small{font-size:13px}
.alert-list>div{font-size:14px}.champion>small,.champion div small,.adopted-head>span small,.reason-grid b{font-size:13px}.reason-grid span{font-size:12px}
.formula-align-right{display:flex;width:100%;justify-content:flex-end}.formula-inline{font-size:16px}.risk-pill,.talent-row{font-size:16px}.panel-title{font-size:14px}
@media(max-width:1100px){.segmented-nav.stretch{width:100%}}
@media(max-width:800px){.employee-strip,.dash-metrics,.reason-grid{grid-template-columns:1fr}.formula-bar{flex-direction:column;align-items:flex-start}.ranking{min-width:900px}.dept-list>div{padding:8px}.adopted-head>div{display:none}}
.block-container{max-width:none;padding:0 12px 94px!important}.tg-nav{height:38px;margin:0 -12px 10px;padding:0 12px}.page-header{padding:16px 4px 10px;margin-bottom:8px}.eyebrow{display:none}.page-header h1{font-size:clamp(31px,3vw,37px);margin-bottom:5px}.page-header p{font-size:15px}.start-hero{min-height:calc(100vh - 48px);margin:-10px -12px -94px}
@media(max-width:800px){.block-container{padding:0 10px 100px!important}.tg-nav{margin:0 -10px 10px}.start-hero{margin:-10px -10px -100px}}
</style>
        """,
        unsafe_allow_html=True,
    )


def top_navigation(active: str) -> None:
    titles = {"salary": "연봉협상 지원", "team": "팀 구성 지원", "people": "인사 지원", "stability": "안정도 모니터링", "models": "ML/DL 성능평가"}
    if active == "start":
        st.markdown('<nav class="tg-nav"><span></span><div class="tg-title tg-brand">Talent<span>Shield</span></div><div class="tg-health">시스템 정상</div></nav>', unsafe_allow_html=True)
        return
    is_hr = active in {"salary", "team", "people", "stability"}
    role = "HR" if is_hr else "DEV"
    st.markdown(f'<nav class="tg-nav"><a class="tg-home" href="/page01" target="_self">홈</a><div class="tg-title">{titles[active]}</div><div class="tg-role"><b>{role}</b></div></nav>', unsafe_allow_html=True)
    links = ([("salary", "/page02", "◎", "연봉협상"), ("team", "/page03", "♙", "팀 구성"), ("people", "/page04", "♜", "인사 지원"), ("stability", "/page05", "◇", "안정도")] if is_hr else [("models", "/page06", "▣", "ML/DL 성능평가")])
    items = "".join(f'<a class="{"active" if key == active else ""}" href="{url}" target="_self"><span class="tab-icon">{icon}</span><span>{label}</span></a>' for key, url, icon, label in links)
    st.markdown(f'<nav class="tg-tabs {"" if is_hr else "dev"}">{items}</nav>', unsafe_allow_html=True)


def page_header(kicker: str, title: str, description: str) -> None:
    st.markdown(f'<header class="page-header"><div class="eyebrow">{kicker}</div><h1>{title}</h1><p>{description}</p></header>', unsafe_allow_html=True)


def segmented_nav(
    options: list[str],
    param: str,
    *,
    default: int = 0,
    badges: list[tuple[str, str] | None] | None = None,
    stretch: bool = False,
) -> str:
    """Render a stable, link-backed segmented control and return its selection."""
    raw_value = st.query_params.get(param, str(default))
    try:
        selected = int(raw_value)
    except (TypeError, ValueError):
        selected = default
    if selected < 0 or selected >= len(options):
        selected = default

    links = []
    for index, label in enumerate(options):
        badge_html = ""
        if badges and badges[index]:
            badge, tone = badges[index]
            badge_html = f'<span class="nav-badge {escape(tone)}">{escape(badge)}</span>'
        active = "active" if index == selected else ""
        links.append(
            f'<a class="{active}" href="?{escape(param)}={index}" target="_self">'
            f'{badge_html}<span>{escape(label)}</span></a>'
        )
    width_class = " stretch" if stretch else ""
    st.markdown(
        f'<nav class="segmented-nav{width_class}" aria-label="{escape(param)}">'
        + "".join(links)
        + "</nav>",
        unsafe_allow_html=True,
    )
    return options[selected]


def style_plotly_chart(fig: go.Figure, height: int = 320) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=35, r=20, t=35, b=35), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(size=15, color="rgba(60,60,67,.70)", family="Noto Sans KR"), legend=dict(orientation="h", y=1.12), hoverlabel=dict(bgcolor="rgba(255,255,255,.94)", bordercolor="rgba(60,60,67,.18)", font_color="#000"))
    fig.update_xaxes(gridcolor="rgba(60,60,67,.10)", zerolinecolor="rgba(60,60,67,.18)")
    fig.update_yaxes(gridcolor="rgba(60,60,67,.10)", zerolinecolor="rgba(60,60,67,.18)")
    return fig
