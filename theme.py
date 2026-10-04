"""theme.py - visual layer for PakProp AI (CSS + small HTML helpers).

Same emerald palette as before, laid out mobile-first: app bar, hero card,
card grids, bottom dock navigation. No emoji; icons are inline SVG or Material.
"""

LOGO = ('<svg viewBox="0 0 32 32" width="34" height="34"><defs><linearGradient id="pl" x1="0" y1="0" x2="1" y2="1">'
        '<stop offset="0" stop-color="#34d399"/><stop offset="1" stop-color="#059669"/></linearGradient></defs>'
        '<rect x="1.5" y="1.5" width="29" height="29" rx="9" fill="#064e3b"/>'
        '<path d="M7 16 L16 8 L25 16" fill="none" stroke="url(#pl)" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/>'
        '<path d="M10 15.5 V24 H22 V15.5" fill="none" stroke="#a7f3d0" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>'
        '<path d="M14 24 V19 H18 V24" fill="none" stroke="#a7f3d0" stroke-width="2" stroke-linejoin="round"/></svg>')


def appbar(status: str = "") -> str:
    pill = f'<div class="live">{status}</div>' if status else ""
    return (f'<div class="appbar">{LOGO}<div><div class="app-title">PakProp AI</div>'
            f'<div class="app-sub">Price intelligence</div></div>{pill}</div>')


def sec(title: str) -> str:
    return f'<div class="section-header">{title}</div>'


def mcard(label: str, value: str, sub: str = "") -> str:
    s = f'<div class="metric-sub">{sub}</div>' if sub else ""
    return (f'<div class="metric-card"><div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div>{s}</div>')


def grid(cards: list, one: bool = False) -> str:
    return f'<div class="grid{" one" if one else ""}">{"".join(cards)}</div>'


def chips(items: list) -> str:
    return '<div class="chips">' + "".join(f'<span class="chip">{i}</span>' for i in items) + '</div>'


def note(html: str) -> str:
    return f'<div class="note">{html}</div>'


def step(n: int, title: str, body: str) -> str:
    return (f'<div class="step"><div class="step-n">{n}</div><div><div class="step-t">{title}</div>'
            f'<div class="step-b">{body}</div></div></div>')


def row(title: str, right: str, stats: list) -> str:
    st_html = "".join(f'<span>{k} <b>{v}</b></span>' for k, v in stats)
    return (f'<div class="srow"><div class="sr-top"><div class="sr-name">{title}</div>'
            f'<div class="sr-tag">{right}</div></div><div class="sr-stats">{st_html}</div></div>')


def range_bar(low: float, mid: float, high: float, low_s: str, high_s: str) -> str:
    """Visual track: low ... estimate ... high (position of estimate inside the range)."""
    span = max(high - low, 1e-9)
    pos = max(4.0, min(96.0, (mid - low) / span * 100))
    return (f'<div class="rb"><div class="rb-track"><i style="left:{pos:.1f}%"></i></div>'
            f'<div class="rb-lbl"><span>{low_s}</span><span>{high_s}</span></div></div>')


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');
:root{--bg:#f9fafb;--card:rgba(255,255,255,.82);--line:#e5e7eb;--txt:#111827;--mut:#6b7280;
--g9:#064e3b;--g8:#065f46;--g6:#059669;--g5:#10b981;--g4:#34d399;--g2:#a7f3d0;--g1:#d1fae5;--g0:#ecfdf5}
html,body,[data-testid="stApp"]{background:var(--bg)!important;color:var(--txt);font-family:'Inter',system-ui,-apple-system,'Segoe UI',sans-serif;-webkit-tap-highlight-color:transparent}
[data-testid="stApp"]::before{content:"";position:fixed;inset:-20%;z-index:0;pointer-events:none;filter:blur(70px);opacity:.9;
 background:radial-gradient(40% 35% at 12% 12%,rgba(52,211,153,.22),transparent 70%),radial-gradient(35% 40% at 90% 8%,rgba(5,150,105,.14),transparent 70%),
 radial-gradient(45% 40% at 75% 95%,rgba(167,243,208,.35),transparent 70%);animation:aur 20s ease-in-out infinite alternate}
@keyframes aur{to{transform:translate3d(-3%,4%,0) rotate(6deg) scale(1.1)}}
header[data-testid="stHeader"],[data-testid="stToolbar"],[data-testid="stDecoration"],[data-testid="stSidebar"],[data-testid="stSidebarCollapsedControl"],footer,#MainMenu{display:none!important}
.block-container{position:relative;z-index:1;max-width:880px!important;padding:max(.6rem,env(safe-area-inset-top)) .9rem 7.5rem!important}
hr{border-color:var(--line)!important}
/* dock nav */
[data-baseweb="tab-list"]{position:fixed;left:50%;transform:translateX(-50%);bottom:max(10px,env(safe-area-inset-bottom));width:min(94vw,560px);z-index:1000;gap:0!important;
 background:rgba(255,255,255,.85);backdrop-filter:blur(20px) saturate(1.5);border:1px solid var(--line);border-radius:24px;padding:4px;justify-content:space-around;overflow:hidden;
 box-shadow:0 14px 40px rgba(6,78,59,.16),0 0 0 1px rgba(5,150,105,.06)}
[data-baseweb="tab-highlight"]{top:4px!important;bottom:4px!important;height:auto!important;border-radius:19px!important;
 background:linear-gradient(135deg,var(--g1),var(--g2))!important;box-shadow:inset 0 0 0 1px rgba(5,150,105,.35)}
[data-baseweb="tab-border"]{display:none!important}
button[role="tab"]{flex:1 1 0;min-width:0;height:auto;padding:9px 0 8px;color:var(--mut);background:none;transition:.25s}
button[role="tab"][aria-selected="true"]{color:var(--g9)}
button[role="tab"] p{display:flex;flex-direction:column;align-items:center;gap:1px;margin:0;font-size:.62rem;font-weight:600;letter-spacing:.3px}
button[role="tab"] [data-testid="stIconMaterial"]{font-size:1.4rem!important}
/* app bar + hero */
.appbar{display:flex;align-items:center;gap:12px;margin:2px 0 12px}
.app-title{font:800 1.2rem Inter;letter-spacing:-.3px;color:var(--g9)}
.app-sub{font:500 .62rem 'JetBrains Mono',monospace;color:var(--mut);letter-spacing:1.4px;text-transform:uppercase}
.live{margin-left:auto;display:flex;align-items:center;gap:7px;font:700 .6rem 'JetBrains Mono',monospace;letter-spacing:1.2px;color:var(--g6);text-transform:uppercase}
.live::before{content:"";width:8px;height:8px;border-radius:50%;background:var(--g5);box-shadow:0 0 0 0 var(--g5);animation:pu 1.8s infinite}
@keyframes pu{70%{box-shadow:0 0 0 9px transparent}100%{box-shadow:0 0 0 0 transparent}}
.hero{position:relative;overflow:hidden;border-radius:26px;padding:22px 20px 20px;margin:10px 0 16px;color:#fff;
 background:linear-gradient(135deg,var(--g9) 0%,var(--g8) 50%,var(--g6) 100%);box-shadow:0 18px 50px rgba(6,78,59,.28)}
.hero::before{content:"";position:absolute;top:-45%;right:-8%;width:300px;height:300px;border-radius:50%;background:rgba(255,255,255,.06)}
.hero::after{content:"";position:absolute;inset:0;background:linear-gradient(105deg,transparent 40%,rgba(255,255,255,.10) 50%,transparent 60%);transform:translateX(-100%);animation:sh 6s ease-in-out infinite}
@keyframes sh{60%,100%{transform:translateX(100%)}}
.hero>*{position:relative;z-index:1}
.hero-badge{display:inline-block;background:rgba(255,255,255,.15);border:1px solid rgba(255,255,255,.25);border-radius:999px;padding:4px 12px;font:700 .62rem 'JetBrains Mono',monospace;letter-spacing:1.2px;text-transform:uppercase}
.hero h1{font:800 clamp(1.5rem,6vw,2.1rem) Inter;letter-spacing:-.8px;line-height:1.15;margin:12px 0 8px;color:#fff;padding:0}
.hero p{color:var(--g2);font-size:.9rem;line-height:1.55;margin:0;max-width:560px}
.hero-by{margin-top:14px;font:500 .64rem 'JetBrains Mono',monospace;letter-spacing:.8px;color:rgba(167,243,208,.85)}
.hero-by b{color:#d1fae5}
.hero-label{font:700 .62rem 'JetBrains Mono',monospace;letter-spacing:1.4px;text-transform:uppercase;color:var(--g2)}
.hero-price{font:800 clamp(2.2rem,10vw,3.4rem) Inter;letter-spacing:-1.5px;line-height:1.05;margin:8px 0 2px;animation:rise .8s cubic-bezier(.2,.9,.3,1) both}
@keyframes rise{from{opacity:0;transform:translateY(14px) scale(.97);filter:blur(5px)}}
.hero-alt{font:500 .8rem 'JetBrains Mono',monospace;color:var(--g1)}
.hero-note{font-size:.76rem;color:var(--g2);margin-top:10px}
.rb{margin-top:16px}.rb-track{position:relative;height:8px;border-radius:8px;background:rgba(255,255,255,.18)}
.rb-track::before{content:"";position:absolute;inset:0;border-radius:8px;background:linear-gradient(90deg,rgba(167,243,208,.35),rgba(167,243,208,.9),rgba(167,243,208,.35))}
.rb-track i{position:absolute;top:50%;width:18px;height:18px;margin:-9px 0 0 -9px;border-radius:50%;background:#fff;border:4px solid var(--g5);box-shadow:0 2px 10px rgba(0,0,0,.25)}
.rb-lbl{display:flex;justify-content:space-between;margin-top:8px;font:600 .7rem 'JetBrains Mono',monospace;color:var(--g1)}
/* cards */
.grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}.grid.one{grid-template-columns:1fr}
.grid .metric-card:last-child:nth-child(odd){grid-column:1/-1}
.metric-card,.step,.srow,.note{position:relative;background:var(--card);border:1px solid var(--line);border-radius:18px;backdrop-filter:blur(12px);
 transition:transform .3s,border-color .3s,box-shadow .3s;animation:fu .55s cubic-bezier(.2,.9,.3,1) both}
.metric-card:hover,.srow:hover,.step:hover{transform:translateY(-3px);border-color:rgba(5,150,105,.5);box-shadow:0 12px 30px rgba(5,150,105,.14)}
.metric-card:nth-child(2){animation-delay:.06s}.metric-card:nth-child(3){animation-delay:.12s}.metric-card:nth-child(4){animation-delay:.18s}.metric-card:nth-child(n+5){animation-delay:.24s}
@keyframes fu{from{opacity:0;transform:translateY(16px)}}
.metric-card{padding:14px 15px;min-height:84px;overflow:hidden}
.metric-card::before{content:"";position:absolute;left:0;top:0;bottom:0;width:3px;background:linear-gradient(var(--g4),var(--g6))}
.metric-label{font:600 .58rem 'JetBrains Mono',monospace;color:var(--mut);text-transform:uppercase;letter-spacing:1.1px;margin-bottom:6px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.metric-value{font:800 1.3rem Inter;letter-spacing:-.4px;line-height:1.15;color:var(--g6)}
.metric-sub{font-size:.68rem;color:var(--mut);margin-top:4px;line-height:1.35}
.section-header{display:flex;align-items:center;gap:10px;font:700 .72rem 'JetBrains Mono',monospace;color:var(--g9);text-transform:uppercase;letter-spacing:1.8px;margin:26px 0 12px}
.section-header::before{content:"";width:9px;height:9px;transform:rotate(45deg);background:linear-gradient(135deg,var(--g4),var(--g6));flex:none}
.section-header::after{content:"";flex:1;height:1px;background:linear-gradient(90deg,var(--g4),transparent)}
.step{display:flex;gap:14px;align-items:flex-start;padding:14px 16px;margin-bottom:9px}
.step-n{flex:none;width:30px;height:30px;border-radius:10px;display:flex;align-items:center;justify-content:center;font:800 .85rem 'JetBrains Mono',monospace;color:#fff;background:linear-gradient(135deg,var(--g5),var(--g8))}
.step-t{font-weight:700;font-size:.92rem;color:var(--g9)}.step-b{font-size:.82rem;color:var(--mut);line-height:1.55;margin-top:2px}
.note{padding:14px 16px;font-size:.84rem;line-height:1.6;color:var(--g9);background:rgba(236,253,245,.9);border-left:3px solid var(--g6);border-radius:6px 18px 18px 6px}
.srow{padding:12px 14px;margin-bottom:9px}.sr-top{display:flex;justify-content:space-between;align-items:center;gap:8px}
.sr-name{font-weight:700;font-size:.95rem}.sr-tag{font:700 .62rem 'JetBrains Mono',monospace;padding:5px 10px;border-radius:999px;background:var(--g1);color:var(--g8);white-space:nowrap}
.sr-stats{display:flex;flex-wrap:wrap;gap:4px 16px;margin-top:8px;font:500 .7rem 'JetBrains Mono',monospace;color:var(--mut)}.sr-stats b{color:var(--g9)}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin:6px 0 10px}
.chip{font:600 .66rem 'JetBrains Mono',monospace;color:var(--g8);background:var(--g0);border:1px solid var(--g1);border-radius:999px;padding:5px 12px}
.foot{text-align:center;margin-top:30px;font:500 .62rem 'JetBrains Mono',monospace;letter-spacing:1.1px;color:var(--mut);line-height:1.9}
.foot b{color:var(--g8)}
/* streamlit widgets */
[data-testid="stHorizontalBlock"]{flex-wrap:wrap!important;gap:.6rem!important}
[data-testid="stHorizontalBlock"]>[data-testid="stColumn"],[data-testid="stHorizontalBlock"]>[data-testid="column"]{flex:1 1 calc(50% - .6rem)!important;min-width:calc(50% - .6rem)!important;width:auto!important}
input,textarea,[data-baseweb="select"] *{font-size:16px!important}
[data-baseweb="select"]>div,[data-baseweb="input"],[data-baseweb="base-input"]{background:rgba(255,255,255,.9)!important;border-color:var(--line)!important;border-radius:14px!important;min-height:48px}
[data-baseweb="select"]>div:focus-within,[data-baseweb="input"]:focus-within{border-color:var(--g6)!important;box-shadow:0 0 0 3px rgba(5,150,105,.16)!important}
[data-testid="stSelectbox"] label,[data-testid="stNumberInput"] label,[data-testid="stSegmentedControl"] label{color:var(--g8)!important;font:600 .72rem 'JetBrains Mono',monospace!important;letter-spacing:.6px}
[data-testid="stNumberInput"] button{border-radius:10px}
[data-testid="stSegmentedControl"] button{border-radius:999px!important;font-weight:600}
[data-testid="stSegmentedControl"] button[aria-checked="true"],[data-testid="stSegmentedControl"] button[aria-pressed="true"]{background:var(--g6)!important;color:#fff!important;border-color:var(--g6)!important}
.stButton>button{position:relative;overflow:hidden;width:100%;min-height:52px;border:0;border-radius:16px;font:800 .9rem Inter;letter-spacing:.4px;color:#fff;
 background:linear-gradient(135deg,var(--g6),var(--g8));box-shadow:0 10px 28px rgba(5,150,105,.34);transition:.25s}
.stButton>button:hover{transform:translateY(-2px);box-shadow:0 14px 34px rgba(5,150,105,.42);color:#fff}.stButton>button:active{transform:scale(.97)}
[data-testid="stExpander"]{background:var(--card);border:1px solid var(--line)!important;border-radius:16px}
[data-testid="stPlotlyChart"]{background:var(--card);border:1px solid var(--line);border-radius:20px;padding:6px;overflow:hidden;backdrop-filter:blur(12px);box-shadow:0 8px 28px rgba(6,78,59,.07)}
[data-testid="stSpinner"] i,[data-testid="stSpinner"] svg{border-top-color:var(--g6)!important}
::-webkit-scrollbar{width:6px;height:6px}::-webkit-scrollbar-thumb{background:var(--g4);border-radius:6px}
@media (prefers-reduced-motion:reduce){*,*::before,*::after{animation:none!important;transition:none!important}}
</style>
"""
