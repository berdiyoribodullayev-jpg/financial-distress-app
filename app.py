import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf

st.set_page_config(
    page_title="Financial Distress App",
    page_icon="📊",
    layout="centered",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Mono:wght@400;500&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color: #0E0F12 !important;
    color: #F0EDE6 !important;
    font-family: 'Syne', sans-serif !important;
}
[data-testid="stSidebar"] { display: none; }
[data-testid="stHeader"] { background: transparent !important; }
.block-container { max-width: 900px !important; padding: 2rem 1.5rem 4rem !important; }
#MainMenu, footer, header { visibility: hidden; }

[data-testid="stTextInput"] input {
    background: #1C1F27 !important;
    border: 0.5px solid rgba(201,168,76,0.25) !important;
    border-radius: 10px !important;
    color: #F0EDE6 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 15px !important;
    font-weight: 500 !important;
    letter-spacing: 2px !important;
    padding: 0 16px !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #C9A84C !important;
    box-shadow: none !important;
}
[data-testid="stButton"] button {
    background: #C9A84C !important;
    color: #0E0F12 !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    height: 46px !important;
    letter-spacing: 0.5px !important;
    width: 100% !important;
}
[data-testid="stButton"] button:hover {
    background: #E8C97A !important;
    color: #0E0F12 !important;
}
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 0.5px solid rgba(201,168,76,0.15) !important;
    gap: 0 !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important;
    color: #5C5850 !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    padding: 10px 20px !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: #C9A84C !important;
    background: transparent !important;
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] { display: none !important; }

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
.fu1 { animation: fadeUp 0.4s ease both; }
.fu2 { animation: fadeUp 0.4s 0.08s ease both; }
.fu3 { animation: fadeUp 0.4s 0.16s ease both; }
.fu4 { animation: fadeUp 0.4s 0.24s ease both; }
.fu5 { animation: fadeUp 0.4s 0.32s ease both; }
.fu6 { animation: fadeUp 0.4s 0.40s ease both; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def fmt(val):
    if val is None: return "N/A"
    abs_val = abs(val)
    sign = "-" if val < 0 else ""
    if abs_val >= 1e12: return f"{sign}${abs_val/1e12:.2f}T"
    if abs_val >= 1e9:  return f"{sign}${abs_val/1e9:.2f}B"
    if abs_val >= 1e6:  return f"{sign}${abs_val/1e6:.2f}M"
    return f"{sign}${abs_val:,.0f}"

def get_val(df, *keys):
    for k in keys:
        if k in df.index:
            try: return float(df.loc[k].iloc[0])
            except: pass
    return None

def compute_zscore(ticker_str):
    stock = yf.Ticker(ticker_str)
    info  = stock.info
    bs    = stock.balance_sheet
    inc   = stock.income_stmt

    mc = info.get("marketCap", 0) or 0
    wc = get_val(bs, "Working Capital", "WorkingCapital")
    ta = get_val(bs, "Total Assets", "TotalAssets")
    re = get_val(bs, "Retained Earnings", "RetainedEarnings")
    tl = get_val(bs, "Total Liabilities Net Minority Interest", "Total Liabilities", "TotalLiabilities")
    eb = get_val(inc, "EBIT", "Ebit", "Operating Income", "OperatingIncome")
    rv = get_val(inc, "Total Revenue", "TotalRevenue", "Revenue")

    if wc is None:
        ca = get_val(bs, "Current Assets", "CurrentAssets")
        cl = get_val(bs, "Current Liabilities", "CurrentLiabilities")
        if ca and cl: wc = ca - cl

    name    = info.get("longName", ticker_str)
    sector  = info.get("sector", "N/A")
    country = info.get("country", "N/A")
    website = info.get("website", "") or ""
    domain  = website.replace("https://","").replace("http://","").split("/")[0]
    logo    = f"https://www.google.com/s2/favicons?domain={domain}&sz=64" if domain else ""

    vals = [wc, ta, re, tl, eb, rv]
    if all(v is not None for v in vals) and ta and ta != 0:
        x1 = wc / ta
        x2 = re / ta
        x3 = eb / ta
        x4 = (mc / tl) if (mc and tl and tl != 0) else 0
        x5 = rv / ta
        z  = 1.2*x1 + 1.4*x2 + 3.3*x3 + 0.6*x4 + 1.0*x5
        x4_warn = (not mc or mc == 0) or x4 > 10
        return dict(ok=True, name=name, sector=sector, country=country,
                    logo=logo, mc=mc, z=z,
                    x1=x1, x2=x2, x3=x3, x4=x4, x5=x5,
                    wc=wc, ta=ta, re=re, tl=tl, eb=eb, rv=rv,
                    x4_warn=x4_warn)
    return dict(ok=False, name=name, sector=sector, country=country, logo=logo, mc=mc)

def zone_info(z):
    if z > 2.99:
        return "Safe Zone",     "Z > 2.99",        "#3FCF8E", "#0D2B1F", "rgba(63,207,142,0.2)"
    if z > 1.81:
        return "Grey Zone",     "1.81 < Z < 2.99", "#F0A030", "#2B1A05", "rgba(240,160,48,0.2)"
    return     "Distress Zone", "Z < 1.81",         "#F06060", "#2B0D0D", "rgba(240,96,96,0.2)"

def gauge_pct(z):
    if z > 2.99: return min(94, 50 + (z - 2.99) * 8)
    if z > 1.81: return 35 + (z - 1.81) * 12
    return max(5, z * 10)

def render_zscore_panel(z, d, uid="zp"):
    """Render Z-score panel with working animation via components.html"""
    zl, zs, zc, zbg, zbd = zone_info(z)
    gp = gauge_pct(z)

    warn_html = ""
    if d.get('x4_warn'):
        if not d['mc'] or d['mc'] == 0:
            warn_html = '<div style="margin-top:12px;font-size:11px;color:#F0A030;">&#9888; Market cap unavailable — X4 set to 0, score may be understated.</div>'
        elif d['x4'] > 10:
            warn_html = '<div style="margin-top:12px;font-size:11px;color:#F0A030;">&#9888; X4 is very large — may inflate Z-score for high-cap firms.</div>'

    html = f"""
<!DOCTYPE html>
<html>
<head>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Mono:wght@500&display=swap" rel="stylesheet">
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ background: transparent; font-family: 'Syne', sans-serif; }}
.panel {{
    background: #14161B;
    border: 0.5px solid rgba(201,168,76,0.2);
    border-radius: 16px;
    padding: 1.4rem 1.5rem;
}}
.top {{ display: flex; align-items: flex-end; justify-content: space-between; flex-wrap: wrap; gap: 12px; margin-bottom: 1.4rem; }}
.label {{ font-size: 10px; color: #5C5850; letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 4px; }}
.zscore {{ font-family: 'DM Mono', monospace; font-size: 56px; font-weight: 500; color: #C9A84C; line-height: 1; }}
.zone-box {{ display: flex; align-items: center; gap: 8px; background: {zbg}; border: 0.5px solid {zbd}; border-radius: 10px; padding: 10px 16px; }}
.zone-dot {{ width: 8px; height: 8px; border-radius: 50%; background: {zc}; flex-shrink: 0; }}
.zone-label {{ font-size: 14px; font-weight: 700; color: {zc}; }}
.zone-sub {{ font-size: 11px; color: {zc}; opacity: 0.6; margin-top: 1px; }}
.gauge-wrap {{ position: relative; height: 6px; background: #242830; border-radius: 3px; margin-bottom: 8px; overflow: visible; }}
.gauge-track {{ position: absolute; left: 0; top: 0; height: 100%; width: 100%; border-radius: 3px; background: linear-gradient(90deg,#F06060 0%,#F0A030 40%,#3FCF8E 100%); }}
.gauge-marker {{ position: absolute; top: -3px; left: 5%; width: 10px; height: 10px; background: #F0EDE6; border-radius: 50%; transform: translateX(-50%); border: 2px solid #14161B; transition: left 1.2s cubic-bezier(.4,0,.2,1); }}
.gauge-labels {{ display: flex; justify-content: space-between; font-size: 10px; color: #5C5850; font-family: 'DM Mono', monospace; }}
</style>
</head>
<body>
<div class="panel">
  <div class="top">
    <div>
      <div class="label">Z-Score</div>
      <div class="zscore" id="znum">0.00</div>
    </div>
    <div class="zone-box">
      <div class="zone-dot"></div>
      <div>
        <div class="zone-label">{zl}</div>
        <div class="zone-sub">{zs}</div>
      </div>
    </div>
  </div>
  <div class="gauge-wrap">
    <div class="gauge-track"></div>
    <div class="gauge-marker" id="gmark"></div>
  </div>
  <div class="gauge-labels">
    <span>Distress &lt;1.81</span><span>Grey 1.81–2.99</span><span>Safe &gt;2.99</span>
  </div>
  {warn_html}
</div>
<script>
var target = {z:.4f};
var gp     = {gp:.1f};
var el     = document.getElementById('znum');
var mk     = document.getElementById('gmark');
var start  = null, dur = 1200;
function step(ts) {{
  if (!start) start = ts;
  var p = Math.min((ts - start) / dur, 1);
  var e = 1 - Math.pow(1 - p, 3);
  el.textContent = (target * e).toFixed(2);
  if (p < 1) {{ requestAnimationFrame(step); }}
  else {{ el.textContent = target.toFixed(2); }}
}}
requestAnimationFrame(step);
setTimeout(function() {{ mk.style.left = gp + '%'; }}, 80);
</script>
</body>
</html>
"""
    components.html(html, height=180)


def render_comparison_card(ticker, d, delay_ms=100):
    """Render a single comparison card with animated Z-score"""
    if not d["ok"]:
        components.html(f"""
        <div style="background:#14161B;border:0.5px solid rgba(240,96,96,0.2);border-radius:14px;
                    padding:1.5rem;text-align:center;font-family:sans-serif;">
          <div style="font-family:monospace;font-size:14px;color:#C9A84C;margin-bottom:8px;">{ticker}</div>
          <div style="color:#F06060;font-size:12px;">Data unavailable</div>
        </div>
        """, height=120)
        return

    z  = d["z"]
    zl, _, zc, zbg, zbd = zone_info(z)
    gp = gauge_pct(z)
    initials_c = ticker[:2]
    if d["logo"]:
        logo_tag = f'<div style="width:36px;height:36px;border-radius:8px;background:#1C1F27;padding:4px;margin:0 auto 8px;display:flex;align-items:center;justify-content:center;"><img src="{d["logo"]}" style="width:28px;height:28px;object-fit:contain;"></div>'
    else:
        logo_tag = f'<div style="width:36px;height:36px;border-radius:8px;background:rgba(201,168,76,0.15);border:0.5px solid rgba(201,168,76,0.3);display:flex;align-items:center;justify-content:center;font-family:monospace;font-size:12px;font-weight:500;color:#C9A84C;margin:0 auto 8px;">{initials_c}</div>'

    components.html(f"""
<!DOCTYPE html>
<html>
<head>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@700&family=DM+Mono:wght@500&display=swap" rel="stylesheet">
<style>
* {{ box-sizing:border-box; margin:0; padding:0; }}
body {{ background:transparent; font-family:'Syne',sans-serif; }}
.card {{ background:#14161B; border:0.5px solid rgba(201,168,76,0.15); border-radius:14px; padding:1.2rem 1rem; text-align:center; }}
.ticker {{ font-family:'DM Mono',monospace; font-size:12px; color:#C9A84C; letter-spacing:1px; margin-bottom:3px; }}
.cname {{ font-size:11px; color:#5C5850; margin-bottom:14px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
.znum {{ font-family:'DM Mono',monospace; font-size:40px; font-weight:500; color:#C9A84C; line-height:1; margin-bottom:10px; }}
.zbadge {{ display:inline-block; background:{zbg}; border:0.5px solid {zbd}; border-radius:8px; padding:5px 12px; margin-bottom:14px; }}
.zbadge span {{ font-size:12px; font-weight:700; color:{zc}; }}
.gbar {{ position:relative; height:4px; background:#242830; border-radius:2px; overflow:visible; margin-bottom:6px; }}
.gtrack {{ position:absolute; left:0; top:0; height:100%; width:100%; background:linear-gradient(90deg,#F06060,#F0A030,#3FCF8E); border-radius:2px; }}
.gmarker {{ position:absolute; top:-3px; left:5%; width:8px; height:8px; background:#F0EDE6; border-radius:50%; transform:translateX(-50%); border:1.5px solid #14161B; transition:left 1.2s cubic-bezier(.4,0,.2,1); }}
.mcap {{ font-size:11px; color:#5C5850; font-family:'DM Mono',monospace; }}
</style>
</head>
<body>
<div class="card">
  {logo_tag}
  <div class="ticker">{ticker}</div>
  <div class="cname">{d['name'][:26]}</div>
  <div class="znum" id="zn">0.00</div>
  <div class="zbadge"><span>{zl}</span></div>
  <div class="gbar"><div class="gtrack"></div><div class="gmarker" id="gm"></div></div>
  <div class="mcap">{fmt(d['mc'])}</div>
</div>
<script>
var target={z:.4f}, gp={gp:.1f};
var el=document.getElementById('zn'), mk=document.getElementById('gm');
var start=null, dur=1100;
function step(ts){{
  if(!start) start=ts;
  var p=Math.min((ts-start)/dur,1), e=1-Math.pow(1-p,3);
  el.textContent=(target*e).toFixed(2);
  if(p<1) requestAnimationFrame(step);
  else el.textContent=target.toFixed(2);
}}
setTimeout(function(){{ requestAnimationFrame(step); mk.style.left=gp+'%'; }},{delay_ms});
</script>
</body>
</html>
""", height=260)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="fu1" style="margin-bottom:2rem;padding-bottom:1.2rem;border-bottom:0.5px solid rgba(201,168,76,0.15);">
  <div style="width:36px;height:36px;border:1.5px solid #C9A84C;border-radius:8px;
              display:flex;align-items:center;justify-content:center;margin-bottom:1rem;">
    <div style="width:14px;height:14px;background:#C9A84C;border-radius:2px;"></div>
  </div>
  <div style="display:flex;align-items:flex-end;justify-content:space-between;flex-wrap:wrap;gap:8px;">
    <div>
      <div style="font-size:28px;font-weight:800;letter-spacing:-0.5px;line-height:1;color:#F0EDE6;">
        Financial <span style="color:#C9A84C;">Distress</span> App
      </div>
      <div style="font-size:13px;color:#9B9589;margin-top:4px;letter-spacing:0.3px;">
        Altman Z-Score · Risk Classification · Competitor Analysis
      </div>
    </div>
    <div style="font-family:'DM Mono',monospace;font-size:10px;color:#8A6E2F;
                background:rgba(201,168,76,0.08);border:0.5px solid rgba(201,168,76,0.15);
                padding:4px 10px;border-radius:4px;letter-spacing:1px;">v1.2</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊  Single Analysis", "⚔️  Competitor Comparison", "🧮  Ohlson O-Score"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single Analysis
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div style="font-size:11px;font-weight:500;color:#9B9589;letter-spacing:1.5px;text-transform:uppercase;margin:1rem 0 6px;">Company Ticker</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([4, 1])
    with c1:
        t1 = st.text_input("t1", placeholder="e.g. AAPL, TSLA, MSFT", label_visibility="collapsed")
    with c2:
        btn1 = st.button("Analyze →", key="btn1")

    if btn1 and t1.strip():
        ticker = t1.strip().upper()
        with st.spinner("Fetching data..."):
            d = compute_zscore(ticker)

        if not d["ok"]:
            st.markdown('<div style="background:#2B0D0D;border:0.5px solid rgba(240,96,96,0.2);border-radius:10px;padding:1rem 1.2rem;color:#F06060;font-size:13px;margin-top:1rem;">Required financial fields missing — Z-score could not be calculated.</div>', unsafe_allow_html=True)
        else:
            z = d["z"]

            # Company banner — avatar always shown, logo overlaid if available
            initials = ticker[:2]
            if d["logo"]:
                logo_html = f'<div style="width:38px;height:38px;border-radius:8px;background:#1C1F27;padding:4px;flex-shrink:0;display:flex;align-items:center;justify-content:center;"><img src="{d["logo"]}" style="width:30px;height:30px;object-fit:contain;"></div>'
            else:
                logo_html = f'<div style="width:38px;height:38px;border-radius:8px;background:rgba(201,168,76,0.15);border:0.5px solid rgba(201,168,76,0.3);display:flex;align-items:center;justify-content:center;font-family:monospace;font-size:13px;font-weight:500;color:#C9A84C;flex-shrink:0;">{initials}</div>'
            st.markdown(f"""
            <div class="fu2" style="background:#14161B;border:0.5px solid rgba(201,168,76,0.2);border-radius:14px;
                        padding:1.2rem 1.5rem;display:flex;align-items:center;
                        justify-content:space-between;flex-wrap:wrap;gap:12px;margin:1rem 0 1.5rem;">
              <div style="display:flex;align-items:center;gap:12px;">
                {logo_html}
                <div style="display:flex;align-items:center;gap:10px;">
                  <div style="background:rgba(201,168,76,0.1);border:0.5px solid rgba(201,168,76,0.2);
                              border-radius:6px;padding:4px 10px;font-family:'DM Mono',monospace;
                              font-size:13px;font-weight:500;color:#C9A84C;letter-spacing:1px;">{ticker}</div>
                  <div>
                    <div style="font-size:17px;font-weight:700;color:#F0EDE6;">{d['name']}</div>
                    <div style="font-size:12px;color:#9B9589;margin-top:1px;">{d['sector']} · {d['country']}</div>
                  </div>
                </div>
              </div>
              <div style="text-align:right;">
                <div style="font-size:10px;color:#5C5850;letter-spacing:1px;text-transform:uppercase;">Market Cap</div>
                <div style="font-family:'DM Mono',monospace;font-size:20px;font-weight:500;color:#F0EDE6;margin-top:2px;">{fmt(d['mc'])}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Raw Financials
            st.markdown('<div class="fu3" style="display:flex;align-items:center;gap:10px;margin-bottom:1rem;"><span style="font-size:10px;font-weight:600;color:#5C5850;letter-spacing:2px;text-transform:uppercase;">Raw Financials</span><div style="flex:1;height:0.5px;background:rgba(201,168,76,0.1);"></div></div>', unsafe_allow_html=True)
            fin_rows = [("Working Capital", d['wc']), ("Total Assets", d['ta']),
                        ("Retained Earnings", d['re']), ("EBIT", d['eb']),
                        ("Total Liabilities", d['tl']), ("Sales / Revenue", d['rv'])]
            cols = st.columns(2)
            for i, (lbl, val) in enumerate(fin_rows):
                with cols[i % 2]:
                    st.markdown(f'<div style="display:flex;justify-content:space-between;align-items:center;background:#14161B;border-radius:8px;padding:10px 14px;border:0.5px solid rgba(201,168,76,0.08);margin-bottom:8px;"><span style="font-size:12px;color:#9B9589;">{lbl}</span><span style="font-family:\'DM Mono\',monospace;font-size:13px;font-weight:500;color:#F0EDE6;">{fmt(val)}</span></div>', unsafe_allow_html=True)

            # Z-Score panel — uses components.html for real JS animation
            st.markdown('<div class="fu4" style="display:flex;align-items:center;gap:10px;margin:1.5rem 0 0.8rem;"><span style="font-size:10px;font-weight:600;color:#5C5850;letter-spacing:2px;text-transform:uppercase;">Altman Z-Score</span><div style="flex:1;height:0.5px;background:rgba(201,168,76,0.1);"></div></div>', unsafe_allow_html=True)
            render_zscore_panel(z, d)

            # Ratios
            st.markdown('<div class="fu5" style="display:flex;align-items:center;gap:10px;margin:1.2rem 0 0.8rem;"><span style="font-size:10px;font-weight:600;color:#5C5850;letter-spacing:2px;text-transform:uppercase;">Ratio Breakdown</span><div style="flex:1;height:0.5px;background:rgba(201,168,76,0.1);"></div></div>', unsafe_allow_html=True)
            ratios = [("X1","Working Capital / Total Assets",d['x1'],"× 1.2"),
                      ("X2","Retained Earnings / Total Assets",d['x2'],"× 1.4"),
                      ("X3","EBIT / Total Assets",d['x3'],"× 3.3"),
                      ("X4","Market Value / Total Liabilities",d['x4'],"× 0.6"),
                      ("X5","Sales / Total Assets",d['x5'],"× 1.0")]
            cols5 = st.columns(5)
            for i, (nm, formula, val, wt) in enumerate(ratios):
                with cols5[i]:
                    st.markdown(f'<div style="background:#1C1F27;border:0.5px solid rgba(201,168,76,0.08);border-radius:10px;padding:0.9rem 0.8rem;text-align:center;"><div style="font-size:11px;font-weight:700;color:#C9A84C;margin-bottom:3px;">{nm}</div><div style="font-size:9px;color:#5C5850;margin-bottom:8px;line-height:1.3;">{formula}</div><div style="font-family:\'DM Mono\',monospace;font-size:17px;font-weight:500;color:#F0EDE6;">{val:.3f}</div><div style="font-size:9px;color:#5C5850;margin-top:3px;font-family:\'DM Mono\',monospace;">{wt}</div></div>', unsafe_allow_html=True)

            # Interpretation
            if z > 2.99:
                interp = f"This company appears financially healthy under the Altman Z-score model. A score of <b>{z:.2f}</b> places it firmly in the Safe Zone (Z > 2.99), indicating low probability of financial distress."
            elif z > 1.81:
                interp = f"This company is in the Grey Zone with a score of <b>{z:.2f}</b>. Risk is neither clearly low nor high — monitor financial trends closely."
            else:
                interp = f"This company may be facing financial distress. A score of <b>{z:.2f}</b> falls in the Distress Zone (Z < 1.81), suggesting elevated risk."

            st.markdown(f'<div class="fu6" style="margin-top:1.2rem;background:#1C1F27;border-left:2px solid #C9A84C;border-radius:0 10px 10px 0;padding:1rem 1.2rem;font-size:13px;color:#9B9589;line-height:1.7;">{interp}</div>', unsafe_allow_html=True)

    elif btn1:
        st.markdown('<div style="background:#2B1A05;border:0.5px solid rgba(240,160,48,0.2);border-radius:10px;padding:1rem 1.2rem;color:#F0A030;font-size:13px;margin-top:1rem;">Please enter a ticker symbol.</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Competitor Comparison
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div style="font-size:11px;font-weight:500;color:#9B9589;letter-spacing:1.5px;text-transform:uppercase;margin:1rem 0 6px;">Enter up to 3 tickers to compare</div>', unsafe_allow_html=True)
    cc1, cc2, cc3, cc4 = st.columns([2, 2, 2, 1])
    with cc1: t_a = st.text_input("ta", placeholder="AAPL", label_visibility="collapsed")
    with cc2: t_b = st.text_input("tb", placeholder="MSFT", label_visibility="collapsed")
    with cc3: t_c = st.text_input("tc", placeholder="GOOGL", label_visibility="collapsed")
    with cc4: btn2 = st.button("Compare →", key="btn2")

    if btn2:
        tickers = [t.strip().upper() for t in [t_a, t_b, t_c] if t.strip()]
        if not tickers:
            st.markdown('<div style="background:#2B1A05;border:0.5px solid rgba(240,160,48,0.2);border-radius:10px;padding:1rem;color:#F0A030;font-size:13px;margin-top:1rem;">Please enter at least one ticker.</div>', unsafe_allow_html=True)
        else:
            results = []
            with st.spinner("Fetching data..."):
                for t in tickers:
                    results.append((t, compute_zscore(t)))

            st.markdown('<div style="display:flex;align-items:center;gap:10px;margin:1.5rem 0 1rem;"><span style="font-size:10px;font-weight:600;color:#5C5850;letter-spacing:2px;text-transform:uppercase;">Z-Score Comparison</span><div style="flex:1;height:0.5px;background:rgba(201,168,76,0.1);"></div></div>', unsafe_allow_html=True)

            comp_cols = st.columns(len(results))
            for i, (ticker, d) in enumerate(results):
                with comp_cols[i]:
                    render_comparison_card(ticker, d, delay_ms=100 + i * 200)

            # Side-by-side ratios table
            valid = [(t, d) for t, d in results if d["ok"]]
            if len(valid) > 1:
                st.markdown('<div style="display:flex;align-items:center;gap:10px;margin:1.5rem 0 1rem;"><span style="font-size:10px;font-weight:600;color:#5C5850;letter-spacing:2px;text-transform:uppercase;">Side-by-Side Ratios</span><div style="flex:1;height:0.5px;background:rgba(201,168,76,0.1);"></div></div>', unsafe_allow_html=True)

                n  = len(valid)
                fr = f"160px {' '.join(['1fr']*n)}"

                def cell(content, accent=False, header=False, center=False):
                    color = "#C9A84C" if accent else ("#9B9589" if header else "#F0EDE6")
                    mono  = "font-family:'DM Mono',monospace;" if not header else ""
                    align = "center" if center else "left"
                    return f'<div style="padding:9px 14px;background:#14161B;color:{color};{mono}font-size:12px;text-align:{align};">{content}</div>'

                html = f'<div style="display:grid;grid-template-columns:{fr};gap:1px;background:rgba(201,168,76,0.08);border-radius:12px;overflow:hidden;margin-bottom:2px;">'
                html += cell("Metric", header=True)
                for t, _ in valid:
                    html += cell(t, accent=True, center=True)
                html += "</div>"

                rows_def = [
                    ("Z-Score",      lambda d: f"{d['z']:.2f}"),
                    ("X1",           lambda d: f"{d['x1']:.3f}"),
                    ("X2",           lambda d: f"{d['x2']:.3f}"),
                    ("X3",           lambda d: f"{d['x3']:.3f}"),
                    ("X4",           lambda d: f"{d['x4']:.3f}"),
                    ("X5",           lambda d: f"{d['x5']:.3f}"),
                    ("Market Cap",   lambda d: fmt(d['mc'])),
                    ("Total Assets", lambda d: fmt(d['ta'])),
                    ("Revenue",      lambda d: fmt(d['rv'])),
                ]

                html += f'<div style="display:grid;grid-template-columns:{fr};gap:1px;background:rgba(201,168,76,0.05);border-radius:12px;overflow:hidden;">'
                for lbl, fn in rows_def:
                    html += cell(lbl, header=True)
                    for _, d in valid:
                        html += cell(fn(d), center=True)
                html += "</div>"

                st.markdown(html, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Ohlson O-Score
# ════════════════════════════════════════════════════════════════════════════

def compute_oscore(ticker_str):
    """
    Ohlson (1980) O-Score formula:
    O = -1.32 - 0.407*X1 + 6.03*X2 - 1.43*X3 + 0.076*X4
            - 1.72*X5 - 2.37*X6 - 1.83*X7 + 0.285*X8 - 0.521*X9
    P(distress) = 1 / (1 + exp(-O))

    X1 = log(Total Assets / GNP Price Index)  → simplified: log(Total Assets)
    X2 = Total Liabilities / Total Assets
    X3 = Working Capital / Total Assets
    X4 = Current Liabilities / Current Assets
    X5 = 1 if Total Liabilities > Total Assets, else 0
    X6 = Net Income / Total Assets
    X7 = Cash Flow from Operations / Total Liabilities
    X8 = 1 if Net Income < 0 for last two years, else 0
    X9 = (Net Income_t - Net Income_t-1) / (|NI_t| + |NI_t-1|)
    """
    import math
    stock = yf.Ticker(ticker_str)
    info  = stock.info
    bs    = stock.balance_sheet
    inc   = stock.income_stmt
    cf    = stock.cash_flow

    name    = info.get("longName", ticker_str)
    sector  = info.get("sector", "N/A")
    country = info.get("country", "N/A")
    mc      = info.get("marketCap", 0) or 0
    website = info.get("website", "") or ""
    domain  = website.replace("https://","").replace("http://","").split("/")[0]
    logo    = f"https://www.google.com/s2/favicons?domain={domain}&sz=64" if domain else ""

    ta  = get_val(bs, "Total Assets", "TotalAssets")
    tl  = get_val(bs, "Total Liabilities Net Minority Interest", "Total Liabilities", "TotalLiabilities")
    wc  = get_val(bs, "Working Capital", "WorkingCapital")
    ca  = get_val(bs, "Current Assets", "CurrentAssets")
    cl  = get_val(bs, "Current Liabilities", "CurrentLiabilities")
    ni  = get_val(inc, "Net Income", "NetIncome")
    cfo = get_val(cf,  "Operating Cash Flow", "Cash Flow From Continuing Operating Activities", "Free Cash Flow")

    if wc is None and ca and cl:
        wc = ca - cl

    # Net Income previous year for X8 and X9
    ni_prev = None
    if inc is not None and len(inc.columns) > 1:
        for k in ["Net Income", "NetIncome"]:
            if k in inc.index:
                try:
                    ni_prev = float(inc.loc[k].iloc[1])
                    break
                except:
                    pass

    required = [ta, tl, wc, ca, cl, ni, cfo]
    if not all(v is not None for v in required) or ta == 0 or tl == 0:
        return dict(ok=False, name=name, sector=sector, country=country, logo=logo, mc=mc)

    x1 = math.log(abs(ta)) if ta > 0 else 0
    x2 = tl / ta
    x3 = wc / ta
    x4 = cl / ca if ca != 0 else 0
    x5 = 1 if tl > ta else 0
    x6 = ni / ta
    x7 = cfo / tl if tl != 0 else 0
    x8 = 1 if (ni_prev is not None and ni < 0 and ni_prev < 0) else 0
    if ni_prev is not None and (abs(ni) + abs(ni_prev)) != 0:
        x9 = (ni - ni_prev) / (abs(ni) + abs(ni_prev))
    else:
        x9 = 0

    o = (-1.32
         - 0.407 * x1
         + 6.03  * x2
         - 1.43  * x3
         + 0.076 * x4
         - 1.72  * x5
         - 2.37  * x6
         - 1.83  * x7
         + 0.285 * x8
         - 0.521 * x9)

    prob = 1 / (1 + math.exp(-o))

    return dict(
        ok=True, name=name, sector=sector, country=country, logo=logo, mc=mc,
        o=o, prob=prob,
        x1=x1, x2=x2, x3=x3, x4=x4, x5=x5,
        x6=x6, x7=x7, x8=x8, x9=x9,
        ta=ta, tl=tl, wc=wc, ca=ca, cl=cl, ni=ni, cfo=cfo,
        ni_prev=ni_prev
    )


def o_zone(prob):
    p = prob * 100
    if p < 20:
        return "Low Risk",    f"{p:.1f}%", "#3FCF8E", "#0D2B1F", "rgba(63,207,142,0.2)"
    if p < 50:
        return "Medium Risk", f"{p:.1f}%", "#F0A030", "#2B1A05", "rgba(240,160,48,0.2)"
    return     "High Risk",   f"{p:.1f}%", "#F06060", "#2B0D0D", "rgba(240,96,96,0.2)"


def render_oscore_panel(prob, o_val):
    zl, pct_str, zc, zbg, zbd = o_zone(prob)
    pct = prob * 100
    html = f"""
<!DOCTYPE html>
<html>
<head>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Mono:wght@500&display=swap" rel="stylesheet">
<style>
* {{ box-sizing:border-box; margin:0; padding:0; }}
body {{ background:transparent; font-family:'Syne',sans-serif; }}
.panel {{ background:#14161B; border:0.5px solid rgba(201,168,76,0.2); border-radius:16px; padding:1.4rem 1.5rem; }}
.top {{ display:flex; align-items:flex-end; justify-content:space-between; flex-wrap:wrap; gap:12px; margin-bottom:1.4rem; }}
.label {{ font-size:10px; color:#5C5850; letter-spacing:1.5px; text-transform:uppercase; margin-bottom:4px; }}
.prob {{ font-family:'DM Mono',monospace; font-size:56px; font-weight:500; color:#C9A84C; line-height:1; }}
.zone-box {{ display:flex; align-items:center; gap:8px; background:{zbg}; border:0.5px solid {zbd}; border-radius:10px; padding:10px 16px; }}
.zone-dot {{ width:8px; height:8px; border-radius:50%; background:{zc}; flex-shrink:0; }}
.zone-label {{ font-size:14px; font-weight:700; color:{zc}; }}
.zone-sub {{ font-size:11px; color:{zc}; opacity:0.6; margin-top:1px; }}
.bar-wrap {{ position:relative; height:6px; background:#242830; border-radius:3px; margin-bottom:8px; overflow:visible; }}
.bar-track {{ position:absolute; left:0; top:0; height:100%; width:100%; border-radius:3px; background:linear-gradient(90deg,#3FCF8E 0%,#F0A030 50%,#F06060 100%); }}
.bar-marker {{ position:absolute; top:-3px; left:5%; width:10px; height:10px; background:#F0EDE6; border-radius:50%; transform:translateX(-50%); border:2px solid #14161B; transition:left 1.2s cubic-bezier(.4,0,.2,1); }}
.bar-labels {{ display:flex; justify-content:space-between; font-size:10px; color:#5C5850; font-family:'DM Mono',monospace; }}
.o-val {{ margin-top:10px; font-size:11px; color:#5C5850; font-family:'DM Mono',monospace; }}
</style>
</head>
<body>
<div class="panel">
  <div class="top">
    <div>
      <div class="label">Distress Probability</div>
      <div class="prob" id="pnum">0.0%</div>
    </div>
    <div class="zone-box">
      <div class="zone-dot"></div>
      <div>
        <div class="zone-label">{zl}</div>
        <div class="zone-sub">O-Score: {o_val:.3f}</div>
      </div>
    </div>
  </div>
  <div class="bar-wrap">
    <div class="bar-track"></div>
    <div class="bar-marker" id="bmark"></div>
  </div>
  <div class="bar-labels">
    <span>Low &lt;20%</span><span>Medium 20–50%</span><span>High &gt;50%</span>
  </div>
  <div class="o-val">Raw O-Score: {o_val:.4f} &nbsp;|&nbsp; P = 1/(1+e^-O)</div>
</div>
<script>
var target={pct:.4f}, mk=document.getElementById('bmark'), el=document.getElementById('pnum');
var start=null, dur=1200;
function step(ts){{
  if(!start) start=ts;
  var p=Math.min((ts-start)/dur,1), e=1-Math.pow(1-p,3);
  el.textContent=(target*e).toFixed(1)+'%';
  if(p<1) requestAnimationFrame(step);
  else el.textContent=target.toFixed(1)+'%';
}}
requestAnimationFrame(step);
setTimeout(function(){{ mk.style.left=Math.min(94,target)+'%'; }}, 80);
</script>
</body>
</html>
"""
    components.html(html, height=190)


with tab3:
    st.markdown('<div style="font-size:11px;font-weight:500;color:#9B9589;letter-spacing:1.5px;text-transform:uppercase;margin:1rem 0 6px;">Company Ticker</div>', unsafe_allow_html=True)
    o1, o2 = st.columns([4, 1])
    with o1:
        t_o = st.text_input("to", placeholder="e.g. AAPL, TSLA, MSFT", label_visibility="collapsed")
    with o2:
        btn3 = st.button("Analyze →", key="btn3")

    if btn3 and t_o.strip():
        ticker_o = t_o.strip().upper()
        with st.spinner("Fetching data..."):
            od = compute_oscore(ticker_o)

        if not od["ok"]:
            st.markdown('<div style="background:#2B0D0D;border:0.5px solid rgba(240,96,96,0.2);border-radius:10px;padding:1rem 1.2rem;color:#F06060;font-size:13px;margin-top:1rem;">Required financial fields missing — O-score could not be calculated.</div>', unsafe_allow_html=True)
        else:
            prob = od["prob"]
            zl, _, zc, _, _ = o_zone(prob)

            # Company banner
            initials_o = ticker_o[:2]
            if od["logo"]:
                logo_o = f'<div style="width:38px;height:38px;border-radius:8px;background:#1C1F27;padding:4px;flex-shrink:0;display:flex;align-items:center;justify-content:center;"><img src="{od["logo"]}" style="width:30px;height:30px;object-fit:contain;"></div>'
            else:
                logo_o = f'<div style="width:38px;height:38px;border-radius:8px;background:rgba(201,168,76,0.15);border:0.5px solid rgba(201,168,76,0.3);display:flex;align-items:center;justify-content:center;font-family:monospace;font-size:13px;font-weight:500;color:#C9A84C;flex-shrink:0;">{initials_o}</div>'

            st.markdown(f"""
            <div class="fu2" style="background:#14161B;border:0.5px solid rgba(201,168,76,0.2);border-radius:14px;
                        padding:1.2rem 1.5rem;display:flex;align-items:center;
                        justify-content:space-between;flex-wrap:wrap;gap:12px;margin:1rem 0 1.5rem;">
              <div style="display:flex;align-items:center;gap:12px;">
                {logo_o}
                <div style="display:flex;align-items:center;gap:10px;">
                  <div style="background:rgba(201,168,76,0.1);border:0.5px solid rgba(201,168,76,0.2);
                              border-radius:6px;padding:4px 10px;font-family:'DM Mono',monospace;
                              font-size:13px;font-weight:500;color:#C9A84C;letter-spacing:1px;">{ticker_o}</div>
                  <div>
                    <div style="font-size:17px;font-weight:700;color:#F0EDE6;">{od['name']}</div>
                    <div style="font-size:12px;color:#9B9589;margin-top:1px;">{od['sector']} · {od['country']}</div>
                  </div>
                </div>
              </div>
              <div style="text-align:right;">
                <div style="font-size:10px;color:#5C5850;letter-spacing:1px;text-transform:uppercase;">Market Cap</div>
                <div style="font-family:'DM Mono',monospace;font-size:20px;font-weight:500;color:#F0EDE6;margin-top:2px;">{fmt(od['mc'])}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Raw financials for O-Score
            st.markdown('<div style="display:flex;align-items:center;gap:10px;margin-bottom:1rem;"><span style="font-size:10px;font-weight:600;color:#5C5850;letter-spacing:2px;text-transform:uppercase;">Raw Financials</span><div style="flex:1;height:0.5px;background:rgba(201,168,76,0.1);"></div></div>', unsafe_allow_html=True)
            o_fin_rows = [
                ("Total Assets",      od['ta']),
                ("Total Liabilities", od['tl']),
                ("Working Capital",   od['wc']),
                ("Current Assets",    od['ca']),
                ("Current Liabilities", od['cl']),
                ("Net Income",        od['ni']),
                ("Net Income (prev)", od['ni_prev']),
                ("Operating Cash Flow", od['cfo']),
            ]
            ocols = st.columns(2)
            for i, (lbl, val) in enumerate(o_fin_rows):
                with ocols[i % 2]:
                    st.markdown(f'<div style="display:flex;justify-content:space-between;align-items:center;background:#14161B;border-radius:8px;padding:10px 14px;border:0.5px solid rgba(201,168,76,0.08);margin-bottom:8px;"><span style="font-size:12px;color:#9B9589;">{lbl}</span><span style="font-family:\'DM Mono\',monospace;font-size:13px;font-weight:500;color:#F0EDE6;">{fmt(val) if val is not None else "N/A"}</span></div>', unsafe_allow_html=True)

            # O-Score panel
            st.markdown('<div style="display:flex;align-items:center;gap:10px;margin:1.5rem 0 0.8rem;"><span style="font-size:10px;font-weight:600;color:#5C5850;letter-spacing:2px;text-transform:uppercase;">Ohlson O-Score</span><div style="flex:1;height:0.5px;background:rgba(201,168,76,0.1);"></div></div>', unsafe_allow_html=True)
            render_oscore_panel(prob, od["o"])

            # Variable breakdown
            st.markdown('<div style="display:flex;align-items:center;gap:10px;margin:1.2rem 0 0.8rem;"><span style="font-size:10px;font-weight:600;color:#5C5850;letter-spacing:2px;text-transform:uppercase;">Variable Breakdown</span><div style="flex:1;height:0.5px;background:rgba(201,168,76,0.1);"></div></div>', unsafe_allow_html=True)

            o_vars = [
                ("X1", "log(Total Assets)",           od['x1'], "−0.407"),
                ("X2", "Total Liabilities / Assets",  od['x2'], "+6.03"),
                ("X3", "Working Capital / Assets",    od['x3'], "−1.43"),
                ("X4", "Current Liab / Current Assets", od['x4'], "+0.076"),
                ("X5", "Insolvent (0/1)",             od['x5'], "−1.72"),
            ]
            o_vars2 = [
                ("X6", "Net Income / Assets",         od['x6'], "−2.37"),
                ("X7", "CFO / Total Liabilities",     od['x7'], "−1.83"),
                ("X8", "2yr Loss (0/1)",              od['x8'], "+0.285"),
                ("X9", "NI Change Ratio",             od['x9'], "−0.521"),
            ]

            cols5o = st.columns(5)
            for i, (nm, formula, val, wt) in enumerate(o_vars):
                with cols5o[i]:
                    st.markdown(f'<div style="background:#1C1F27;border:0.5px solid rgba(201,168,76,0.08);border-radius:10px;padding:0.9rem 0.8rem;text-align:center;"><div style="font-size:11px;font-weight:700;color:#C9A84C;margin-bottom:3px;">{nm}</div><div style="font-size:9px;color:#5C5850;margin-bottom:8px;line-height:1.3;">{formula}</div><div style="font-family:\'DM Mono\',monospace;font-size:17px;font-weight:500;color:#F0EDE6;">{val:.3f}</div><div style="font-size:9px;color:#5C5850;margin-top:3px;font-family:\'DM Mono\',monospace;">{wt}</div></div>', unsafe_allow_html=True)

            cols4o = st.columns(4)
            for i, (nm, formula, val, wt) in enumerate(o_vars2):
                with cols4o[i]:
                    st.markdown(f'<div style="background:#1C1F27;border:0.5px solid rgba(201,168,76,0.08);border-radius:10px;padding:0.9rem 0.8rem;text-align:center;margin-top:8px;"><div style="font-size:11px;font-weight:700;color:#C9A84C;margin-bottom:3px;">{nm}</div><div style="font-size:9px;color:#5C5850;margin-bottom:8px;line-height:1.3;">{formula}</div><div style="font-family:\'DM Mono\',monospace;font-size:17px;font-weight:500;color:#F0EDE6;">{val:.3f}</div><div style="font-size:9px;color:#5C5850;margin-top:3px;font-family:\'DM Mono\',monospace;">{wt}</div></div>', unsafe_allow_html=True)

            # Interpretation
            pct = prob * 100
            if pct < 20:
                interp = f"With a distress probability of <b>{pct:.1f}%</b>, this company shows <b>low risk</b> of financial distress under the Ohlson O-Score model. The company's leverage, liquidity, and profitability indicators are within healthy ranges."
            elif pct < 50:
                interp = f"A distress probability of <b>{pct:.1f}%</b> places this company in the <b>medium risk</b> zone. Some financial indicators warrant monitoring — particularly leverage and cash flow metrics."
            else:
                interp = f"A distress probability of <b>{pct:.1f}%</b> signals <b>high risk</b> of financial distress. The Ohlson model identifies significant stress in this company's financial structure. Caution is advised."

            st.markdown(f'<div style="margin-top:1.2rem;background:#1C1F27;border-left:2px solid #C9A84C;border-radius:0 10px 10px 0;padding:1rem 1.2rem;font-size:13px;color:#9B9589;line-height:1.7;">{interp}</div>', unsafe_allow_html=True)

    elif btn3:
        st.markdown('<div style="background:#2B1A05;border:0.5px solid rgba(240,160,48,0.2);border-radius:10px;padding:1rem 1.2rem;color:#F0A030;font-size:13px;margin-top:1rem;">Please enter a ticker symbol.</div>', unsafe_allow_html=True)