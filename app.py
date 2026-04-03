import streamlit as st
import yfinance as yf

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Financial Distress App",
    page_icon="📊",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Mono:wght@400;500&display=swap');

/* ── Root & body ── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color: #0E0F12 !important;
    color: #F0EDE6 !important;
    font-family: 'Syne', sans-serif !important;
}
[data-testid="stSidebar"] { display: none; }
[data-testid="stHeader"] { background: transparent !important; }
.block-container { max-width: 860px !important; padding: 2rem 1.5rem 4rem !important; }

/* ── Hide Streamlit branding ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Input ── */
[data-testid="stTextInput"] input {
    background: #1C1F27 !important;
    border: 0.5px solid rgba(201,168,76,0.25) !important;
    border-radius: 10px !important;
    color: #F0EDE6 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 16px !important;
    font-weight: 500 !important;
    letter-spacing: 2px !important;
    height: 48px !important;
    padding: 0 16px !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #C9A84C !important;
    box-shadow: none !important;
}

/* ── Button ── */
[data-testid="stButton"] button {
    background: #C9A84C !important;
    color: #0E0F12 !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    height: 48px !important;
    letter-spacing: 0.5px !important;
    width: 100% !important;
    transition: background 0.15s !important;
}
[data-testid="stButton"] button:hover {
    background: #E8C97A !important;
    color: #0E0F12 !important;
}

/* ── Divider ── */
hr { border-color: rgba(201,168,76,0.12) !important; margin: 1.5rem 0 !important; }

/* ── Streamlit alerts override ── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
}
</style>
""", unsafe_allow_html=True)

# ── Helper: format large numbers ──────────────────────────────────────────────
def fmt(val):
    if val is None:
        return "N/A"
    abs_val = abs(val)
    sign = "-" if val < 0 else ""
    if abs_val >= 1e12:
        return f"{sign}${abs_val/1e12:.2f}T"
    elif abs_val >= 1e9:
        return f"{sign}${abs_val/1e9:.2f}B"
    elif abs_val >= 1e6:
        return f"{sign}${abs_val/1e6:.2f}M"
    else:
        return f"{sign}${abs_val:,.0f}"

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:2rem; padding-bottom:1.2rem; border-bottom:0.5px solid rgba(201,168,76,0.15);">
  <div style="width:36px;height:36px;border:1.5px solid #C9A84C;border-radius:8px;
              display:flex;align-items:center;justify-content:center;margin-bottom:1rem;">
    <div style="width:14px;height:14px;background:#C9A84C;border-radius:2px;"></div>
  </div>
  <div style="display:flex;align-items:flex-end;justify-content:space-between;flex-wrap:wrap;gap:8px;">
    <div>
      <div style="font-size:28px;font-weight:800;letter-spacing:-0.5px;line-height:1;color:#F0EDE6;">
        Financial <span style="color:#C9A84C;">Distress</span> App
      </div>
      <div style="font-size:13px;color:#9B9589;margin-top:4px;font-weight:400;letter-spacing:0.3px;">
        Altman Z-Score · Ohlson O-Score · Risk Classification
      </div>
    </div>
    <div style="font-family:'DM Mono',monospace;font-size:10px;color:#8A6E2F;
                background:rgba(201,168,76,0.08);border:0.5px solid rgba(201,168,76,0.15);
                padding:4px 10px;border-radius:4px;letter-spacing:1px;">v1.0</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Ticker input ──────────────────────────────────────────────────────────────
st.markdown('<div style="font-size:11px;font-weight:500;color:#9B9589;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:6px;">Company Ticker</div>', unsafe_allow_html=True)

col1, col2 = st.columns([4, 1])
with col1:
    ticker_input = st.text_input("", placeholder="e.g. AAPL, TSLA, MSFT", label_visibility="collapsed")
with col2:
    analyze = st.button("Analyze →")

# ── Main logic ────────────────────────────────────────────────────────────────
if analyze and ticker_input.strip():
    ticker = ticker_input.strip().upper()

    with st.spinner(""):
        stock = yf.Ticker(ticker)
        info  = stock.info
        bs    = stock.balance_sheet
        inc   = stock.income_stmt

    name       = info.get("longName", ticker)
    sector     = info.get("sector", "N/A")
    country    = info.get("country", "N/A")
    market_cap = info.get("marketCap", 0)

    # ── Company banner ──
    st.markdown(f"""
    <div style="background:#14161B;border:0.5px solid rgba(201,168,76,0.2);border-radius:14px;
                padding:1.2rem 1.5rem;display:flex;align-items:center;
                justify-content:space-between;flex-wrap:wrap;gap:12px;margin-bottom:1.5rem;">
      <div style="display:flex;align-items:center;gap:14px;">
        <div style="background:rgba(201,168,76,0.1);border:0.5px solid rgba(201,168,76,0.2);
                    border-radius:6px;padding:4px 10px;font-family:'DM Mono',monospace;
                    font-size:13px;font-weight:500;color:#C9A84C;letter-spacing:1px;">{ticker}</div>
        <div>
          <div style="font-size:18px;font-weight:700;color:#F0EDE6;">{name}</div>
          <div style="font-size:12px;color:#9B9589;margin-top:2px;">{sector} · {country}</div>
        </div>
      </div>
      <div style="text-align:right;">
        <div style="font-size:10px;color:#5C5850;letter-spacing:1px;text-transform:uppercase;">Market Cap</div>
        <div style="font-family:'DM Mono',monospace;font-size:20px;font-weight:500;color:#F0EDE6;margin-top:2px;">{fmt(market_cap)}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Pull financial values ──
    def get_val(df, *keys):
        for k in keys:
            if k in df.index:
                v = df.loc[k].iloc[0]
                try:
                    return float(v)
                except:
                    pass
        return None

    working_capital   = get_val(bs, "Working Capital", "WorkingCapital")
    total_assets      = get_val(bs, "Total Assets", "TotalAssets")
    retained_earnings = get_val(bs, "Retained Earnings", "RetainedEarnings")
    total_liabilities = get_val(bs, "Total Liabilities Net Minority Interest", "Total Liabilities", "TotalLiabilities")
    ebit              = get_val(inc, "EBIT", "Ebit", "Operating Income", "OperatingIncome")
    revenue           = get_val(inc, "Total Revenue", "TotalRevenue", "Revenue")

    # If Working Capital not direct, compute from Current Assets - Current Liabilities
    if working_capital is None:
        ca = get_val(bs, "Current Assets", "CurrentAssets")
        cl = get_val(bs, "Current Liabilities", "CurrentLiabilities")
        if ca and cl:
            working_capital = ca - cl

    # ── Raw Financials section ──
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:1rem;">
      <span style="font-size:10px;font-weight:600;color:#5C5850;letter-spacing:2px;text-transform:uppercase;">Raw Financials</span>
      <div style="flex:1;height:0.5px;background:rgba(201,168,76,0.1);"></div>
    </div>
    """, unsafe_allow_html=True)

    fin_rows = [
        ("Working Capital",   working_capital),
        ("Total Assets",      total_assets),
        ("Retained Earnings", retained_earnings),
        ("EBIT",              ebit),
        ("Total Liabilities", total_liabilities),
        ("Sales / Revenue",   revenue),
    ]

    cols = st.columns(2)
    for i, (label, value) in enumerate(fin_rows):
        with cols[i % 2]:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                        background:#14161B;border-radius:8px;padding:10px 14px;
                        border:0.5px solid rgba(201,168,76,0.08);margin-bottom:8px;">
              <span style="font-size:12px;color:#9B9589;">{label}</span>
              <span style="font-family:'DM Mono',monospace;font-size:13px;font-weight:500;color:#F0EDE6;">{fmt(value)}</span>
            </div>
            """, unsafe_allow_html=True)

    # ── Altman Z-Score ──
    required = [working_capital, total_assets, retained_earnings, ebit, total_liabilities, revenue]

    if all(v is not None for v in required) and total_assets != 0:

        mv = market_cap if market_cap and market_cap > 0 else None

        x1 = working_capital   / total_assets
        x2 = retained_earnings / total_assets
        x3 = ebit              / total_assets
        x4 = (mv / total_liabilities) if (mv and total_liabilities and total_liabilities != 0) else 0
        x5 = revenue           / total_assets

        z = 1.2*x1 + 1.4*x2 + 3.3*x3 + 0.6*x4 + 1.0*x5

        # Zone
        if z > 2.99:
            zone_label = "Safe Zone"
            zone_sub   = "Z > 2.99 · Financially healthy"
            zone_color = "#3FCF8E"
            zone_bg    = "#0D2B1F"
            zone_border= "rgba(63,207,142,0.2)"
            interp     = f"This company appears financially healthy under the Altman Z-score model. A score of <b>{z:.2f}</b> places it firmly in the Safe Zone (Z > 2.99), indicating low probability of financial distress in the near term."
            gauge_pct  = min(95, 50 + (z - 2.99) * 8)
        elif z > 1.81:
            zone_label = "Grey Zone"
            zone_sub   = "1.81 < Z < 2.99 · Moderate risk"
            zone_color = "#F0A030"
            zone_bg    = "#2B1A05"
            zone_border= "rgba(240,160,48,0.2)"
            interp     = f"This company is in the Grey Zone with a score of <b>{z:.2f}</b>. Risk is neither clearly low nor high. Monitor financial trends closely."
            gauge_pct  = 35 + (z - 1.81) * 12
        else:
            zone_label = "Distress Zone"
            zone_sub   = "Z < 1.81 · High risk"
            zone_color = "#F06060"
            zone_bg    = "#2B0D0D"
            zone_border= "rgba(240,96,96,0.2)"
            interp     = f"This company may be facing financial distress. A score of <b>{z:.2f}</b> falls in the Distress Zone (Z < 1.81), suggesting elevated risk of financial difficulty."
            gauge_pct  = max(5, z * 10)

        # X4 warning
        x4_warning = ""
        if not mv or market_cap == 0:
            x4_warning = "<br><span style='color:#F0A030;font-size:11px;'>⚠ Market cap unavailable — X4 set to 0, Z-score may be understated.</span>"
        elif x4 > 10:
            x4_warning = "<br><span style='color:#F0A030;font-size:11px;'>⚠ X4 is very large, which may inflate the Z-score for high-cap firms.</span>"

        # ── Z-Score section ──
        st.markdown("""
        <div style="display:flex;align-items:center;gap:10px;margin:1.5rem 0 1rem;">
          <span style="font-size:10px;font-weight:600;color:#5C5850;letter-spacing:2px;text-transform:uppercase;">Altman Z-Score</span>
          <div style="flex:1;height:0.5px;background:rgba(201,168,76,0.1);"></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background:#14161B;border:0.5px solid rgba(201,168,76,0.2);border-radius:16px;padding:1.5rem;margin-bottom:1.5rem;">
          <div style="display:flex;align-items:flex-end;justify-content:space-between;flex-wrap:wrap;gap:12px;margin-bottom:1.5rem;">
            <div>
              <div style="font-size:10px;color:#5C5850;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:4px;">Z-Score</div>
              <div style="font-family:'DM Mono',monospace;font-size:56px;font-weight:500;color:#C9A84C;line-height:1;">{z:.2f}</div>
            </div>
            <div style="display:flex;align-items:center;gap:8px;background:{zone_bg};border:0.5px solid {zone_border};border-radius:10px;padding:10px 16px;">
              <div style="width:8px;height:8px;border-radius:50%;background:{zone_color};flex-shrink:0;"></div>
              <div>
                <div style="font-size:14px;font-weight:700;color:{zone_color};">{zone_label}</div>
                <div style="font-size:11px;color:{zone_color};opacity:0.6;margin-top:1px;">{zone_sub}</div>
              </div>
            </div>
          </div>
          <div style="position:relative;height:6px;background:#242830;border-radius:3px;margin-bottom:8px;overflow:hidden;">
            <div style="position:absolute;left:0;top:0;height:100%;width:100%;border-radius:3px;
                        background:linear-gradient(90deg,#F06060 0%,#F0A030 40%,#3FCF8E 100%);"></div>
            <div style="position:absolute;top:-2px;left:{gauge_pct}%;width:10px;height:10px;
                        background:#F0EDE6;border-radius:50%;transform:translateX(-50%);
                        border:2px solid #14161B;"></div>
          </div>
          <div style="display:flex;justify-content:space-between;font-size:10px;color:#5C5850;font-family:'DM Mono',monospace;">
            <span>Distress &lt;1.81</span><span>Grey 1.81–2.99</span><span>Safe &gt;2.99</span>
          </div>
          {x4_warning}
        </div>
        """, unsafe_allow_html=True)

        # ── Ratio Breakdown ──
        st.markdown("""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:1rem;">
          <span style="font-size:10px;font-weight:600;color:#5C5850;letter-spacing:2px;text-transform:uppercase;">Ratio Breakdown</span>
          <div style="flex:1;height:0.5px;background:rgba(201,168,76,0.1);"></div>
        </div>
        """, unsafe_allow_html=True)

        ratios = [
            ("X1", "Working Capital / Total Assets",          x1, "× 1.2"),
            ("X2", "Retained Earnings / Total Assets",        x2, "× 1.4"),
            ("X3", "EBIT / Total Assets",                     x3, "× 3.3"),
            ("X4", "Market Value / Total Liabilities",        x4, "× 0.6"),
            ("X5", "Sales / Total Assets",                    x5, "× 1.0"),
        ]

        cols5 = st.columns(5)
        for i, (name_r, formula, val, weight) in enumerate(ratios):
            with cols5[i]:
                st.markdown(f"""
                <div style="background:#1C1F27;border:0.5px solid rgba(201,168,76,0.08);
                            border-radius:10px;padding:0.9rem 0.8rem;text-align:center;">
                  <div style="font-size:11px;font-weight:700;color:#C9A84C;margin-bottom:3px;">{name_r}</div>
                  <div style="font-size:9px;color:#5C5850;margin-bottom:8px;line-height:1.3;">{formula}</div>
                  <div style="font-family:'DM Mono',monospace;font-size:17px;font-weight:500;color:#F0EDE6;">{val:.3f}</div>
                  <div style="font-size:9px;color:#5C5850;margin-top:3px;font-family:'DM Mono',monospace;">{weight}</div>
                </div>
                """, unsafe_allow_html=True)

        # ── Interpretation ──
        st.markdown(f"""
        <div style="margin-top:1.5rem;background:#1C1F27;border-left:2px solid #C9A84C;
                    border-radius:0 10px 10px 0;padding:1rem 1.2rem;
                    font-size:13px;color:#9B9589;line-height:1.7;">
          {interp}
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="background:#2B0D0D;border:0.5px solid rgba(240,96,96,0.2);border-radius:10px;
                    padding:1rem 1.2rem;color:#F06060;font-size:13px;">
          Some required financial fields are missing — Z-score could not be calculated.
        </div>
        """, unsafe_allow_html=True)

elif analyze and not ticker_input.strip():
    st.markdown("""
    <div style="background:#2B1A05;border:0.5px solid rgba(240,160,48,0.2);border-radius:10px;
                padding:1rem 1.2rem;color:#F0A030;font-size:13px;">
      Please enter a ticker symbol.
    </div>
    """, unsafe_allow_html=True)