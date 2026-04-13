import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
import math
import sys
from pathlib import Path

# ── ML model integration (optional — graceful fallback if not trained yet) ───
_ML_DIR = Path(__file__).parent / "ml_model"
if str(_ML_DIR) not in sys.path:
    sys.path.insert(0, str(_ML_DIR))

try:
    from ml_model.predict_ticker import predict_ticker as _ml_predict, model_available as _ml_available  # type: ignore
    _ML_IMPORTED = True
except Exception:
    try:
        from predict_ticker import predict_ticker as _ml_predict, model_available as _ml_available  # type: ignore
        _ML_IMPORTED = True
    except Exception:
        # ── Inline fallback: pure JSON + NumPy prediction (no joblib/xgboost) ──
        import json as _json, numpy as _np
        _ML_IMPORTED = False
        _ML_MODEL_PATH = Path(__file__).parent / "ml_model" / "model" / "distress_model.json"
        _ML_MEDS_PATH  = Path(__file__).parent / "ml_model" / "data"  / "feature_medians.json"
        _ML_THR_PATH   = Path(__file__).parent / "ml_model" / "model" / "threshold.json"
        _ml_bundle_cache = {}

        def _ml_available():
            return _ML_MODEL_PATH.exists()

        def _load_ml_bundle():
            if not _ml_bundle_cache:
                with open(_ML_MODEL_PATH) as f:
                    _ml_bundle_cache["bundle"] = _json.load(f)
                _ml_bundle_cache["meds"] = {}
                if _ML_MEDS_PATH.exists():
                    with open(_ML_MEDS_PATH) as f:
                        _ml_bundle_cache["meds"] = _json.load(f)
                _ml_bundle_cache["thr"] = 0.5
                if _ML_THR_PATH.exists():
                    with open(_ML_THR_PATH) as f:
                        _ml_bundle_cache["thr"] = _json.load(f).get("threshold", 0.5)

        def _ml_logit_predict(feature_row):
            _load_ml_bundle()
            b = _ml_bundle_cache["bundle"]
            w   = _np.array(b["weights"],      dtype=_np.float64)
            bi  = float(b["bias"])
            mu  = _np.array(b["feature_mu"],   dtype=_np.float64)
            std = _np.array(b["feature_std"],  dtype=_np.float64)
            x   = (_np.array(feature_row, dtype=_np.float64) - mu) / std
            logit = float(_np.dot(x, w) + bi)
            logit = max(-50.0, min(50.0, logit))
            return 1.0 / (1.0 + math.exp(-logit))

        def _ml_predict(ticker: str) -> dict:
            if not _ml_available():
                return {"error": "Model not trained yet.", "probability": None}
            _load_ml_bundle()
            b    = _ml_bundle_cache["bundle"]
            meds = _ml_bundle_cache["meds"]
            thr  = _ml_bundle_cache["thr"]
            feat_cols = b["feature_cols"]

            # Build features using the same yfinance logic as the full script
            stock = yf.Ticker(ticker)
            info  = stock.info or {}

            def _gb(row, col=0):
                try:
                    bs = stock.balance_sheet
                    if bs is None or bs.empty or row not in bs.index: return None
                    return float(bs.loc[row].iloc[col])
                except: return None
            def _gi(row, col=0):
                try:
                    inc = stock.income_stmt
                    if inc is None or inc.empty or row not in inc.index: return None
                    return float(inc.loc[row].iloc[col])
                except: return None
            def _gc(row, col=0):
                try:
                    cf = stock.cashflow
                    if cf is None or cf.empty or row not in cf.index: return None
                    return float(cf.loc[row].iloc[col])
                except: return None
            def _p(*keys, src="bs", col=0):
                fn = {"bs":_gb,"inc":_gi,"cf":_gc}[src]
                for k in keys:
                    v = fn(k, col)
                    if v is not None and math.isfinite(v): return v
                return math.nan

            at0  = _p("Total Assets","TotalAssets")
            lt0  = _p("Total Liabilities Net Minority Interest","Total Liab")
            act0 = _p("Current Assets","Total Current Assets","CurrentAssets")
            lct0 = _p("Current Liabilities","Total Current Liabilities","CurrentLiabilities")
            re0  = _p("Retained Earnings","RetainedEarnings")
            ebit0= _p("EBIT","Ebit",src="inc")
            sale0= _p("Total Revenue","Revenue","Net Revenues",src="inc")
            ni0  = _p("Net Income","NetIncome",src="inc")
            ib0  = _p("Net Income Common Stockholders","Net Income","NetIncome",src="inc")
            cfo0 = _p("Operating Cash Flow",src="cf")
            dltt0= _p("Long Term Debt","LongTermDebt")
            dlc0 = _p("Current Debt","Short Long Term Debt")
            dp0  = _p("Depreciation And Amortization","Depreciation",src="cf")
            rect0= _p("Receivables","Net Receivables","Accounts Receivable")
            cogs0= _p("Cost Of Revenue","Cost Of Goods Sold",src="inc")
            xsga0= _p("Selling General Administrative",src="inc")
            ppent0=_p("Net PPE","Property Plant Equipment")
            che0 = _p("Cash And Cash Equivalents","Cash Cash Equivalents And Short Term Investments")
            ceq0 = _p("Stockholders Equity","Common Stock Equity","Total Stockholders Equity")
            mkt0 = float(info.get("marketCap") or max(0, ceq0) if not math.isnan(ceq0) else math.nan)

            at1  = _p("Total Assets",col=1); lt1=_p("Total Liabilities Net Minority Interest","Total Liab",col=1)
            sale1= _p("Total Revenue","Revenue",src="inc",col=1); ni1=_p("Net Income","NetIncome",src="inc",col=1)
            cfo1 = _p("Operating Cash Flow",src="cf",col=1); dltt1=_p("Long Term Debt",col=1); dlc1=_p("Current Debt",col=1)
            act1 = _p("Current Assets","Total Current Assets",col=1); lct1=_p("Current Liabilities","Total Current Liabilities",col=1)
            cogs1= _p("Cost Of Revenue","Cost Of Goods Sold",src="inc",col=1); dp1=_p("Depreciation And Amortization",src="cf",col=1)
            ppent1=_p("Net PPE","Property Plant Equipment",col=1); xsga1=_p("Selling General Administrative",src="inc",col=1)
            rect1= _p("Receivables","Net Receivables",col=1)

            def sd(a,b_): return a/b_ if (not math.isnan(a) and not math.isnan(b_) and b_!=0) else math.nan
            wcap=act0-lct0 if not(math.isnan(act0) or math.isnan(lct0)) else math.nan
            ib=ib0 if not math.isnan(ib0) else ni0

            feat = {}
            feat["z_x1"]=sd(wcap,at0); feat["z_x2"]=sd(re0,at0); feat["z_x3"]=sd(ebit0,at0)
            feat["z_x4"]=sd(mkt0,lt0); feat["z_x5"]=sd(sale0,at0)
            feat["z_score"]=1.2*feat["z_x1"]+1.4*feat["z_x2"]+3.3*feat["z_x3"]+0.6*feat["z_x4"]+feat["z_x5"]
            feat["o_x1"]=math.log(at0) if at0>0 else math.nan
            feat["o_x2"]=sd(lt0,at0); feat["o_x3"]=sd(wcap,at0)
            feat["o_x4"]=1. if lt0>at0 else 0.
            feat["o_x5"]=sd(ib+dp0,lt0); feat["o_x6"]=sd(ib,at0); feat["o_x7"]=sd(ib+dp0,at0)
            feat["o_x8"]=1. if (ni0<0 and ni1<0) else 0.
            dn9=abs(ni0)+abs(ni1)
            feat["o_x9"]=(ni0-ni1)/dn9 if (dn9>0 and not math.isnan(dn9)) else math.nan
            o_r=(-1.32-0.407*feat["o_x1"]+6.03*feat["o_x2"]-1.43*feat["o_x3"]+0.076*feat["o_x4"]
                 -1.72*feat["o_x5"]-2.37*feat["o_x6"]-1.83*feat["o_x7"]+0.285*feat["o_x8"]-0.521*feat["o_x9"])
            feat["o_score_raw"]=o_r; feat["o_prob"]=1/(1+math.exp(-max(-50,min(50,o_r)))) if math.isfinite(o_r) else math.nan
            roa0v=sd(ni0,at0); roa1v=sd(ni1,at1); cfo_ta=sd(cfo0,at0)
            lev0v=sd(dltt0,at0); lev1v=sd(dltt1,at1)
            cr0v=sd(act0,lct0); cr1v=sd(act1,lct1)
            gm0v=sd(sale0-cogs0,sale0); gm1v=sd(sale1-cogs1,sale1)
            at0v=sd(sale0,at0); at1v=sd(sale1,at1)
            feat["f1"]=1. if roa0v>0 else 0.; feat["f2"]=1. if cfo0>0 else 0.
            feat["f3"]=1. if (not math.isnan(roa1v) and roa0v>roa1v) else 0.
            feat["f4"]=1. if (not math.isnan(cfo_ta) and cfo_ta>roa0v) else 0.
            feat["f5"]=1. if (not math.isnan(lev1v) and lev0v<lev1v) else 0.
            feat["f6"]=1. if (not math.isnan(cr1v) and cr0v>cr1v) else 0.
            feat["f7"]=0.; feat["f8"]=1. if (not math.isnan(gm1v) and gm0v>gm1v) else 0.
            feat["f9"]=1. if (not math.isnan(at1v) and at0v>at1v) else 0.
            feat["f_score"]=sum(feat[f"f{i}"] for i in range(1,10))
            dsri=sd(sd(rect0,sale0),sd(rect1,sale1)); gmi=sd(sd(sale1-cogs1,sale1),sd(sale0-cogs0,sale0))
            np0v=at0-ppent0-che0; np1v=at1-ppent1
            aqi=sd(sd(np0v,at0),sd(np1v,at1)); sgi=sd(sale0,sale1)
            dep0v=sd(dp0,dp0+ppent0); dep1v=sd(dp1,dp1+ppent1); depi=sd(dep1v,dep0v)
            sgai=sd(sd(xsga0,sale0),sd(xsga1,sale1)); tata=sd(ib-cfo0,at0)
            dbt0v=(dltt0+dlc0)/at0 if at0 else math.nan; dbt1v=(dltt1+dlc1)/at1 if (not math.isnan(at1) and at1) else math.nan
            lvgi=sd(dbt0v,dbt1v)
            feat.update(dict(dsri=dsri,gmi=gmi,aqi=aqi,sgi=sgi,depi=depi,sgai=sgai,tata=tata,lvgi=lvgi))
            m=(-4.84+0.920*dsri+0.528*gmi+0.404*aqi+0.892*sgi+0.115*depi-0.172*sgai+4.679*tata-0.327*lvgi)
            feat["m_score"]=m if math.isfinite(m) else math.nan
            feat.update(dict(roa=roa0v,cfo_ta=cfo_ta,cr=cr0v,lev=lev0v,gm=gm0v,asset_turn=at0v,
                             delta_roa=roa0v-roa1v if not math.isnan(roa1v) else math.nan,
                             delta_cr=cr0v-cr1v if not math.isnan(cr1v) else math.nan,
                             delta_lev=lev0v-lev1v if not math.isnan(lev1v) else math.nan,
                             delta_gm=gm0v-gm1v if not math.isnan(gm1v) else math.nan,
                             delta_sale=sd(sale0-sale1,sale1), delta_at=sd(at0-at1,at1),
                             debt_ratio=sd(lt0,at0), equity_ratio=sd(ceq0,at0),
                             log_at=math.log(at0) if at0>0 else math.nan,
                             log_sale=math.log(sale0) if sale0>0 else math.nan))

            row = []
            n_miss = 0
            for fc in feat_cols:
                v = feat.get(fc, math.nan)
                if math.isnan(float(v) if v is not None else math.nan):
                    v = meds.get(fc, 0.0); n_miss += 1
                row.append(float(v))

            prob = _ml_logit_predict(row)
            if prob < 0.15:   rl, rc = "Low Risk",      "#3FCF8E"
            elif prob < 0.40: rl, rc = "Elevated Risk", "#F0A030"
            else:              rl, rc = "High Risk",     "#E85555"

            return dict(probability=prob, prediction=int(prob>=thr), threshold=thr,
                        risk_label=rl, risk_color=rc, features=feat,
                        missing_pct=n_miss/len(feat_cols), error=None)

st.set_page_config(
    page_title="Financial Distress App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session-state navigation (no browser page reloads) ────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "home"
if "history" not in st.session_state:
    st.session_state.history = []

current_page = st.session_state.page

def go(page):
    """Navigate to a page without any browser reload."""
    if page != st.session_state.page:
        st.session_state.history.append(st.session_state.page)
    st.session_state.page = page
    st.rerun()

def go_back():
    """Go to the previous page."""
    if st.session_state.history:
        st.session_state.page = st.session_state.history.pop()
    else:
        st.session_state.page = "home"
    st.rerun()

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Mono:wght@400;500&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color: #0E0F12 !important;
    color: #F0EDE6 !important;
    font-family: 'Syne', sans-serif !important;
}
[data-testid="stHeader"] { background: transparent !important; }
.block-container { padding: 2rem 2.5rem 4rem !important; max-width: 100% !important; }
#MainMenu, footer, header { visibility: hidden; }

/* ── Sidebar ──────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    display: flex !important;
    visibility: visible !important;
    background-color: #0B0C0F !important;
    border-right: 0.5px solid rgba(201,168,76,0.12) !important;
    min-width: 220px !important;
    max-width: 220px !important;
    width: 220px !important;
    transform: none !important;
}
[data-testid="stSidebarContent"] { padding: 1.8rem 0.8rem 1rem !important; }
[data-testid="stSidebarCollapseButton"],
button[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"] { display: none !important; }
[data-testid="stSidebarUserContent"] { padding: 0 !important; }

.brand {
    display: flex; align-items: center; gap: 12px;
    margin-bottom: 2.2rem; padding: 0 0.4rem;
}
.brand-icon {
    width: 38px; height: 38px; border: 1.5px solid #C9A84C;
    border-radius: 9px; display: flex; align-items: center;
    justify-content: center; flex-shrink: 0;
}
.brand-icon-inner { width: 16px; height: 16px; background: #C9A84C; border-radius: 3px; }
.brand-name-1 { font-size: 15px; font-weight: 800; color: #F0EDE6; line-height: 1.2; }
.brand-name-2 { font-size: 15px; font-weight: 800; color: #C9A84C; line-height: 1.2; }

.nav-section-label {
    font-size: 10px; color: #5C5850; letter-spacing: 2px; font-weight: 600;
    text-transform: uppercase; margin: 0 0 0.5rem 0; padding: 0 0.5rem;
}

/* ── Sidebar nav buttons (inactive items) ─────────────────────────────────── */
[data-testid="stSidebar"] [data-testid="stButton"] > button {
    background: transparent !important;
    color: #6B6560 !important;
    border: none !important;
    text-align: left !important;
    justify-content: flex-start !important;
    padding: 9px 12px !important;
    border-radius: 9px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    height: auto !important;
    min-height: 38px !important;
    letter-spacing: 0 !important;
    line-height: 1.4 !important;
    box-shadow: none !important;
    margin-bottom: 3px !important;
    width: 100% !important;
}
[data-testid="stSidebar"] [data-testid="stButton"] > button:hover {
    background: rgba(201,168,76,0.07) !important;
    color: #D4CECC !important;
}
[data-testid="stSidebar"] [data-testid="stButton"] > button:focus {
    box-shadow: none !important;
    outline: none !important;
}

/* ── Main area inputs & buttons ───────────────────────────────────────────── */
[data-testid="stTextInput"] input {
    background: #1C1F27 !important;
    border: 0.5px solid rgba(201,168,76,0.25) !important;
    border-radius: 10px !important; color: #F0EDE6 !important;
    font-family: 'DM Mono', monospace !important; font-size: 15px !important;
    font-weight: 500 !important; letter-spacing: 2px !important; padding: 0 16px !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #C9A84C !important; box-shadow: none !important;
}
/* Main-area buttons (gold) — only outside the sidebar */
.main [data-testid="stButton"] > button,
[data-testid="stMain"] [data-testid="stButton"] > button {
    background: #C9A84C !important; color: #0E0F12 !important;
    border: none !important; border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important; font-size: 14px !important;
    font-weight: 700 !important; height: 46px !important;
    letter-spacing: 0.5px !important; width: 100% !important;
}
.main [data-testid="stButton"] > button:hover,
[data-testid="stMain"] [data-testid="stButton"] > button:hover {
    background: #E8C97A !important; color: #0E0F12 !important;
}
/* Back button — subtle style */
.back-btn [data-testid="stButton"] > button {
    background: transparent !important;
    color: #9B9589 !important;
    border: 0.5px solid rgba(201,168,76,0.2) !important;
    font-size: 13px !important;
    height: 36px !important;
    width: auto !important;
    padding: 0 14px !important;
    letter-spacing: 0 !important;
}
.back-btn [data-testid="stButton"] > button:hover {
    background: rgba(201,168,76,0.07) !important;
    color: #F0EDE6 !important;
}
/* Home card open buttons — smaller, outlined */
.card-open-btn [data-testid="stButton"] > button {
    background: rgba(201,168,76,0.06) !important;
    color: #C9A84C !important;
    border: 0.5px solid rgba(201,168,76,0.25) !important;
    font-size: 12px !important;
    height: 36px !important;
    letter-spacing: 0.3px !important;
    border-top-left-radius: 0 !important;
    border-top-right-radius: 0 !important;
}
.card-open-btn [data-testid="stButton"] > button:hover {
    background: rgba(201,168,76,0.12) !important;
    color: #E8C97A !important;
}

/* ── Home page model cards ────────────────────────────────────────────────── */
.model-card {
    background: #14161B;
    border: 0.5px solid rgba(201,168,76,0.15); border-radius: 14px;
    padding: 1.3rem 1.4rem; color: inherit !important;
}
.model-card-live {
    border-bottom-left-radius: 0 !important;
    border-bottom-right-radius: 0 !important;
    border-bottom: none !important;
    transition: border-color 0.2s, background 0.2s;
    cursor: pointer;
}
.model-card-live:hover { border-color: rgba(201,168,76,0.35); background: #1A1C23; }
.model-card-header {
    display: flex; align-items: center;
    justify-content: space-between; margin-bottom: 10px;
}
.model-card-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
.model-card-name { font-size: 16px; font-weight: 700; color: #F0EDE6; }
.model-card-badge {
    font-size: 9px; font-weight: 700; letter-spacing: 0.5px;
    padding: 3px 8px; border-radius: 5px; font-family: 'DM Mono', monospace;
}
.model-card-desc { font-size: 12px; color: #6B6560; line-height: 1.5; margin-top: 4px; }
.model-card-soon { opacity: 0.55; margin-top: 1rem; }

.qs-box {
    background: #14161B; border: 0.5px solid rgba(201,168,76,0.12);
    border-radius: 14px; padding: 1.3rem 1.6rem;
}
.qs-step {
    display: flex; align-items: flex-start; gap: 12px;
    margin-bottom: 10px; font-size: 13px; color: #9B9589; line-height: 1.5;
}
.qs-step:last-child { margin-bottom: 0; }
.qs-num {
    font-family: 'DM Mono', monospace; font-size: 12px; font-weight: 500;
    color: #C9A84C; background: rgba(201,168,76,0.1); border-radius: 50%;
    width: 22px; height: 22px; display: flex; align-items: center;
    justify-content: center; flex-shrink: 0; margin-top: 1px;
}

/* ── Section dividers ─────────────────────────────────────────────────────── */
.sec-hdr { display: flex; align-items: center; gap: 10px; margin: 1.5rem 0 0.8rem; }
.sec-lbl { font-size: 10px; font-weight: 600; color: #5C5850; letter-spacing: 2px; text-transform: uppercase; white-space: nowrap; }
.sec-line { flex: 1; height: 0.5px; background: rgba(201,168,76,0.1); }

/* ── Animations ───────────────────────────────────────────────────────────── */
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


# ── Models ────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
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


@st.cache_data(ttl=3600, show_spinner=False)
def compute_oscore(ticker_str):
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

    if wc is None and ca and cl: wc = ca - cl

    ni_prev = None
    if inc is not None and len(inc.columns) > 1:
        for k in ["Net Income", "NetIncome"]:
            if k in inc.index:
                try:
                    ni_prev = float(inc.loc[k].iloc[1])
                    break
                except: pass

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

    o = (-1.32 - 0.407*x1 + 6.03*x2 - 1.43*x3 + 0.076*x4
         - 1.72*x5 - 2.37*x6 - 1.83*x7 + 0.285*x8 - 0.521*x9)
    prob = 1 / (1 + math.exp(-o))

    return dict(
        ok=True, name=name, sector=sector, country=country, logo=logo, mc=mc,
        o=o, prob=prob,
        x1=x1, x2=x2, x3=x3, x4=x4, x5=x5,
        x6=x6, x7=x7, x8=x8, x9=x9,
        ta=ta, tl=tl, wc=wc, ca=ca, cl=cl, ni=ni, cfo=cfo, ni_prev=ni_prev
    )


@st.cache_data(ttl=3600, show_spinner=False)
def compute_fscore(ticker_str):
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

    def _g(df, *keys, yr=0):
        for k in keys:
            if df is not None and k in df.index:
                try:
                    if len(df.columns) > yr:
                        v = float(df.loc[k].iloc[yr])
                        if not math.isnan(v):
                            return v
                except: pass
        return None

    ta0  = _g(bs,  "Total Assets",  "TotalAssets")
    ta1  = _g(bs,  "Total Assets",  "TotalAssets",                                        yr=1)
    ni0  = _g(inc, "Net Income",    "NetIncome")
    ni1  = _g(inc, "Net Income",    "NetIncome",                                           yr=1)
    cfo0 = _g(cf,  "Operating Cash Flow", "Cash Flow From Continuing Operating Activities")
    ltd0 = _g(bs,  "Long Term Debt", "LongTermDebt", "Long Term Debt And Capital Lease Obligation")
    ltd1 = _g(bs,  "Long Term Debt", "LongTermDebt", "Long Term Debt And Capital Lease Obligation", yr=1)
    ca0  = _g(bs,  "Current Assets",      "CurrentAssets")
    ca1  = _g(bs,  "Current Assets",      "CurrentAssets",       yr=1)
    cl0  = _g(bs,  "Current Liabilities", "CurrentLiabilities")
    cl1  = _g(bs,  "Current Liabilities", "CurrentLiabilities",  yr=1)
    gp0  = _g(inc, "Gross Profit",  "GrossProfit")
    gp1  = _g(inc, "Gross Profit",  "GrossProfit",               yr=1)
    rv0  = _g(inc, "Total Revenue", "TotalRevenue", "Revenue")
    rv1  = _g(inc, "Total Revenue", "TotalRevenue", "Revenue",   yr=1)
    sh0  = _g(bs,  "Ordinary Shares Number", "Share Issued")
    sh1  = _g(bs,  "Ordinary Shares Number", "Share Issued",     yr=1)

    if ta0 is None or ni0 is None or cfo0 is None:
        return dict(ok=False, name=name, sector=sector, country=country, logo=logo, mc=mc)

    roa0   = ni0 / ta0 if ta0 else 0
    roa1   = (ni1 / ta1) if (ni1 is not None and ta1) else None
    cfo_ta = cfo0 / ta0 if ta0 else 0

    def _pct(a, b): return f"{a:.3f} vs {b:.3f}"

    flags = {}
    # ── A: Profitability ──────────────────────────────────────────────────────
    flags["F1"] = ("ROA > 0",       "Net income / assets positive",       roa0 > 0,         f"{roa0:.3f}")
    flags["F2"] = ("CFO > 0",       "Operating cash flow positive",       cfo0 > 0,         fmt(cfo0))
    flags["F3"] = ("ΔROA > 0",      "ROA improved year-over-year",
                   (roa0 > roa1) if roa1 is not None else False,
                   (_pct(roa0, roa1) if roa1 is not None else "N/A"))
    flags["F4"] = ("Accruals",      "CFO / TA > ROA  (earnings quality)", cfo_ta > roa0,    _pct(cfo_ta, roa0))
    # ── B: Leverage & Liquidity ───────────────────────────────────────────────
    if ltd0 is not None and ltd1 is not None and ta0 and ta1:
        lev0, lev1 = ltd0/ta0, ltd1/ta1
        flags["F5"] = ("ΔLeverage",    "Long-term leverage decreased",   lev0 < lev1,       _pct(lev0, lev1))
    else:
        flags["F5"] = ("ΔLeverage",    "Long-term leverage decreased",   False,             "N/A")
    if ca0 and cl0 and ca1 and cl1:
        cr0, cr1 = ca0/cl0, ca1/cl1
        flags["F6"] = ("ΔCurr Ratio",  "Current ratio improved",         cr0 > cr1,         f"{cr0:.2f} vs {cr1:.2f}")
    else:
        flags["F6"] = ("ΔCurr Ratio",  "Current ratio improved",         False,             "N/A")
    if sh0 is not None and sh1 is not None:
        flags["F7"] = ("No Dilution",  "Shares outstanding not increased", sh0 <= sh1,      f"{sh0/1e6:.1f}M vs {sh1/1e6:.1f}M")
    else:
        flags["F7"] = ("No Dilution",  "Shares outstanding not increased", False,           "N/A")
    # ── C: Operating Efficiency ───────────────────────────────────────────────
    if gp0 and gp1 and rv0 and rv1:
        gm0, gm1 = gp0/rv0, gp1/rv1
        flags["F8"] = ("ΔGross Margin","Gross margin improved",           gm0 > gm1,        _pct(gm0, gm1))
    else:
        flags["F8"] = ("ΔGross Margin","Gross margin improved",           False,            "N/A")
    if rv0 and rv1 and ta0 and ta1:
        at0, at1 = rv0/ta0, rv1/ta1
        flags["F9"] = ("ΔAsset Turn.", "Asset turnover improved",         at0 > at1,        _pct(at0, at1))
    else:
        flags["F9"] = ("ΔAsset Turn.", "Asset turnover improved",         False,            "N/A")

    score = sum(1 for v in flags.values() if v[2])
    return dict(
        ok=True, name=name, sector=sector, country=country, logo=logo, mc=mc,
        score=score, flags=flags,
        ta0=ta0, ni0=ni0, cfo0=cfo0, ltd0=ltd0, ca0=ca0, cl0=cl0,
        gp0=gp0, rv0=rv0, roa0=roa0
    )


@st.cache_data(ttl=3600, show_spinner=False)
def compute_mscore(ticker_str):
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

    def _g(df, *keys, yr=0):
        for k in keys:
            if df is not None and k in df.index:
                try:
                    if len(df.columns) > yr:
                        v = float(df.loc[k].iloc[yr])
                        if not math.isnan(v):
                            return v
                except: pass
        return None

    rec0  = _g(bs,  "Net Receivables",   "Accounts Receivable",    "Receivables")
    rec1  = _g(bs,  "Net Receivables",   "Accounts Receivable",    "Receivables",                         yr=1)
    rv0   = _g(inc, "Total Revenue",     "TotalRevenue",            "Revenue")
    rv1   = _g(inc, "Total Revenue",     "TotalRevenue",            "Revenue",                             yr=1)
    cogs0 = _g(inc, "Cost Of Revenue",   "CostOfRevenue",           "Cost Of Goods Sold")
    cogs1 = _g(inc, "Cost Of Revenue",   "CostOfRevenue",           "Cost Of Goods Sold",                  yr=1)
    ca0   = _g(bs,  "Current Assets",    "CurrentAssets")
    ca1   = _g(bs,  "Current Assets",    "CurrentAssets",                                                  yr=1)
    ppe0  = _g(bs,  "Net PPE",           "Property Plant Equipment Net", "Net Property Plant And Equipment")
    ppe1  = _g(bs,  "Net PPE",           "Property Plant Equipment Net", "Net Property Plant And Equipment", yr=1)
    ta0   = _g(bs,  "Total Assets",      "TotalAssets")
    ta1   = _g(bs,  "Total Assets",      "TotalAssets",                                                    yr=1)
    dep0  = _g(cf,  "Depreciation And Amortization", "Depreciation", "Depreciation Amortization Depletion")
    dep1  = _g(cf,  "Depreciation And Amortization", "Depreciation", "Depreciation Amortization Depletion", yr=1)
    sga0  = _g(inc, "Selling General And Administrative", "SGAExpense", "Selling And Marketing Expense")
    sga1  = _g(inc, "Selling General And Administrative", "SGAExpense", "Selling And Marketing Expense",   yr=1)
    ni0   = _g(inc, "Net Income",        "NetIncome")
    cfo0  = _g(cf,  "Operating Cash Flow", "Cash Flow From Continuing Operating Activities")
    ltd0  = _g(bs,  "Long Term Debt",    "LongTermDebt",            "Long Term Debt And Capital Lease Obligation")
    ltd1  = _g(bs,  "Long Term Debt",    "LongTermDebt",            "Long Term Debt And Capital Lease Obligation", yr=1)
    cl0   = _g(bs,  "Current Liabilities", "CurrentLiabilities")
    cl1   = _g(bs,  "Current Liabilities", "CurrentLiabilities",                                           yr=1)

    if any(v is None for v in [rv0, rv1, ta0, ta1]):
        return dict(ok=False, name=name, sector=sector, country=country, logo=logo, mc=mc)

    idx = {}  # key → (full_name, value, elevated_flag)

    if rec0 and rec1 and rv0 and rv1:
        v = (rec0/rv0) / (rec1/rv1)
        idx["DSRI"] = ("Days Sales Receivable Index",     v,    v > 1.031)
    else:
        idx["DSRI"] = ("Days Sales Receivable Index",     None, False)

    if cogs0 and cogs1 and rv0 and rv1:
        gm0 = (rv0 - cogs0) / rv0;  gm1 = (rv1 - cogs1) / rv1
        v   = (gm1 / gm0) if gm0 != 0 else None
        idx["GMI"]  = ("Gross Margin Index",              v,    (v > 1.014) if v else False)
    else:
        idx["GMI"]  = ("Gross Margin Index",              None, False)

    if ca0 and ppe0 and ta0 and ca1 and ppe1 and ta1:
        aq0 = 1 - (ca0+ppe0)/ta0;  aq1 = 1 - (ca1+ppe1)/ta1
        v   = (aq0 / aq1) if aq1 != 0 else None
        idx["AQI"]  = ("Asset Quality Index",             v,    (v > 1.039) if v else False)
    else:
        idx["AQI"]  = ("Asset Quality Index",             None, False)

    v = rv0/rv1 if rv1 != 0 else None
    idx["SGI"]  = ("Sales Growth Index",              v,    (v > 1.134) if v else False)

    if dep0 and ppe0 and dep1 and ppe1:
        d0 = dep0/(dep0+ppe0) if (dep0+ppe0) != 0 else None
        d1 = dep1/(dep1+ppe1) if (dep1+ppe1) != 0 else None
        v  = (d1/d0) if (d0 and d0 != 0) else None
        idx["DEPI"] = ("Depreciation Index",              v,    (v > 1.001) if v else False)
    else:
        idx["DEPI"] = ("Depreciation Index",              None, False)

    if sga0 and sga1 and rv0 and rv1:
        v = (sga0/rv0) / (sga1/rv1)
        idx["SGAI"] = ("SG&A Expense Index",              v,    v > 1.054)
    else:
        idx["SGAI"] = ("SG&A Expense Index",              None, False)

    if ni0 is not None and cfo0 is not None and ta0:
        v = (ni0 - cfo0) / ta0
        idx["TATA"] = ("Total Accruals / Total Assets",   v,    v > 0.018)
    else:
        idx["TATA"] = ("Total Accruals / Total Assets",   None, False)

    if ltd0 is not None and cl0 and ta0 and ltd1 is not None and cl1 and ta1:
        lev0 = (ltd0+cl0)/ta0;  lev1 = (ltd1+cl1)/ta1
        v    = (lev0/lev1) if lev1 != 0 else None
        idx["LVGI"] = ("Leverage Index",                  v,    (v > 1.0) if v else False)
    else:
        idx["LVGI"] = ("Leverage Index",                  None, False)

    def _v(k): return idx[k][1] if idx[k][1] is not None else 1.0

    m = (-4.84
         + 0.920 * _v("DSRI")
         + 0.528 * _v("GMI")
         + 0.404 * _v("AQI")
         + 0.892 * _v("SGI")
         + 0.115 * _v("DEPI")
         - 0.172 * _v("SGAI")
         + 4.679 * _v("TATA")
         - 0.327 * _v("LVGI"))

    return dict(
        ok=True, name=name, sector=sector, country=country, logo=logo, mc=mc,
        m=m, idx=idx, rv0=rv0, ta0=ta0, ni0=ni0, cfo0=cfo0
    )


# ── Zone helpers ──────────────────────────────────────────────────────────────
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

def o_zone(prob):
    p = prob * 100
    if p < 20:
        return "Low Risk",    f"{p:.1f}%", "#3FCF8E", "#0D2B1F", "rgba(63,207,142,0.2)"
    if p < 50:
        return "Medium Risk", f"{p:.1f}%", "#F0A030", "#2B1A05", "rgba(240,160,48,0.2)"
    return     "High Risk",   f"{p:.1f}%", "#F06060", "#2B0D0D", "rgba(240,96,96,0.2)"


# ── Render components ─────────────────────────────────────────────────────────
def render_zscore_panel(z, d):
    zl, zs, zc, zbg, zbd = zone_info(z)
    gp = gauge_pct(z)
    warn_html = ""
    if d.get('x4_warn'):
        if not d['mc'] or d['mc'] == 0:
            warn_html = '<div style="margin-top:12px;font-size:11px;color:#F0A030;">&#9888; Market cap unavailable — X4 set to 0, score may be understated.</div>'
        elif d['x4'] > 10:
            warn_html = '<div style="margin-top:12px;font-size:11px;color:#F0A030;">&#9888; X4 is very large — may inflate Z-score for high-cap firms.</div>'
    html = f"""<!DOCTYPE html><html>
<head><link href="https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Mono:wght@500&display=swap" rel="stylesheet">
<style>*{{box-sizing:border-box;margin:0;padding:0;}}body{{background:transparent;font-family:'Syne',sans-serif;}}
.panel{{background:#14161B;border:0.5px solid rgba(201,168,76,0.2);border-radius:16px;padding:1.4rem 1.5rem;}}
.top{{display:flex;align-items:flex-end;justify-content:space-between;flex-wrap:wrap;gap:12px;margin-bottom:1.4rem;}}
.label{{font-size:10px;color:#5C5850;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:4px;}}
.zscore{{font-family:'DM Mono',monospace;font-size:56px;font-weight:500;color:#C9A84C;line-height:1;}}
.zone-box{{display:flex;align-items:center;gap:8px;background:{zbg};border:0.5px solid {zbd};border-radius:10px;padding:10px 16px;}}
.zone-dot{{width:8px;height:8px;border-radius:50%;background:{zc};flex-shrink:0;}}
.zone-label{{font-size:14px;font-weight:700;color:{zc};}}.zone-sub{{font-size:11px;color:{zc};opacity:0.6;margin-top:1px;}}
.gauge-wrap{{position:relative;height:6px;background:#242830;border-radius:3px;margin-bottom:8px;overflow:visible;}}
.gauge-track{{position:absolute;left:0;top:0;height:100%;width:100%;border-radius:3px;background:linear-gradient(90deg,#F06060 0%,#F0A030 40%,#3FCF8E 100%);}}
.gauge-marker{{position:absolute;top:-3px;left:5%;width:10px;height:10px;background:#F0EDE6;border-radius:50%;transform:translateX(-50%);border:2px solid #14161B;transition:left 1.2s cubic-bezier(.4,0,.2,1);}}
.gauge-labels{{display:flex;justify-content:space-between;font-size:10px;color:#5C5850;font-family:'DM Mono',monospace;}}</style></head>
<body><div class="panel">
  <div class="top"><div><div class="label">Z-Score</div><div class="zscore" id="znum">0.00</div></div>
    <div class="zone-box"><div class="zone-dot"></div>
      <div><div class="zone-label">{zl}</div><div class="zone-sub">{zs}</div></div></div></div>
  <div class="gauge-wrap"><div class="gauge-track"></div><div class="gauge-marker" id="gmark"></div></div>
  <div class="gauge-labels"><span>Distress &lt;1.81</span><span>Grey 1.81–2.99</span><span>Safe &gt;2.99</span></div>
  {warn_html}
</div>
<script>
var target={z:.4f},gp={gp:.1f};
var el=document.getElementById('znum'),mk=document.getElementById('gmark');
var start=null,dur=1200;
function step(ts){{if(!start)start=ts;var p=Math.min((ts-start)/dur,1),e=1-Math.pow(1-p,3);
  el.textContent=(target*e).toFixed(2);if(p<1)requestAnimationFrame(step);else el.textContent=target.toFixed(2);}}
requestAnimationFrame(step);setTimeout(function(){{mk.style.left=gp+'%';}},80);
</script></body></html>"""
    components.html(html, height=180)


def render_comparison_card(ticker, d, delay_ms=100):
    if not d["ok"]:
        components.html(f"""<div style="background:#14161B;border:0.5px solid rgba(240,96,96,0.2);border-radius:14px;
                    padding:1.5rem;text-align:center;font-family:sans-serif;">
          <div style="font-family:monospace;font-size:14px;color:#C9A84C;margin-bottom:8px;">{ticker}</div>
          <div style="color:#F06060;font-size:12px;">Data unavailable</div></div>""", height=120)
        return
    z  = d["z"]
    zl, _, zc, zbg, zbd = zone_info(z)
    gp = gauge_pct(z)
    initials_c = ticker[:2]
    if d["logo"]:
        logo_tag = f'<div style="width:36px;height:36px;border-radius:8px;background:#1C1F27;padding:4px;margin:0 auto 8px;display:flex;align-items:center;justify-content:center;"><img src="{d["logo"]}" style="width:28px;height:28px;object-fit:contain;"></div>'
    else:
        logo_tag = f'<div style="width:36px;height:36px;border-radius:8px;background:rgba(201,168,76,0.15);border:0.5px solid rgba(201,168,76,0.3);display:flex;align-items:center;justify-content:center;font-family:monospace;font-size:12px;font-weight:500;color:#C9A84C;margin:0 auto 8px;">{initials_c}</div>'
    components.html(f"""<!DOCTYPE html><html>
<head><link href="https://fonts.googleapis.com/css2?family=Syne:wght@700&family=DM+Mono:wght@500&display=swap" rel="stylesheet">
<style>*{{box-sizing:border-box;margin:0;padding:0;}}body{{background:transparent;font-family:'Syne',sans-serif;}}
.card{{background:#14161B;border:0.5px solid rgba(201,168,76,0.15);border-radius:14px;padding:1.2rem 1rem;text-align:center;}}
.ticker{{font-family:'DM Mono',monospace;font-size:12px;color:#C9A84C;letter-spacing:1px;margin-bottom:3px;}}
.cname{{font-size:11px;color:#5C5850;margin-bottom:14px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}}
.znum{{font-family:'DM Mono',monospace;font-size:40px;font-weight:500;color:#C9A84C;line-height:1;margin-bottom:10px;}}
.zbadge{{display:inline-block;background:{zbg};border:0.5px solid {zbd};border-radius:8px;padding:5px 12px;margin-bottom:14px;}}
.zbadge span{{font-size:12px;font-weight:700;color:{zc};}}
.gbar{{position:relative;height:4px;background:#242830;border-radius:2px;overflow:visible;margin-bottom:6px;}}
.gtrack{{position:absolute;left:0;top:0;height:100%;width:100%;background:linear-gradient(90deg,#F06060,#F0A030,#3FCF8E);border-radius:2px;}}
.gmarker{{position:absolute;top:-3px;left:5%;width:8px;height:8px;background:#F0EDE6;border-radius:50%;transform:translateX(-50%);border:1.5px solid #14161B;transition:left 1.2s cubic-bezier(.4,0,.2,1);}}
.mcap{{font-size:11px;color:#5C5850;font-family:'DM Mono',monospace;}}</style></head>
<body><div class="card">
  {logo_tag}
  <div class="ticker">{ticker}</div><div class="cname">{d['name'][:26]}</div>
  <div class="znum" id="zn">0.00</div>
  <div class="zbadge"><span>{zl}</span></div>
  <div class="gbar"><div class="gtrack"></div><div class="gmarker" id="gm"></div></div>
  <div class="mcap">{fmt(d['mc'])}</div>
</div>
<script>
var target={z:.4f},gp={gp:.1f};
var el=document.getElementById('zn'),mk=document.getElementById('gm');
var start=null,dur=1100;
function step(ts){{if(!start)start=ts;var p=Math.min((ts-start)/dur,1),e=1-Math.pow(1-p,3);
  el.textContent=(target*e).toFixed(2);if(p<1)requestAnimationFrame(step);else el.textContent=target.toFixed(2);}}
setTimeout(function(){{requestAnimationFrame(step);mk.style.left=gp+'%';}},{delay_ms});
</script></body></html>""", height=260)


def render_oscore_panel(prob, o_val):
    zl, pct_str, zc, zbg, zbd = o_zone(prob)
    pct = prob * 100
    html = f"""<!DOCTYPE html><html>
<head><link href="https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Mono:wght@500&display=swap" rel="stylesheet">
<style>*{{box-sizing:border-box;margin:0;padding:0;}}body{{background:transparent;font-family:'Syne',sans-serif;}}
.panel{{background:#14161B;border:0.5px solid rgba(201,168,76,0.2);border-radius:16px;padding:1.4rem 1.5rem;}}
.top{{display:flex;align-items:flex-end;justify-content:space-between;flex-wrap:wrap;gap:12px;margin-bottom:1.4rem;}}
.label{{font-size:10px;color:#5C5850;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:4px;}}
.prob{{font-family:'DM Mono',monospace;font-size:56px;font-weight:500;color:#C9A84C;line-height:1;}}
.zone-box{{display:flex;align-items:center;gap:8px;background:{zbg};border:0.5px solid {zbd};border-radius:10px;padding:10px 16px;}}
.zone-dot{{width:8px;height:8px;border-radius:50%;background:{zc};flex-shrink:0;}}
.zone-label{{font-size:14px;font-weight:700;color:{zc};}}.zone-sub{{font-size:11px;color:{zc};opacity:0.6;margin-top:1px;}}
.bar-wrap{{position:relative;height:6px;background:#242830;border-radius:3px;margin-bottom:8px;overflow:visible;}}
.bar-track{{position:absolute;left:0;top:0;height:100%;width:100%;border-radius:3px;background:linear-gradient(90deg,#3FCF8E 0%,#F0A030 50%,#F06060 100%);}}
.bar-marker{{position:absolute;top:-3px;left:5%;width:10px;height:10px;background:#F0EDE6;border-radius:50%;transform:translateX(-50%);border:2px solid #14161B;transition:left 1.2s cubic-bezier(.4,0,.2,1);}}
.bar-labels{{display:flex;justify-content:space-between;font-size:10px;color:#5C5850;font-family:'DM Mono',monospace;}}
.o-val{{margin-top:10px;font-size:11px;color:#5C5850;font-family:'DM Mono',monospace;}}</style></head>
<body><div class="panel">
  <div class="top"><div><div class="label">Distress Probability</div><div class="prob" id="pnum">0.0%</div></div>
    <div class="zone-box"><div class="zone-dot"></div>
      <div><div class="zone-label">{zl}</div><div class="zone-sub">O-Score: {o_val:.3f}</div></div></div></div>
  <div class="bar-wrap"><div class="bar-track"></div><div class="bar-marker" id="bmark"></div></div>
  <div class="bar-labels"><span>Low &lt;20%</span><span>Medium 20–50%</span><span>High &gt;50%</span></div>
  <div class="o-val">Raw O-Score: {o_val:.4f} &nbsp;|&nbsp; P = 1/(1+e^-O)</div>
</div>
<script>
var target={pct:.4f},mk=document.getElementById('bmark'),el=document.getElementById('pnum');
var start=null,dur=1200;
function step(ts){{if(!start)start=ts;var p=Math.min((ts-start)/dur,1),e=1-Math.pow(1-p,3);
  el.textContent=(target*e).toFixed(1)+'%';if(p<1)requestAnimationFrame(step);else el.textContent=target.toFixed(1)+'%';}}
requestAnimationFrame(step);setTimeout(function(){{mk.style.left=Math.min(94,target)+'%';}},80);
</script></body></html>"""
    components.html(html, height=190)


def render_mscore_panel(m):
    if m > -1.78:
        zone_lbl, zc, zbg, zbd = "Manipulator",     "#F06060", "#2B0D0D", "rgba(240,96,96,0.2)"
    elif m > -2.22:
        zone_lbl, zc, zbg, zbd = "Grey Zone",       "#F0A030", "#2B1A05", "rgba(240,160,48,0.2)"
    else:
        zone_lbl, zc, zbg, zbd = "Non-Manipulator", "#3FCF8E", "#0D2B1F", "rgba(63,207,142,0.2)"

    # Map m to gauge position: range -4.5 → +0.5 = 0%→100%
    gp = max(3.0, min(97.0, (m + 4.5) / 5.0 * 100.0))

    html = f"""<!DOCTYPE html><html>
<head><link href="https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Mono:wght@500&display=swap" rel="stylesheet">
<style>*{{box-sizing:border-box;margin:0;padding:0;}}body{{background:transparent;font-family:'Syne',sans-serif;}}
.panel{{background:#14161B;border:0.5px solid rgba(201,168,76,0.2);border-radius:16px;padding:1.4rem 1.5rem;}}
.top{{display:flex;align-items:flex-end;justify-content:space-between;flex-wrap:wrap;gap:12px;margin-bottom:1.4rem;}}
.lbl{{font-size:10px;color:#5C5850;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:4px;}}
.mnum{{font-family:'DM Mono',monospace;font-size:56px;font-weight:500;color:#C9A84C;line-height:1;}}
.zbox{{display:flex;align-items:center;gap:8px;background:{zbg};border:0.5px solid {zbd};border-radius:10px;padding:10px 16px;}}
.zdot{{width:8px;height:8px;border-radius:50%;background:{zc};flex-shrink:0;}}
.zlbl{{font-size:14px;font-weight:700;color:{zc};}}.zsub{{font-size:11px;color:{zc};opacity:0.6;margin-top:1px;}}
.gw{{position:relative;height:6px;background:#242830;border-radius:3px;margin-bottom:8px;overflow:visible;}}
.gt{{position:absolute;left:0;top:0;height:100%;width:100%;border-radius:3px;
     background:linear-gradient(90deg,#3FCF8E 0%,#3FCF8E 40%,#F0A030 50%,#F06060 100%);}}
.gm{{position:absolute;top:-3px;left:5%;width:10px;height:10px;background:#F0EDE6;border-radius:50%;
     transform:translateX(-50%);border:2px solid #14161B;transition:left 1.2s cubic-bezier(.4,0,.2,1);}}
.gl{{display:flex;justify-content:space-between;font-size:10px;color:#5C5850;font-family:'DM Mono',monospace;}}
</style></head>
<body><div class="panel">
  <div class="top">
    <div><div class="lbl">M-Score</div><div class="mnum" id="mn">0.00</div></div>
    <div class="zbox"><div class="zdot"></div>
      <div><div class="zlbl">{zone_lbl}</div>
           <div class="zsub">Below −2.22: safe &middot; −2.22–−1.78: grey &middot; Above −1.78: risk</div></div>
    </div>
  </div>
  <div class="gw"><div class="gt"></div><div class="gm" id="gmark"></div></div>
  <div class="gl"><span>Non-Manipulator &lt;−2.22</span><span>Grey −2.22–−1.78</span><span>Manipulator &gt;−1.78</span></div>
</div>
<script>
var target={m:.4f},gp={gp:.1f};
var el=document.getElementById('mn'),mk=document.getElementById('gmark');
var start=null,dur=1200;
function step(ts){{if(!start)start=ts;var p=Math.min((ts-start)/dur,1),e=1-Math.pow(1-p,3);
  el.textContent=(target*e).toFixed(2);if(p<1)requestAnimationFrame(step);else el.textContent=target.toFixed(2);}}
requestAnimationFrame(step);
setTimeout(function(){{mk.style.left=gp+'%';}},80);
</script></body></html>"""
    components.html(html, height=185)


def render_criteria_group(group_title, keys, flags):
    rows = ""
    for k in keys:
        short, desc, passes, val = flags[k]
        ic     = "✓" if passes else "✗"
        ic_col = "#3FCF8E" if passes else "#F06060"
        ic_bg  = "rgba(63,207,142,0.1)" if passes else "rgba(240,96,96,0.08)"
        ic_bd  = "rgba(63,207,142,0.3)" if passes else "rgba(240,96,96,0.25)"
        rows += (
            f'<div style="display:flex;align-items:center;gap:10px;padding:9px 12px;'
            f'background:#1C1F27;border-radius:8px;margin-bottom:5px;">'
            f'<div style="width:24px;height:24px;border-radius:50%;background:{ic_bg};'
            f'border:0.5px solid {ic_bd};display:flex;align-items:center;justify-content:center;'
            f'font-size:11px;color:{ic_col};flex-shrink:0;">{ic}</div>'
            f'<div style="font-family:\'DM Mono\',monospace;font-size:10px;font-weight:500;'
            f'color:#C9A84C;width:22px;flex-shrink:0;">{k}</div>'
            f'<div style="flex:1;min-width:0;">'
            f'<div style="font-size:12px;font-weight:600;color:#F0EDE6;line-height:1.2;">{short}</div>'
            f'<div style="font-size:10px;color:#5C5850;margin-top:1px;">{desc}</div>'
            f'</div>'
            f'<div style="font-family:\'DM Mono\',monospace;font-size:10px;color:#9B9589;'
            f'text-align:right;flex-shrink:0;max-width:95px;overflow:hidden;text-overflow:ellipsis;">{val}</div>'
            f'</div>'
        )
    return (
        f'<div style="margin-bottom:1rem;">'
        f'<div style="font-size:10px;font-weight:700;color:#5C5850;letter-spacing:1.5px;'
        f'text-transform:uppercase;margin-bottom:8px;padding:0 2px;">{group_title}</div>'
        f'{rows}</div>'
    )


def render_fscore_panel(score, flags):
    if score >= 7:
        zone_lbl, zc, zbg, zbd = "Strong",  "#3FCF8E", "#0D2B1F", "rgba(63,207,142,0.2)"
    elif score >= 3:
        zone_lbl, zc, zbg, zbd = "Neutral", "#F0A030", "#2B1A05", "rgba(240,160,48,0.2)"
    else:
        zone_lbl, zc, zbg, zbd = "Weak",    "#F06060", "#2B0D0D", "rgba(240,96,96,0.2)"

    dots_js = "[" + ",".join("1" if flags[f"F{i+1}"][2] else "0" for i in range(9)) + "]"
    dl_html = "".join(f'<div class="dl">F{i+1}</div>' for i in range(9))

    html = f"""<!DOCTYPE html><html>
<head><link href="https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Mono:wght@500&display=swap" rel="stylesheet">
<style>*{{box-sizing:border-box;margin:0;padding:0;}}body{{background:transparent;font-family:'Syne',sans-serif;}}
.panel{{background:#14161B;border:0.5px solid rgba(201,168,76,0.2);border-radius:16px;padding:1.4rem 1.5rem;}}
.top{{display:flex;align-items:flex-end;justify-content:space-between;flex-wrap:wrap;gap:12px;margin-bottom:1.2rem;}}
.lbl{{font-size:10px;color:#5C5850;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:4px;}}
.snum{{font-family:'DM Mono',monospace;font-size:56px;font-weight:500;color:#C9A84C;line-height:1;}}
.sden{{font-family:'DM Mono',monospace;font-size:28px;font-weight:500;color:#5C5850;line-height:1;vertical-align:bottom;padding-bottom:6px;}}
.zbox{{display:flex;align-items:center;gap:8px;background:{zbg};border:0.5px solid {zbd};border-radius:10px;padding:10px 16px;}}
.zdot{{width:8px;height:8px;border-radius:50%;background:{zc};flex-shrink:0;}}
.zlbl{{font-size:14px;font-weight:700;color:{zc};}}
.zsub{{font-size:11px;color:{zc};opacity:0.6;margin-top:1px;}}
.dots{{display:flex;gap:8px;align-items:center;margin-bottom:5px;flex-wrap:nowrap;}}
.dot{{width:30px;height:30px;border-radius:50%;border:1.5px solid #2A2D35;background:#1C1F27;
      display:flex;align-items:center;justify-content:center;font-size:12px;
      opacity:0;transition:opacity 0.25s;flex-shrink:0;}}
.dot-labels{{display:flex;gap:8px;}}
.dl{{width:30px;text-align:center;font-size:9px;font-family:'DM Mono',monospace;color:#5C5850;flex-shrink:0;}}
</style></head>
<body><div class="panel">
  <div class="top">
    <div><div class="lbl">F-Score</div>
      <span class="snum" id="sn">0</span><span class="sden">&thinsp;/9</span>
    </div>
    <div class="zbox"><div class="zdot"></div>
      <div><div class="zlbl">{zone_lbl}</div>
           <div class="zsub">0–2 Weak &middot; 3–6 Neutral &middot; 7–9 Strong</div></div>
    </div>
  </div>
  <div class="dots" id="dr"></div>
  <div class="dot-labels">{dl_html}</div>
</div>
<script>
var d={dots_js},t={score};
var row=document.getElementById('dr');
for(var i=0;i<9;i++){{
  var el=document.createElement('div');el.className='dot';el.id='d'+i;
  if(d[i]){{el.style.background='rgba(63,207,142,0.15)';el.style.borderColor='#3FCF8E';
            el.style.color='#3FCF8E';el.textContent='✓';}}
  else{{el.style.background='rgba(240,96,96,0.08)';el.style.borderColor='rgba(240,96,96,0.3)';
        el.style.color='#F06060';el.textContent='✗';}}
  row.appendChild(el);
}}
for(var j=0;j<9;j++){{
  (function(idx){{setTimeout(function(){{document.getElementById('d'+idx).style.opacity='1';}},60+idx*70);}})(j);
}}
var sn=document.getElementById('sn'),st2=null,dur=700;
function step(ts){{if(!st2)st2=ts;var p=Math.min((ts-st2)/dur,1);
  sn.textContent=Math.round(p*t);if(p<1)requestAnimationFrame(step);else sn.textContent=t;}}
requestAnimationFrame(step);
</script></body></html>"""
    components.html(html, height=200)


# ── Sidebar (session-state based, no href links) ──────────────────────────────
NAV_ITEMS = [
    ("home",       "Home",       "#C9A84C", None,   None,      None),
    ("zscore",     "Z-Score",    "#3FCF8E", "LIVE", "#3FCF8E", "rgba(63,207,142,0.12)"),
    ("oscore",     "O-Score",    "#F0A030", "LIVE", "#F0A030", "rgba(240,160,48,0.12)"),
    ("fscore",     "F-Score",    "#9B6FD4", "LIVE", "#9B6FD4", "rgba(155,111,212,0.12)"),
    ("mscore",     "M-Score",    "#E85555", "LIVE", "#E85555", "rgba(232,85,85,0.12)"),
    ("comparison", "Comparison", "#4A9EF0", "LIVE", "#4A9EF0", "rgba(74,158,240,0.12)"),
    ("mlscore",   "ML Score",   "#C96BE8", "ML",   "#C96BE8", "rgba(201,107,232,0.12)"),
]

with st.sidebar:
    st.markdown("""
    <div class="brand">
      <div class="brand-icon"><div class="brand-icon-inner"></div></div>
      <div><div class="brand-name-1">Financial</div><div class="brand-name-2">Distress</div></div>
    </div>
    <div class="nav-section-label">Navigation</div>
    """, unsafe_allow_html=True)

    for page_key, label, dot_color, badge, badge_color, badge_bg in NAV_ITEMS:
        if current_page == page_key:
            # Active item — rendered as styled HTML div
            badge_html = f'<span class="nav-badge" style="background:{badge_bg};color:{badge_color};">{badge}</span>' if badge else ""
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;padding:9px 12px;border-radius:9px;
                        background:rgba(201,168,76,0.09);margin-bottom:3px;">
              <span style="font-size:9px;color:{dot_color};">●</span>
              <span style="flex:1;font-size:14px;font-weight:600;color:#F0EDE6;
                           font-family:'Syne',sans-serif;">{label}</span>
              {badge_html}
            </div>
            """, unsafe_allow_html=True)
        else:
            # Inactive item — rendered as a styled st.button (triggers st.rerun, no page reload)
            badge_suffix = f"  ·  {badge}" if badge else ""
            if st.button(f"● {label}{badge_suffix}", key=f"nav_{page_key}"):
                go(page_key)


# ── Shared: back button ───────────────────────────────────────────────────────
def render_back_button():
    if st.session_state.history:
        st.markdown('<div class="back-btn">', unsafe_allow_html=True)
        if st.button("← Back", key=f"back_{current_page}"):
            go_back()
        st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: Home
# ════════════════════════════════════════════════════════════════════════════
def page_home():
    st.markdown("""
    <div class="fu1" style="margin-bottom:2.5rem;">
      <div style="font-size:32px;font-weight:800;letter-spacing:-0.5px;line-height:1.1;
                  color:#F0EDE6;margin-bottom:8px;">
        Financial <span style="color:#C9A84C;">Distress</span> App
      </div>
      <div style="font-size:14px;color:#6B6560;line-height:1.6;max-width:520px;">
        Analyze financial distress risk of any public company using multiple established models.
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-hdr fu2"><span class="sec-lbl">Available Models</span><div class="sec-line"></div></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="medium")

    with col1:
        # Z-Score card (LIVE) — flat bottom connects to button
        st.markdown("""
        <div class="model-card model-card-live fu3">
          <div class="model-card-header">
            <div style="display:flex;align-items:center;gap:8px;">
              <div class="model-card-dot" style="background:#3FCF8E;"></div>
              <div class="model-card-name">Altman Z-Score</div>
            </div>
            <span class="model-card-badge" style="background:rgba(63,207,142,0.1);color:#3FCF8E;">LIVE</span>
          </div>
          <div class="model-card-desc">5 financial ratios · Safe / Grey / Distress zones</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="card-open-btn">', unsafe_allow_html=True)
        if st.button("Open Altman Z-Score →", key="home_zscore"):
            go("zscore")
        st.markdown('</div>', unsafe_allow_html=True)

        # Beneish M-Score (LIVE)
        st.markdown("""
        <div class="model-card model-card-live fu5">
          <div class="model-card-header">
            <div style="display:flex;align-items:center;gap:8px;">
              <div class="model-card-dot" style="background:#E85555;"></div>
              <div class="model-card-name">Beneish M-Score</div>
            </div>
            <span class="model-card-badge" style="background:rgba(232,85,85,0.1);color:#E85555;">LIVE</span>
          </div>
          <div class="model-card-desc">8 indices · Earnings manipulation detection</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="card-open-btn">', unsafe_allow_html=True)
        if st.button("Open Beneish M-Score →", key="home_mscore"):
            go("mscore")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        # O-Score card (LIVE) — flat bottom connects to button
        st.markdown("""
        <div class="model-card model-card-live fu4">
          <div class="model-card-header">
            <div style="display:flex;align-items:center;gap:8px;">
              <div class="model-card-dot" style="background:#F0A030;"></div>
              <div class="model-card-name">Ohlson O-Score</div>
            </div>
            <span class="model-card-badge" style="background:rgba(240,160,48,0.1);color:#F0A030;">LIVE</span>
          </div>
          <div class="model-card-desc">9 variables · Distress probability as a percentage</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="card-open-btn">', unsafe_allow_html=True)
        if st.button("Open Ohlson O-Score →", key="home_oscore"):
            go("oscore")
        st.markdown('</div>', unsafe_allow_html=True)

        # Piotroski F-Score (LIVE)
        st.markdown("""
        <div class="model-card model-card-live fu6">
          <div class="model-card-header">
            <div style="display:flex;align-items:center;gap:8px;">
              <div class="model-card-dot" style="background:#9B6FD4;"></div>
              <div class="model-card-name">Piotroski F-Score</div>
            </div>
            <span class="model-card-badge" style="background:rgba(155,111,212,0.1);color:#9B6FD4;">LIVE</span>
          </div>
          <div class="model-card-desc">9 criteria · Financial strength score from 0 to 9</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="card-open-btn">', unsafe_allow_html=True)
        if st.button("Open Piotroski F-Score →", key="home_fscore"):
            go("fscore")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-hdr" style="margin-top:2rem;"><span class="sec-lbl">Machine Learning Model</span><div class="sec-line"></div></div>', unsafe_allow_html=True)
    ml_col1, ml_col2 = st.columns(2, gap="medium")
    ml_status = "LIVE" if _ml_available() else "TRAIN FIRST"
    ml_badge_color = "#C96BE8" if _ml_available() else "#9B9589"
    ml_badge_bg = "rgba(201,107,232,0.1)" if _ml_available() else "rgba(155,149,137,0.1)"
    with ml_col1:
        st.markdown(f"""
        <div class="model-card model-card-live fu3">
          <div class="model-card-header">
            <div style="display:flex;align-items:center;gap:8px;">
              <div class="model-card-dot" style="background:{ml_badge_color};"></div>
              <div class="model-card-name">ML Distress Score</div>
            </div>
            <span class="model-card-badge" style="background:{ml_badge_bg};color:{ml_badge_color};">{ml_status}</span>
          </div>
          <div class="model-card-desc">XGBoost · Trained on WRDS/Compustat 1990–2025 · Calibrated probability</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="card-open-btn">', unsafe_allow_html=True)
        if st.button("Open ML Score →", key="home_mlscore"):
            go("mlscore")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-hdr" style="margin-top:2rem;"><span class="sec-lbl">Quick Start</span><div class="sec-line"></div></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="qs-box">
      <div class="qs-step">
        <div class="qs-num">1</div>
        <div>Select a model from the left sidebar</div>
      </div>
      <div class="qs-step">
        <div class="qs-num">2</div>
        <div>Enter a ticker — e.g. <b style="color:#F0EDE6;">AAPL</b>,
             <b style="color:#F0EDE6;">MSFT</b>, <b style="color:#F0EDE6;">TSLA</b></div>
      </div>
      <div class="qs-step">
        <div class="qs-num">3</div>
        <div>Click <b style="color:#F0EDE6;">Analyze →</b> and review the results</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: Z-Score
# ════════════════════════════════════════════════════════════════════════════
def page_zscore():
    render_back_button()
    st.markdown("""
    <div class="fu1" style="margin-bottom:1.5rem;padding-bottom:1.2rem;
                            border-bottom:0.5px solid rgba(201,168,76,0.12);">
      <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;">
        <div>
          <div style="font-size:24px;font-weight:800;letter-spacing:-0.3px;color:#F0EDE6;">
            Altman <span style="color:#C9A84C;">Z-Score</span>
          </div>
          <div style="font-size:12px;color:#6B6560;margin-top:3px;">5-variable bankruptcy prediction model</div>
        </div>
        <div style="font-family:'DM Mono',monospace;font-size:10px;color:#8A6E2F;
                    background:rgba(201,168,76,0.08);border:0.5px solid rgba(201,168,76,0.15);
                    padding:4px 10px;border-radius:4px;letter-spacing:1px;">v1.2</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

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

            st.markdown('<div class="fu3 sec-hdr"><span class="sec-lbl">Raw Financials</span><div class="sec-line"></div></div>', unsafe_allow_html=True)
            fin_rows = [("Working Capital", d['wc']), ("Total Assets", d['ta']),
                        ("Retained Earnings", d['re']), ("EBIT", d['eb']),
                        ("Total Liabilities", d['tl']), ("Sales / Revenue", d['rv'])]
            cols = st.columns(2)
            for i, (lbl, val) in enumerate(fin_rows):
                with cols[i % 2]:
                    st.markdown(f'<div style="display:flex;justify-content:space-between;align-items:center;background:#14161B;border-radius:8px;padding:10px 14px;border:0.5px solid rgba(201,168,76,0.08);margin-bottom:8px;"><span style="font-size:12px;color:#9B9589;">{lbl}</span><span style="font-family:\'DM Mono\',monospace;font-size:13px;font-weight:500;color:#F0EDE6;">{fmt(val)}</span></div>', unsafe_allow_html=True)

            st.markdown('<div class="fu4 sec-hdr" style="margin-top:1.5rem;"><span class="sec-lbl">Altman Z-Score</span><div class="sec-line"></div></div>', unsafe_allow_html=True)
            render_zscore_panel(z, d)

            st.markdown('<div class="fu5 sec-hdr" style="margin-top:1.2rem;"><span class="sec-lbl">Ratio Breakdown</span><div class="sec-line"></div></div>', unsafe_allow_html=True)
            ratios = [("X1","Working Capital / Total Assets",d['x1'],"× 1.2"),
                      ("X2","Retained Earnings / Total Assets",d['x2'],"× 1.4"),
                      ("X3","EBIT / Total Assets",d['x3'],"× 3.3"),
                      ("X4","Market Value / Total Liabilities",d['x4'],"× 0.6"),
                      ("X5","Sales / Total Assets",d['x5'],"× 1.0")]
            cols5 = st.columns(5)
            for i, (nm, formula, val, wt) in enumerate(ratios):
                with cols5[i]:
                    st.markdown(f'<div style="background:#1C1F27;border:0.5px solid rgba(201,168,76,0.08);border-radius:10px;padding:0.9rem 0.8rem;text-align:center;"><div style="font-size:11px;font-weight:700;color:#C9A84C;margin-bottom:3px;">{nm}</div><div style="font-size:9px;color:#5C5850;margin-bottom:8px;line-height:1.3;">{formula}</div><div style="font-family:\'DM Mono\',monospace;font-size:17px;font-weight:500;color:#F0EDE6;">{val:.3f}</div><div style="font-size:9px;color:#5C5850;margin-top:3px;font-family:\'DM Mono\',monospace;">{wt}</div></div>', unsafe_allow_html=True)

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
# PAGE: O-Score
# ════════════════════════════════════════════════════════════════════════════
def page_oscore():
    render_back_button()
    st.markdown("""
    <div class="fu1" style="margin-bottom:1.5rem;padding-bottom:1.2rem;
                            border-bottom:0.5px solid rgba(201,168,76,0.12);">
      <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;">
        <div>
          <div style="font-size:24px;font-weight:800;letter-spacing:-0.3px;color:#F0EDE6;">
            Ohlson <span style="color:#C9A84C;">O-Score</span>
          </div>
          <div style="font-size:12px;color:#6B6560;margin-top:3px;">9-variable logistic regression distress model</div>
        </div>
        <div style="font-family:'DM Mono',monospace;font-size:10px;color:#8A6E2F;
                    background:rgba(201,168,76,0.08);border:0.5px solid rgba(201,168,76,0.15);
                    padding:4px 10px;border-radius:4px;letter-spacing:1px;">v1.0</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

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

            st.markdown('<div class="sec-hdr"><span class="sec-lbl">Raw Financials</span><div class="sec-line"></div></div>', unsafe_allow_html=True)
            o_fin_rows = [
                ("Total Assets", od['ta']), ("Total Liabilities", od['tl']),
                ("Working Capital", od['wc']), ("Current Assets", od['ca']),
                ("Current Liabilities", od['cl']), ("Net Income", od['ni']),
                ("Net Income (prev)", od['ni_prev']), ("Operating Cash Flow", od['cfo']),
            ]
            ocols = st.columns(2)
            for i, (lbl, val) in enumerate(o_fin_rows):
                with ocols[i % 2]:
                    st.markdown(f'<div style="display:flex;justify-content:space-between;align-items:center;background:#14161B;border-radius:8px;padding:10px 14px;border:0.5px solid rgba(201,168,76,0.08);margin-bottom:8px;"><span style="font-size:12px;color:#9B9589;">{lbl}</span><span style="font-family:\'DM Mono\',monospace;font-size:13px;font-weight:500;color:#F0EDE6;">{fmt(val) if val is not None else "N/A"}</span></div>', unsafe_allow_html=True)

            st.markdown('<div class="sec-hdr" style="margin-top:1.5rem;"><span class="sec-lbl">Ohlson O-Score</span><div class="sec-line"></div></div>', unsafe_allow_html=True)
            render_oscore_panel(prob, od["o"])

            st.markdown('<div class="sec-hdr" style="margin-top:1.2rem;"><span class="sec-lbl">Variable Breakdown</span><div class="sec-line"></div></div>', unsafe_allow_html=True)
            o_vars = [
                ("X1","log(Total Assets)",od['x1'],"−0.407"),
                ("X2","Total Liabilities / Assets",od['x2'],"+6.03"),
                ("X3","Working Capital / Assets",od['x3'],"−1.43"),
                ("X4","Current Liab / Current Assets",od['x4'],"+0.076"),
                ("X5","Insolvent (0/1)",od['x5'],"−1.72"),
            ]
            o_vars2 = [
                ("X6","Net Income / Assets",od['x6'],"−2.37"),
                ("X7","CFO / Total Liabilities",od['x7'],"−1.83"),
                ("X8","2yr Loss (0/1)",od['x8'],"+0.285"),
                ("X9","NI Change Ratio",od['x9'],"−0.521"),
            ]
            cols5o = st.columns(5)
            for i, (nm, formula, val, wt) in enumerate(o_vars):
                with cols5o[i]:
                    st.markdown(f'<div style="background:#1C1F27;border:0.5px solid rgba(201,168,76,0.08);border-radius:10px;padding:0.9rem 0.8rem;text-align:center;"><div style="font-size:11px;font-weight:700;color:#C9A84C;margin-bottom:3px;">{nm}</div><div style="font-size:9px;color:#5C5850;margin-bottom:8px;line-height:1.3;">{formula}</div><div style="font-family:\'DM Mono\',monospace;font-size:17px;font-weight:500;color:#F0EDE6;">{val:.3f}</div><div style="font-size:9px;color:#5C5850;margin-top:3px;font-family:\'DM Mono\',monospace;">{wt}</div></div>', unsafe_allow_html=True)
            cols4o = st.columns(4)
            for i, (nm, formula, val, wt) in enumerate(o_vars2):
                with cols4o[i]:
                    st.markdown(f'<div style="background:#1C1F27;border:0.5px solid rgba(201,168,76,0.08);border-radius:10px;padding:0.9rem 0.8rem;text-align:center;margin-top:8px;"><div style="font-size:11px;font-weight:700;color:#C9A84C;margin-bottom:3px;">{nm}</div><div style="font-size:9px;color:#5C5850;margin-bottom:8px;line-height:1.3;">{formula}</div><div style="font-family:\'DM Mono\',monospace;font-size:17px;font-weight:500;color:#F0EDE6;">{val:.3f}</div><div style="font-size:9px;color:#5C5850;margin-top:3px;font-family:\'DM Mono\',monospace;">{wt}</div></div>', unsafe_allow_html=True)

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


# ════════════════════════════════════════════════════════════════════════════
# PAGE: Comparison
# ════════════════════════════════════════════════════════════════════════════
def page_comparison():
    render_back_button()
    st.markdown("""
    <div class="fu1" style="margin-bottom:1.5rem;padding-bottom:1.2rem;
                            border-bottom:0.5px solid rgba(201,168,76,0.12);">
      <div>
        <div style="font-size:24px;font-weight:800;letter-spacing:-0.3px;color:#F0EDE6;">
          Competitor <span style="color:#C9A84C;">Comparison</span>
        </div>
        <div style="font-size:12px;color:#6B6560;margin-top:3px;">Side-by-side analysis across all 4 models for up to 3 companies</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

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
            all_data = {}
            with st.spinner("Fetching all model data..."):
                for t in tickers:
                    all_data[t] = {
                        "z": compute_zscore(t),
                        "o": compute_oscore(t),
                        "f": compute_fscore(t),
                        "m": compute_mscore(t),
                    }

            n = len(tickers)

            def _logo_tag(d, ticker):
                if d.get("logo"):
                    return (f'<div style="width:32px;height:32px;border-radius:7px;background:#1C1F27;'
                            f'padding:3px;margin:0 auto 6px;display:flex;align-items:center;justify-content:center;">'
                            f'<img src="{d["logo"]}" style="width:26px;height:26px;object-fit:contain;"></div>')
                return (f'<div style="width:32px;height:32px;border-radius:7px;background:rgba(201,168,76,0.15);'
                        f'border:0.5px solid rgba(201,168,76,0.3);display:flex;align-items:center;'
                        f'justify-content:center;font-family:monospace;font-size:11px;font-weight:500;'
                        f'color:#C9A84C;margin:0 auto 6px;">{ticker[:2]}</div>')

            def _mini_card(ticker, d, value_str, zone_label, zone_color, zone_bg, zone_bd):
                if not d["ok"]:
                    return (f'<div style="background:#14161B;border:0.5px solid rgba(240,96,96,0.15);'
                            f'border-radius:14px;padding:1.2rem 1rem;text-align:center;">'
                            f'<div style="font-family:\'DM Mono\',monospace;font-size:12px;color:#C9A84C;margin-bottom:8px;">{ticker}</div>'
                            f'<div style="color:#F06060;font-size:11px;">Data unavailable</div></div>')
                logo   = _logo_tag(d, ticker)
                cname  = d.get("name", ticker)[:24]
                return (
                    f'<div style="background:#14161B;border:0.5px solid rgba(201,168,76,0.15);'
                    f'border-radius:14px;padding:1.2rem 1rem;text-align:center;">'
                    f'{logo}'
                    f'<div style="font-family:\'DM Mono\',monospace;font-size:11px;color:#C9A84C;'
                    f'letter-spacing:1px;margin-bottom:2px;">{ticker}</div>'
                    f'<div style="font-size:10px;color:#5C5850;margin-bottom:12px;overflow:hidden;'
                    f'text-overflow:ellipsis;white-space:nowrap;">{cname}</div>'
                    f'<div style="font-family:\'DM Mono\',monospace;font-size:34px;font-weight:500;'
                    f'color:#C9A84C;line-height:1;margin-bottom:10px;">{value_str}</div>'
                    f'<div style="display:inline-block;background:{zone_bg};border:0.5px solid {zone_bd};'
                    f'border-radius:8px;padding:5px 12px;">'
                    f'<span style="font-size:11px;font-weight:700;color:{zone_color};">{zone_label}</span>'
                    f'</div></div>'
                )

            # ── Z-Score ───────────────────────────────────────────────────
            st.markdown('<div class="sec-hdr" style="margin-top:1.5rem;"><span class="sec-lbl">Z-Score Comparison</span><div class="sec-line"></div></div>', unsafe_allow_html=True)
            zcols = st.columns(n)
            for i, t in enumerate(tickers):
                with zcols[i]:
                    render_comparison_card(t, all_data[t]["z"], delay_ms=100 + i * 200)

            # ── O-Score ───────────────────────────────────────────────────
            st.markdown('<div class="sec-hdr" style="margin-top:1.8rem;"><span class="sec-lbl">O-Score Comparison</span><div class="sec-line"></div></div>', unsafe_allow_html=True)
            ocols = st.columns(n)
            for i, t in enumerate(tickers):
                with ocols[i]:
                    od = all_data[t]["o"]
                    if od["ok"]:
                        zl, _, zc, zbg, zbd = o_zone(od["prob"])
                        card = _mini_card(t, od, f"{od['prob']*100:.1f}%", zl, zc, zbg, zbd)
                    else:
                        card = _mini_card(t, od, "N/A", "N/A", "#5C5850", "#1C1F27", "#2A2D35")
                    st.markdown(card, unsafe_allow_html=True)

            # ── F-Score ───────────────────────────────────────────────────
            st.markdown('<div class="sec-hdr" style="margin-top:1.8rem;"><span class="sec-lbl">F-Score Comparison</span><div class="sec-line"></div></div>', unsafe_allow_html=True)
            fcols = st.columns(n)
            for i, t in enumerate(tickers):
                with fcols[i]:
                    fd = all_data[t]["f"]
                    if fd["ok"]:
                        sc = fd["score"]
                        if sc >= 7:   fzl, fzc, fzbg, fzbd = "Strong",  "#3FCF8E", "#0D2B1F", "rgba(63,207,142,0.2)"
                        elif sc >= 3: fzl, fzc, fzbg, fzbd = "Neutral", "#F0A030", "#2B1A05", "rgba(240,160,48,0.2)"
                        else:         fzl, fzc, fzbg, fzbd = "Weak",    "#F06060", "#2B0D0D", "rgba(240,96,96,0.2)"
                        card = _mini_card(t, fd, f"{sc}/9", fzl, fzc, fzbg, fzbd)
                    else:
                        card = _mini_card(t, fd, "N/A", "N/A", "#5C5850", "#1C1F27", "#2A2D35")
                    st.markdown(card, unsafe_allow_html=True)

            # ── M-Score ───────────────────────────────────────────────────
            st.markdown('<div class="sec-hdr" style="margin-top:1.8rem;"><span class="sec-lbl">M-Score Comparison</span><div class="sec-line"></div></div>', unsafe_allow_html=True)
            mcols = st.columns(n)
            for i, t in enumerate(tickers):
                with mcols[i]:
                    mrd = all_data[t]["m"]
                    if mrd["ok"]:
                        mv = mrd["m"]
                        if mv > -1.78:   mzl, mzc, mzbg, mzbd = "Manipulator",     "#F06060", "#2B0D0D", "rgba(240,96,96,0.2)"
                        elif mv > -2.22: mzl, mzc, mzbg, mzbd = "Grey Zone",       "#F0A030", "#2B1A05", "rgba(240,160,48,0.2)"
                        else:            mzl, mzc, mzbg, mzbd = "Non-Manipulator", "#3FCF8E", "#0D2B1F", "rgba(63,207,142,0.2)"
                        card = _mini_card(t, mrd, f"{mv:.2f}", mzl, mzc, mzbg, mzbd)
                    else:
                        card = _mini_card(t, mrd, "N/A", "N/A", "#5C5850", "#1C1F27", "#2A2D35")
                    st.markdown(card, unsafe_allow_html=True)

            # ── Summary Table ─────────────────────────────────────────────
            if n > 1:
                st.markdown('<div class="sec-hdr" style="margin-top:1.8rem;"><span class="sec-lbl">Summary Table</span><div class="sec-line"></div></div>', unsafe_allow_html=True)
                fr = f"180px {' '.join(['1fr']*n)}"

                def cell(content, accent=False, header=False, center=False, clr=None):
                    c = clr if clr else ("#C9A84C" if accent else ("#9B9589" if header else "#F0EDE6"))
                    mono  = "font-family:'DM Mono',monospace;" if not header else ""
                    align = "center" if center else "left"
                    return f'<div style="padding:9px 14px;background:#14161B;color:{c};{mono}font-size:12px;text-align:{align};">{content}</div>'

                def sep_row(lbl, dot_col):
                    row = (f'<div style="padding:6px 14px;background:#111316;color:{dot_col};'
                           f'font-size:10px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;">{lbl}</div>')
                    row += f'<div style="background:#111316;"></div>' * n
                    return row

                # header
                html = (f'<div style="display:grid;grid-template-columns:{fr};gap:1px;'
                        f'background:rgba(201,168,76,0.08);border-radius:12px;overflow:hidden;margin-bottom:2px;">')
                html += cell("Model / Metric", header=True)
                for t in tickers:
                    html += cell(t, accent=True, center=True)
                html += "</div>"

                # body
                html += (f'<div style="display:grid;grid-template-columns:{fr};gap:1px;'
                         f'background:rgba(201,168,76,0.05);border-radius:12px;overflow:hidden;">')

                # Z-Score block
                html += sep_row("Altman Z-Score", "#3FCF8E")
                html += cell("Z-Score", header=True)
                for t in tickers:
                    zd = all_data[t]["z"]
                    html += cell(f"{zd['z']:.2f}" if zd["ok"] else "N/A", center=True)
                html += cell("Zone", header=True)
                for t in tickers:
                    zd = all_data[t]["z"]
                    if zd["ok"]:
                        zl2, _, zc2, _, _ = zone_info(zd["z"])
                        html += cell(zl2, center=True, clr=zc2)
                    else:
                        html += cell("N/A", center=True)
                html += cell("Market Cap", header=True)
                for t in tickers:
                    zd = all_data[t]["z"]
                    html += cell(fmt(zd["mc"]) if zd["ok"] else "N/A", center=True)

                # O-Score block
                html += sep_row("Ohlson O-Score", "#F0A030")
                html += cell("Distress Prob.", header=True)
                for t in tickers:
                    od = all_data[t]["o"]
                    html += cell(f"{od['prob']*100:.1f}%" if od["ok"] else "N/A", center=True)
                html += cell("Zone", header=True)
                for t in tickers:
                    od = all_data[t]["o"]
                    if od["ok"]:
                        zl2, _, zc2, _, _ = o_zone(od["prob"])
                        html += cell(zl2, center=True, clr=zc2)
                    else:
                        html += cell("N/A", center=True)

                # F-Score block
                html += sep_row("Piotroski F-Score", "#9B6FD4")
                html += cell("F-Score", header=True)
                for t in tickers:
                    fd = all_data[t]["f"]
                    html += cell(f"{fd['score']}/9" if fd["ok"] else "N/A", center=True)
                html += cell("Zone", header=True)
                for t in tickers:
                    fd = all_data[t]["f"]
                    if fd["ok"]:
                        sc = fd["score"]
                        fzl2 = "Strong" if sc >= 7 else ("Neutral" if sc >= 3 else "Weak")
                        fzc2 = "#3FCF8E" if sc >= 7 else ("#F0A030" if sc >= 3 else "#F06060")
                        html += cell(fzl2, center=True, clr=fzc2)
                    else:
                        html += cell("N/A", center=True)

                # M-Score block
                html += sep_row("Beneish M-Score", "#E85555")
                html += cell("M-Score", header=True)
                for t in tickers:
                    mrd = all_data[t]["m"]
                    html += cell(f"{mrd['m']:.2f}" if mrd["ok"] else "N/A", center=True)
                html += cell("Zone", header=True)
                for t in tickers:
                    mrd = all_data[t]["m"]
                    if mrd["ok"]:
                        mv = mrd["m"]
                        mzl2 = "Manipulator" if mv > -1.78 else ("Grey Zone" if mv > -2.22 else "Non-Manip.")
                        mzc2 = "#F06060"     if mv > -1.78 else ("#F0A030"   if mv > -2.22 else "#3FCF8E")
                        html += cell(mzl2, center=True, clr=mzc2)
                    else:
                        html += cell("N/A", center=True)

                html += "</div>"
                st.markdown(html, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: M-Score
# ════════════════════════════════════════════════════════════════════════════
def page_mscore():
    render_back_button()
    st.markdown("""
    <div class="fu1" style="margin-bottom:1.5rem;padding-bottom:1.2rem;
                            border-bottom:0.5px solid rgba(201,168,76,0.12);">
      <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;">
        <div>
          <div style="font-size:24px;font-weight:800;letter-spacing:-0.3px;color:#F0EDE6;">
            Beneish <span style="color:#C9A84C;">M-Score</span>
          </div>
          <div style="font-size:12px;color:#6B6560;margin-top:3px;">8-index earnings manipulation detection model</div>
        </div>
        <div style="font-family:'DM Mono',monospace;font-size:10px;color:#8A6E2F;
                    background:rgba(201,168,76,0.08);border:0.5px solid rgba(201,168,76,0.15);
                    padding:4px 10px;border-radius:4px;letter-spacing:1px;">v1.0</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="font-size:11px;font-weight:500;color:#9B9589;letter-spacing:1.5px;text-transform:uppercase;margin:1rem 0 6px;">Company Ticker</div>', unsafe_allow_html=True)
    pm1, pm2 = st.columns([4, 1])
    with pm1:
        t_m = st.text_input("tm", placeholder="e.g. AAPL, TSLA, MSFT", label_visibility="collapsed")
    with pm2:
        btn_m = st.button("Analyze →", key="btn_m")

    if btn_m and t_m.strip():
        ticker_m = t_m.strip().upper()
        with st.spinner("Fetching data..."):
            md = compute_mscore(ticker_m)

        if not md["ok"]:
            st.markdown('<div style="background:#2B0D0D;border:0.5px solid rgba(240,96,96,0.2);border-radius:10px;padding:1rem 1.2rem;color:#F06060;font-size:13px;margin-top:1rem;">Required financial fields missing — M-score could not be calculated.</div>', unsafe_allow_html=True)
        else:
            m    = md["m"]
            idx  = md["idx"]
            initials_m = ticker_m[:2]
            if md["logo"]:
                logo_m = f'<div style="width:38px;height:38px;border-radius:8px;background:#1C1F27;padding:4px;flex-shrink:0;display:flex;align-items:center;justify-content:center;"><img src="{md["logo"]}" style="width:30px;height:30px;object-fit:contain;"></div>'
            else:
                logo_m = f'<div style="width:38px;height:38px;border-radius:8px;background:rgba(201,168,76,0.15);border:0.5px solid rgba(201,168,76,0.3);display:flex;align-items:center;justify-content:center;font-family:monospace;font-size:13px;font-weight:500;color:#C9A84C;flex-shrink:0;">{initials_m}</div>'

            st.markdown(f"""
            <div class="fu2" style="background:#14161B;border:0.5px solid rgba(201,168,76,0.2);border-radius:14px;
                        padding:1.2rem 1.5rem;display:flex;align-items:center;
                        justify-content:space-between;flex-wrap:wrap;gap:12px;margin:1rem 0 1.5rem;">
              <div style="display:flex;align-items:center;gap:12px;">
                {logo_m}
                <div style="display:flex;align-items:center;gap:10px;">
                  <div style="background:rgba(201,168,76,0.1);border:0.5px solid rgba(201,168,76,0.2);
                              border-radius:6px;padding:4px 10px;font-family:'DM Mono',monospace;
                              font-size:13px;font-weight:500;color:#C9A84C;letter-spacing:1px;">{ticker_m}</div>
                  <div>
                    <div style="font-size:17px;font-weight:700;color:#F0EDE6;">{md['name']}</div>
                    <div style="font-size:12px;color:#9B9589;margin-top:1px;">{md['sector']} · {md['country']}</div>
                  </div>
                </div>
              </div>
              <div style="text-align:right;">
                <div style="font-size:10px;color:#5C5850;letter-spacing:1px;text-transform:uppercase;">Market Cap</div>
                <div style="font-family:'DM Mono',monospace;font-size:20px;font-weight:500;color:#F0EDE6;margin-top:2px;">{fmt(md['mc'])}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # M-Score panel
            st.markdown('<div class="fu3 sec-hdr"><span class="sec-lbl">Beneish M-Score</span><div class="sec-line"></div></div>', unsafe_allow_html=True)
            render_mscore_panel(m)

            # Index breakdown — 4-column grid (2 rows × 4)
            st.markdown('<div class="fu4 sec-hdr" style="margin-top:1.2rem;"><span class="sec-lbl">Index Breakdown</span><div class="sec-line"></div></div>', unsafe_allow_html=True)
            INDEX_ORDER = ["DSRI","GMI","AQI","SGI","DEPI","SGAI","TATA","LVGI"]
            # Coefficients for display
            COEFF = {"DSRI":"+0.920","GMI":"+0.528","AQI":"+0.404","SGI":"+0.892",
                     "DEPI":"+0.115","SGAI":"−0.172","TATA":"+4.679","LVGI":"−0.327"}
            icols = st.columns(4)
            for i, key in enumerate(INDEX_ORDER):
                full_name, val, elevated = idx[key]
                with icols[i % 4]:
                    val_str  = f"{val:.3f}" if val is not None else "N/A"
                    dot_col  = "#E85555" if elevated else "#3FCF8E"
                    dot_bg   = "rgba(232,85,85,0.12)" if elevated else "rgba(63,207,142,0.1)"
                    dot_bd   = "rgba(232,85,85,0.3)"  if elevated else "rgba(63,207,142,0.3)"
                    flag_lbl = "Elevated" if elevated else "Normal"
                    st.markdown(
                        f'<div style="background:#1C1F27;border:0.5px solid rgba(201,168,76,0.08);'
                        f'border-radius:10px;padding:0.85rem 0.9rem;margin-bottom:8px;">'
                        f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;">'
                        f'<span style="font-family:\'DM Mono\',monospace;font-size:12px;font-weight:500;color:#C9A84C;">{key}</span>'
                        f'<span style="font-family:\'DM Mono\',monospace;font-size:9px;color:#5C5850;">{COEFF[key]}</span>'
                        f'</div>'
                        f'<div style="font-size:9px;color:#5C5850;margin-bottom:8px;line-height:1.3;">{full_name}</div>'
                        f'<div style="font-family:\'DM Mono\',monospace;font-size:20px;font-weight:500;color:#F0EDE6;margin-bottom:6px;">{val_str}</div>'
                        f'<div style="display:inline-flex;align-items:center;gap:5px;background:{dot_bg};'
                        f'border:0.5px solid {dot_bd};border-radius:5px;padding:2px 8px;">'
                        f'<div style="width:5px;height:5px;border-radius:50%;background:{dot_col};"></div>'
                        f'<span style="font-size:9px;font-weight:600;color:{dot_col};">{flag_lbl}</span>'
                        f'</div></div>',
                        unsafe_allow_html=True
                    )

            # Interpretation
            if m > -1.78:
                interp = f"An M-Score of <b>{m:.2f}</b> (above −1.78) signals a <b>high probability of earnings manipulation</b>. The model identifies abnormal growth in receivables, declining margins, or aggressive accrual accounting. Independent scrutiny of financial statements is strongly advised."
            elif m > -2.22:
                interp = f"An M-Score of <b>{m:.2f}</b> falls in the <b>grey zone</b> (−2.22 to −1.78). Some indices appear elevated — the risk of earnings management cannot be ruled out. Monitor receivables, accruals, and margin trends closely."
            else:
                interp = f"An M-Score of <b>{m:.2f}</b> (below −2.22) suggests a <b>low probability of earnings manipulation</b>. The company's financial reporting appears consistent with non-manipulative accounting practices."
            st.markdown(f'<div class="fu5" style="margin-top:1.2rem;background:#1C1F27;border-left:2px solid #C9A84C;border-radius:0 10px 10px 0;padding:1rem 1.2rem;font-size:13px;color:#9B9589;line-height:1.7;">{interp}</div>', unsafe_allow_html=True)

    elif btn_m:
        st.markdown('<div style="background:#2B1A05;border:0.5px solid rgba(240,160,48,0.2);border-radius:10px;padding:1rem 1.2rem;color:#F0A030;font-size:13px;margin-top:1rem;">Please enter a ticker symbol.</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: F-Score
# ════════════════════════════════════════════════════════════════════════════
def page_fscore():
    render_back_button()
    st.markdown("""
    <div class="fu1" style="margin-bottom:1.5rem;padding-bottom:1.2rem;
                            border-bottom:0.5px solid rgba(201,168,76,0.12);">
      <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;">
        <div>
          <div style="font-size:24px;font-weight:800;letter-spacing:-0.3px;color:#F0EDE6;">
            Piotroski <span style="color:#C9A84C;">F-Score</span>
          </div>
          <div style="font-size:12px;color:#6B6560;margin-top:3px;">9-point financial strength scoring model</div>
        </div>
        <div style="font-family:'DM Mono',monospace;font-size:10px;color:#8A6E2F;
                    background:rgba(201,168,76,0.08);border:0.5px solid rgba(201,168,76,0.15);
                    padding:4px 10px;border-radius:4px;letter-spacing:1px;">v1.0</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="font-size:11px;font-weight:500;color:#9B9589;letter-spacing:1.5px;text-transform:uppercase;margin:1rem 0 6px;">Company Ticker</div>', unsafe_allow_html=True)
    pf1, pf2 = st.columns([4, 1])
    with pf1:
        t_f = st.text_input("tf", placeholder="e.g. AAPL, TSLA, MSFT", label_visibility="collapsed")
    with pf2:
        btn_f = st.button("Analyze →", key="btn_f")

    if btn_f and t_f.strip():
        ticker_f = t_f.strip().upper()
        with st.spinner("Fetching data..."):
            fd = compute_fscore(ticker_f)

        if not fd["ok"]:
            st.markdown('<div style="background:#2B0D0D;border:0.5px solid rgba(240,96,96,0.2);border-radius:10px;padding:1rem 1.2rem;color:#F06060;font-size:13px;margin-top:1rem;">Required financial fields missing — F-score could not be calculated.</div>', unsafe_allow_html=True)
        else:
            score = fd["score"]
            flags = fd["flags"]
            initials_f = ticker_f[:2]
            if fd["logo"]:
                logo_f = f'<div style="width:38px;height:38px;border-radius:8px;background:#1C1F27;padding:4px;flex-shrink:0;display:flex;align-items:center;justify-content:center;"><img src="{fd["logo"]}" style="width:30px;height:30px;object-fit:contain;"></div>'
            else:
                logo_f = f'<div style="width:38px;height:38px;border-radius:8px;background:rgba(201,168,76,0.15);border:0.5px solid rgba(201,168,76,0.3);display:flex;align-items:center;justify-content:center;font-family:monospace;font-size:13px;font-weight:500;color:#C9A84C;flex-shrink:0;">{initials_f}</div>'

            st.markdown(f"""
            <div class="fu2" style="background:#14161B;border:0.5px solid rgba(201,168,76,0.2);border-radius:14px;
                        padding:1.2rem 1.5rem;display:flex;align-items:center;
                        justify-content:space-between;flex-wrap:wrap;gap:12px;margin:1rem 0 1.5rem;">
              <div style="display:flex;align-items:center;gap:12px;">
                {logo_f}
                <div style="display:flex;align-items:center;gap:10px;">
                  <div style="background:rgba(201,168,76,0.1);border:0.5px solid rgba(201,168,76,0.2);
                              border-radius:6px;padding:4px 10px;font-family:'DM Mono',monospace;
                              font-size:13px;font-weight:500;color:#C9A84C;letter-spacing:1px;">{ticker_f}</div>
                  <div>
                    <div style="font-size:17px;font-weight:700;color:#F0EDE6;">{fd['name']}</div>
                    <div style="font-size:12px;color:#9B9589;margin-top:1px;">{fd['sector']} · {fd['country']}</div>
                  </div>
                </div>
              </div>
              <div style="text-align:right;">
                <div style="font-size:10px;color:#5C5850;letter-spacing:1px;text-transform:uppercase;">Market Cap</div>
                <div style="font-family:'DM Mono',monospace;font-size:20px;font-weight:500;color:#F0EDE6;margin-top:2px;">{fmt(fd['mc'])}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="fu3 sec-hdr"><span class="sec-lbl">Raw Financials</span><div class="sec-line"></div></div>', unsafe_allow_html=True)
            f_fin_rows = [
                ("Total Assets",        fd['ta0']),
                ("Net Income",          fd['ni0']),
                ("Operating Cash Flow", fd['cfo0']),
                ("Long-term Debt",      fd.get('ltd0')),
                ("Current Assets",      fd.get('ca0')),
                ("Current Liabilities", fd.get('cl0')),
                ("Gross Profit",        fd.get('gp0')),
                ("Revenue",             fd.get('rv0')),
            ]
            fcols = st.columns(2)
            for i, (lbl, val) in enumerate(f_fin_rows):
                with fcols[i % 2]:
                    disp = fmt(val) if val is not None else "N/A"
                    st.markdown(f'<div style="display:flex;justify-content:space-between;align-items:center;background:#14161B;border-radius:8px;padding:10px 14px;border:0.5px solid rgba(201,168,76,0.08);margin-bottom:8px;"><span style="font-size:12px;color:#9B9589;">{lbl}</span><span style="font-family:\'DM Mono\',monospace;font-size:13px;font-weight:500;color:#F0EDE6;">{disp}</span></div>', unsafe_allow_html=True)

            st.markdown('<div class="fu4 sec-hdr" style="margin-top:1.5rem;"><span class="sec-lbl">Piotroski F-Score</span><div class="sec-line"></div></div>', unsafe_allow_html=True)
            render_fscore_panel(score, flags)

            st.markdown('<div class="fu5 sec-hdr" style="margin-top:1.2rem;"><span class="sec-lbl">Criteria Breakdown</span><div class="sec-line"></div></div>', unsafe_allow_html=True)
            ga, gb, gc = st.columns(3, gap="medium")
            with ga:
                st.markdown(render_criteria_group("A · Profitability", ["F1","F2","F3","F4"], flags), unsafe_allow_html=True)
            with gb:
                st.markdown(render_criteria_group("B · Leverage &amp; Liquidity", ["F5","F6","F7"], flags), unsafe_allow_html=True)
            with gc:
                st.markdown(render_criteria_group("C · Operating Efficiency", ["F8","F9"], flags), unsafe_allow_html=True)

            if score >= 7:
                interp = f"A Piotroski F-Score of <b>{score}/9</b> signals <b>strong financial health</b>. The company performs well across profitability, leverage, and operating efficiency — historically associated with positive stock performance."
            elif score >= 3:
                interp = f"A score of <b>{score}/9</b> is <b>neutral</b>. Some financial indicators are positive, but the company shows weaknesses in certain areas. Monitor trends across the next few reporting periods."
            else:
                interp = f"A score of <b>{score}/9</b> flags <b>financial weakness</b>. The Piotroski model identifies deteriorating fundamentals across multiple dimensions — elevated caution is advised."
            st.markdown(f'<div class="fu6" style="margin-top:1.2rem;background:#1C1F27;border-left:2px solid #C9A84C;border-radius:0 10px 10px 0;padding:1rem 1.2rem;font-size:13px;color:#9B9589;line-height:1.7;">{interp}</div>', unsafe_allow_html=True)

    elif btn_f:
        st.markdown('<div style="background:#2B1A05;border:0.5px solid rgba(240,160,48,0.2);border-radius:10px;padding:1rem 1.2rem;color:#F0A030;font-size:13px;margin-top:1rem;">Please enter a ticker symbol.</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: ML Score
# ════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600, show_spinner=False)
def compute_mlscore(ticker_str):
    """Wrapper around the trained XGBoost model — cacheable via Streamlit."""
    return _ml_predict(ticker_str)


def render_mlscore_panel(result: dict):
    """Animated probability gauge for the ML distress score."""
    prob       = result["probability"]
    risk_label = result["risk_label"]
    risk_color = result["risk_color"]
    threshold  = result.get("threshold", 0.5)
    missing    = result.get("missing_pct", 0.0)
    pct        = round(prob * 100, 1)
    # Gauge: 0 = left (low risk), 100 = right (high risk)
    gp = max(3.0, min(97.0, pct))

    html = f"""
<div id="mlpanel" style="background:#14161B;border:0.5px solid rgba(201,107,232,0.2);
     border-radius:14px;padding:1.8rem 2rem 1.5rem;margin-bottom:1rem;">

  <div style="display:flex;align-items:center;gap:10px;margin-bottom:1.4rem;">
    <div style="font-size:11px;font-weight:600;letter-spacing:2px;color:#5C5850;text-transform:uppercase;">ML Distress Probability</div>
    <div style="flex:1;height:0.5px;background:rgba(201,107,232,0.1);"></div>
    <div style="font-size:10px;font-weight:600;letter-spacing:1px;
                background:rgba(201,107,232,0.1);color:#C96BE8;
                padding:3px 9px;border-radius:5px;font-family:'DM Mono',monospace;">XGBoost</div>
  </div>

  <!-- Big probability number -->
  <div style="text-align:center;margin-bottom:1.6rem;">
    <div id="ml-pct" style="font-family:'DM Mono',monospace;font-size:52px;font-weight:500;
                              color:{risk_color};line-height:1;">0.0%</div>
    <div style="font-size:13px;font-weight:600;color:{risk_color};margin-top:6px;
                letter-spacing:0.5px;">{risk_label}</div>
    <div style="font-size:11px;color:#5C5850;margin-top:4px;">
      Decision threshold: {threshold:.0%}
    </div>
  </div>

  <!-- Gradient bar gauge -->
  <div style="position:relative;height:10px;border-radius:999px;margin:0 0.5rem 0.4rem;
              background:linear-gradient(to right,#3FCF8E 0%,#F0A030 40%,#E85555 80%,#B22020 100%);
              box-shadow:0 0 12px rgba(201,107,232,0.15);">
    <div id="ml-marker" style="position:absolute;top:50%;transform:translate(-50%,-50%);
                                left:0%;transition:left 1s cubic-bezier(.22,1,.36,1);
                                width:16px;height:16px;border-radius:50%;
                                background:{risk_color};
                                box-shadow:0 0 8px {risk_color},0 0 0 3px rgba(0,0,0,0.6);"></div>
  </div>
  <div style="display:flex;justify-content:space-between;font-family:'DM Mono',monospace;
              font-size:10px;color:#5C5850;padding:0 0.5rem;margin-bottom:1.4rem;">
    <span>0% · Low</span>
    <span>{threshold:.0%} · Threshold</span>
    <span>100% · High</span>
  </div>

  <!-- Interpretation bar -->
  <div style="background:#1C1F27;border-left:2px solid {risk_color};border-radius:0 8px 8px 0;
              padding:0.7rem 1rem;font-size:12px;color:#9B9589;line-height:1.6;">
    {"<b style='color:" + risk_color + ";'>⚠ High distress probability.</b> The model assigns a &gt;50% chance of financial distress within 1 year." if prob >= 0.5
    else "<b style='color:" + risk_color + ";'>⚡ Elevated risk.</b> Probability between 20–50% — monitor closely and corroborate with other models." if prob >= 0.2
    else "<b style='color:" + risk_color + ";'>✓ Low distress risk.</b> The ML model finds no strong signals of near-term financial distress."}
    {"<span style='color:#5C5850;'> &nbsp;·&nbsp; " + f"{missing:.0%} of features imputed from training-set medians.</span>" if missing > 0.1 else ""}
  </div>

</div>

<script>
(function(){{
  var target = {pct};
  var marker = document.getElementById('ml-marker');
  var numEl  = document.getElementById('ml-pct');
  if (!marker || !numEl) return;
  var start = null;
  var dur   = 1000;
  function ease(t){{ return t<0.5 ? 4*t*t*t : 1-Math.pow(-2*t+2,3)/2; }}
  function step(ts){{
    if (!start) start = ts;
    var p = Math.min((ts - start) / dur, 1);
    var v = ease(p) * target;
    numEl.textContent  = v.toFixed(1) + '%';
    marker.style.left  = Math.max(1, Math.min(99, v)) + '%';
    if (p < 1) requestAnimationFrame(step);
    else {{ numEl.textContent = target.toFixed(1) + '%'; marker.style.left = '{gp}%'; }}
  }}
  requestAnimationFrame(step);
}})();
</script>
"""
    components.html(html, height=320, scrolling=False)


def page_mlscore():
    render_back_button()
    st.markdown("""
    <div class="fu1" style="margin-bottom:1.5rem;padding-bottom:1.2rem;
                            border-bottom:0.5px solid rgba(201,168,76,0.12);">
      <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;">
        <div>
          <div style="font-size:24px;font-weight:800;letter-spacing:-0.3px;color:#F0EDE6;">
            ML <span style="color:#C9A84C;">Distress Score</span>
          </div>
          <div style="font-size:12px;color:#6B6560;margin-top:3px;">
            XGBoost classifier trained on WRDS/Compustat 1990–2025 data
          </div>
        </div>
        <div style="font-family:'DM Mono',monospace;font-size:10px;color:#8A4FBF;
                    background:rgba(201,107,232,0.08);border:0.5px solid rgba(201,107,232,0.2);
                    padding:4px 10px;border-radius:4px;letter-spacing:1px;">v1.0 · XGBoost</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if not _ml_available():
        st.markdown("""
        <div style="background:#1A1025;border:0.5px solid rgba(201,107,232,0.2);border-radius:14px;
                    padding:1.5rem 1.8rem;margin-top:1rem;">
          <div style="font-size:16px;font-weight:700;color:#C96BE8;margin-bottom:0.7rem;">
            ⚙ Model Not Trained Yet
          </div>
          <div style="font-size:13px;color:#9B9589;line-height:1.7;">
            To enable this feature, run the ML training pipeline from the
            <code style="background:#0E0F12;padding:2px 6px;border-radius:4px;color:#C9A84C;">ml_model/</code> directory:
          </div>
          <div style="background:#0E0F12;border-radius:8px;padding:1rem 1.2rem;margin:0.8rem 0;
                      font-family:'DM Mono',monospace;font-size:12px;color:#9B9589;line-height:2;">
            <span style="color:#5C5850;"># Step 1 — Pull WRDS data (requires WRDS account)</span><br>
            <span style="color:#C9A84C;">python</span> ml_model/01_pull_wrds_data.py<br><br>
            <span style="color:#5C5850;"># Step 2 — Engineer features</span><br>
            <span style="color:#C9A84C;">python</span> ml_model/02_prepare_features.py<br><br>
            <span style="color:#5C5850;"># Step 3 — Train the model (~5 min)</span><br>
            <span style="color:#C9A84C;">python</span> ml_model/03_train_model.py
          </div>
          <div style="font-size:12px;color:#5C5850;line-height:1.5;">
            After training, restart the app and this page will show live predictions.
          </div>
        </div>
        """, unsafe_allow_html=True)
        return

    st.markdown('<div style="font-size:11px;font-weight:500;color:#9B9589;letter-spacing:1.5px;text-transform:uppercase;margin:1rem 0 6px;">Company Ticker</div>', unsafe_allow_html=True)
    pml1, pml2 = st.columns([4, 1])
    with pml1:
        t_ml = st.text_input("t_ml", placeholder="e.g. AAPL, TSLA, MSFT", label_visibility="collapsed")
    with pml2:
        btn_ml = st.button("Analyze →", key="btn_ml")

    if btn_ml and t_ml.strip():
        ticker_ml = t_ml.strip().upper()
        with st.spinner("Running ML model..."):
            result = compute_mlscore(ticker_ml)

        if result.get("error") or result.get("probability") is None:
            err_msg = result.get("error", "Unknown error")
            st.markdown(f'<div style="background:#2B0D0D;border:0.5px solid rgba(240,96,96,0.2);border-radius:10px;padding:1rem 1.2rem;color:#F06060;font-size:13px;margin-top:1rem;">{err_msg}</div>', unsafe_allow_html=True)
        else:
            prob       = result["probability"]
            risk_color = result["risk_color"]
            risk_label = result["risk_label"]

            # ── Company header (from yfinance info) ──────────────────────────
            try:
                info   = yf.Ticker(ticker_ml).info
                name   = info.get("longName", ticker_ml)
                sector = info.get("sector", "N/A")
                country= info.get("country", "N/A")
                mc     = info.get("marketCap", 0) or 0
                website= info.get("website", "") or ""
                domain = website.replace("https://","").replace("http://","").split("/")[0]
                logo   = f"https://www.google.com/s2/favicons?domain={domain}&sz=64" if domain else ""
            except Exception:
                name = ticker_ml; sector = "N/A"; country = "N/A"; mc = 0; logo = ""

            initials_ml = ticker_ml[:2]
            if logo:
                logo_ml = f'<div style="width:38px;height:38px;border-radius:8px;background:#1C1F27;padding:4px;flex-shrink:0;display:flex;align-items:center;justify-content:center;"><img src="{logo}" style="width:30px;height:30px;object-fit:contain;"></div>'
            else:
                logo_ml = f'<div style="width:38px;height:38px;border-radius:8px;background:rgba(201,107,232,0.15);border:0.5px solid rgba(201,107,232,0.3);display:flex;align-items:center;justify-content:center;font-family:monospace;font-size:13px;font-weight:500;color:#C96BE8;flex-shrink:0;">{initials_ml}</div>'

            st.markdown(f"""
            <div class="fu2" style="background:#14161B;border:0.5px solid rgba(201,168,76,0.2);border-radius:14px;
                        padding:1.2rem 1.5rem;display:flex;align-items:center;
                        justify-content:space-between;flex-wrap:wrap;gap:12px;margin:1rem 0 1.5rem;">
              <div style="display:flex;align-items:center;gap:12px;">
                {logo_ml}
                <div style="display:flex;align-items:center;gap:10px;">
                  <div style="background:rgba(201,168,76,0.1);border:0.5px solid rgba(201,168,76,0.2);
                              border-radius:6px;padding:4px 10px;font-family:'DM Mono',monospace;
                              font-size:13px;font-weight:500;color:#C9A84C;letter-spacing:1px;">{ticker_ml}</div>
                  <div>
                    <div style="font-size:17px;font-weight:700;color:#F0EDE6;">{name}</div>
                    <div style="font-size:12px;color:#9B9589;margin-top:1px;">{sector} · {country}</div>
                  </div>
                </div>
              </div>
              <div style="text-align:right;">
                <div style="font-size:10px;color:#5C5850;letter-spacing:1px;text-transform:uppercase;">Market Cap</div>
                <div style="font-family:'DM Mono',monospace;font-size:20px;font-weight:500;color:#F0EDE6;margin-top:2px;">{fmt(mc)}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="fu3 sec-hdr"><span class="sec-lbl">ML Distress Probability</span><div class="sec-line"></div></div>', unsafe_allow_html=True)
            render_mlscore_panel(result)

            # ── Key feature values ────────────────────────────────────────────
            feats = result.get("features", {})
            if feats:
                st.markdown('<div class="fu4 sec-hdr" style="margin-top:1.2rem;"><span class="sec-lbl">Key Inputs Used by Model</span><div class="sec-line"></div></div>', unsafe_allow_html=True)

                display_feats = [
                    ("z_score",    "Altman Z-Score",          "{:.2f}"),
                    ("o_prob",     "Ohlson Distress Prob",     "{:.1%}"),
                    ("f_score",    "Piotroski F-Score",        "{:.0f}/9"),
                    ("m_score",    "Beneish M-Score",          "{:.3f}"),
                    ("roa",        "Return on Assets",         "{:.1%}"),
                    ("cfo_ta",     "Cash Flow / Assets",       "{:.1%}"),
                    ("lev",        "Long-term Leverage",       "{:.2f}"),
                    ("cr",         "Current Ratio",            "{:.2f}"),
                    ("delta_sale", "YoY Revenue Growth",       "{:.1%}"),
                    ("debt_ratio", "Debt Ratio",               "{:.2f}"),
                    ("log_at",     "Log Total Assets",         "{:.2f}"),
                    ("delta_roa",  "ΔROA (YoY change)",        "{:.3f}"),
                ]

                feat_cols = st.columns(3)
                for i, (key, label, fmt_str) in enumerate(display_feats):
                    val = feats.get(key)
                    if val is not None and not math.isnan(float(val)):
                        disp = fmt_str.format(float(val))
                        color_val = "#F0EDE6"
                        if key == "z_score":
                            color_val = "#3FCF8E" if val >= 2.99 else "#E85555" if val < 1.81 else "#F0A030"
                        elif key == "o_prob":
                            color_val = "#E85555" if val >= 0.5 else "#F0A030" if val >= 0.2 else "#3FCF8E"
                        elif key == "f_score":
                            color_val = "#3FCF8E" if val >= 7 else "#E85555" if val <= 2 else "#F0A030"
                        elif key == "m_score":
                            color_val = "#E85555" if val > -1.78 else "#F0A030" if val > -2.22 else "#3FCF8E"
                    else:
                        disp = "N/A"
                        color_val = "#5C5850"
                    with feat_cols[i % 3]:
                        st.markdown(
                            f'<div style="background:#14161B;border:0.5px solid rgba(201,107,232,0.1);border-radius:8px;'
                            f'padding:10px 14px;margin-bottom:8px;display:flex;justify-content:space-between;align-items:center;">'
                            f'<span style="font-size:11px;color:#9B9589;">{label}</span>'
                            f'<span style="font-family:\'DM Mono\',monospace;font-size:13px;font-weight:500;color:{color_val};">{disp}</span>'
                            f'</div>', unsafe_allow_html=True)

            # ── Model info footer ─────────────────────────────────────────────
            st.markdown(f"""
            <div class="fu5" style="margin-top:1rem;background:#1C1F27;border-left:2px solid #C96BE8;
                         border-radius:0 10px 10px 0;padding:0.9rem 1.2rem;font-size:12px;
                         color:#9B9589;line-height:1.7;">
              <b style="color:#C96BE8;">About this model:</b> XGBoost classifier trained on
              Compustat annual fundamentals (1990–2025), validated on held-out data from 2016–2025.
              Features include all Z/O/F/M model components plus YoY delta ratios.
              Probability is calibrated via isotonic regression for reliability.
              <br><span style="color:#5C5850;">
              Prediction horizon: 1-year forward distress.
              Missing inputs are imputed with training-set medians.
              </span>
            </div>
            """, unsafe_allow_html=True)

    elif btn_ml:
        st.markdown('<div style="background:#2B1A05;border:0.5px solid rgba(240,160,48,0.2);border-radius:10px;padding:1rem 1.2rem;color:#F0A030;font-size:13px;margin-top:1rem;">Please enter a ticker symbol.</div>', unsafe_allow_html=True)


# ── Router ────────────────────────────────────────────────────────────────────
if current_page == "zscore":
    page_zscore()
elif current_page == "oscore":
    page_oscore()
elif current_page == "fscore":
    page_fscore()
elif current_page == "mscore":
    page_mscore()
elif current_page == "comparison":
    page_comparison()
elif current_page == "mlscore":
    page_mlscore()
else:
    page_home()
