"""
Rare Earth Exposure Research Dashboard
=======================================
Presentation dashboard for mentor meeting.
Loads pre-computed results and displays them interactively.

HOW TO RUN:
    pip3 install streamlit plotly
    streamlit run dashboard.py

REQUIRES (in same folder):
    dollar_exposure.csv
    dollar_exposure_events.csv
    delta_results_smm.csv
    delta_results_events.csv
    rare_earth_prices.csv
    event_months.csv
    charts/                    — chart images from delta_regression.py
    charts/event_windows/      — event window charts
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import glob
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Rare Earth Exposure — Duke Research",
    page_icon="⛏",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stTabs [data-baseweb="tab-list"] { gap: 0px; border-bottom: 1px solid #222; }
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px; letter-spacing: 0.05em;
    color: #555; padding: 12px 28px;
}
.stTabs [aria-selected="true"] {
    color: #ffffff !important;
    border-bottom: 2px solid #ffffff !important;
    background: transparent !important;
}
.kpi-row { display: flex; gap: 16px; margin-bottom: 2rem; }
.kpi {
    flex: 1; background: #111; border: 1px solid #222;
    border-radius: 8px; padding: 1.2rem 1.5rem;
}
.kpi-label {
    font-size: 11px; color: #555; letter-spacing: 0.1em;
    text-transform: uppercase; font-family: 'IBM Plex Mono', monospace;
    margin-bottom: 6px;
}
.kpi-value {
    font-size: 26px; font-weight: 500; color: #ffffff;
    font-family: 'IBM Plex Mono', monospace;
}
.kpi-sub { font-size: 11px; color: #444; margin-top: 4px; }
.section-label {
    font-size: 11px; font-weight: 500; color: #555;
    letter-spacing: 0.1em; text-transform: uppercase;
    border-bottom: 1px solid #1a1a1a; padding-bottom: 8px;
    margin-bottom: 1.5rem; font-family: 'IBM Plex Mono', monospace;
}
.finding {
    background: #111; border-left: 3px solid #ffffff;
    padding: 1rem 1.25rem; margin: 0.75rem 0;
    border-radius: 0 6px 6px 0; font-size: 14px;
    color: #cccccc; line-height: 1.7;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_all():
    data = {}
    loaders = {
        "exposure"       : ("dollar_exposure.csv", False),
        "exposure_events": ("dollar_exposure_events.csv", False),
        "delta_smm"      : ("delta_results_smm.csv", False),
        "delta_events"   : ("delta_results_events.csv", False),
        "re_prices"      : ("rare_earth_prices.csv", True),
        "event_months"   : ("event_months.csv", True),
        "sample"         : ("event_study_sample.csv", False),
    }
    for key, (fname, has_index) in loaders.items():
        try:
            if has_index:
                data[key] = pd.read_csv(fname, index_col=0, parse_dates=True)
            else:
                data[key] = pd.read_csv(fname)
            data[f"{key}_ok"] = True
        except FileNotFoundError:
            data[key] = None
            data[f"{key}_ok"] = False
    return data

def get_charts(folder, pattern="*.png"):
    return sorted(glob.glob(os.path.join(folder, pattern)))

NASDAQ = [
    "AMD","NVDA","TSLA","AAPL","INTC","QCOM","MCHP","NXPI",
    "SWKS","ON","MPWR","KLAC","LRCX","AMAT","TER","KEYS",
    "COHR","FSLR","RIVN","CSCO","ANET","AVGO","GLW","ADI",
    "TXN","MU","AMZN","GOOGL","GOOG","APP","ISRG","STX","WDC",
    "ADBE","ADSK","CDNS","DDOG","EBAY","IDXX","ALGN","DXCM",
    "PODD","HOLX","CIEN","LITE","NTAP","SNDK","SMCI","DELL",
    "HPQ","HPE","ABNB",
]

data   = load_all()
exp    = data.get("exposure")
exp_ev = data.get("exposure_events")
d_smm  = data.get("delta_smm")
d_ev   = data.get("delta_events")
sample = data.get("sample")

if sample is not None:
    sample = sample.copy()
    sample["ticker"] = sample["Symbol"].apply(
        lambda x: f"{x}.O" if x in NASDAQ else f"{x}.N"
    )

st.markdown("# Rare Earth Metal Exposure")
st.markdown("*S&P 500 company exposure analysis · Duke University · 2026*")
st.markdown("---")

n_co  = len(exp) if exp is not None else "—"
n_met = len(data["re_prices"].columns) if data.get("re_prices") is not None else "—"
sig_f = int(d_smm["significant"].sum()) if d_smm is not None else "—"
sig_e = int(d_ev["significant"].sum()) if d_ev is not None else "—"

st.markdown(f"""
<div class="kpi-row">
  <div class="kpi">
    <div class="kpi-label">Companies analyzed</div>
    <div class="kpi-value">{n_co}</div>
    <div class="kpi-sub">S&P 500 — high / medium / control tiers</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Rare earth metals</div>
    <div class="kpi-value">{n_met}</div>
    <div class="kpi-sub">SMM spot prices via Refinitiv</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Significant deltas (full)</div>
    <div class="kpi-value">{sig_f}</div>
    <div class="kpi-sub">p &lt; 0.05 — full sample</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Significant deltas (events)</div>
    <div class="kpi-value">{sig_e}</div>
    <div class="kpi-sub">p &lt; 0.05 — shock months only</div>
  </div>
</div>
""", unsafe_allow_html=True)

t1, t2, t3, t4, t5 = st.tabs([
    "01 — RE Prices",
    "02 — Delta Regression",
    "03 — Dollar Exposure",
    "04 — Sector Sensitivity",
    "05 — Event Windows",
])

with t1:
    st.markdown('<p class="section-label">Rare earth spot prices — Shanghai Metals Market via Refinitiv</p>', unsafe_allow_html=True)
    if data["re_prices"] is not None:
        re = data["re_prices"].copy()
        metals = st.multiselect("Select metals", options=re.columns.tolist(),
                                default=[c for c in re.columns if c != "lithium"][:4])
        if metals:
            col1, col2 = st.columns([3, 1])
            with col1:
                re_norm = re[metals].div(re[metals].dropna().iloc[0]) * 100
                fig = go.Figure()
                colors_re = ["#ffffff","#4ECDC4","#FF6B6B","#F7DC6F","#BB8FCE","#85C1E9"]
                for i, m in enumerate(metals):
                    fig.add_trace(go.Scatter(x=re_norm.index, y=re_norm[m].dropna(),
                                             name=m.capitalize(),
                                             line=dict(color=colors_re[i % len(colors_re)], width=1.5)))
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  font_color="#888", xaxis=dict(gridcolor="#111", color="#444"),
                                  yaxis=dict(gridcolor="#111", color="#444", title="Index (base=100)"),
                                  legend=dict(font=dict(color="#888"), bgcolor="rgba(0,0,0,0)"),
                                  margin=dict(t=20,b=20,l=20,r=20), height=420, hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.markdown("**Change since 2015**")
                latest = re[metals].dropna().iloc[-1]
                first  = re[metals].dropna().iloc[0]
                for m in metals:
                    chg = (latest[m] / first[m] - 1) * 100
                    color = "green" if chg > 0 else "red"
                    st.markdown(f"**{m.capitalize()}**  \n`{latest[m]:,.0f}`  "
                                f"<span style='color:{color}'>{chg:+.0f}%</span>",
                                unsafe_allow_html=True)
                    st.markdown("---")
        if data["event_months"] is not None:
            st.markdown('<p class="section-label">Event months (rolling z-score threshold = 1.5sigma)</p>', unsafe_allow_html=True)
            ev = data["event_months"]
            evs = ev[ev["is_event"] == True][["re_return","z_score","direction"]].copy()
            evs["re_return"] = (evs["re_return"] * 100).round(1)
            evs["z_score"]   = evs["z_score"].round(2)
            evs.index        = evs.index.strftime("%Y-%m")
            evs.columns      = ["RE Return (%)","Z-Score","Direction"]
            st.dataframe(evs, use_container_width=True, height=300)
            img = "charts/event_months_re_index.png"
            if os.path.exists(img):
                st.image(img, use_container_width=True)
    else:
        st.warning("rare_earth_prices.csv not found.")

with t2:
    st.markdown('<p class="section-label">Delta regression results</p>', unsafe_allow_html=True)
    st.markdown("""
    **Model:** `Stock return = alpha + delta_RE x RE_index + delta_M x Market_return + epsilon`

    **delta_RE** — sensitivity to a 1% move in rare earth prices, controlling for the market.
    **epsilon** — residual unexplained risk. **R2** — % of variance explained.
    """)
    col1, col2 = st.columns(2)
    def show_delta(df, label, key):
        st.markdown(f"**{label}**")
        if df is not None and sample is not None:
            show = df.merge(sample[["ticker","Security","tier"]], on="ticker", how="left")
            cols = [c for c in ["ticker","Security","tier","delta_RE","p_value","significant","r2","epsilon"] if c in show.columns]
            if st.checkbox("Significant only", key=key):
                show = show[show["significant"] == True]
            st.dataframe(show[cols].sort_values("delta_RE").round(4), use_container_width=True, height=380)
        else:
            st.warning("Data not found.")
    with col1:
        show_delta(d_smm, "Full sample (135 months)", "s1")
    with col2:
        show_delta(d_ev, "Event-based (shock months only)", "s2")
    st.markdown("---")
    if os.path.exists("charts/full_vs_event_delta.png"):
        st.markdown('<p class="section-label">Full sample vs event-based comparison</p>', unsafe_allow_html=True)
        st.image("charts/full_vs_event_delta.png", use_container_width=True)
    st.markdown("""
    <div class="finding">
    <b>Why two regressions?</b> The full-sample regression shows mostly positive deltas —
    counterintuitive for manufacturers hurt by rising input costs. Most historical RE price
    spikes coincided with macro booms that drove both RE prices AND stocks upward simultaneously
    (omitted variable bias). The event-based regression isolates genuine shock months using a
    rolling 12-month z-score threshold (1.5sigma), revealing a cleaner negative relationship
    for high-exposure companies.
    </div>
    """, unsafe_allow_html=True)

with t3:
    st.markdown('<p class="section-label">Implied dollar exposure to rare earth price risk</p>', unsafe_allow_html=True)
    st.markdown("""
    **Formula:** `Dollar exposure = |delta_RE| x sigma_RE x Market Cap`
    = dollar change in market value for a **1 standard deviation annual move** in RE prices.
    RE index annualized volatility = **15.9%**
    """)
    reg_choice = st.radio("Regression used", ["Full sample","Event-based"], horizontal=True, key="exp_r")
    df_exp = exp if reg_choice == "Full sample" else exp_ev
    if df_exp is not None and "dollar_exposure_usd" in df_exp.columns:
        top_n  = st.slider("Top N companies", 10, 50, 20)
        tier_f = st.multiselect("Filter by tier", ["high","medium","control"], default=["high","medium"])
        display = df_exp.copy()
        if tier_f and "tier" in display.columns:
            display = display[display["tier"].isin(tier_f)]
        top = display.dropna(subset=["Security","dollar_exposure_usd"]).sort_values("dollar_exposure_usd", ascending=False).head(top_n)
        col1, col2 = st.columns([3, 1])
        with col1:
            fig = px.bar(top, x="exposure_billions", y="Security", color="tier", orientation="h",
                         color_discrete_map={"high":"#FF6B6B","medium":"#F7DC6F","control":"#4ECDC4"},
                         labels={"exposure_billions":"Implied exposure ($B)","Security":""},
                         hover_data=["ticker","delta_RE","exposure_pct_mcap","significant"])
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font_color="#888", xaxis=dict(gridcolor="#111",color="#444"),
                              yaxis=dict(gridcolor="#111",color="#888",autorange="reversed"),
                              legend=dict(font=dict(color="#888"),bgcolor="rgba(0,0,0,0)"),
                              margin=dict(t=10,b=20,l=10,r=20), height=max(350, top_n*22))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.metric("Total exposure", f"${display['dollar_exposure_usd'].dropna().sum()/1e9:.0f}B")
            st.metric("Avg % of mkt cap", f"{display['exposure_pct_mcap'].dropna().mean():.1f}%")
            st.metric("Companies with -delta", f"{(display['delta_RE']<0).mean():.0%}")
            if "tier" in display.columns:
                st.markdown("---")
                st.markdown("**By tier**")
                ts = display.dropna(subset=["tier"]).groupby("tier")["exposure_billions"].agg(avg="mean",total="sum").round(1)
                st.dataframe(ts, use_container_width=True)
        st.markdown("---")
        cols_t = [c for c in ["Security","ticker","tier","delta_RE","significant","exposure_billions","exposure_pct_mcap","market_cap_billions"] if c in display.columns]
        st.dataframe(display[cols_t].sort_values("exposure_billions",ascending=False).round(3), use_container_width=True, height=400)
        st.download_button("Download as CSV", display[cols_t].to_csv(index=False), "rare_earth_exposure.csv","text/csv")
    else:
        st.warning("Dollar exposure data not found. Run delta_regression.py first.")

with t4:
    st.markdown('<p class="section-label">Average delta by GICS sub-industry</p>', unsafe_allow_html=True)
    st.markdown("""
    **Red** = hurt when RE prices rise (negative delta — manufacturers)
    **Teal** = benefits when RE prices rise (positive delta — miners)
    """)
    src = st.radio("Regression source", ["Full sample (SMM)","Event-based"], horizontal=True, key="sec_src")
    img = "charts/delta_by_sector_smm.png" if src == "Full sample (SMM)" else "charts/delta_by_sector_event.png"
    if os.path.exists(img):
        st.image(img, use_container_width=True)
    else:
        st.warning(f"Chart not found: {img}")
    st.markdown("---")
    st.markdown('<p class="section-label">Top 15 companies vs RE index</p>', unsafe_allow_html=True)
    v   = st.radio("Version", ["All companies","Excluding Nvidia"], horizontal=True, key="t15v")
    sfx = "all" if v == "All companies" else "no_nvidia"
    p15 = f"charts/top15_vs_re_smm_{sfx}.png"
    if os.path.exists(p15):
        st.image(p15, use_container_width=True)
    else:
        st.warning(f"Chart not found: {p15}")

with t5:
    st.markdown('<p class="section-label">Stock price reaction around rare earth price shocks</p>', unsafe_allow_html=True)
    st.markdown("""
    **Window:** 15 trading days before to 30 days after each shock. Normalized to 100 at event day.
    **White line** = RE composite index. **Colored lines** = top 15 most exposed companies.
    **Hypothesis:** RE price spike -> company stocks go DOWN after day 0.
    """)
    event_charts = get_charts("charts/event_windows")
    if event_charts:
        def format_event_name(filename):
            name = os.path.basename(filename).replace(".png","").replace("event_","")
            parts = name.split("_")
            if len(parts) >= 2:
                date_str = parts[0]
                direction = parts[1].upper()
                year = date_str[:4]
                month_num = int(date_str[4:6])
                months = ["Jan","Feb","Mar","Apr","May","Jun",
                          "Jul","Aug","Sep","Oct","Nov","Dec"]
                month_name = months[month_num - 1]
                arrow = "↑" if direction == "UP" else "↓"
                return f"{month_name} {year} — RE Price {arrow} {direction}"
            return name

        chart_names = [format_event_name(p) for p in event_charts]
        sel = st.selectbox("Select event", range(len(event_charts)), format_func=lambda i: chart_names[i])
        st.image(event_charts[sel], use_container_width=True)
        st.markdown("---")
        st.markdown("**All events overview**")
        cols = st.columns(2)
        for i, path in enumerate(event_charts):
            with cols[i % 2]:
                st.caption(format_event_name(path))
                st.image(path, use_container_width=True)
    else:
        st.warning("No event charts found. Run event_window_chart.py first.")
    st.markdown("""
    <div class="finding">
    <b>Key finding:</b> Most historical RE shocks coincided with macro events (2019 trade war,
    2020 vaccine rally) masking the inverse relationship. January 2026 is the clearest pure
    RE shock — several high-exposure companies declined as RE prices rose, confirming the
    inverse relationship exists but is difficult to isolate from macro noise.
    </div>
    """, unsafe_allow_html=True)