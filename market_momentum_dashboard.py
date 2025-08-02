# market_momentum_dashboard.py — Streamlit v2.0

import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import streamlit as st
from pytrends.request import TrendReq
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('future.no_silent_downcasting', True)

st.set_page_config(layout="wide")

# === CONFIG ===
DEFAULT_ETFS = [
    "SPY", "QQQ", "DIA", "IWM",
    "XLF", "XLK", "XLE", "XLY", "XLI", "XLP", "XLV", "XLU", "XLB", "XLRE", "XLC"
]
TODAY = datetime.date.today()
START = TODAY - datetime.timedelta(days=180)

# === USER INPUT INLINE ===
with st.container():
    cols = st.columns([6, 1])
    with cols[0]:
        user_input = st.text_input("Add Custom Tickers", value="TSLA, AAPL", label_visibility="collapsed")
    with cols[1]:
        st.markdown("&nbsp;")  # space filler
ALL_SYMBOLS = list(set(DEFAULT_ETFS + [x.strip().upper() for x in user_input.split(",") if x.strip()]))

# === DATA ===
@st.cache_data(ttl=3600)
def get_price_data(symbols, start, end):
    return yf.download(symbols, start=start, end=end, group_by='ticker', auto_adjust=True)

data = get_price_data(ALL_SYMBOLS, START.isoformat(), TODAY.isoformat())

# === VIX / VIX3M ===
vix_data = yf.download(['^VIX', '^VIX3M'], start=START, end=TODAY)
vix_ratio = vix_data['Close']['^VIX'].iloc[-1] / vix_data['Close']['^VIX3M'].iloc[-1]

# === ZWEIG BREADTH THRUST ===
breadth_data = yf.download("^GSPC", start=START, end=TODAY)
zweig_signal = "N/A"
try:
    breadth_data["Advance"] = breadth_data["Close"].pct_change() > 0
    breadth_ratio = breadth_data["Advance"].rolling(10).mean()
    latest_zweig = breadth_ratio.iloc[-1]
    zweig_signal = "✅ Breadth Thrust" if latest_zweig > 0.615 else "⚠️ Normal"
except:
    latest_zweig = np.nan

# === INDICATORS ===
def compute_indicators(symbol):
    df = data[symbol].copy()
    df['Advance'] = df['Close'].pct_change() > 0
    df['Decline'] = df['Close'].pct_change() < 0
    df['NetAdv'] = df['Advance'].astype(int) - df['Decline'].astype(int)
    df['EMA19'] = df['NetAdv'].ewm(span=19).mean()
    df['EMA39'] = df['NetAdv'].ewm(span=39).mean()
    df['McClellan'] = df['EMA19'] - df['EMA39']
    return df

# === METRICS ===
st.title("📊 Modern Market Momentum Dashboard")
col1, col2 = st.columns(2)
with col1:
    st.metric("VIX/VIX3M Ratio", value=round(vix_ratio, 2), delta="⚠️ High" if vix_ratio > 1.2 else "✅ Stable")
    st.metric("Zweig Breadth Thrust", value=f"{latest_zweig:.3f}" if pd.notna(latest_zweig) else "N/A", delta=zweig_signal)

# === CHART STACK ===
st.subheader("📈 McClellan Oscillator + Price")
row = st.columns(3)
for i, symbol in enumerate(ALL_SYMBOLS):
    df = compute_indicators(symbol)
    df.dropna(inplace=True)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.4],
                        vertical_spacing=0.03,
                        subplot_titles=(f"{symbol} Price", "McClellan Oscillator"))

    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'],
                                 name="Price"), row=1, col=1)

    colors = ['green' if val > 0 else 'red' for val in df['McClellan']]
    fig.add_trace(go.Bar(x=df.index, y=df['McClellan'], marker_color=colors, name="McClellan"), row=2, col=1)

    fig.update_layout(
        height=600, width=450, showlegend=False,
        xaxis_rangeslider_visible=False,
        xaxis=dict(tickformat='%b %Y'),
        title=f"{symbol} Technical Stack"
    )

    row[i % 3].plotly_chart(fig, use_container_width=True)

# === RRG CHART ===
st.markdown("---")
st.subheader("🔄 Relative Rotation Graph (RRG) — Sector Momentum vs Strength")
view_mode = st.radio("Select Timeframe", ["Daily", "Weekly"], horizontal=True)

@st.cache_data(ttl=3600)
def get_rrg_data(interval, days_back):
    prices = yf.download(DEFAULT_ETFS, start=(TODAY - datetime.timedelta(days=days_back)).isoformat(), end=TODAY.isoformat(), interval=interval)['Close']
    returns = prices.pct_change().dropna()
    benchmark = returns['SPY']
    rel_strength = returns.div(benchmark, axis=0)
    rs = rel_strength.rolling(10).mean()
    mom = rs.diff()
    rs_norm = 100 + (rs - rs.mean())
    mom_norm = 100 + (mom - mom.mean())
    return rs_norm, mom_norm

interval = '1d' if view_mode == "Daily" else '1wk'
days_back = 60 if view_mode == "Daily" else 280
rs_norm, mom_norm = get_rrg_data(interval, days_back)

fig = go.Figure()
for symbol in DEFAULT_ETFS:
    x = rs_norm[symbol].iloc[-10:]
    y = mom_norm[symbol].iloc[-10:]
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines+markers+text", name=symbol,
        text=[symbol]*len(x),
        textposition="top center",
        line=dict(width=2),
        marker=dict(size=8)
    ))

# === QUADRANTS ===
fig.add_shape(type="line", x0=100, x1=100, y0=90, y1=110, line=dict(color="gray", dash="dot"))
fig.add_shape(type="line", y0=100, y1=100, x0=90, x1=110, line=dict(color="gray", dash="dot"))
for x0, y0, x1, y1, color in [(100,100,110,110,"lightgreen"), (90,100,100,110,"lightblue"),
                              (90,90,100,100,"lightcoral"), (100,90,110,100,"khaki")]:
    fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1, fillcolor=color, opacity=0.2, line_width=0)

fig.update_layout(
    title=f"📊 RRG Flow ({view_mode}) — JDK RS vs Momentum",
    xaxis_title="JDK RS-Ratio (Normalized)", yaxis_title="JDK Momentum (Normalized)",
    xaxis_range=[90,110], yaxis_range=[90,110], height=600
)
st.plotly_chart(fig, use_container_width=True)

# === GOOGLE TRENDS ===
st.markdown("---")
st.subheader("📈 Google Trends Sentiment Tracker")
trends = TrendReq(hl='en-US', tz=360)
keywords = ["how to buy stocks", "stock market crash", "AI stocks", "bitcoin"]
trends.build_payload(kw_list=keywords, timeframe='now 7-d')
trend_data = trends.interest_over_time().infer_objects(copy=False)
if not trend_data.empty:
    st.line_chart(trend_data[keywords])
else:
    st.warning("Google Trends data could not be loaded.")

st.caption("Dashboard v2.0 — Optimized RRG and Chart Layout")
