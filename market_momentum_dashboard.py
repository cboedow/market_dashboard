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
import scipy.interpolate as si
import warnings
from RRGPy import RRG
from RRG.universe import Universe


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

# === RRG Section (RRGPy Integrated) ===
st.markdown("---")
st.subheader("🔄 RRG — Sector Rotation via RRGPy")

try:
    view_mode = st.radio("RRG View", ["Daily", "Weekly"], horizontal=True)
    with st.spinner(f"Generating {view_mode} RRG..."):
        days_back = 60 if view_mode == "Daily" else 280
        interval = '1d' if view_mode == "Daily" else '1wk'

        @st.cache_data(ttl=3600)
        def get_rrg_data(etfs, interval, days_back):
            df = yf.download(etfs,
                             start=(datetime.date.today() - datetime.timedelta(days=days_back)).isoformat(),
                             end=TODAY.isoformat(),
                             interval=interval,
                             auto_adjust=True)['Close']
            return df.dropna()

        rrg_prices = get_rrg_data(DEFAULT_ETFS + ['SPY'], interval, days_back)

        # Construct RRG universe and model
        universe = Universe(rrg_prices, benchmark='SPY')
        rrg = RRG(universe=universe)

        # Plot using RRGPy (plotly backend)
        fig = rrg.plot_plotly(
            tickers=[etf for etf in DEFAULT_ETFS if etf != 'SPY'],
            lookback=10,
            show_tails=True,
            quadrant_colors=True,
            annotations=True,
            annotate_latest=True,
            scale_axes=True
        )

        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Failed to generate RRG: {e}")
# === GOOGLE TRENDS ===
st.markdown("---")
st.subheader("📈 Google Trends Sentiment Tracker")
trends = TrendReq(hl='en-US', tz=360)
keywords = ["GEV", "stocks", "TSLA", "NVDA", "AI stocks", "bitcoin"]
trends.build_payload(kw_list=keywords, timeframe='now 7-d')
trend_data = trends.interest_over_time().infer_objects(copy=False)
if not trend_data.empty:
    st.line_chart(trend_data[keywords])
else:
    st.warning("Google Trends data could not be loaded.")

st.caption("Dashboard v2.0 — Optimized RRG and Chart Layout")
