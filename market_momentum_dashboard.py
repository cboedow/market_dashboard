# market_momentum_dashboard.py — Streamlit Prototype

import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import streamlit as st
from pytrends.request import TrendReq
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from bs4 import BeautifulSoup

st.set_page_config(layout="wide")

# === CONFIG ===
DEFAULT_ETFS = [
    "SPY", "QQQ", "DIA", "IWM",
    "XLF", "XLK", "XLE", "XLY", "XLI", "XLP", "XLV", "XLU", "XLB", "XLRE", "XLC"
]

# === USER STOCK INPUT ===
user_input = st.text_input("📥 Add Custom Tickers (comma-separated)", value="TSLA, AAPL", help="Type tickers separated by commas (e.g., TSLA, AAPL)")
custom_tickers = [s.strip().upper() for s in user_input.split(",") if s.strip()]
ALL_SYMBOLS = list(set(DEFAULT_ETFS + custom_tickers))

START = (datetime.date.today() - datetime.timedelta(days=180)).isoformat()
TODAY = datetime.date.today().isoformat()

# === DOWNLOAD DATA ===
data = yf.download(ALL_SYMBOLS, start=START, end=TODAY, group_by='ticker', auto_adjust=False)

# === VIX / VIX3M ===
vix_data = yf.download(['^VIX', '^VIX3M'], start=START, end=TODAY, auto_adjust=True)
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



# === LAYOUT ===
st.title("📊 Modern Market Momentum Dashboard")

col1, col2 = st.columns(2)

with col1:
    st.metric("VIX/VIX3M Ratio", value=round(vix_ratio, 2), delta="⚠️ High" if vix_ratio > 1.2 else "✅ Stable")
    st.metric("Zweig Breadth Thrust", value=f"{latest_zweig:.3f}" if pd.notna(latest_zweig) else "N/A", delta=zweig_signal)

# === CHARTS ===
st.subheader("📊 McClellan Oscillator + Price")

row = st.columns(3)
for i, symbol in enumerate(ALL_SYMBOLS):
    df = compute_indicators(symbol)
    df.dropna(inplace=True)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.4],
                        vertical_spacing=0.03,
                        subplot_titles=(f"{symbol} Price", "McClellan Oscillator")))

    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'],
                                 name=f"{symbol} Price"), row=1, col=1)

    colors = ['green' if v > 0 else 'red' for v in df['McClellan']]
    fig.add_trace(go.Bar(x=df.index, y=df['McClellan'], marker_color=colors, name="McClellan"), row=2, col=1)

    
    fig.update_layout(height=850, width=500, showlegend=False, title=f"{symbol} Technical Stack", xaxis_rangeslider_visible=False, xaxis=dict(tickformat='%b %Y'))

    row[i % 3].plotly_chart(fig, use_container_width=True)

# === RRG Chart (Sector Rotation View) with View Toggle ===

st.markdown("---")
st.subheader("🔄 Relative Rotation Graph (RRG) — Sector Momentum vs Strength")
try:
    view_mode = st.radio("Select RRG Timeframe", options=["Daily", "Weekly"], horizontal=True)
days_back = 60 if view_mode == "Daily" else 280
interval = '1d' if view_mode == "Daily" else '1wk'
sector_data = yf.download(DEFAULT_ETFS, start=(datetime.date.today() - datetime.timedelta(days=days_back)).isoformat(), end=TODAY, interval=interval)['Close']
    returns = sector_data.pct_change().dropna()
    benchmark = returns['SPY']
    rel_strength = returns.div(benchmark, axis=0)
    jdk_rs = rel_strength.rolling(window=10).mean().iloc[-1]
    jdk_momentum = rel_strength.rolling(window=10).mean().diff().rolling(5).mean().iloc[-1]

    rrg_df = pd.DataFrame({
        'Symbol': jdk_rs.index,
        'JDK RS': jdk_rs.values,
        'JDK Momentum': jdk_momentum.values
    })

    rrg_fig = px.scatter(rrg_df, x='JDK RS', y='JDK Momentum', text='Symbol', color='Symbol', size_max=60,
                         title="RRG: Sector Relative Strength vs Momentum")
    rrg_fig.update_traces(textposition='top center')
    st.plotly_chart(rrg_fig, use_container_width=True)

    # === RRG with quadrants and path lines ===
    st.subheader("🔄 RRG — Quadrants + Trajectories")
    rrg_traj_fig = go.Figure()
    for symbol in rrg_df['Symbol']:
        rs_series = rel_strength[symbol].rolling(window=10).mean().iloc[-10:]
        mom_series = rs_series.diff().rolling(5).mean().iloc[-10:]
        phase_color = 'blue' if rs_series.iloc[-1] > 0 and mom_series.iloc[-1] > 0 else \
                      'green' if rs_series.iloc[-1] < 0 and mom_series.iloc[-1] > 0 else \
                      'red' if rs_series.iloc[-1] < 0 and mom_series.iloc[-1] < 0 else 'orange'
        rrg_traj_fig.add_trace(go.Scatter(
            x=rs_series,
            y=mom_series,
            mode='lines+markers',
            name=symbol,
            text=[f"{symbol}<br>RS: {rs:.2f}<br>Mom: {mo:.2f}" for rs, mo in zip(rs_series, mom_series)],
            hoverinfo='text',
            line=dict(color=phase_color)
        ))

    # Add quadrant lines
    rrg_traj_fig.add_shape(type="line", x0=0, x1=0, y0=rrg_df['JDK Momentum'].min(), y1=rrg_df['JDK Momentum'].max(), line=dict(color="gray", dash="dash"))
    rrg_traj_fig.add_shape(type="line", y0=0, y1=0, x0=rrg_df['JDK RS'].min(), x1=rrg_df['JDK RS'].max(), line=dict(color="gray", dash="dash"))

    rrg_traj_fig.update_layout(title="RRG Flow — Momentum vs Relative Strength (with Quadrants)", xaxis_title="JDK RS", yaxis_title="JDK Momentum")
    st.plotly_chart(rrg_traj_fig, use_container_width=True)
except:
    st.warning("Unable to load RRG sector chart.")

# === GOOGLE TRENDS ===
st.markdown("---")
trends = TrendReq(hl='en-US', tz=360)
keywords = ["how to buy stocks", "stock market crash", "AI stocks", "bitcoin"]
trends.build_payload(kw_list=keywords, timeframe='now 7-d')
trend_data = trends.interest_over_time()

st.subheader("📈 Google Trends Sentiment Tracker")
if not trend_data.empty:
    st.line_chart(trend_data[keywords])
else:
    st.warning("Google Trends data could not be loaded. Try again later.")

st.caption("Dashboard prototype v1.8 — Cleaned metrics")
