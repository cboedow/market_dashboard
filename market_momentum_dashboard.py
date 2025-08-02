# market_momentum_dashboard.py — Streamlit Prototype

import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import streamlit as st
from pytrends.request import TrendReq
import plotly.express as px

st.set_page_config(layout="wide")

# === CONFIG ===
ETFS = [
    "SPY", "QQQ", "DIA", "IWM",  # Major indices
    "XLF", "XLK", "XLE", "XLY", "XLI", "XLP", "XLV", "XLU", "XLB", "XLRE", "XLC"  # Sector SPDRs
]
START = (datetime.date.today() - datetime.timedelta(days=60)).isoformat()
TODAY = datetime.date.today().isoformat()

# === DOWNLOAD ETF DATA ===
data = yf.download(ETFS, start=START, end=TODAY, group_by='ticker', auto_adjust=True)

# === VIX / VIX3M ===
vix_data = yf.download(['^VIX', '^VIX3M'], start=START, end=TODAY, auto_adjust=True)
vix_ratio = vix_data['Close']['^VIX'].iloc[-1] / vix_data['Close']['^VIX3M'].iloc[-1]

# === BUILD ETF SUMMARY ===
summary = []
for symbol in ETFS:
    df = data[symbol].copy()
    df['Return'] = df['Close'].pct_change()
    df['RSI2'] = df['Return'].rolling(2).mean() / df['Return'].rolling(2).std()
    df['GapUp'] = (df['Open'] > df['Close'].shift(1)) & ((df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1) > 0.02)

    move_2pct = (df['Return'].abs() > 0.02).sum() / len(df)
    latest_rsi2 = df['RSI2'].iloc[-1] if not df['RSI2'].isnull().all() else np.nan
    latest_return = df['Return'].iloc[-1]
    recent_gapups = df['GapUp'].tail(10).sum()

    summary.append({
        'Symbol': symbol,
        '% ±2% Days': round(move_2pct * 100, 2),
        'Latest Return': round(latest_return * 100, 2),
        'RSI(2)': round(latest_rsi2, 2),
        'Recent GapUps (10d)': int(recent_gapups)
    })

summary_df = pd.DataFrame(summary)

# === SENTIMENT FROM GOOGLE TRENDS ===
trends = TrendReq(hl='en-US', tz=360)
keywords = ["how to buy stocks", "stock market crash", "AI stocks", "bitcoin"]
trends.build_payload(kw_list=keywords, timeframe='now 7-d')
trend_data = trends.interest_over_time()

# === STREAMLIT LAYOUT ===
st.title("📊 Modern Market Momentum Dashboard")

col1, col2 = st.columns(2)

with col1:
    st.metric("VIX/VIX3M Ratio", value=round(vix_ratio, 2), delta="⚠️ High" if vix_ratio > 1.2 else "✅ Stable")
    st.dataframe(summary_df.set_index("Symbol"))

with col2:
    fig = px.bar(summary_df.sort_values(by='% ±2% Days', ascending=False), 
                 x='Symbol', y='% ±2% Days', color='RSI(2)', 
                 hover_data=['Latest Return', 'Recent GapUps (10d)'],
                 title="Momentum by ETF: % ±2% Days + RSI")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("📈 Google Trends Sentiment Tracker")
if not trend_data.empty:
    st.line_chart(trend_data[keywords])
else:
    st.warning("Google Trends data could not be loaded. Try again later.")

# === TODO ===
# - Add Put/Call Ratio via external API (e.g. CBOE scrape)
# - Add GEX signals from SqueezeMetrics or SpotGamma API
# - Set up daily auto-run via scheduler
# - Deploy to Streamlit Cloud or alternative

st.caption("Dashboard prototype v1.0 — Inspired by Michael Marcus, built with modern tools")
