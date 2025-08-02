# market_momentum_dashboard.py — Streamlit Prototype

import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import streamlit as st
from pytrends.request import TrendReq
import plotly.express as px
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup

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

# === CBOE PUT/CALL RATIO (SCRAPED) ===
def get_put_call_ratios():
    url = "https://www.cboe.com/us/options/market_statistics/daily/"
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find("table")
        df = pd.read_html(str(table))[0]
        df.columns = df.columns.droplevel(0) if isinstance(df.columns, pd.MultiIndex) else df.columns
        latest = df.iloc[0]
        return {
            "Equity P/C": float(latest.get("Equity P/C Ratio", np.nan)),
            "Total P/C": float(latest.get("Total P/C Ratio", np.nan))
        }
    except Exception as e:
        return {"Equity P/C": np.nan, "Total P/C": np.nan}

put_call_data = get_put_call_ratios()

# === GAMMA EXPOSURE (CSV SOURCE) ===
def load_gex_csv():
    try:
        df = pd.read_csv("https://raw.githubusercontent.com/gexmetrics/gexdata/main/gex_daily.csv")
        df['date'] = pd.to_datetime(df['date'])
        latest_gex = df.sort_values('date').iloc[-1]['GEX_SPX']  # Adjust for SPX or SPY column
        return latest_gex
    except:
        return np.nan

GEX_level = load_gex_csv()

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

# === MCCLELLAN OSCILLATOR ===
def compute_mcclellan(symbol):
    df = data[symbol].copy()
    df['Advance'] = df['Close'].pct_change() > 0
    df['Decline'] = df['Close'].pct_change() < 0
    df['NetAdv'] = df['Advance'].astype(int) - df['Decline'].astype(int)
    df['EMA19'] = df['NetAdv'].ewm(span=19).mean()
    df['EMA39'] = df['NetAdv'].ewm(span=39).mean()
    df['McClellan'] = df['EMA19'] - df['EMA39']
    return df[['McClellan']]

mcclellan_charts = []
for symbol in ETFS:
    osc = compute_mcclellan(symbol)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=osc.index, y=osc['McClellan'], mode='lines', name=symbol))
    fig.update_layout(title=f"McClellan Oscillator - {symbol}", height=300)
    mcclellan_charts.append(fig)

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
    st.metric("Equity Put/Call Ratio", value=put_call_data['Equity P/C'])
    st.metric("Total Put/Call Ratio", value=put_call_data['Total P/C'])
    st.metric("Gamma Exposure (GEX)", value=f"{GEX_level/1e6:.1f}M", delta="⚠️ Risky" if GEX_level < 0 else "✅ Positive")
    st.metric("Zweig Breadth Thrust", value=round(latest_zweig, 3), delta=zweig_signal)
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

st.markdown("---")
st.subheader("🧭 McClellan Oscillator for Each ETF")
for chart in mcclellan_charts:
    st.plotly_chart(chart, use_container_width=True)

st.caption("Dashboard prototype v1.3 — Now with real GEX, McClellan Oscillator, and Zweig Breadth Thrust")
