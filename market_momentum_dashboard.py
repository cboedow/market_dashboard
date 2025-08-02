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
user_input = st.sidebar.text_area("📥 Add Custom Tickers (comma-separated)", value="TSLA, AAPL")
custom_tickers = [s.strip().upper() for s in user_input.split(",") if s.strip()]
ALL_SYMBOLS = list(set(DEFAULT_ETFS + custom_tickers))

START = (datetime.date.today() - datetime.timedelta(days=180)).isoformat()
TODAY = datetime.date.today().isoformat()

# === DOWNLOAD DATA ===
data = yf.download(ALL_SYMBOLS, start=START, end=TODAY, group_by='ticker', auto_adjust=False)

# === VIX / VIX3M ===
vix_data = yf.download(['^VIX', '^VIX3M'], start=START, end=TODAY, auto_adjust=True)
vix_ratio = vix_data['Close']['^VIX'].iloc[-1] / vix_data['Close']['^VIX3M'].iloc[-1]

# === CBOE PUT/CALL RATIO ===
def get_put_call_ratios():
    url = "https://www.cboe.com/us/options/market_statistics/daily/"
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find("table", class_="table")
        df = pd.read_html(str(table))[0]
        df.columns = df.columns.droplevel(0) if isinstance(df.columns, pd.MultiIndex) else df.columns
        df.columns = [col.strip().replace("\n", " ") for col in df.columns]
        latest = df.iloc[0]
        equity_pc = pd.to_numeric(latest.get("Equity Put/Call Ratio"), errors='coerce')
        total_pc = pd.to_numeric(latest.get("Total Put/Call Ratio"), errors='coerce')
        return {
            "Equity P/C": equity_pc,
            "Total P/C": total_pc
        }
    except:
        return {"Equity P/C": np.nan, "Total P/C": np.nan}

put_call_data = get_put_call_ratios()

# === GAMMA EXPOSURE ===
def load_gex_csv():
    try:
        df = pd.read_csv("https://raw.githubusercontent.com/gexmetrics/gexdata/main/gex_daily.csv")
        df['date'] = pd.to_datetime(df['date'])
        latest_gex = df.sort_values('date').iloc[-1]['GEX_SPX']
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

# === INDICATORS ===
def compute_indicators(symbol):
    df = data[symbol].copy()
    df['Advance'] = df['Close'].pct_change() > 0
    df['Decline'] = df['Close'].pct_change() < 0
    df['NetAdv'] = df['Advance'].astype(int) - df['Decline'].astype(int)
    df['EMA19'] = df['NetAdv'].ewm(span=19).mean()
    df['EMA39'] = df['NetAdv'].ewm(span=39).mean()
    df['McClellan'] = df['EMA19'] - df['EMA39']
    df['RSI'] = compute_rsi(df['Close'])
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# === LAYOUT ===
st.title("📊 Modern Market Momentum Dashboard")

col1, col2 = st.columns(2)

with col1:
    st.metric("VIX/VIX3M Ratio", value=round(vix_ratio, 2), delta="⚠️ High" if vix_ratio > 1.2 else "✅ Stable")
    st.metric("Equity Put/Call Ratio", value=put_call_data['Equity P/C'])
    st.metric("Total Put/Call Ratio", value=put_call_data['Total P/C'])
    st.metric("Gamma Exposure (GEX)", value=f"{GEX_level/1e6:.1f}M", delta="⚠️ Risky" if GEX_level < 0 else "✅ Positive")
    st.metric("Zweig Breadth Thrust", value=round(latest_zweig, 3), delta=zweig_signal)

# === CHARTS ===
st.subheader("📊 McClellan Oscillator + Price + RSI")

row = st.columns(3)
for i, symbol in enumerate(ALL_SYMBOLS):
    df = compute_indicators(symbol)
    df.dropna(inplace=True)

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.45, 0.30, 0.25],
                        vertical_spacing=0.03,
                        subplot_titles=(f"{symbol} Price", "McClellan Oscillator", "RSI"))

    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'],
                                 name=f"{symbol} Price"), row=1, col=1)

    colors = ['green' if v > 0 else 'red' for v in df['McClellan']]
    fig.add_trace(go.Bar(x=df.index, y=df['McClellan'], marker_color=colors, name="McClellan"), row=2, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI", mode="lines"), row=3, col=1)
    fig.update_yaxes(range=[0, 100], row=3, col=1)
    fig.update_layout(height=800, width=500, showlegend=False, title=f"{symbol} Technical Stack")

    row[i % 3].plotly_chart(fig, use_container_width=True)

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

st.caption("Dashboard prototype v1.7 — 6mo charts stacked vertically and better sized")
