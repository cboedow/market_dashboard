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

# === RRG Section ===
st.markdown("---")
st.subheader("🔄 RRG — Sector Rotation (JDK RS + Momentum)")

try:
    view_mode = st.radio("View", ["Daily", "Weekly"], horizontal=True)
    with st.spinner(f"Loading {view_mode} RRG..."):
        days_back = 60 if view_mode == "Daily" else 280
        interval = '1d' if view_mode == "Daily" else '1wk'

        @st.cache_data(ttl=3600)
        def fetch_sector_data(interval, days_back):
            return yf.download(DEFAULT_ETFS,
                               start=(datetime.date.today() - datetime.timedelta(days=days_back)).isoformat(),
                               end=TODAY,
                               interval=interval,
                               auto_adjust=True)['Close']

        price_df = fetch_sector_data(interval, days_back)
        log_returns = np.log(price_df / price_df.shift(1)).dropna()
        rel_strength = log_returns.subtract(log_returns['SPY'], axis=0)

        jdk_rs = rel_strength.rolling(window=10).mean()
        jdk_mom = jdk_rs.diff().rolling(window=5).mean()

        jdk_rs_scaled = jdk_rs / jdk_rs.std()
        jdk_mom_scaled = jdk_mom / jdk_mom.std()

        fig = go.Figure()

        # === Background Quadrants ===
        quads = {
            "Leading":    {"x0": 0, "x1": 1.5, "y0": 0, "y1": 1.5, "color": "rgba(173,216,230,0.15)", "label": "Leading", "label_x": 0.75, "label_y": 1.3, "color_txt": "blue"},
            "Weakening":  {"x0": 0, "x1": 1.5, "y0": -1.5, "y1": 0, "color": "rgba(255,255,153,0.15)", "label": "Weakening", "label_x": 0.75, "label_y": -1.3, "color_txt": "orange"},
            "Lagging":    {"x0": -1.5, "x1": 0, "y0": -1.5, "y1": 0, "color": "rgba(255,160,122,0.15)", "label": "Lagging", "label_x": -0.75, "label_y": -1.3, "color_txt": "red"},
            "Improving":  {"x0": -1.5, "x1": 0, "y0": 0, "y1": 1.5, "color": "rgba(144,238,144,0.15)", "label": "Improving", "label_x": -0.75, "label_y": 1.3, "color_txt": "green"},
        }

        for q in quads.values():
            fig.add_shape(type="rect", xref="x", yref="y",
                          x0=q["x0"], x1=q["x1"], y0=q["y0"], y1=q["y1"],
                          fillcolor=q["color"], line_width=0)
            fig.add_annotation(
                x=q["label_x"], y=q["label_y"],
                text=q["label"],
                showarrow=False,
                font=dict(size=13, color=q["color_txt"])
            )

        # Axes
        fig.add_shape(type="line", x0=-1.5, x1=1.5, y0=0, y1=0,
                      line=dict(color="gray", dash="dash"))
        fig.add_shape(type="line", x0=0, x1=0, y0=-1.5, y1=1.5,
                      line=dict(color="gray", dash="dash"))

        trail_length = 3

        for symbol in DEFAULT_ETFS:
            rs_series = jdk_rs_scaled[symbol].dropna().iloc[-trail_length:]
            mom_series = jdk_mom_scaled[symbol].dropna().iloc[-trail_length:]
            if len(rs_series) < trail_length:
                continue

            # Interpolate smoother path (cubic)
            x = rs_series.values
            y = mom_series.values
            t = np.arange(len(x))
            try:
                t_smooth = np.linspace(t.min(), t.max(), 100)
                x_smooth = si.interp1d(t, x, kind='cubic')(t_smooth)
                y_smooth = si.interp1d(t, y, kind='cubic')(t_smooth)
            except Exception:
                x_smooth, y_smooth = x, y

            color = f"rgba({np.random.randint(0,255)}, {np.random.randint(0,255)}, {np.random.randint(0,255)}, 0.8)"

            # Path
            fig.add_trace(go.Scatter(
                x=x_smooth,
                y=y_smooth,
                mode="lines",
                line=dict(width=2, color=color),
                name=symbol,
                hoverinfo='skip',
                showlegend=True
            ))

            # Final arrowhead
            fig.add_trace(go.Scatter(
                x=[x[-1]],
                y=[y[-1]],
                mode="markers+text",
                marker=dict(size=11, color=color, line=dict(color="black", width=1)),
                text=[symbol],
                textposition="top center",
                hoverinfo='skip',
                showlegend=False
            ))

        # Final Layout
        fig.update_layout(
            title="RRG: Sector Rotation via JDK RS & Momentum",
            xaxis_title="JDK RS (Relative Strength vs SPY) - Scaled",
            yaxis_title="JDK Momentum - Scaled",
            xaxis=dict(range=[-1.5, 1.5], zeroline=False),
            yaxis=dict(range=[-1.5, 1.5], zeroline=False),
            height=750,
            legend_title="ETF",
            legend=dict(orientation="v", yanchor="top", y=0.98, xanchor="left", x=1.02)
        )
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"RRG chart error: {e}")

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
