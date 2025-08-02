import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

class RRG:
    def __init__(self, tickers=None, benchmark='SPY', period='6mo', interval='1wk', price_df=None, window=10):
        self.tickers = tickers
        self.benchmark = benchmark
        self.period = period
        self.interval = interval
        self.window = window
        self.price_df = price_df
        self.rsr = {}
        self.rsm = {}

    def fetch_data(self):
        raw = yf.download(self.tickers, period=self.period, interval=self.interval, group_by='ticker', auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex):
            self.tickers_data = raw.loc[:, (slice(None), 'Close')].droplevel(1, axis=1)
        else:
            self.tickers_data = raw[['Close']]

        benchmark_raw = yf.download(self.benchmark, period=self.period, interval=self.interval, auto_adjust=True)
        if 'Close' not in benchmark_raw.columns:
            raise ValueError("Benchmark data missing 'Close'")
        self.benchmark_data = benchmark_raw['Close']

    def calculate_indicators(self):
        for ticker in self.tickers:
            if ticker not in self.tickers_data.columns:
                continue
            price = self.tickers_data[ticker]
            if price.isnull().sum() > 5:
                continue
            rs = price / self.benchmark_data
            rsm = rs.pct_change().rolling(self.window).mean()
            rsr = (rs - rs.rolling(self.window).mean()) / rs.rolling(self.window).std()
            self.rsr[ticker] = rsr.dropna()
            self.rsm[ticker] = rsm.dropna()

    def get_status_color(self, rs, rsm):
        if rs >= 0 and rsm >= 0:
            return "Leading", "green"
        elif rs < 0 and rsm >= 0:
            return "Improving", "blue"
        elif rs < 0 and rsm < 0:
            return "Lagging", "red"
        else:
            return "Weakening", "orange"

    def prepare_dataframe(self):
        records = []
        for ticker in self.tickers:
            if ticker not in self.rsr or self.rsr[ticker].empty:
                continue
            rs_last = self.rsr[ticker].values[-1]
            rsm_last = self.rsm[ticker].values[-1]
            status, color = self.get_status_color(rs_last, rsm_last)
            records.append({
                'Ticker': ticker,
                'RS-Ratio': rs_last,
                'RS-Momentum': rsm_last,
                'Status': status,
                'Color': color
            })
        return pd.DataFrame(records)

    def plot_plotly(self, tickers=None, trail_length=12, scaled=True, title="RRG", show_legend=True):
        df = self.prepare_dataframe()
        fig = go.Figure()
        for _, row in df.iterrows():
            fig.add_trace(go.Scatter(
                x=[row['RS-Ratio']],
                y=[row['RS-Momentum']],
                mode='markers+text',
                marker=dict(size=14, color=row['Color']),
                name=row['Ticker'],
                text=row['Ticker'],
                textposition="top center"
            ))

        fig.update_layout(
            title=title,
            xaxis=dict(title="RS-Ratio", zeroline=True),
            yaxis=dict(title="RS-Momentum", zeroline=True),
            height=600
        )
        return fig
