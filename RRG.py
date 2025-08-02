import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

class RRG:
    def __init__(self, tickers, benchmark='SPY', period='1y', interval='1wk', window=14):
        self.tickers = tickers
        self.benchmark = benchmark
        self.period = period
        self.interval = interval
        self.window = window

        self.tickers_data = None
        self.benchmark_data = None
        self.rs = []
        self.rsr = []
        self.rsm = []

        self.dataframe = None

    def fetch_data(self):
        raw = yf.download(self.tickers, period=self.period, interval=self.interval, auto_adjust=True, group_by='ticker')
        if isinstance(raw.columns, pd.MultiIndex):
            # For multiple tickers: extract each 'Close' series into a unified DataFrame
            self.tickers_data = pd.concat({ticker: raw[ticker]['Close'] for ticker in self.tickers}, axis=1)
        else:
            # For single ticker: just get the Close column directly
            self.tickers_data = raw[['Close']].rename(columns={'Close': self.tickers[0]})
        # Fetch benchmark
        benchmark_raw = yf.download(self.benchmark, period=self.period, interval=self.interval, auto_adjust=True)
        if 'Close' in benchmark_raw.columns:
            self.benchmark_data = benchmark_raw['Close']
        else:
            raise ValueError("Benchmark data has no 'Close' column.")

     def calculate_indicators(self):
        for ticker in self.tickers:
            rs = 100 * (self.tickers_data[ticker] / self.benchmark_data)
            rsr = (100 + (rs - rs.rolling(window=self.window).mean()) / rs.rolling(window=self.window).std(ddof=0)).dropna()
            rsm = (101 + ((rsr - rsr.rolling(window=self.window).mean()) / rsr.rolling(window=self.window).std(ddof=0))).dropna()
            # Align lengths
            rsr = rsr[rsr.index.isin(rsm.index)]
            rsm = rsm[rsm.index.isin(rsr.index)]
            self.rs.append(rs)
            self.rsr.append(rsr)
            self.rsm.append(rsm)

    def get_status_color(self, x, y):
        if x < 100 and y < 100:
            return 'lagging', 'red'
        elif x > 100 and y > 100:
            return 'leading', 'green'
        elif x < 100 and y > 100:
            return 'improving', 'blue'
        elif x > 100 and y < 100:
            return 'weakening', 'orange'
        return 'unknown', 'gray'

    def prepare_dataframe(self):
        records = []
        for i, ticker in enumerate(self.tickers):
            rs_last = self.rsr[i].values[-1]
            rsm_last = self.rsm[i].values[-1]
            status, color = self.get_status_color(rs_last, rsm_last)
            records.append({
                'Ticker': ticker,
                'RS': rs_last,
                'RSM': rsm_last,
                'Status': status,
                'Color': color
            })
        self.dataframe = pd.DataFrame(records)
        return self.dataframe

    def plot_plotly(self):
        fig = go.Figure()

        # Add quadrants
        fig.add_shape(type="rect", x0=100, x1=200, y0=100, y1=200, fillcolor="green", opacity=0.1, line_width=0)
        fig.add_shape(type="rect", x0=0, x1=100, y0=100, y1=200, fillcolor="blue", opacity=0.1, line_width=0)
        fig.add_shape(type="rect", x0=0, x1=100, y0=0, y1=100, fillcolor="red", opacity=0.1, line_width=0)
        fig.add_shape(type="rect", x0=100, x1=200, y0=0, y1=100, fillcolor="orange", opacity=0.1, line_width=0)

        for _, row in self.dataframe.iterrows():
            fig.add_trace(go.Scatter(
                x=[row['RS']],
                y=[row['RSM']],
                mode='markers+text',
                marker=dict(size=12, color=row['Color']),
                text=row['Ticker'],
                textposition='top center',
                name=row['Ticker']
            ))

        fig.update_layout(
            title='Relative Rotation Graph (RRG)',
            xaxis_title='JdK RS Ratio',
            yaxis_title='JdK RS Momentum',
            xaxis=dict(range=[90, 110]),
            yaxis=dict(range=[90, 110]),
            width=800,
            height=600,
            showlegend=False
        )

        return fig
