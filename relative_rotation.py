import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta


class RRGData:
    def __init__(self, symbols, benchmark):
        self.symbols = symbols
        self.benchmark = benchmark
        self.today = datetime.today()
        self.lookback = 252
        self.tail = 30
        self.data = None
        self.rs_ratios = {}
        self.rs_momentum = {}

    def fetch_data(self):
        all_tickers = self.symbols + [self.benchmark]
        df = yf.download(all_tickers, period="1y", interval="1d", group_by="ticker", auto_adjust=True)
        prices = {}
        for ticker in all_tickers:
            if (ticker,) in df.columns:
                prices[ticker] = df[(ticker,)]['Close']
            else:
                prices[ticker] = df[ticker]['Close']
        self.data = pd.DataFrame(prices).dropna()

    def calculate_indicators(self):
        rel_strength = self.data[self.symbols].div(self.data[self.benchmark], axis=0)
        self.rs_ratios = rel_strength
        self.rs_momentum = rel_strength.pct_change(periods=5)

    def show(self):
        import plotly.express as px

        last = self.rs_ratios.index[-1]
        df = pd.DataFrame({
            "symbol": self.symbols,
            "RS": self.rs_ratios.loc[last].values,
            "Momentum": self.rs_momentum.loc[last].values,
        })

        fig = px.scatter(df, x="RS", y="Momentum", text="symbol",
                         title="Relative Rotation Graph", width=800, height=600)
        fig.update_traces(textposition="top center")
        fig.update_layout(xaxis_title="Relative Strength", yaxis_title="Momentum")
        return fig
