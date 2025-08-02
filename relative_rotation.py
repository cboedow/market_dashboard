# relative_rotation.py

import pandas as pd
import yfinance as yf
from datetime import datetime


class RRGData:
    def __init__(self, symbols, benchmark):
        self.symbols = symbols
        self.benchmark = benchmark
        self.today = datetime.today()
        self.rs_ratios = pd.DataFrame()
        self.rs_momentum = pd.DataFrame()
        self.data = pd.DataFrame()

    def fetch_data(self):
        all_tickers = list(set(self.symbols + [self.benchmark]))
        df = yf.download(all_tickers, period="1y", interval="1d", group_by="ticker", auto_adjust=True)

        prices = {}
        for ticker in all_tickers:
            try:
                # Handle both multi-index and flat DataFrame cases
                if isinstance(df.columns, pd.MultiIndex):
                    prices[ticker] = df[(ticker,)]['Close']
                else:
                    prices[ticker] = df[ticker]['Close']
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
                continue

        self.data = pd.DataFrame(prices).dropna()

    def calculate_indicators(self):
        if self.data.empty:
            raise ValueError("No price data loaded. Run fetch_data() first.")

        rel_strength = self.data[self.symbols].div(self.data[self.benchmark], axis=0)
        self.rs_ratios = rel_strength
        self.rs_momentum = rel_strength.pct_change(periods=5)

    def show(self):
        import plotly.express as px

        if self.rs_ratios.empty or self.rs_momentum.empty:
            raise ValueError("Indicators not calculated. Run calculate_indicators() first.")

        last = self.rs_ratios.index[-1]
        df = pd.DataFrame({
            "Symbol": self.symbols,
            "RS": self.rs_ratios.loc[last].values,
            "Momentum": self.rs_momentum.loc[last].values,
        })

        fig = px.scatter(df, x="RS", y="Momentum", text="Symbol",
                         title="Relative Rotation Graph",
                         width=800, height=600)
        fig.update_traces(textposition="top center")
        fig.update_layout(xaxis_title="Relative Strength",
                          yaxis_title="Momentum",
                          template="plotly_white")
        return fig
