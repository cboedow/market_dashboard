from datetime import datetime
from typing import List, Optional

import pandas as pd
import plotly.graph_objects as go
from openbb_core.app.model.obbject import OBBject
from openbb_core.app.model.obbject_utils import basemodel_to_df
from openbb_core.app.provider_interface import ProviderChoices
from openbb_core.app.utils import get_data

class RelativeRotationGraph:
    def __init__(
        self,
        symbols: List[str],
        benchmark: str,
        study: str = "price",
        date: Optional[datetime] = None,
        long_period: int = 252,
        short_period: int = 21,
        window: int = 21,
        trading_periods: int = 252,
        tail_periods: int = 30,
        tail_interval: str = "week",
        provider: str = "yfinance",
    ):
        self.symbols = symbols
        self.benchmark = benchmark
        self.study = study
        self.date = date or datetime.today()
        self.long_period = long_period
        self.short_period = short_period
        self.window = window
        self.trading_periods = trading_periods
        self.tail_periods = tail_periods
        self.tail_interval = tail_interval
        self.provider = provider

        self.rrg: Optional[OBBject] = None

    async def fetch(self):
        self.rrg = await get_data(
            "economy/rrg",
            provider=ProviderChoices(self.provider),
            symbols=self.symbols,
            benchmark=self.benchmark,
            study=self.study,
            date=self.date,
            long_period=self.long_period,
            short_period=self.short_period,
            window=self.window,
            trading_periods=self.trading_periods,
            tail_periods=self.tail_periods,
            tail_interval=self.tail_interval,
        )

    def show(
        self,
        date: Optional[datetime] = None,
        show_tails: bool = False,
        tail_periods: int = 30,
        tail_interval: str = "week",
        external: bool = True,
    ):
        if self.rrg is None:
            raise ValueError("No RRG data. Call `fetch()` first.")

        return self.rrg.chart(
            date=date,
            show_tails=show_tails,
            tail_periods=tail_periods,
            tail_interval=tail_interval,
            external=external,
        )

    @property
    def symbols_data(self):
        return self.rrg.symbols_data if self.rrg else None

    @property
    def benchmark_data(self):
        return self.rrg.benchmark_data if self.rrg else None

    @property
    def rs_ratios(self):
        return self.rrg.rs_ratios if self.rrg else None

    @property
    def rs_momentum(self):
        return self.rrg.rs_momentum if self.rrg else None

def SPDRS():
    return [
        "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLB",
        "XLRE", "XLK", "XLU", "XLC"
    ]

def create(**kwargs):
    graph = RelativeRotationGraph(**kwargs)
    await graph.fetch()
    return graph
