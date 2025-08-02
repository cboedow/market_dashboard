import asyncio
import importlib.util
from datetime import datetime
import pandas as pd
import streamlit as st
from openbb_core.app.utils import basemodel_to_df

st.set_page_config(layout="wide", page_title="Relative Rotation", initial_sidebar_state="expanded")

# === CONFIG ===
DEFAULT_ETFS = [
    "SPY", "QQQ", "DIA", "IWM",
    "XLF", "XLK", "XLE", "XLY", "XLI", "XLP", "XLV", "XLU", "XLB", "XLRE", "XLC"
]
TODAY = datetime.today()

# === SYMBOLS ===
symbols = DEFAULT_ETFS
benchmark = "SPY"

# === LOAD MODULE ===
def import_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

module = import_from_file("relative_rotation", "relative_rotation.py")

async def load_rrg():
    return await module.create(
        symbols=symbols,
        benchmark=benchmark,
        study="price",
        date=pd.to_datetime(TODAY),
        long_period=252,
        short_period=21,
        window=21,
        trading_periods=252,
        tail_periods=30,
        tail_interval="week",
        provider="yfinance",
    )

st.markdown("---")
st.subheader("📊 Relative Rotation Graph (OpenBB RRG)")

try:
    rrg_data = asyncio.run(load_rrg())

    fig = rrg_data.show(
        date=TODAY,
        show_tails=True,
        tail_periods=30,
        tail_interval="week",
        external=True,
    )
    fig.update_layout(height=600, margin=dict(l=0, r=20, b=0, t=50, pad=0))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Study Data Table", expanded=False):
        symbols_data = (
            basemodel_to_df(rrg_data.symbols_data).join(
                basemodel_to_df(rrg_data.benchmark_data)[benchmark]
            )
        ).set_index("date")
        symbols_data.index = pd.to_datetime(symbols_data.index).strftime("%Y-%m-%d")
        st.dataframe(symbols_data)

    with st.expander("Relative Strength Ratio Table", expanded=False):
        ratios_data = basemodel_to_df(rrg_data.rs_ratios).set_index("date")
        ratios_data.index = pd.to_datetime(ratios_data.index).strftime("%Y-%m-%d")
        st.dataframe(ratios_data)

    with st.expander("Relative Strength Momentum Table", expanded=False):
        momentum_data = basemodel_to_df(rrg_data.rs_momentum).set_index("date")
        momentum_data.index = pd.to_datetime(momentum_data.index).strftime("%Y-%m-%d")
        st.dataframe(momentum_data)

except Exception as e:
    st.error(f"OpenBB RRG Error: {e}")
