"""Streamlit dashboard for India CPI inflation: history, backtest, forecast.

Run from the repo root:  streamlit run app/streamlit_app.py
Requires the pipeline outputs (data_ingestion -> backtest -> forecast).
"""
from __future__ import annotations

import pathlib
import sys

import pandas as pd
import streamlit as st

sys.path.append("src")

st.set_page_config(page_title="India Inflation Forecasting", layout="wide")
st.title("India CPI Inflation: Forecasting Dashboard")

PROCESSED = pathlib.Path("data/processed/cpi_india.csv")
BACKTEST = pathlib.Path("reports/backtest_predictions.csv")
SUMMARY = pathlib.Path("reports/backtest_summary.csv")
FORECAST = pathlib.Path("reports/forecast.csv")

if not PROCESSED.exists():
    st.error("Run `python src/data_ingestion.py` first to fetch the data.")
    st.stop()

df = pd.read_csv(PROCESSED, parse_dates=["date"]).set_index("date")

col1, col2, col3 = st.columns(3)
col1.metric("Latest YoY inflation", f"{df['inflation_yoy'].iloc[-1]:.2f}%")
col2.metric("12m average", f"{df['inflation_yoy'].tail(12).mean():.2f}%")
in_band = 2 <= df["inflation_yoy"].iloc[-1] <= 6
col3.metric("RBI band (2-6%)", "inside" if in_band else "outside")

tab_hist, tab_bt, tab_fc = st.tabs(["History", "Backtest", "Forecast"])

with tab_hist:
    years = st.slider("Years of history", 5, 35, 15)
    chunk = df.tail(years * 12)
    st.line_chart(chunk[["inflation_yoy"]], height=350)
    st.caption("YoY CPI inflation (%). Source: OECD MEI via FRED / IMF via DBnomics.")

with tab_bt:
    if BACKTEST.exists():
        bt = pd.read_csv(BACKTEST, parse_dates=["date"]).set_index("date")
        models = [c for c in bt.columns if c != "actual"]
        chosen = st.multiselect("Models", models, default=["naive", "sarima", "lgbm"])
        st.line_chart(bt[chosen + ["actual"]], height=350)
        if SUMMARY.exists():
            st.dataframe(pd.read_csv(SUMMARY).round(3), use_container_width=True)
    else:
        st.info("Run `python src/backtest.py` to generate backtest results.")

with tab_fc:
    if FORECAST.exists():
        fc = pd.read_csv(FORECAST, index_col=0, parse_dates=True)
        hist = df["inflation_yoy"].tail(36).rename("actual")
        st.line_chart(pd.concat([hist, fc], axis=1), height=350)
        st.dataframe(fc.round(2), use_container_width=True)
    else:
        st.info("Run `python src/forecast.py` to generate the forecast.")
