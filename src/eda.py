"""Exploratory analysis of Indian CPI inflation.

Produces figures in reports/figures/: level and inflation trends, seasonal
profile of month-on-month changes, STL decomposition, ACF/PACF, and a view
of inflation regimes (pre-reform, high-inflation 90s, inflation targeting).
"""
from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL

from data_ingestion import load_config

FIGDIR = pathlib.Path("reports/figures")

# Monetary-policy regimes for India
REGIMES = [
    ("1990-01-01", "1997-12-31", "Post-liberalisation high inflation"),
    ("1998-01-01", "2013-12-31", "Multiple-indicator approach"),
    ("2014-01-01", None, "Flexible inflation targeting (4% +/- 2%)"),
]


def load_processed() -> pd.DataFrame:
    cfg = load_config()
    df = pd.read_csv(cfg["data"]["processed_path"], parse_dates=["date"])
    return df.set_index("date")


def plot_trends(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(df.index, df["cpi_index"], color="tab:blue")
    axes[0].set_title("India CPI index (all items)")
    axes[1].plot(df.index, df["inflation_yoy"], color="tab:red")
    axes[1].axhspan(2, 6, color="green", alpha=0.12, label="RBI target band (2-6%)")
    axes[1].axhline(4, color="green", ls="--", lw=1)
    axes[1].set_title("YoY inflation (%)")
    axes[1].legend()
    for start, end, label in REGIMES:
        axes[1].axvline(pd.Timestamp(start), color="grey", ls=":", lw=1)
    fig.tight_layout()
    fig.savefig(FIGDIR / "trends.png", dpi=150)
    plt.close(fig)


def plot_seasonality(df: pd.DataFrame) -> None:
    tmp = df.copy()
    tmp["month"] = tmp.index.month
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=tmp.reset_index(), x="month", y="inflation_mom", ax=ax)
    ax.set_title("MoM inflation by calendar month (seasonal pattern)")
    ax.axhline(0, color="grey", lw=1)
    fig.tight_layout()
    fig.savefig(FIGDIR / "seasonality.png", dpi=150)
    plt.close(fig)


def plot_stl(df: pd.DataFrame) -> None:
    series = df["inflation_yoy"].dropna()
    res = STL(series, period=12, robust=True).fit()
    fig = res.plot()
    fig.set_size_inches(11, 8)
    fig.suptitle("STL decomposition of YoY inflation", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGDIR / "stl_decomposition.png", dpi=150)
    plt.close(fig)


def plot_acf_pacf(df: pd.DataFrame) -> None:
    series = df["inflation_yoy"].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(series, lags=36, ax=axes[0])
    plot_pacf(series, lags=36, ax=axes[1], method="ywm")
    fig.tight_layout()
    fig.savefig(FIGDIR / "acf_pacf.png", dpi=150)
    plt.close(fig)


def regime_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for start, end, label in REGIMES:
        chunk = df.loc[start:end, "inflation_yoy"].dropna()
        rows.append({
            "regime": label,
            "mean": chunk.mean(),
            "std": chunk.std(),
            "months_above_6pct": int((chunk > 6).sum()),
            "share_in_band_2_6": float(((chunk >= 2) & (chunk <= 6)).mean()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    df = load_processed()
    plot_trends(df)
    plot_seasonality(df)
    plot_stl(df)
    plot_acf_pacf(df)
    summary = regime_summary(df)
    summary.to_csv("reports/regime_summary.csv", index=False)
    print(summary.round(2).to_string(index=False))
    print(f"figures written to {FIGDIR}/")


if __name__ == "__main__":
    main()
