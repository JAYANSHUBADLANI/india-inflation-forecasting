"""Leakage-safe feature engineering for one-month-ahead inflation forecasting.

All features at time t use information available up to t only; the target is
inflation_yoy at t+1. Includes autoregressive lags, rolling statistics,
calendar dummies, and a base-effect term (YoY inflation mechanically moves
when the year-ago month had an unusual print).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data_ingestion import load_config


def build_features(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Return a supervised frame with target = next month's YoY inflation."""
    cfg = cfg or load_config()
    lags = cfg["features"]["lags"]
    windows = cfg["features"]["rolling_windows"]

    out = df.copy().sort_index()
    y = out["inflation_yoy"]

    for lag in lags:
        out[f"yoy_lag{lag}"] = y.shift(lag - 1)  # lag1 = current month, known at t

    for w in windows:
        out[f"yoy_rollmean{w}"] = y.rolling(w).mean()
        out[f"yoy_rollstd{w}"] = y.rolling(w).std()

    out["mom_lag1"] = out["inflation_mom"]
    out["mom_rollsum3"] = out["inflation_mom"].rolling(3).sum()

    # Base effect: the year-ago MoM print that will drop out of the YoY window
    out["base_effect"] = out["inflation_mom"].shift(11)

    # Momentum: 3m annualised vs current YoY
    out["momentum_gap"] = out["mom_rollsum3"] * 4 - y

    out["month"] = out.index.month
    month_dummies = pd.get_dummies(out["month"], prefix="m", drop_first=True)
    out = pd.concat([out, month_dummies.astype(float)], axis=1)

    out["target"] = y.shift(-1)
    feature_cols = [c for c in out.columns
                    if c.startswith(("yoy_", "mom_", "base_", "momentum", "m_"))]
    return out[feature_cols + ["target"]].dropna()


def main() -> None:
    cfg = load_config()
    df = pd.read_csv(cfg["data"]["processed_path"], parse_dates=["date"]).set_index("date")
    feat = build_features(df, cfg)
    feat.to_csv("data/processed/features.csv")
    print(f"feature matrix: {feat.shape[0]} rows x {feat.shape[1] - 1} features")
    print("columns:", ", ".join(feat.columns[:8]), "...")


if __name__ == "__main__":
    main()
