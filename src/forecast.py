"""Produce the final 12-month-ahead inflation forecast with intervals.

Uses the UCM (structural) model for the multi-step path with prediction
intervals, and LightGBM quantile models for a cross-check at h=1.
Outputs reports/forecast.csv and a chart in reports/figures/.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_ingestion import load_config
from features import build_features
from models import fit_lgbm_set, fit_ucm, forecast_ucm


def main() -> None:
    cfg = load_config()
    horizon = cfg["forecast"]["horizon"]
    quantiles = cfg["forecast"]["quantiles"]

    df = pd.read_csv(cfg["data"]["processed_path"], parse_dates=["date"]).set_index("date")
    y = df["inflation_yoy"].dropna()

    ucm = fit_ucm(y)
    mean, ci = forecast_ucm(ucm, steps=horizon, alpha=0.2)
    future_idx = pd.date_range(y.index[-1] + pd.DateOffset(months=1),
                               periods=horizon, freq="MS")
    fc = pd.DataFrame({"forecast": mean, "p10": ci[:, 0], "p90": ci[:, 1]},
                      index=future_idx)

    # LightGBM quantile cross-check at h=1
    feat = build_features(df, cfg)
    models = fit_lgbm_set(feat.drop(columns="target"), feat["target"],
                          cfg["lightgbm"], quantiles=quantiles)
    x_last = feat.iloc[[-1]].drop(columns="target")
    lgbm_h1 = {f"q{q}": float(models[f"q{q}"].predict(x_last)[0])
               for q in quantiles}
    lgbm_h1["point"] = float(models["point"].predict(x_last)[0])

    fc.to_csv("reports/forecast.csv")
    print("12-month UCM forecast (YoY inflation, %):")
    print(fc.round(2).to_string())
    print("\nLightGBM h=1 cross-check:", {k: round(v, 2) for k, v in lgbm_h1.items()})

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(y.tail(60).index, y.tail(60), label="actual", color="tab:blue")
    ax.plot(fc.index, fc["forecast"], label="UCM forecast", color="tab:red")
    ax.fill_between(fc.index, fc["p10"], fc["p90"], color="tab:red", alpha=0.2,
                    label="80% interval")
    ax.axhspan(2, 6, color="green", alpha=0.1, label="RBI band")
    ax.set_title("India YoY CPI inflation: 12-month forecast")
    ax.legend()
    fig.tight_layout()
    fig.savefig("reports/figures/forecast.png", dpi=150)
    print("chart saved to reports/figures/forecast.png")


if __name__ == "__main__":
    main()
