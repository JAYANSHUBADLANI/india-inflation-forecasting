"""Rolling-origin backtest for one-month-ahead YoY inflation forecasts.

Design:
  * Expanding window: at each origin t, models see data up to t and predict t+1.
  * SARIMA order is re-selected, and SARIMA/UCM are refit, every `refit_every`
    origins (between refits, cheaper parameter updates via apply/extend).
  * LightGBM is refit at every refit point on the feature matrix.
  * Metrics: MAE, RMSE, bias; Diebold-Mariano test compares each model
    against the random-walk baseline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from data_ingestion import load_config
from features import build_features
from models import (fit_lgbm_set, fit_sarima, fit_ucm, forecast_naive,
                    forecast_sarima, forecast_seasonal_naive, forecast_ucm,
                    select_sarima_order)


def diebold_mariano(e1: np.ndarray, e2: np.ndarray) -> tuple[float, float]:
    """DM test with squared-error loss, small-sample t approximation."""
    d = e1 ** 2 - e2 ** 2
    n = len(d)
    dbar = d.mean()
    var = d.var(ddof=1) / n
    if var == 0:
        return 0.0, 1.0
    dm = dbar / np.sqrt(var)
    p = 2 * (1 - stats.t.cdf(abs(dm), df=n - 1))
    return float(dm), float(p)


def run_backtest() -> pd.DataFrame:
    cfg = load_config()
    df = pd.read_csv(cfg["data"]["processed_path"], parse_dates=["date"]).set_index("date")
    y = df["inflation_yoy"].dropna()
    feat = build_features(df, cfg)

    test_start = pd.Timestamp(cfg["backtest"]["test_start"])
    refit_every = cfg["backtest"]["refit_every"]
    origins = y.index[(y.index >= test_start) & (y.index < y.index[-1])]

    preds: dict[str, list] = {m: [] for m in
                              ["naive", "seasonal_naive", "sarima", "ucm", "lgbm"]}
    actuals, dates = [], []
    sarima_fit = ucm_fit = lgbm_models = None
    order = seasonal_order = None

    for i, origin in enumerate(origins):
        train = y.loc[:origin]
        target_date = origin + pd.DateOffset(months=1)
        if target_date not in y.index:
            continue

        if i % refit_every == 0:
            order, seasonal_order = select_sarima_order(
                train, cfg["sarima"]["max_p"], cfg["sarima"]["max_q"])
            sarima_fit = fit_sarima(train, order, seasonal_order)
            ucm_fit = fit_ucm(train)
            ftrain = feat.loc[:origin]
            lgbm_models = fit_lgbm_set(ftrain.drop(columns="target"),
                                       ftrain["target"], cfg["lightgbm"],
                                       quantiles=[])
        else:
            sarima_fit = sarima_fit.apply(train)
            ucm_fit = ucm_fit.apply(train)

        preds["naive"].append(forecast_naive(train))
        preds["seasonal_naive"].append(forecast_seasonal_naive(train))
        preds["sarima"].append(forecast_sarima(sarima_fit, 1)[0])
        preds["ucm"].append(forecast_ucm(ucm_fit, 1)[0][0])
        x_row = feat.loc[[origin]].drop(columns="target")
        preds["lgbm"].append(float(lgbm_models["point"].predict(x_row)[0]))

        actuals.append(y.loc[target_date])
        dates.append(target_date)

    res = pd.DataFrame(preds, index=pd.DatetimeIndex(dates, name="date"))
    res["actual"] = actuals
    return res


def summarise(res: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base_err = (res["naive"] - res["actual"]).values
    for model in [c for c in res.columns if c != "actual"]:
        err = (res[model] - res["actual"]).values
        dm, p = (np.nan, np.nan) if model == "naive" else diebold_mariano(err, base_err)
        rows.append({"model": model,
                     "MAE": np.abs(err).mean(),
                     "RMSE": np.sqrt((err ** 2).mean()),
                     "bias": err.mean(),
                     "DM_vs_naive": dm,
                     "p_value": p})
    return pd.DataFrame(rows).sort_values("RMSE")


def main() -> None:
    res = run_backtest()
    res.to_csv("reports/backtest_predictions.csv")
    summary = summarise(res)
    summary.to_csv("reports/backtest_summary.csv", index=False)
    print(f"backtest: {len(res)} one-step forecasts, "
          f"{res.index.min():%Y-%m} to {res.index.max():%Y-%m}")
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
