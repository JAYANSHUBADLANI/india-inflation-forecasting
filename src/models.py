"""Model zoo for one-month-ahead YoY inflation forecasting.

Four families, deliberately spanning classical -> Bayesian -> ML:
  1. Naive baselines: random walk (last value), seasonal mean
  2. SARIMA with AIC-based order selection on a grid
  3. Unobserved Components Model (Bayesian structural time series flavour):
     local level + stochastic seasonal, estimated by MLE with smoothed states
  4. LightGBM on engineered features, plus quantile models for intervals
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.structural import UnobservedComponents

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------- baselines
def forecast_naive(train: pd.Series) -> float:
    """Random-walk forecast: next month equals this month."""
    return float(train.iloc[-1])


def forecast_seasonal_naive(train: pd.Series) -> float:
    """Average of the same calendar month over the last three years."""
    target_month = (train.index[-1] + pd.DateOffset(months=1)).month
    same_month = train[train.index.month == target_month]
    return float(same_month.tail(3).mean())


# ------------------------------------------------------------------ SARIMA
def select_sarima_order(train: pd.Series, max_p: int = 3, max_q: int = 3,
                        m: int = 12) -> tuple:
    """Small AIC grid search over (p,d,q)(P,D,Q,m)."""
    best_aic, best = np.inf, ((1, 0, 1), (0, 0, 0, m))
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            for P in (0, 1):
                for Q in (0, 1):
                    if p == q == 0:
                        continue
                    try:
                        res = SARIMAX(train, order=(p, 1, q),
                                      seasonal_order=(P, 0, Q, m),
                                      enforce_stationarity=False,
                                      enforce_invertibility=False,
                                      ).fit(disp=False, maxiter=100)
                        if res.aic < best_aic:
                            best_aic, best = res.aic, ((p, 1, q), (P, 0, Q, m))
                    except Exception:
                        continue
    return best


def fit_sarima(train: pd.Series, order: tuple, seasonal_order: tuple):
    return SARIMAX(train, order=order, seasonal_order=seasonal_order,
                   enforce_stationarity=False, enforce_invertibility=False,
                   ).fit(disp=False, maxiter=200)


def forecast_sarima(fitted, steps: int = 1) -> np.ndarray:
    return np.asarray(fitted.forecast(steps=steps))


# --------------------------------------------------- structural time series
def fit_ucm(train: pd.Series):
    """Local level + stochastic seasonal: the workhorse BSTS specification."""
    model = UnobservedComponents(train, level="local level",
                                 seasonal=12, stochastic_seasonal=True)
    return model.fit(disp=False, maxiter=200)


def forecast_ucm(fitted, steps: int = 1, alpha: float = 0.2):
    """Point forecast plus a (1 - alpha) prediction interval."""
    fc = fitted.get_forecast(steps=steps)
    mean = np.asarray(fc.predicted_mean)
    ci = fc.conf_int(alpha=alpha)
    return mean, np.asarray(ci)


# ---------------------------------------------------------------- LightGBM
def make_lgbm(cfg: dict, quantile: float | None = None) -> LGBMRegressor:
    params = dict(num_leaves=cfg["num_leaves"],
                  learning_rate=cfg["learning_rate"],
                  n_estimators=cfg["n_estimators"],
                  min_child_samples=cfg["min_child_samples"],
                  verbose=-1)
    if quantile is not None:
        return LGBMRegressor(objective="quantile", alpha=quantile, **params)
    return LGBMRegressor(objective="regression", **params)


def fit_lgbm_set(X: pd.DataFrame, y: pd.Series, cfg: dict,
                 quantiles: list[float]) -> dict:
    """Fit a point model and one model per quantile."""
    models = {"point": make_lgbm(cfg).fit(X, y)}
    for q in quantiles:
        models[f"q{q}"] = make_lgbm(cfg, quantile=q).fit(X, y)
    return models
