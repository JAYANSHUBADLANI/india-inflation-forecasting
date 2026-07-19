# India Inflation Forecasting

One-month-ahead and 12-month-ahead forecasting of India's headline CPI (YoY)
inflation, comparing classical, Bayesian-structural, and machine-learning
approaches against naive benchmarks: evaluated honestly with a rolling-origin
backtest and Diebold-Mariano significance tests.

## Why this problem

Inflation is the single most-watched macro variable in India: the RBI's
Monetary Policy Committee targets 4% (+/- 2%) CPI inflation, and every basis
point matters for rate decisions. Forecasting it well is hard because the
series mixes strong seasonality (food prices), structural breaks (the 2016
shift to flexible inflation targeting), and base effects that mechanically
move YoY numbers.

## Data

| Source | Series | Notes |
|---|---|---|
| FRED (primary) | `INDCPIALLMINMEI` | India CPI all-items index, monthly, from OECD MEI |
| DBnomics (fallback) | `IMF/CPI/M.IN.PCPI_IX` | IMF CPI database mirror |

Both sources are free, keyless, and fetched programmatically: the entire
pipeline reproduces from a fresh clone with no manual downloads.

## Models

1. **Naive baselines**: random walk and 3-year seasonal mean. Any model that
   cannot beat these is not worth deploying.
2. **SARIMA**: AIC grid search over (p,1,q)(P,0,Q)12.
3. **Unobserved Components (structural / BSTS-style)**: local level +
   stochastic seasonal, state-space estimation; provides principled
   prediction intervals.
4. **LightGBM**: gradient boosting on leakage-safe features: AR lags,
   rolling statistics, calendar dummies, momentum gap, and an explicit
   **base-effect** term (the year-ago MoM print about to drop out of the
   YoY window). Quantile objectives give P10/P90 intervals.

## Evaluation

Rolling-origin (expanding window) backtest from 2015 onwards: at each monthly
origin, models see only past data and predict the next month. SARIMA order is
re-selected and all models refit every 12 origins. Reported: MAE, RMSE, bias,
and the Diebold-Mariano test versus the random-walk baseline.

## Reproduce

```bash
pip install -r requirements.txt
python src/data_ingestion.py   # fetch + clean CPI, derive YoY/MoM inflation
python src/eda.py              # trends, seasonality, STL, ACF/PACF, regimes
python src/features.py         # leakage-safe feature matrix
python src/backtest.py         # rolling-origin backtest + DM tests
python src/forecast.py         # 12-month forecast with 80% intervals
streamlit run app/streamlit_app.py
```

## Repository layout

```
src/            pipeline modules (ingestion -> eda -> features -> backtest -> forecast)
app/            Streamlit dashboard
reports/        backtest results, forecasts, figures (generated)
data/           raw and processed series (generated, gitignored)
config.yaml     every knob in one place
```

## Honest limitations

* Headline CPI only: a food/fuel/core decomposition would sharpen the story
  but MoSPI component series need manual assembly.
* Univariate + calendar features; no exchange rate, crude oil, or monsoon
  covariates yet (natural next step).
* YoY inflation is a smoothed target; MoM SAAR forecasting is harder and
  more operationally useful for policy desks.
