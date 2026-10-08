"""Per-stock Prophet forecasts turned into expected-return scores."""
from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)


def forecast_return(series: pd.Series, horizon: int = 20) -> float:
    """Forecast % change of ``series`` ``horizon`` business days after its last date.

    Uses Prophet's point forecast (``yhat``) relative to the last observed price.
    """
    from prophet import Prophet

    s = series.dropna()
    df = pd.DataFrame({"ds": s.index.tz_localize(None), "y": s.values})
    model = Prophet(daily_seasonality=False)
    model.fit(df)
    future = model.make_future_dataframe(periods=horizon, freq="B")
    yhat = model.predict(future)["yhat"].iloc[-1]
    return float(yhat / s.iloc[-1] - 1)


def _safe(args):
    ticker, series, horizon = args
    try:
        return ticker, forecast_return(series, horizon)
    except Exception:  # a single bad series must not kill the whole run
        return ticker, float("nan")


def forecast_returns(prices: pd.DataFrame, horizon: int = 20, workers: int = 1) -> pd.Series:
    """Expected return over ``horizon`` days for every column of ``prices``."""
    jobs = [(c, prices[c], horizon) for c in prices.columns]
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            res = list(ex.map(_safe, jobs))
    else:
        res = [_safe(j) for j in jobs]
    return pd.Series(dict(res), name=f"fc_ret_{horizon}d").dropna()


def select_universe(expected: pd.Series, size: int = 20, min_return: float = 0.0) -> list[str]:
    """Top ``size`` stocks by forecast return, keeping only those above ``min_return``."""
    ranked = expected[expected > min_return].sort_values(ascending=False)
    return ranked.head(size).index.tolist()
