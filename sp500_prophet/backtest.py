"""Walk-forward backtest: forecast -> pick -> optimise -> hold -> repeat.

Only data up to each rebalance date is used (no look-ahead).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .forecast import forecast_returns, select_universe
from .portfolio import optimise_weights

Scorer = Callable[[pd.DataFrame], pd.Series]


def prophet_scorer(horizon: int = 20, workers: int = 1) -> Scorer:
    return lambda hist: forecast_returns(hist, horizon, workers)


def momentum_scorer(window: int = 120) -> Scorer:
    """Cheap baseline: trailing return over ``window`` days."""
    return lambda hist: hist.iloc[-1] / hist.iloc[-window] - 1


def backtest(prices: pd.DataFrame, scorer: Scorer, lookback: int = 500, hold: int = 20,
             universe_size: int = 20, objective: str = "min_volatility",
             cutoff: float = 0.05, max_weight: float = 0.3, cost_bps: float = 5.0) -> pd.DataFrame:
    """Daily equity curve (start = 1.0) with per-rebalance turnover costs.

    Returns a frame with columns ``strategy`` and ``benchmark`` (equal-weight of all stocks,
    rebalanced at the same dates).
    """
    rets = prices.pct_change().fillna(0.0)
    prev_w = pd.Series(dtype=float)
    strat, bench = [], []
    for t in range(lookback, len(prices) - 1, hold):
        hist = prices.iloc[t - lookback:t + 1]
        scores = scorer(hist)
        picks = select_universe(scores, universe_size, min_return=-np.inf)
        w = pd.Series(optimise_weights(hist[picks], objective, cutoff, max_weight))
        w = w[w > 0]
        w /= w.sum()
        turnover = w.sub(prev_w, fill_value=0).abs().sum()
        window = rets.iloc[t + 1:t + 1 + hold]
        daily = (window[w.index] * w).sum(axis=1)
        daily.iloc[0] -= turnover * cost_bps / 1e4
        strat.append(daily)
        bench.append(window.mean(axis=1))
        prev_w = w
    out = pd.DataFrame({"strategy": pd.concat(strat), "benchmark": pd.concat(bench)})
    return (1 + out).cumprod()


def summarise(curve: pd.DataFrame, periods: int = 252) -> pd.DataFrame:
    """CAGR, volatility, Sharpe (rf=0) and max drawdown per column of an equity curve."""
    r = curve.pct_change().dropna()
    years = len(r) / periods
    return pd.DataFrame({
        "CAGR": curve.iloc[-1] ** (1 / years) - 1,
        "Volatility": r.std() * np.sqrt(periods),
        "Sharpe": r.mean() / r.std() * np.sqrt(periods),
        "MaxDrawdown": (curve / curve.cummax() - 1).min(),
    }).round(4)
