"""Portfolio construction with PyPortfolioOpt."""
from __future__ import annotations

import pandas as pd
from pypfopt import EfficientFrontier, risk_models
from pypfopt.discrete_allocation import DiscreteAllocation


def optimise_weights(prices: pd.DataFrame, objective: str = "min_volatility",
                     cutoff: float = 0.05, max_weight: float = 0.3) -> dict[str, float]:
    """Optimal weights for the columns of ``prices``.

    Risk comes from a Ledoit-Wolf shrunk covariance. ``max_sharpe`` uses the
    historical mean return; ``min_volatility`` needs no return estimate.
    """
    cov = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
    if objective == "max_sharpe":
        from pypfopt import expected_returns

        mu = expected_returns.capm_return(prices)
        ef = EfficientFrontier(mu, cov, weight_bounds=(0, max_weight))
        ef.max_sharpe()
    else:
        ef = EfficientFrontier(None, cov, weight_bounds=(0, max_weight))
        ef.min_volatility()
    return dict(ef.clean_weights(cutoff=cutoff))


def allocate(weights: dict[str, float], latest_prices: pd.Series, value: float) -> tuple[pd.DataFrame, float]:
    """Whole-share allocation. Returns (table with shares/price/value, leftover cash)."""
    w = {k: v for k, v in weights.items() if v > 0}
    shares, cash = DiscreteAllocation(w, latest_prices[list(w)], total_portfolio_value=value).greedy_portfolio()
    table = pd.DataFrame({
        "weight": pd.Series(w),
        "shares": pd.Series(shares),
    }).dropna()
    table["price"] = latest_prices[table.index].round(2)
    table["value"] = (table["shares"] * table["price"]).round(2)
    return table, float(cash)
