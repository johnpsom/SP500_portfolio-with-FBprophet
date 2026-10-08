"""Command line: ``python -m sp500_prophet {portfolio,backtest}``."""
from __future__ import annotations

import argparse
from datetime import date, timedelta

from .backtest import backtest, momentum_scorer, prophet_scorer, summarise
from .data import clean_prices, download_prices, sp500_tickers
from .forecast import forecast_returns, select_universe
from .portfolio import allocate, optimise_weights


def main(argv=None):
    p = argparse.ArgumentParser(prog="sp500_prophet", description=__doc__)
    p.add_argument("command", choices=["portfolio", "backtest"])
    p.add_argument("--days", type=int, default=1500, help="history length in calendar days")
    p.add_argument("--cache", default="prices.csv", help="CSV price cache")
    p.add_argument("--max-tickers", type=int, default=None, help="limit universe (faster Prophet runs)")
    p.add_argument("--size", type=int, default=20, help="stocks in portfolio")
    p.add_argument("--horizon", type=int, default=20, help="forecast horizon (business days)")
    p.add_argument("--value", type=float, default=50000, help="portfolio value for share allocation")
    p.add_argument("--objective", choices=["min_volatility", "max_sharpe"], default="min_volatility")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--scorer", choices=["prophet", "momentum"], default="prophet", help="backtest only")
    a = p.parse_args(argv)

    tickers = sp500_tickers()[: a.max_tickers]
    start = (date.today() - timedelta(days=a.days)).isoformat()
    prices = clean_prices(download_prices(tickers, start, cache=a.cache))

    if a.command == "portfolio":
        expected = forecast_returns(prices, a.horizon, a.workers)
        picks = select_universe(expected, a.size)
        weights = optimise_weights(prices[picks], a.objective)
        table, cash = allocate(weights, prices.iloc[-1], a.value)
        table["forecast_ret"] = expected[table.index].round(4)
        print(table.sort_values("weight", ascending=False).to_string())
        print(f"\ninvested {table['value'].sum():.2f}, leftover cash {cash:.2f}")
    else:
        scorer = prophet_scorer(a.horizon, a.workers) if a.scorer == "prophet" else momentum_scorer()
        curve = backtest(prices, scorer, lookback=min(500, len(prices) // 2), hold=a.horizon,
                         universe_size=a.size, objective=a.objective)
        print(summarise(curve).to_string())


if __name__ == "__main__":
    main()
