# SP500_portfolio-with-FBprophet

Limit an S&P 500 portfolio to stocks that Prophet expects to rise, then size them with
PyPortfolioOpt, and test the idea with a walk-forward backtest.

## Pipeline
1. `data.py` – S&P 500 tickers + adjusted prices (yfinance, CSV cache).
2. `forecast.py` – fit Prophet per stock, score = forecast % change over `--horizon` business days; keep the top `--size`.
3. `portfolio.py` – Ledoit-Wolf covariance, min-volatility (or CAPM max-Sharpe) weights, whole-share allocation.
4. `backtest.py` – rebalance every `hold` days using only past data, 5 bps cost per unit turnover, compared to an equal-weight S&P 500 benchmark. A cheap momentum scorer is included as a baseline the Prophet signal has to beat.

## Usage
```
pip install -r requirements.txt
python -m sp500_prophet portfolio --size 20 --value 50000      # today's picks and share counts
python -m sp500_prophet backtest --scorer momentum              # fast baseline
python -m sp500_prophet backtest --scorer prophet --max-tickers 100 --workers 8   # slow
pytest
```
Prophet backtests refit hundreds of models per rebalance, so limit tickers or use `--workers`.

## Changes from the original prototype (in `legacy/`)
The old scripts did not run: missing `momentum_score`/`download_button`, `fbprophet` and
`DataFrame.append` no longer exist, the loop forecast the `Date` column, and the backtest called
`get_portfolio` with the wrong arguments. Also, the old screen ranked on Prophet's *lower bound*
(`trend_lower`); this version uses the point forecast `yhat`.

Not investment advice. Results are untested on real data in this environment (Yahoo was unreachable); the tests use synthetic prices.
