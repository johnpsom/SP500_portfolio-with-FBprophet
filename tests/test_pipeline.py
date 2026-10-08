import numpy as np
import pandas as pd

from sp500_prophet.backtest import backtest, momentum_scorer, summarise
from sp500_prophet.forecast import forecast_returns, select_universe
from sp500_prophet.portfolio import allocate, optimise_weights


def synth(n=900, k=30, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    drift = rng.normal(0.0003, 0.0004, k)
    r = rng.normal(drift, 0.015, (n, k))
    return pd.DataFrame(100 * np.exp(r.cumsum(0)), index=idx, columns=[f"S{i}" for i in range(k)])


def test_select_universe():
    s = pd.Series({"A": 0.1, "B": -0.2, "C": 0.05})
    assert select_universe(s, 5) == ["A", "C"]


def test_weights_and_allocation():
    p = synth()
    w = optimise_weights(p.iloc[:, :10])
    assert abs(sum(w.values()) - 1) < 1e-6
    table, cash = allocate(w, p.iloc[-1], 50000)
    assert table["value"].sum() + cash <= 50000 + 1e-6


def test_backtest_runs_and_has_no_lookahead_nan():
    p = synth()
    curve = backtest(p, momentum_scorer(60), lookback=300, hold=20, universe_size=8)
    assert curve.notna().all().all()
    assert {"CAGR", "Sharpe"} <= set(summarise(curve).columns)


def test_prophet_forecast_smoke():
    p = synth(n=300, k=2)
    out = forecast_returns(p, horizon=10)
    assert len(out) == 2 and np.isfinite(out).all()
