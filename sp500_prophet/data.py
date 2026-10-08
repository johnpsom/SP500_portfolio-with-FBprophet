"""Price download helpers (yfinance) with a simple CSV cache."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def sp500_tickers() -> list[str]:
    """Current S&P 500 constituents from Wikipedia, in Yahoo ticker format."""
    import io
    import urllib.request

    req = urllib.request.Request(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        headers={"User-Agent": "Mozilla/5.0"},
    )
    html = urllib.request.urlopen(req, timeout=30).read().decode()
    table = pd.read_html(io.StringIO(html))[0]
    return [s.replace(".", "-") for s in table["Symbol"]]


def download_prices(tickers, start, end=None, cache: str | None = None) -> pd.DataFrame:
    """Adjusted close prices (rows = dates, columns = tickers).

    If ``cache`` is a path and exists it is read instead of downloading.
    """
    if cache and Path(cache).exists():
        return pd.read_csv(cache, index_col=0, parse_dates=True)
    import yfinance as yf

    raw = yf.download(list(tickers), start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"].dropna(axis=1, how="all")
    if cache:
        prices.to_csv(cache)
    return prices


def clean_prices(prices: pd.DataFrame, max_missing: float = 0.05) -> pd.DataFrame:
    """Drop tickers with too many gaps, forward-fill the rest."""
    keep = prices.columns[prices.isna().mean() <= max_missing]
    return prices[keep].ffill().dropna()
