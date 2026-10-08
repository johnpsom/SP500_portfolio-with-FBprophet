#!/usr/bin/env python
"""Download ~3 years of adjusted daily closes for the S&P 500 and save prices.csv.

    pip install yfinance pandas lxml
    python scripts/fetch_prices.py            # writes prices.csv in the current folder
    python scripts/fetch_prices.py --years 4 --out my_prices.csv

Upload the resulting CSV (dates as rows, tickers as columns) to the Claude session.
"""
import argparse
import io
import sys
import time
import urllib.request
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

WIKI = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
FALLBACK = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=30).read().decode()


def tickers():
    try:
        syms = pd.read_html(io.StringIO(fetch(WIKI)))[0]["Symbol"]
    except Exception as e:
        print(f"Wikipedia failed ({e}); using GitHub list", file=sys.stderr)
        syms = pd.read_csv(io.StringIO(fetch(FALLBACK)))["Symbol"]
    return sorted({s.strip().replace(".", "-") for s in syms})  # BRK.B -> BRK-B for Yahoo


def download(syms, start, batch=100, retries=3):
    frames = []
    for i in range(0, len(syms), batch):
        chunk = syms[i:i + batch]
        for attempt in range(retries):
            try:
                raw = yf.download(chunk, start=start, auto_adjust=True, progress=False, threads=True)
                close = raw["Close"]
                if isinstance(close, pd.Series):
                    close = close.to_frame(chunk[0])
                frames.append(close.dropna(axis=1, how="all"))
                print(f"batch {i // batch + 1}: {close.shape[1]} tickers", file=sys.stderr)
                break
            except Exception as e:
                print(f"batch {i // batch + 1} attempt {attempt + 1} failed: {e}", file=sys.stderr)
                time.sleep(5 * (attempt + 1))
    return pd.concat(frames, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=3.0)
    ap.add_argument("--out", default="prices.csv")
    a = ap.parse_args()
    start = (date.today() - timedelta(days=int(a.years * 365.25) + 10)).isoformat()
    syms = tickers()
    print(f"{len(syms)} tickers, from {start}", file=sys.stderr)
    prices = download(syms, start)
    prices = prices.loc[:, ~prices.columns.duplicated()].sort_index()
    prices.index.name = "Date"
    prices.round(4).to_csv(a.out)
    missing = sorted(set(syms) - set(prices.columns))
    print(f"saved {a.out}: {prices.shape[0]} days x {prices.shape[1]} tickers")
    if missing:
        print(f"no data for {len(missing)}: {', '.join(missing)}")


if __name__ == "__main__":
    main()
