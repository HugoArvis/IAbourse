"""Download daily prices for all candidate tickers, cached per ticker and source.

Sources: yfinance (no strict quota, few delisted tickers) and Tiingo (more delisted
tickers, free quota ~50 requests/hour). Both are stored with the same schema:
  open, high, low, close   split-adjusted, NOT dividend-adjusted
  adj_close                split- and dividend-adjusted (total return series)
  volume, dividends, splits

Usage: python data_pipeline/prices/download.py
"""

import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sources import tiingo  # noqa: E402

CACHE = ROOT / "cache"
YF_DIR = CACHE / "prices" / "yfinance"
TIINGO_DIR = CACHE / "prices" / "tiingo"
SOURCES = ("yfinance", "tiingo")
START = "1995-01-01"
BATCH = 40
PAUSE = 3.0  # seconds between batches


def yf_symbol(ticker: str) -> str:
    return ticker.replace(".", "-")


def _path(ticker: str, source: str = "yfinance") -> Path:
    return (YF_DIR if source == "yfinance" else TIINGO_DIR) / f"{ticker}.parquet"


def download(tickers: list[str]) -> None:
    YF_DIR.mkdir(parents=True, exist_ok=True)
    todo = [t for t in tickers if not _path(t).exists()]
    print(f"{len(tickers) - len(todo)} cached, {len(todo)} to download", flush=True)
    for i in range(0, len(todo), BATCH):
        batch = todo[i:i + BATCH]
        data = yf.download([yf_symbol(t) for t in batch], start=START, auto_adjust=False,
                           actions=True, group_by="ticker", progress=False, threads=True)
        for t in batch:
            sym = yf_symbol(t)
            if data.empty or sym not in data.columns.get_level_values(0):
                df = pd.DataFrame()
            else:
                df = data[sym].dropna(how="all", subset=["Close"])
                df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close",
                                        "Adj Close": "adj_close", "Volume": "volume",
                                        "Dividends": "dividends", "Stock Splits": "splits"})
                df.index.name = "date"
            df.to_parquet(_path(t))  # empty frame = "no data" marker, avoids re-asking
        done = min(i + BATCH, len(todo))
        print(f"  {done}/{len(todo)}", flush=True)
        time.sleep(PAUSE)


def download_tiingo(ticker: str) -> bool:
    """Fetch one ticker from Tiingo into the common schema. Returns False if no data.

    Tiingo gives raw prices plus splitFactor; convert to split-adjusted OHLC like yfinance.
    Raises tiingo.RateLimited when the free quota is exhausted (nothing is written then).
    """
    path = _path(ticker, "tiingo")
    if path.exists():
        return not pd.read_parquet(path).empty
    rows = tiingo.prices(ticker.replace(".", "-"), start=START)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df = df.set_index("date").sort_index()
        split = df["splitFactor"].where(df["splitFactor"] > 0, 1.0)
        after = split[::-1].cumprod()[::-1].shift(-1, fill_value=1.0)  # splits strictly after t
        df = pd.DataFrame({
            "open": df["open"] / after, "high": df["high"] / after, "low": df["low"] / after,
            "close": df["close"] / after, "adj_close": df["adjClose"], "volume": df["volume"] * after,
            "dividends": df["divCash"] / after, "splits": df["splitFactor"].where(df["splitFactor"] != 1.0, 0.0),
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return not df.empty


def load(ticker: str, source: str = "yfinance") -> pd.DataFrame:
    """Daily prices; drops today's row (session possibly not closed yet)."""
    path = _path(ticker, source)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    return df[df.index < pd.Timestamp.today().normalize()] if not df.empty else df


if __name__ == "__main__":
    cand = pd.read_parquet(CACHE / "price_ticker_candidates.parquet")
    download(sorted(cand["ticker"].unique()))
    have = [t for t in cand["ticker"].unique() if not load(t).empty]
    print(f"tickers with data: {len(have)}/{cand['ticker'].nunique()}")
    sys.exit(0)
