"""Loaders for the two Wikipedia-derived S&P 500 membership sources."""

from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "cache" / "raw"
HANSHOF_CSV = RAW / "sp_500_historical_components.csv"
HANSHOF_URL = "https://raw.githubusercontent.com/hanshof/sp500_constituents/main/sp_500_historical_components.csv"


def load_hanshof() -> pd.Series:
    """hanshof/sp500_constituents: Series indexed by date, values = frozenset of tickers."""
    if not HANSHOF_CSV.exists():
        import requests
        HANSHOF_CSV.parent.mkdir(parents=True, exist_ok=True)
        HANSHOF_CSV.write_bytes(requests.get(HANSHOF_URL, timeout=120).content)
    df = pd.read_csv(HANSHOF_CSV, parse_dates=["date"]).sort_values("date")
    return df.set_index("date")["tickers"].map(lambda s: frozenset(s.split(",")))


def hanshof_ticker_periods(hans: pd.Series) -> pd.DataFrame:
    """First and last date each ticker appears in hanshof."""
    long = hans.explode().rename("ticker").reset_index()
    return long.groupby("ticker")["date"].agg(first="min", last="max").reset_index()


def ic_members_at(hist: pd.DataFrame, day: pd.Timestamp) -> frozenset:
    """index-constitution symbols that were members on `day`."""
    active = (hist["opt-in"] <= day) & (hist["opt-out"].isna() | (hist["opt-out"] > day))
    return frozenset(hist.loc[active, "symbol"])
