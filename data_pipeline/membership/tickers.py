"""Ticker -> CIK mapping for the Wikipedia-derived membership sources.

Layers, most reliable first:
  1. SEC company_tickers.json: official mapping for currently listed tickers.
  2. index-constitution former members: name -> CIK within their membership period.
  3. Remaining historical tickers (hanshof): company name from the Wikipedia changes
     table, else from Tiingo metadata, then name -> CIK within the period the ticker
     was in the index. Tiingo names the *current* holder of a reused ticker; the
     period constraint rejects those (the CIK must file annual reports at the time).
"""

import json
from io import StringIO
from pathlib import Path

import index_constitution as ic
import pandas as pd

from membership.entity_match import match_names
from sources import sec, tiingo

CACHE = Path(__file__).resolve().parents[1] / "cache"
WIKI_CHANGES_HTML = CACHE / "raw" / "wikipedia_sp500_historical.html"


def key(ticker: str) -> str:
    return ticker.upper().replace("-", ".")


def sec_current_tickers() -> dict[str, int]:
    current = json.loads(sec.get("https://www.sec.gov/files/company_tickers.json"))
    return {key(r["ticker"]): int(r["cik_str"]) for r in current.values()}


def ic_with_cik() -> pd.DataFrame:
    """index-constitution history with a CIK per row (layers 1 and 2)."""
    hist = ic.history("sp500").copy()
    current = sec_current_tickers()
    active = hist["opt-out"].isna()
    hist["cik"] = hist["symbol"].map(lambda s: current.get(key(s))).where(active)
    todo = hist[hist["cik"].isna()]
    names = pd.DataFrame({
        "name": todo["name"],
        "first_seen": todo["opt-in"].clip(lower=pd.Timestamp("1994-01-01")),
        "last_seen": todo["opt-out"].fillna(pd.Timestamp.today().normalize()),
    }).drop_duplicates("name")
    by_name = match_names(names).set_index("name")["cik"]
    hist.loc[hist["cik"].isna(), "cik"] = todo["name"].map(by_name)
    return hist


def wikipedia_change_names() -> dict[str, str]:
    """Ticker -> security name from Wikipedia's 'Historical components' changes table."""
    if not WIKI_CHANGES_HTML.exists():
        return {}
    ch = pd.read_html(StringIO(WIKI_CHANGES_HTML.read_text()))[0]
    ch.columns = ["date", "add_t", "add_n", "rem_t", "rem_n", "reason", "refs"]
    pairs = pd.concat([ch[["add_t", "add_n"]].set_axis(["t", "n"], axis=1),
                       ch[["rem_t", "rem_n"]].set_axis(["t", "n"], axis=1)]).dropna()
    return dict(zip(pairs["t"].map(key), pairs["n"]))


def ticker_to_cik(ic_hist: pd.DataFrame, ticker_periods: pd.DataFrame | None = None) -> dict[str, int]:
    """Ticker -> CIK.

    `ticker_periods` (columns ticker, first, last) lists extra historical tickers to
    resolve with layer 3, e.g. every ticker that appears in hanshof.
    """
    mapping = sec_current_tickers()
    former = ic_hist[ic_hist["opt-out"].notna() & ic_hist["cik"].notna()]
    for sym, cik in zip(former["symbol"], former["cik"]):
        mapping.setdefault(key(sym), int(cik))  # don't override a current listing
    if ticker_periods is None:
        return mapping

    todo = ticker_periods[~ticker_periods["ticker"].map(key).isin(mapping)]
    wiki = wikipedia_change_names()
    rows = []
    tiingo_available = True
    for t, first, last in zip(todo["ticker"], todo["first"], todo["last"]):
        name = wiki.get(key(t))
        if name is None and tiingo_available:
            try:
                m = tiingo.meta(t.replace(".", "-"))
                name = (m or {}).get("name")
            except tiingo.RateLimited:
                tiingo_available = False  # keep going with cached / Wikipedia names only
                print("[tickers] Tiingo quota reached: remaining tickers resolved from cache only")
        elif name is None:
            cached = tiingo.CACHE / "meta" / f"{t.replace('.', '-').upper()}.json"
            if cached.exists():
                name = (json.loads(cached.read_text()) or {}).get("name")
        if name:
            rows.append({"ticker": t, "name": name.strip(), "first_seen": first, "last_seen": last})
    if rows:
        resolved = match_names(pd.DataFrame(rows))
        for t, cik in zip(resolved["ticker"], resolved["cik"]):
            if pd.notna(cik):
                mapping[key(t)] = int(cik)
    return mapping
