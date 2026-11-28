"""Ticker -> CIK mapping for the Wikipedia-derived membership sources.

Layers, most reliable first:
  1. SEC company_tickers.json: official mapping for currently listed
     tickers.
  2. index-constitution former members: name -> CIK within their
     membership period.
  3. Remaining historical tickers (hanshof): company names from the
     Wikipedia changes table and from Tiingo metadata (both tried),
     then name -> CIK within the period the ticker was in the index.
     Tiingo names the *current* holder of a reused ticker; the period
     constraint rejects those (the CIK must file annual reports at the
     time).
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
    current = json.loads(
        sec.get("https://www.sec.gov/files/company_tickers.json")
    )
    return {key(r["ticker"]): int(r["cik_str"]) for r in current.values()}


def ic_with_cik() -> pd.DataFrame:
    """index-constitution history with a CIK per row (layers 1, 2)."""
    hist = ic.history("sp500").copy()
    current = sec_current_tickers()
    active = hist["opt-out"].isna()
    hist["cik"] = (
        hist["symbol"].map(lambda s: current.get(key(s))).where(active)
    )
    todo = hist[hist["cik"].isna()]
    names = pd.DataFrame(
        {
            "name": todo["name"],
            "first_seen": (
                todo["opt-in"].clip(lower=pd.Timestamp("1994-01-01"))
            ),
            "last_seen": (
                todo["opt-out"].fillna(pd.Timestamp.today().normalize())
            ),
        }
    ).drop_duplicates("name")
    by_name = match_names(names).set_index("name")["cik"]
    hist.loc[hist["cik"].isna(), "cik"] = todo["name"].map(by_name)
    return hist


def wikipedia_change_names() -> dict[str, str]:
    """Ticker -> security name from the Wikipedia changes table.

    It is the 'Historical components' table of the S&P 500 page.
    """
    if not WIKI_CHANGES_HTML.exists():
        return {}
    ch = pd.read_html(StringIO(WIKI_CHANGES_HTML.read_text()))[0]
    ch.columns = ["date", "add_t", "add_n", "rem_t", "rem_n", "reason", "refs"]
    pairs = pd.concat(
        [
            ch[["add_t", "add_n"]].set_axis(["t", "n"], axis=1),
            ch[["rem_t", "rem_n"]].set_axis(["t", "n"], axis=1),
        ]
    ).dropna()
    return dict(zip(pairs["t"].map(key), pairs["n"]))


def ticker_to_cik(
    ic_hist: pd.DataFrame, ticker_periods: pd.DataFrame | None = None
) -> dict[str, int]:
    """Ticker -> CIK.

    `ticker_periods` (columns ticker, first, last) lists extra
    historical tickers to resolve with layer 3, e.g. every ticker that
    appears in hanshof.
    """
    mapping = sec_current_tickers()
    former = ic_hist[ic_hist["opt-out"].notna() & ic_hist["cik"].notna()]
    for sym, cik in zip(former["symbol"], former["cik"]):
        # don't override a current listing
        mapping.setdefault(key(sym), int(cik))
    if ticker_periods is None:
        return mapping

    todo = ticker_periods[~ticker_periods["ticker"].map(key).isin(mapping)]
    names = historical_ticker_names(todo)
    if not names.empty:
        resolved = (
            match_names(names).dropna(subset=["cik"]).sort_values("pref")
        )
        for t, cik in resolved.drop_duplicates("ticker")[
            ["ticker", "cik"]
        ].itertuples(index=False):
            mapping[key(t)] = int(cik)
    return mapping


def historical_ticker_names(
    ticker_periods: pd.DataFrame, fetch: bool = True
) -> pd.DataFrame:
    """Company names of historical tickers.

    `ticker_periods` has columns ticker, first, last.

    Both sources are kept, Wikipedia first (pref 0): its wording
    sometimes fails to match EDGAR ("Interpublic Group of Companies")
    where Tiingo's (pref 1) matches. Tiingo names are fetched when
    missing from cache, unless fetch=False.
    """
    wiki = wikipedia_change_names()
    rows = []
    for t, first, last in zip(
        ticker_periods["ticker"],
        ticker_periods["first"],
        ticker_periods["last"],
    ):
        names = [wiki.get(key(t)), _tiingo_name(t, fetch=fetch)]
        if names[1] is False:
            fetch = False  # keep going with cached / Wikipedia names only
            print(
                "[tickers] Tiingo quota reached: remaining tickers resolved"
                " from cache only"
            )
            names[1] = None
        for pref, name in enumerate(names):
            if name:
                rows.append(
                    {
                        "ticker": t,
                        "name": name.strip(),
                        "first_seen": first,
                        "last_seen": last,
                        "pref": pref,
                    }
                )
    return pd.DataFrame(
        rows, columns=["ticker", "name", "first_seen", "last_seen", "pref"]
    )


def _tiingo_name(ticker: str, fetch: bool) -> str | None | bool:
    """Company name from Tiingo metadata.

    Cached; fetched if allowed. False = quota reached.
    """
    cached = tiingo.CACHE / "meta" / f"{ticker.replace('.', '-').upper()}.json"
    if cached.exists():
        return (json.loads(cached.read_text()) or {}).get("name")
    if not fetch:
        return None
    try:
        return (tiingo.meta(ticker.replace(".", "-")) or {}).get("name")
    except tiingo.RateLimited:
        return False
