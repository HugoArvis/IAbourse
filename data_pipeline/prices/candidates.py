"""Candidate tickers for every company of the membership table.

A company (company_id) can have several tickers over time (renames,
reused symbols). Candidates come from:
  - index-constitution: canonical (latest) ticker, mapped to CIK,
  - hanshof: ticker as listed at the time, mapped to CIK,
  - SEC company_tickers.json: current ticker(s) of the CIK,
  - for companies still without any: tickers declared in their XBRL
    reports, and hanshof tickers whose name matches while the company
    was held.
Which candidate is right for which period is decided later by
validate.py, using Vanguard's implied share prices.
"""

import json
import sys
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from membership.build_membership import (  # noqa: E402
    assign_company_ids,
    load_holdings,
)
from membership.entity_match import normalize  # noqa: E402
from membership.tickers import (  # noqa: E402
    historical_ticker_names,
    ic_with_cik,
    key,
    ticker_to_cik,
)
from membership.wiki_sources import (  # noqa: E402
    hanshof_ticker_periods,
    load_hanshof,
)
from sources import sec  # noqa: E402

CACHE = ROOT / "cache"
NAME_MATCH = 90  # rapidfuzz token_sort_ratio on normalized names


def sec_primary_tickers() -> pd.DataFrame:
    """One row per CIK: primary ticker and company title.

    The primary ticker is the first one listed by the SEC.

    The SEC file also lists preferreds / secondary lines under the same
    CIK (e.g. FMCC plus FMCCG, FMCCH...): only the first entry is kept.
    """
    data = json.loads(
        sec.get("https://www.sec.gov/files/company_tickers.json")
    )
    df = pd.DataFrame(data.values()).rename(columns={"cik_str": "cik"})
    df["ticker"] = df["ticker"].map(key)
    return df.drop_duplicates("cik")[["cik", "ticker", "title"]]


def _fallback_candidates(
    holdings: pd.DataFrame, have: set, periods: pd.DataFrame
) -> pd.DataFrame:
    """Candidates for companies with no ticker from the CIK mappings.

    A ticker maps to a single CIK above, so when two entities shared it
    (Allergan Inc. then Allergan plc on AGN, banks re-registered outside
    the SEC like Zions) the second one gets nothing. Two looser sources,
    both checked later by validate.py:
      sec_xbrl     tickers the company declared in its own XBRL
                   reports;
      name_period  hanshof tickers listed while the company was held,
                   whose Wikipedia / Tiingo name matches one of the
                   company's names.
    """
    missing = holdings[~holdings["company_id"].isin(have)]
    rows = []
    for comp, g in missing.groupby("company_id"):
        for cik in g["cik"].dropna().astype(int).unique():
            rows += [(comp, t, "sec_xbrl") for t in sec.trading_symbols(cik)]

    names = historical_ticker_names(periods, fetch=False)
    names["key"] = names["name"].map(normalize)
    span = (
        missing.groupby(["company_id", "key"])["date"]
        .agg(["min", "max"])
        .reset_index()
    )
    for s in span.itertuples():
        near = names[
            (names["first_seen"] <= s.max) & (names["last_seen"] >= s.min)
        ]
        for t, k in zip(near["ticker"], near["key"]):
            if fuzz.token_sort_ratio(k, s.key) >= NAME_MATCH:
                rows.append((s.company_id, key(t), "name_period"))
    return pd.DataFrame(
        rows, columns=["company_id", "ticker", "source"]
    ).assign(entity=None)


def build_candidates() -> pd.DataFrame:
    holdings = load_holdings()
    company = assign_company_ids(holdings)  # entity ("cik:123") -> company_id
    ic_hist = ic_with_cik()
    hans = load_hanshof()
    periods = hanshof_ticker_periods(hans)
    t2c = ticker_to_cik(ic_hist, periods)

    rows = []
    for sym, cik in zip(ic_hist["symbol"], ic_hist["cik"]):
        if pd.notna(cik):
            rows.append((f"cik:{int(cik)}", key(sym), "ic"))
    for t in periods["ticker"]:
        cik = t2c.get(key(t))
        if cik:
            rows.append((f"cik:{cik}", key(t), "hanshof"))
    sec_primary = sec_primary_tickers()
    for cik, t in zip(sec_primary["cik"], sec_primary["ticker"]):
        rows.append((f"cik:{cik}", t, "sec_current"))

    cand = pd.DataFrame(rows, columns=["entity", "ticker", "source"])
    cand["company_id"] = cand["entity"].map(company)
    cand = cand.dropna(subset=["company_id"])

    # Companies still without candidate: same normalized name as a
    # current SEC listing (e.g. Exxon Mobil whose ticker now points to a
    # new holding-company CIK).
    holdings["company_id"] = holdings["entity"].map(company)
    missing = holdings[~holdings["company_id"].isin(cand["company_id"])]

    # "EXXONMOBIL HOLDINGS" == "EXXON MOBIL"
    def compact(name_key: str) -> str:
        return (
            name_key.replace("HOLDINGS", "")
            .replace("GROUP", "")
            .replace(" ", "")
        )

    by_key = dict(
        zip(
            sec_primary["title"].map(lambda t: compact(normalize(t))),
            sec_primary["ticker"],
        )
    )
    extra = [
        (None, by_key[compact(k)], "sec_name", c)
        for c, k in (
            missing[["company_id", "key"]]
            .drop_duplicates()
            .itertuples(index=False)
        )
        if compact(k) in by_key
    ]
    if extra:
        cand = pd.concat(
            [
                cand,
                pd.DataFrame(
                    extra, columns=["entity", "ticker", "source", "company_id"]
                ),
            ]
        )
    cand = pd.concat(
        [
            cand,
            _fallback_candidates(holdings, set(cand["company_id"]), periods),
        ]
    )
    # SEC lists many secondary lines (preferreds, warrants: "BAC-PL",
    # "XYZ.WS"): keep plain symbols plus share classes like BRK.B /
    # BF.B.
    cand = cand[cand["ticker"].str.fullmatch(r"[A-Z]{1,5}(\.[A-Z])?")]
    return (
        cand.groupby(["company_id", "ticker"])["source"]
        .agg(lambda s: ",".join(sorted(set(s))))
        .reset_index()
    )


if __name__ == "__main__":
    cand = build_candidates()
    cand.to_parquet(CACHE / "price_ticker_candidates.parquet")
    table = pd.read_parquet(CACHE / "sp500_membership.parquet")
    covered = table["company_id"].isin(cand["company_id"])
    print(
        f"candidates: {len(cand)} rows, {cand['ticker'].nunique()} tickers, "
        f"{cand['company_id'].nunique()} companies"
    )
    print(
        "companies of the membership table with >=1 candidate:"
        f" {covered.mean():.1%} ({(~covered).sum()} without)"
    )
    print(
        "per company:",
        cand.groupby("company_id").size().describe().round(1).to_dict(),
    )
    print(
        "no candidate:",
        table.loc[~covered, "name"].drop_duplicates().head(40).tolist(),
    )
