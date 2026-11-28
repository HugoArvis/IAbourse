"""Link companies that changed CIK without changing their shares.

Example: Google Inc. (CIK 1288776) became Alphabet Inc. (CIK 1652044) in
2015 through a holding-company reorganization: one Google share became
one Alphabet share, same ticker, same price. The SEC sees two issuers;
economically it is one company.

A successor issuer announces itself with an 8-K12B / 8-K12G3 filing.
Acquirers also file these forms when they take over a target's
registered securities (Wells Fargo for Wachovia), so the filing alone is
not enough. A pair (predecessor P, successor S) is linked only when:
  1. S filed an 8-K12B / 8-K12G3;
  2. P's last Vanguard snapshot is the last one before the filing (P
     vanished then);
  3. S's share price matches P's actual price (Vanguard implied price)
     at P's snapshots: one old share = one new share. An acquisition
     with any other exchange ratio fails this test.

Run after prices/validate.py inputs exist (candidates + downloaded
prices); the links are then used by build_membership.assign_company_ids,
so re-run build_membership.py, prices/candidates.py and
prices/validate.py afterwards.

Output: cache/successor_links.parquet
Usage: python data_pipeline/membership/successors.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from membership.build_membership import (  # noqa: E402
    assign_company_ids,
    load_holdings,
)
from prices.download import SOURCES, load  # noqa: E402
from prices.validate import TOLERANCE, price_on, unadjusted_close  # noqa: E402
from sources import sec  # noqa: E402

CACHE = ROOT / "cache"
OUT = CACHE / "successor_links.parquet"
FORMS = r"8-K12[BG]3?(/A)?"
MIN_MATCHES = 3  # snapshots where S's price equals P's actual price
MIN_SHARE = 0.8  # ... out of the snapshots where S has a price


def succession_filings(ciks) -> pd.DataFrame:
    rows = []
    for cik in ciks:
        f = sec.filings(cik)
        for r in f[f["form"].str.fullmatch(FORMS)].itertuples():
            rows.append((cik, r.form, r.filingDate))
    return (
        pd.DataFrame(rows, columns=["cik", "form", "filed"])
        .drop_duplicates(["cik", "filed"])
        .reset_index(drop=True)
    )


def _price_continuity(p_rows: pd.DataFrame, tickers, cache: dict) -> tuple:
    """Best of S's tickers against P's implied prices.

    Returns (ticker, source, n_ok, n_px).

    A snapshot counts as matched when any of P's share classes matches.
    """
    best = (None, None, 0, 0)
    for t in tickers:
        for src in SOURCES:
            if (t, src) not in cache:
                px = load(t, src)
                cache[t, src] = unadjusted_close(px) if not px.empty else None
            raw = cache[t, src]
            if raw is None:
                continue
            per_date = {}
            for r in p_rows.itertuples():
                p = price_on(raw, r.date)
                if p is None:
                    continue
                ok = abs(p / r.implied - 1) <= TOLERANCE
                per_date[r.date] = per_date.get(r.date, False) or ok
            n_ok, n_px = sum(per_date.values()), len(per_date)
            if (n_ok, n_px) > best[2:]:
                best = (t, src, n_ok, n_px)
    return best


def find_links() -> pd.DataFrame:
    h = load_holdings()
    company = assign_company_ids(h, use_links=False)
    h["company_id"] = h["entity"].map(company)
    h = h[h["shares"] > 0].copy()
    h["implied"] = h["value_k"] * 1000 / h["shares"]
    dates = sorted(h["date"].unique())
    last = h.groupby("company_id")["date"].max()
    cand = pd.read_parquet(CACHE / "price_ticker_candidates.parquet")
    tickers = cand.groupby("company_id")["ticker"].apply(list)

    filings = succession_filings(
        sorted(h["cik"].dropna().astype(int).unique())
    )
    cache: dict = {}
    links = []
    for f in filings.itertuples():
        succ = company.get(f"cik:{f.cik}")
        before = [d for d in dates if d < f.filed]
        if succ is None or not before or succ not in tickers:
            continue
        for pred in last.index[(last == before[-1]) & (last.index != succ)]:
            t, src, n_ok, n_px = _price_continuity(
                h[h["company_id"] == pred], tickers[succ], cache
            )
            if n_ok >= MIN_MATCHES and n_ok >= MIN_SHARE * n_px:
                links.append(
                    {
                        "pred": pred,
                        "succ": succ,
                        "cik": f.cik,
                        "form": f.form,
                        "filed": f.filed,
                        "ticker": t,
                        "source": src,
                        "n_ok": n_ok,
                        "n_px": n_px,
                    }
                )
    return (
        pd.DataFrame(links)
        .drop_duplicates(["pred", "succ"])
        .reset_index(drop=True)
    )


def load_links() -> pd.DataFrame:
    return (
        pd.read_parquet(OUT)
        if OUT.exists()
        else pd.DataFrame(columns=["pred", "succ"])
    )


if __name__ == "__main__":
    links = find_links()
    links.to_parquet(OUT)
    names = load_holdings().groupby("entity")["name"].last()
    print(f"{len(links)} successor links")
    print(
        links.assign(
            pred=links["pred"].map(names), succ=links["succ"].map(names)
        )[
            ["filed", "form", "pred", "succ", "ticker", "n_ok", "n_px"]
        ].to_string(
            index=False
        )
    )
