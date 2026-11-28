"""Fill price gaps from Tiingo within its free quota (~50 requests/hour). Resumable.

Phases (everything is cached, so the script can be stopped and restarted anytime):
  A. prices of candidate tickers that yfinance lacks, when Tiingo lists them over
     the period the company was in the index;
  B. company names of historical tickers still unmapped (Tiingo metadata), then
     rebuild the ticker candidates;
  C. same as A for the new candidates;
  then re-run the price validation.
When the quota is reached the script sleeps one hour and continues.

Usage: python data_pipeline/prices/fill_tiingo.py
"""

import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prices.download import download, download_tiingo, load  # noqa: E402
from sources import tiingo  # noqa: E402

CACHE = ROOT / "cache"
PY = sys.executable
WAIT = 3600


def _with_quota(fn, item):
    """Call fn(item), sleeping through Tiingo quota exhaustion and network outages."""
    while True:
        try:
            return fn(item)
        except tiingo.RateLimited:
            print(
                f"  quota reached at {time.strftime('%H:%M')}, sleeping"
                f" {WAIT // 60} min",
                flush=True,
            )
            time.sleep(WAIT)
        except requests.ConnectionError:
            print("  network unavailable, retrying in 1 min", flush=True)
            time.sleep(60)


def tiingo_gap_tickers() -> list[str]:
    """Candidate tickers with no yfinance data whose Tiingo listing covers the company's period."""
    cand = pd.read_parquet(CACHE / "price_ticker_candidates.parquet")
    table = pd.read_parquet(CACHE / "sp500_membership.parquet")
    first_in = (
        table.groupby("company_id")["start"]
        .min()
        .clip(lower=pd.Timestamp("2004-01-01"))
    )
    listing = pd.read_csv(
        CACHE / "raw" / "supported_tickers.zip",
        parse_dates=["startDate", "endDate"],
    )
    listing = listing[listing["assetType"] == "Stock"]
    listing["ticker"] = (
        listing["ticker"].str.upper().str.replace("-", ".", regex=False)
    )
    span = listing.groupby("ticker").agg(
        t_start=("startDate", "min"), t_end=("endDate", "max")
    )
    cand = cand.join(span, on="ticker").join(
        first_in.rename("first_in"), on="company_id"
    )
    gap = cand[cand["ticker"].map(lambda t: load(t).empty)]
    covers = (gap["t_start"] <= gap["first_in"] + pd.Timedelta(days=7)) & (
        gap["t_end"] >= gap["first_in"]
    )
    already = {p.stem for p in (CACHE / "prices" / "tiingo").glob("*.parquet")}
    return sorted(set(gap.loc[covers, "ticker"]) - already)


def unnamed_tickers() -> list[str]:
    path = CACHE / "unmapped_hanshof_tickers.txt"
    todo = path.read_text().split() if path.exists() else []
    return [
        t
        for t in todo
        if not (
            tiingo.CACHE / "meta" / f"{t.replace('.', '-').upper()}.json"
        ).exists()
    ]


def fill_prices(label: str) -> None:
    todo = tiingo_gap_tickers()
    print(f"[{label}] Tiingo prices to fetch: {len(todo)}", flush=True)
    for i, t in enumerate(todo, 1):
        got = _with_quota(download_tiingo, t)
        print(
            f"  {i}/{len(todo)} {t}: {'ok' if got else 'no data'}", flush=True
        )


def main() -> None:
    fill_prices("A")
    todo = unnamed_tickers()
    print(f"[B] ticker names to fetch: {len(todo)}", flush=True)
    for i, t in enumerate(todo, 1):
        meta = _with_quota(tiingo.meta, t.replace(".", "-"))
        print(f"  {i}/{len(todo)} {t}: {(meta or {}).get('name')}", flush=True)
    print("[B] rebuilding candidates", flush=True)
    subprocess.run(
        [PY, "-I", str(ROOT / "prices" / "candidates.py")], check=True
    )
    cand = pd.read_parquet(CACHE / "price_ticker_candidates.parquet")
    # yfinance for new candidates (cached otherwise)
    download(sorted(cand["ticker"].unique()))
    fill_prices("C")
    print("[done] validating", flush=True)
    subprocess.run(
        [PY, "-I", str(ROOT / "prices" / "validate.py")], check=True
    )


if __name__ == "__main__":
    main()
