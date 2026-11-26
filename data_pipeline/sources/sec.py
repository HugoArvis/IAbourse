"""Minimal SEC EDGAR client with on-disk cache and rate limiting.

SEC fair-access policy: at most 10 requests/second and a declared User-Agent.
"""

import hashlib
import time
from pathlib import Path

import pandas as pd
import requests

CACHE = Path(__file__).resolve().parents[1] / "cache" / "sec"
USER_AGENT = "IAbourse-research-script admin@example.com"
MIN_INTERVAL = 0.15  # seconds between requests (< 10 req/s)

_session = requests.Session()
_session.headers["User-Agent"] = USER_AGENT
_last_call = 0.0


def get(url: str, binary: bool = False) -> str | bytes:
    """GET a SEC URL, served from cache when already downloaded."""
    global _last_call
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / hashlib.sha1(url.encode()).hexdigest()
    if path.exists():
        data = path.read_bytes()
        return data if binary else data.decode("utf-8", errors="replace")
    for attempt in range(4):
        wait = MIN_INTERVAL - (time.monotonic() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()
        resp = _session.get(url, timeout=60)
        if resp.status_code == 200:
            path.write_bytes(resp.content)
            return resp.content if binary else resp.text
        if resp.status_code in (429, 503):
            time.sleep(2 ** attempt)
            continue
        resp.raise_for_status()
    raise RuntimeError(f"SEC request failed after retries: {url}")


def filings(cik: int) -> pd.DataFrame:
    """All filings of an entity (recent + paginated history)."""
    import json

    base = json.loads(get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json"))
    frames = [pd.DataFrame(base["filings"]["recent"])]
    for page in base["filings"].get("files", []):
        frames.append(pd.DataFrame(json.loads(get("https://data.sec.gov/submissions/" + page["name"]))))
    df = pd.concat(frames, ignore_index=True)
    df["filingDate"] = pd.to_datetime(df["filingDate"])
    df["reportDate"] = pd.to_datetime(df["reportDate"], errors="coerce")
    return df.sort_values("filingDate").reset_index(drop=True)


def archive_url(cik: int, accession: str, document: str) -> str:
    return f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}/{document}"


def trading_symbols(cik: int, max_filings: int = 6) -> set[str]:
    """Tickers the issuer itself declared (dei:TradingSymbol) in its XBRL periodic reports.

    Read from the SEC rendering of the cover page (R1.htm), on up to `max_filings`
    reports spread over the entity's history. XBRL started in 2009-2011 and the field
    is optional, so older or smaller filers often return nothing."""
    import re

    f = filings(cik)
    f = f[f["form"].isin(["10-K", "10-Q", "20-F", "40-F"]) & (f["isXBRL"] == 1)]
    step = max(1, len(f) // max_filings)
    found = set()
    for r in f.iloc[::step].head(max_filings).itertuples():
        try:
            page = get(archive_url(cik, r.accessionNumber, "R1.htm"))
        except requests.HTTPError:
            continue  # early XBRL filings have no rendered pages
        for value in re.findall(r"Trading Symbol.*?</td>\s*<td[^>]*>\s*([^<]*)<", page, re.S | re.I):
            if re.fullmatch(r"[A-Za-z]{1,5}([.\-][A-Za-z])?", value.strip()):
                found.add(value.strip().upper().replace("-", "."))
    return found
