"""Minimal Tiingo client (end-of-day prices and ticker metadata) with on-disk cache.

API key: TIINGO_API_KEY in backend/.env (never committed).
"""

import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(__file__).resolve().parents[1] / "cache" / "tiingo"
BASE = "https://api.tiingo.com/tiingo/daily"
MIN_INTERVAL = 1.5  # seconds between requests (free plan limits are not published per endpoint)

load_dotenv(ROOT / "backend" / ".env")
_session = requests.Session()
_last_call = 0.0


class RateLimited(RuntimeError):
    """Free plan quota reached (observed: ~50 requests/hour). Retry later; cache keeps progress."""


def _get(url: str, params: dict | None = None):
    global _last_call
    key = os.environ.get("TIINGO_API_KEY")
    if not key:
        raise RuntimeError("TIINGO_API_KEY missing from backend/.env")
    wait = MIN_INTERVAL - (time.monotonic() - _last_call)
    if wait > 0:
        time.sleep(wait)
    _last_call = time.monotonic()
    resp = _session.get(url, params=params, headers={"Authorization": f"Token {key}"}, timeout=60)
    if resp.status_code == 429:
        raise RateLimited(url)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


def meta(ticker: str) -> dict | None:
    """Ticker metadata (name, startDate, endDate...). Known but price-less tickers
    (e.g. delisted ones) still return a name. Cached forever."""
    path = CACHE / "meta" / f"{ticker.upper()}.json"
    if path.exists():
        return json.loads(path.read_text())
    data = _get(f"{BASE}/{ticker.lower()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return data


def prices(ticker: str, start: str = "1990-01-01", end: str | None = None) -> list[dict]:
    """Daily OHLCV with adjusted fields. Cached per ticker (re-download to refresh)."""
    path = CACHE / "prices" / f"{ticker.upper()}.json"
    if path.exists():
        return json.loads(path.read_text())
    params = {"startDate": start}
    if end:
        params["endDate"] = end
    data = _get(f"{BASE}/{ticker.lower()}/prices", params) or []
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return data
