"""Validate price series against Vanguard 500 holdings and pick each company's ticker.

At every snapshot, Vanguard reports shares held and market value, i.e. the actual
closing price of the stock that day (implied = value / shares). A candidate ticker
is validated for a company at a snapshot when its unadjusted close matches the
implied price within TOLERANCE. This proves both that the ticker is the right
company at that time (reused tickers fail) and that the price data are sound.

Outputs:
  cache/price_checks.parquet    one row per (company, ticker, snapshot) check
  cache/company_tickers.parquet validated ticker per company and period
Usage: python data_pipeline/prices/validate.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from membership.build_membership import assign_company_ids, load_holdings  # noqa: E402
from prices.download import SOURCES, load  # noqa: E402

CACHE = ROOT / "cache"
TOLERANCE = 0.03


def unadjusted_close(px: pd.DataFrame) -> pd.Series:
    """Undo yfinance split adjustment: raw(t) = close(t) * product of split ratios after t."""
    ratios = px["splits"].where(px["splits"] > 0, 1.0)
    after = ratios[::-1].cumprod()[::-1].shift(-1, fill_value=1.0)  # product of splits strictly after t
    return px["close"] * after


def price_on(series: pd.Series, day: pd.Timestamp) -> float | None:
    """Close of the last trading day <= day, if within 5 days."""
    s = series.loc[:day]
    if s.empty or (day - s.index[-1]).days > 5:
        return None
    return float(s.iloc[-1])


def run() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    holdings = load_holdings()
    holdings["company_id"] = holdings["entity"].map(assign_company_ids(holdings))
    holdings = holdings[holdings["shares"] > 0]
    holdings["implied"] = holdings["value_k"] * 1000 / holdings["shares"]
    cand = pd.read_parquet(CACHE / "price_ticker_candidates.parquet")

    raw_cache: dict[tuple[str, str], pd.Series] = {}
    checks = []
    for comp, tickers in cand.groupby("company_id")["ticker"]:
        rows = holdings[holdings["company_id"] == comp]
        for t in tickers:
            for src in SOURCES:
                if (t, src) not in raw_cache:
                    px = load(t, src)
                    raw_cache[t, src] = unadjusted_close(px) if not px.empty else pd.Series(dtype=float)
                raw = raw_cache[t, src]
                if raw.empty and src != "yfinance":
                    continue  # absent secondary source: no row (keeps yfinance as the baseline)
                for r in rows.itertuples():
                    p = price_on(raw, r.date) if not raw.empty else None
                    checks.append({
                        "company_id": comp, "ticker": t, "source": src, "date": r.date, "name": r.name,
                        "weight": r.weight, "implied": r.implied, "market": p,
                        "ok": p is not None and abs(p / r.implied - 1) <= TOLERANCE,
                    })
    checks = pd.DataFrame(checks)

    # Per company and snapshot: the validated (ticker, source); most often validated wins,
    # yfinance preferred on ties.
    ok = checks[checks["ok"]].copy()
    score = ok.groupby(["company_id", "ticker", "source"]).size().rename("n_ok")
    ok = ok.join(score, on=["company_id", "ticker", "source"])
    ok["src_rank"] = ok["source"].map({s: i for i, s in enumerate(SOURCES)})
    best = (ok.sort_values(["n_ok", "src_rank"], ascending=[False, True])
              .drop_duplicates(["company_id", "date"]))
    periods = (best.groupby(["company_id", "ticker", "source"])["date"]
                   .agg(first_ok="min", last_ok="max", n_ok="size").reset_index())
    return checks, periods, holdings


def coverage_report(checks: pd.DataFrame, holdings: pd.DataFrame) -> None:
    """Share of ALL index members (Vanguard holdings) with a validated price, per year."""
    held = holdings.groupby(["company_id", "date"], as_index=False)["weight"].sum()
    ok = checks[checks["ok"]].drop_duplicates(["company_id", "date"])[["company_id", "date"]]
    held = held.merge(ok.assign(ok=True), on=["company_id", "date"], how="left")
    held["ok"] = held["ok"].notna()
    by_year = held.groupby(held["date"].dt.year).apply(
        lambda g: pd.Series({"members": len(g) / g["date"].nunique(),
                             "validated": g["ok"].mean(),
                             "validated_weight": (g["weight"] * g["ok"]).sum() / g["weight"].sum()}),
        include_groups=False)
    by_year["members"] = by_year["members"].round().astype(int)
    print("Index members whose price matches Vanguard's implied price (share of members / of index weight):")
    print(by_year.to_string(formatters={"validated": "{:.1%}".format, "validated_weight": "{:.1%}".format}))


if __name__ == "__main__":
    checks, periods, holdings = run()
    checks.to_parquet(CACHE / "price_checks.parquet")
    periods.to_parquet(CACHE / "company_tickers.parquet")
    coverage_report(checks, holdings)
    mismatch = checks[checks["market"].notna() & ~checks["ok"]]
    print(f"\nchecks: {len(checks)}, with market price: {checks['market'].notna().mean():.1%}, "
          f"ok: {checks['ok'].mean():.1%}, price present but mismatching: {len(mismatch)}")
