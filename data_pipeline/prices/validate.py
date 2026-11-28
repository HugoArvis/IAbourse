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

from membership.build_membership import (  # noqa: E402
    assign_company_ids, load_holdings,
)
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
    return checks, pick_periods(checks), holdings


def pick_periods(checks: pd.DataFrame) -> pd.DataFrame:
    """Validated (ticker, source) per company as runs of consecutive held snapshots.

    Selection is by continuity, not by total count: providers sometimes splice two
    companies under one ticker (Yahoo's AA = Alcoa Inc. until 2016, then Alcoa Corp.),
    and an unrelated price can match within the tolerance by chance on one date.
      - per snapshot, the candidate whose run of consecutive validations is longest wins
        (then most validations overall, then yfinance);
      - a single isolated validation contradicted by a neighbouring snapshot (price
        present but different) is a coincidence and is dropped;
      - a period never spans a snapshot where it was not picked.
    """
    keys = ["company_id", "ticker", "source"]
    agg = (checks.groupby(keys + ["date"])
                 .agg(ok=("ok", "any"), has_px=("market", lambda s: s.notna().any()))
                 .reset_index().sort_values(keys + ["date"]))
    g = agg.groupby(keys)
    agg["run"] = g["ok"].transform(lambda s: (s != s.shift()).cumsum())
    agg["run_len"] = agg.groupby(keys + ["run"])["ok"].transform("size")

    def contradicted(col):
        return g[col].shift().fillna(False).astype(bool)

    agg["prev_bad"] = contradicted("has_px") & ~contradicted("ok")
    agg["next_bad"] = (g["has_px"].shift(-1).fillna(False).astype(bool)
                       & ~g["ok"].shift(-1).fillna(False).astype(bool))
    ok = agg[agg["ok"] & ~((agg["run_len"] == 1) & (agg["prev_bad"] | agg["next_bad"]))].copy()
    ok["n_ok"] = ok.groupby(keys)["date"].transform("size")
    ok["src_rank"] = ok["source"].map({s: i for i, s in enumerate(SOURCES)})
    best = (ok.sort_values(["run_len", "n_ok", "src_rank"], ascending=[False, False, True])
              .drop_duplicates(["company_id", "date"]))

    # Index of each held snapshot within its company, to detect breaks between picks.
    held = agg[["company_id", "date"]].drop_duplicates().sort_values(["company_id", "date"])
    held["pos"] = held.groupby("company_id").cumcount()
    best = best.merge(held, on=["company_id", "date"]).sort_values(["company_id", "date"])
    same = ((best["company_id"] == best["company_id"].shift())
            & (best["ticker"] == best["ticker"].shift())
            & (best["source"] == best["source"].shift())
            & (best["pos"] == best["pos"].shift() + 1))
    best["period"] = (~same).cumsum()
    return (best.groupby(["period"] + keys)["date"]
                .agg(first_ok="min", last_ok="max", n_ok="size")
                .reset_index().drop(columns="period"))


def coverage_report(periods: pd.DataFrame, holdings: pd.DataFrame) -> None:
    """Share of ALL index members (Vanguard holdings) with a validated price, per year."""
    held = holdings.groupby(["company_id", "date"], as_index=False)["weight"].sum()
    ok = held.merge(periods, on="company_id")
    ok = ok[(ok["date"] >= ok["first_ok"]) & (ok["date"] <= ok["last_ok"])]
    ok = ok.drop_duplicates(["company_id", "date"])[["company_id", "date"]]
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
    coverage_report(periods, holdings)
    mismatch = checks[checks["market"].notna() & ~checks["ok"]]
    print(f"\nchecks: {len(checks)}, with market price: {checks['market'].notna().mean():.1%}, "
          f"ok: {checks['ok'].mean():.1%}, price present but mismatching: {len(mismatch)}")
