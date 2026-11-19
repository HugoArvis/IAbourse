"""Check whether Tiingo has daily prices for every ticker that was ever in the S&P 500.

Uses Tiingo's supported_tickers list (one file, no per-ticker API calls) and
membership periods from both free membership sources.

Usage: python data_pipeline/audits/tiingo_coverage_audit.py
"""

from pathlib import Path

import index_constitution as ic
import pandas as pd

from membership_audit import load_hanshof

RAW = Path(__file__).resolve().parents[1] / "cache" / "raw"
TIINGO_LIST = RAW / "supported_tickers.zip"
US_EXCHANGES = {"NYSE", "NASDAQ", "NYSE ARCA", "NYSE MKT", "AMEX", "BATS", "PINK", "OTCMKTS", "OTCGREY"}


def membership_periods() -> pd.DataFrame:
    """One row per (source, ticker) with first and last membership date."""
    hans = load_hanshof()
    long = hans.explode().rename("ticker").reset_index()
    h = long.groupby("ticker")["date"].agg(first="min", last="max").reset_index()
    h["source"] = "hanshof"
    hist = ic.history("sp500")
    i = pd.DataFrame({
        "ticker": hist["symbol"],
        "first": hist["opt-in"].clip(lower=pd.Timestamp("1996-01-01")),
        "last": hist["opt-out"].fillna(pd.Timestamp.today().normalize()),
        "source": "ic",
    })
    i = i[i["last"] >= pd.Timestamp("1996-01-01")]  # same window as hanshof
    return pd.concat([h, i], ignore_index=True)


def main() -> None:
    tiingo = pd.read_csv(TIINGO_LIST, parse_dates=["startDate", "endDate"])
    tiingo = tiingo[(tiingo["assetType"] == "Stock") & tiingo["exchange"].isin(US_EXCHANGES)]
    tiingo["ticker"] = tiingo["ticker"].str.upper()
    # A ticker can appear several times (reused by different companies): keep widest span.
    span = tiingo.groupby("ticker").agg(t_start=("startDate", "min"), t_end=("endDate", "max"))

    per = membership_periods()
    per["key"] = per["ticker"].str.upper().str.replace(".", "-", regex=False)
    per = per.join(span, on="key")
    per["found"] = per["t_start"].notna()
    # Allow a few days of slack at both ends (listing date vs index inclusion).
    slack = pd.Timedelta(days=7)
    per["full_cover"] = per["found"] & (per["t_start"] <= per["first"] + slack) & (per["t_end"] >= per["last"] - slack)
    today = pd.Timestamp.today().normalize()
    per["still_member"] = per["last"] >= today - pd.Timedelta(days=120)

    for src, g in per.groupby("source"):
        print(f"\n== {src}: {len(g)} tickers")
        for label, sub in [("current members", g[g["still_member"]]), ("former members", g[~g["still_member"]])]:
            if len(sub) == 0:
                continue
            print(f"  {label:16s} n={len(sub):4d}  found={sub['found'].mean():6.1%}  full period covered={sub['full_cover'].mean():6.1%}")
        former = g[~g["still_member"]]
        missing = former[~former["found"]]["ticker"].sort_values()
        print(f"  former members not found ({len(missing)}): {list(missing)[:40]}")
        partial = former[former["found"] & ~former["full_cover"]].copy()
        partial["gap_start_days"] = (partial["t_start"] - partial["first"]).dt.days
        print(f"  former found but partial ({len(partial)}), sample:")
        print(partial.sort_values("gap_start_days", ascending=False)
              [["ticker", "first", "last", "t_start", "t_end"]].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
