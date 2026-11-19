"""Compare two free sources of historical S&P 500 membership.

Sources:
  - hanshof/sp500_constituents: one row per date with the full member list.
  - index-constitution (PyPI): one row per company with opt-in / opt-out dates.

Usage: python data_pipeline/audits/membership_audit.py
"""

from pathlib import Path

import index_constitution as ic
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "cache" / "raw"
HANSHOF_CSV = RAW / "sp_500_historical_components.csv"


def load_hanshof() -> pd.Series:
    """Return a Series indexed by date, values = frozenset of tickers."""
    df = pd.read_csv(HANSHOF_CSV, parse_dates=["date"]).sort_values("date")
    return df.set_index("date")["tickers"].map(lambda s: frozenset(s.split(",")))


def ic_members_at(hist: pd.DataFrame, day: pd.Timestamp) -> frozenset:
    active = (hist["opt-in"] <= day) & (hist["opt-out"].isna() | (hist["opt-out"] > day))
    return frozenset(hist.loc[active, "symbol"])


def main() -> None:
    hans = load_hanshof()
    hist = ic.history("sp500")

    print("== hanshof")
    sizes = hans.map(len)
    print(f"dates: {len(hans)}  range: {hans.index.min().date()} -> {hans.index.max().date()}")
    print(f"members per date: min={sizes.min()} median={sizes.median():.0f} max={sizes.max()}")
    print(f"unique tickers: {len(frozenset().union(*hans))}")
    gaps = hans.index.to_series().diff().dt.days
    print(f"largest gap between snapshots: {gaps.max():.0f} days (at {gaps.idxmax().date()})")

    print("\n== index-constitution")
    print(f"companies: {len(hist)}  symbols unique: {hist['symbol'].nunique()}")
    print(f"opt-in range: {hist['opt-in'].min().date()} -> {hist['opt-in'].max().date()}")
    print(f"missing opt-in: {hist['opt-in'].isna().sum()}")

    # Compare on month-end snapshots (last hanshof date of each month).
    month_end = hans.groupby(hans.index.to_period("M")).tail(1)
    rows = []
    for day, h_set in month_end.items():
        i_set = ic_members_at(hist, day)
        rows.append({
            "date": day.date(),
            "hanshof": len(h_set),
            "ic": len(i_set),
            "common": len(h_set & i_set),
            "only_hanshof": len(h_set - i_set),
            "only_ic": len(i_set - h_set),
        })
    cmp_df = pd.DataFrame(rows)
    cmp_df["jaccard"] = cmp_df["common"] / (cmp_df["hanshof"] + cmp_df["ic"] - cmp_df["common"])

    print("\n== month-end comparison (yearly sample: January)")
    jan = cmp_df[pd.to_datetime(cmp_df["date"]).dt.month == 1]
    print(jan.to_string(index=False, float_format="%.3f"))
    print(f"\nmean jaccard: {cmp_df['jaccard'].mean():.3f}  last: {cmp_df['jaccard'].iloc[-1]:.3f}")

    # Show concrete disagreements on a few dates.
    for day in [pd.Timestamp("2005-01-31"), pd.Timestamp("2015-01-30"), hans.index.max()]:
        d = hans.index[hans.index <= day].max()
        h_set, i_set = hans[d], ic_members_at(hist, d)
        print(f"\n-- {d.date()}  only_hanshof ({len(h_set - i_set)}): {sorted(h_set - i_set)[:25]}")
        print(f"-- {d.date()}  only_ic ({len(i_set - h_set)}): {sorted(i_set - h_set)[:25]}")


if __name__ == "__main__":
    main()
