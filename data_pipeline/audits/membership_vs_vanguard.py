"""Validate the Wikipedia-derived membership sources against Vanguard 500 holdings.

Vanguard 500 Index Fund holdings (SEC filings) are the reference. Everything is
compared in CIK space so that ticker changes and reuse do not create false gaps.

Prerequisites (cached by membership/ modules):
  cache/vanguard500_snapshots.parquet, cache/vanguard500_name_cik.parquet
Usage: python data_pipeline/audits/membership_vs_vanguard.py
"""

import json
import sys
from pathlib import Path

import index_constitution as ic
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "audits"))

from membership.entity_match import match_names  # noqa: E402
from membership_audit import ic_members_at, load_hanshof  # noqa: E402
from sources import sec  # noqa: E402

CACHE = ROOT / "cache"


def _key(ticker: str) -> str:
    return ticker.upper().replace("-", ".")


def sec_current_tickers() -> dict[str, int]:
    current = json.loads(sec.get("https://www.sec.gov/files/company_tickers.json"))
    return {_key(r["ticker"]): int(r["cik_str"]) for r in current.values()}


def ic_with_cik() -> pd.DataFrame:
    """index-constitution history with a CIK per row.

    Current members: official SEC ticker -> CIK. Former members (ticker possibly
    reused since): name matching restricted to their membership period.
    """
    hist = ic.history("sp500").copy()
    current = sec_current_tickers()
    active = hist["opt-out"].isna()
    hist["cik"] = hist["symbol"].map(lambda s: current.get(_key(s))).where(active)
    todo = hist[hist["cik"].isna()]
    names = pd.DataFrame({
        "name": todo["name"],
        "first_seen": todo["opt-in"].clip(lower=pd.Timestamp("1994-01-01")),
        "last_seen": todo["opt-out"].fillna(pd.Timestamp.today().normalize()),
    }).drop_duplicates("name")
    by_name = match_names(names).set_index("name")["cik"]
    hist.loc[hist["cik"].isna(), "cik"] = todo["name"].map(by_name)
    return hist


def ticker_to_cik(ic_hist: pd.DataFrame) -> dict[str, int]:
    """Ticker -> CIK: SEC current tickers, overridden by index-constitution's former members."""
    mapping = sec_current_tickers()
    former = ic_hist[ic_hist["opt-out"].notna() & ic_hist["cik"].notna()]
    for sym, cik in zip(former["symbol"], former["cik"]):
        mapping.setdefault(_key(sym), int(cik))  # don't override a current listing
    return mapping


def main() -> None:
    snaps = pd.read_parquet(CACHE / "vanguard500_snapshots.parquet")
    names = pd.read_parquet(CACHE / "vanguard500_name_cik.parquet")[["name", "cik"]]
    snaps = snaps.merge(names, on="name", how="left")
    hans = load_hanshof()
    ic_hist = ic_with_cik()
    t2c = ticker_to_cik(ic_hist)

    rows = []
    for day, g in snaps.groupby("date"):
        v = set(g["cik"].dropna().astype(int))
        i_syms = ic_members_at(ic_hist, day)
        i = set(ic_hist[ic_hist["symbol"].isin(i_syms)]["cik"].dropna().astype(int))
        row = {
            "date": day.date(), "vanguard": len(g), "v_matched": len(v),
            "ic_in_v": len(i & v) / len(v), "ic_extra": len(i - v),
        }
        if day <= hans.index.max():
            h_day = hans.index[hans.index <= day].max()
            h = {t2c.get(_key(t)) for t in hans[h_day]} - {None}
            row.update({"hanshof_in_v": len(h & v) / len(v), "hanshof_extra": len(h - v),
                        "h_unmapped": sum(t2c.get(_key(t)) is None for t in hans[h_day])})
        rows.append(row)
    out = pd.DataFrame(rows)
    for col in ("ic_in_v", "hanshof_in_v"):
        out[col] = out[col].map(lambda x: f"{x:.1%}" if pd.notna(x) else "-")
    for col in ("hanshof_extra", "h_unmapped"):
        out[col] = out[col].map(lambda x: f"{x:.0f}" if pd.notna(x) else "-")
    print("Share of Vanguard holdings (matched to CIK) also present in each source:")
    print(out.to_string(index=False))

    unmatched = snaps[snaps["cik"].isna()]
    print(f"\nVanguard holdings without CIK: {len(unmatched)} rows, "
          f"{unmatched['name'].nunique()} names, weight share {unmatched['weight'].sum() / snaps['date'].nunique():.2%} per snapshot")
    print(unmatched.drop_duplicates("name")["name"].head(40).tolist())


if __name__ == "__main__":
    main()
