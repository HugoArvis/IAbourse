"""Build the point-in-time S&P 500 membership table.

Reference: Vanguard 500 Index Fund holdings (SEC filings), see vanguard_500.py.
Exact entry/exit dates come from the Wikipedia-derived sources only when they
fall inside the window bracketed by two Vanguard snapshots; otherwise:
  - entry = first snapshot where the company is held (conservative),
  - exit  = first snapshot where it is no longer held (keeps the decline period
            before removal, which matters against survivorship bias).

Output: cache/sp500_membership.parquet, one row per (company, membership spell).
Usage: python data_pipeline/membership/build_membership.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from membership.entity_match import normalize  # noqa: E402
from membership.tickers import ic_with_cik, key, ticker_to_cik  # noqa: E402
from membership.wiki_sources import hanshof_ticker_periods, load_hanshof  # noqa: E402

CACHE = ROOT / "cache"
MAX_LINK_GAP = pd.Timedelta(days=400)  # successive CIKs of one company: gap <= ~1 snapshot interval


class UnionFind:
    def __init__(self):
        self.parent: dict = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        self.parent[self.find(a)] = self.find(b)


def load_holdings() -> pd.DataFrame:
    snaps = pd.read_parquet(CACHE / "vanguard500_snapshots.parquet")
    names = pd.read_parquet(CACHE / "vanguard500_name_cik.parquet")[["name", "cik"]]
    df = snaps.merge(names, on="name", how="left")
    df["key"] = df["name"].map(normalize)
    # Holdings without CIK (bank filers, tracking stocks) keep a name-based id.
    df["entity"] = df["cik"].map(lambda c: f"cik:{int(c)}", na_action="ignore").fillna("name:" + df["key"])
    return df


def assign_company_ids(df: pd.DataFrame, use_links: bool = True) -> pd.Series:
    """Group entities (CIKs) that are the same company; returns entity -> company_id."""
    uf = UnionFind()
    for e in df["entity"].unique():
        uf.find(e)
    # Rule 1: same security identifier held under different entities (NPORT era).
    # Non-US issuers have CUSIP "N/A": fall back to ISIN.
    valid_cusip = df["cusip"].str.fullmatch(r"[0-9A-Z]{8}[0-9]", na=False) & (df["cusip"] != "000000000")
    valid_isin = df["isin"].str.fullmatch(r"[A-Z]{2}[0-9A-Z]{9}[0-9]", na=False)
    df = df.assign(sec_id=df["cusip"].where(valid_cusip, df["isin"].where(valid_isin)))
    for _, g in df.dropna(subset=["sec_id"]).groupby("sec_id"):
        ents = g["entity"].unique()
        for e in ents[1:]:
            uf.union(ents[0], e)
    # Rule 2: same normalized name, contiguous non-overlapping periods (re-incorporations).
    # The old entity must vanish and the new one must be new: otherwise it is an acquirer
    # taking over the target's name (ACE -> Chubb) or a rename + spin-off (News Corp).
    span = df.groupby(["key", "entity"])["date"].agg(["min", "max"]).reset_index()
    ent_first = df.groupby("entity")["date"].min()
    ent_last = df.groupby("entity")["date"].max()
    for _, g in span.groupby("key"):
        g = g.sort_values("min")
        for prev, cur in zip(g.itertuples(), g.iloc[1:].itertuples()):
            contiguous = prev.max < cur.min and cur.min - prev.max <= MAX_LINK_GAP
            succession = ent_last[prev.entity] == prev.max and ent_first[cur.entity] == cur.min
            if contiguous and succession:
                uf.union(prev.entity, cur.entity)
    # Rule 3: successor issuers proven by an 8-K12B and price continuity (successors.py).
    if use_links:
        from membership.successors import load_links
        known = set(df["entity"])
        for pred, succ in zip(*load_links()[["pred", "succ"]].values.T):
            if pred in known and succ in known:
                uf.union(pred, succ)
    ids = {e: uf.find(e) for e in df["entity"].unique()}
    # Company id = most recent entity of the group (readable and stable going forward).
    last_seen = df.groupby("entity")["date"].max()
    groups: dict = {}
    for e, root in ids.items():
        groups.setdefault(root, []).append(e)
    rep = {root: max(members, key=lambda m: last_seen[m]) for root, members in groups.items()}
    return pd.Series({e: rep[root] for e, root in ids.items()}, name="company_id")


def wiki_events(entity_to_company: pd.Series) -> pd.DataFrame:
    """Candidate entry/exit dates from both Wikipedia-derived sources, per company."""
    ic_hist = ic_with_cik()
    rows = []
    known = ic_hist.dropna(subset=["cik"])
    for cik, opt_in, opt_out in zip(known["cik"], known["opt-in"], known["opt-out"]):
        comp = entity_to_company.get(f"cik:{int(cik)}")
        if comp is None:
            continue
        rows.append((comp, "in", opt_in, "ic"))
        if pd.notna(opt_out):
            rows.append((comp, "out", opt_out, "ic"))
    hans = load_hanshof()
    t2c = ticker_to_cik(ic_hist, hanshof_ticker_periods(hans))
    dates = hans.index
    # Members on hanshof's first date: member since at least that date (lower bound).
    for t in hans.iloc[0]:
        cik = t2c.get(key(t))
        comp = entity_to_company.get(f"cik:{cik}") if cik else None
        if comp is not None:
            rows.append((comp, "in", dates[0], "hanshof_since"))
    for i in range(1, len(dates)):
        before, after = hans.iloc[i - 1], hans.iloc[i]
        for t, kind in [(t, "in") for t in after - before] + [(t, "out") for t in before - after]:
            cik = t2c.get(key(t))
            comp = entity_to_company.get(f"cik:{cik}") if cik else None
            if comp is not None:
                rows.append((comp, kind, dates[i], "hanshof"))
    return pd.DataFrame(rows, columns=["company_id", "kind", "date", "source"])


def build() -> pd.DataFrame:
    df = load_holdings()
    company = assign_company_ids(df)
    df["company_id"] = df["entity"].map(company)
    snap_dates = sorted(df["date"].unique())
    held = df.groupby("company_id")["date"].apply(set)
    events = wiki_events(company)

    def pick(comp: str, kind: str, lo, hi):
        """Wikipedia date in (lo, hi]; if both sources give one, prefer agreement then ic."""
        ev = events[(events["company_id"] == comp) & (events["kind"] == kind)
                    & (events["date"] > lo) & (events["date"] <= hi)]
        if ev.empty:
            return None, None
        if ev["source"].nunique() == 2 and (ev["date"].max() - ev["date"].min()).days <= 7:
            return ev["date"].min(), "wiki_both"
        ic_ev = ev[ev["source"] == "ic"]
        best = ic_ev if not ic_ev.empty else ev
        return best["date"].iloc[0], f"wiki_{best['source'].iloc[0]}"

    spells = []
    for comp, dates_held in held.items():
        present = [d in dates_held for d in snap_dates]
        k = 0
        while k < len(snap_dates):
            if not present[k]:
                k += 1
                continue
            start_k = k
            while k + 1 < len(snap_dates) and present[k + 1]:
                k += 1
            end_k = k
            # Entry, bracketed by previous snapshot (or open-ended before the first one).
            lo = snap_dates[start_k - 1] if start_k > 0 else pd.Timestamp("1900-01-01")
            start, s_src = pick(comp, "in", lo, snap_dates[start_k])
            if start is None:
                start, s_src = snap_dates[start_k], ("snapshot" if start_k > 0 else "left_censored")
            # Exit, bracketed by next snapshot (or open-ended after the last one).
            if end_k + 1 < len(snap_dates):
                end, e_src = pick(comp, "out", snap_dates[end_k], snap_dates[end_k + 1])
                if end is None:
                    end, e_src = snap_dates[end_k + 1], "snapshot"
            else:
                end, e_src = pick(comp, "out", snap_dates[end_k], pd.Timestamp.today())
                e_src = e_src or "current"
            names = df[(df["company_id"] == comp) & df["date"].between(snap_dates[start_k], snap_dates[end_k])]
            spells.append({
                "company_id": comp,
                "name": names.sort_values("date")["name"].iloc[-1],
                "ciks": sorted({int(c) for c in names["cik"].dropna()}),
                "start": pd.Timestamp(start), "start_source": s_src,
                "end": pd.Timestamp(end) if end is not None else pd.NaT, "end_source": e_src,
                "n_snapshots": end_k - start_k + 1,
            })
            k += 1
    return pd.DataFrame(spells).sort_values(["start", "company_id"]).reset_index(drop=True)


def members_at(table: pd.DataFrame, day) -> pd.DataFrame:
    day = pd.Timestamp(day)
    return table[(table["start"] <= day) & (table["end"].isna() | (table["end"] > day))]


if __name__ == "__main__":
    table = build()
    table.to_parquet(CACHE / "sp500_membership.parquet")
    print(f"spells: {len(table)}  companies: {table['company_id'].nunique()}")
    print(table["start_source"].value_counts().to_string())
    print(table["end_source"].value_counts().to_string())
