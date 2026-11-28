"""Labels: cross-sectional terciles of forward open-to-open returns (5 and 21 days).

Timing: the decision on day t uses data up to the close of t; orders fill at the
open of t+1. The label of (company, t) for horizon h is the total return from the
open of t+1 to the open of t+1+h (trading days of the market calendar), minus the
mean of the index members labelled that day, cut into terciles per date:
    0 = worst third, 1 = middle, 2 = best third.

Rules (decided 2026-11-28):
  - only index members on day t with a price on t and on t+1 get a label;
  - series ending inside the window (acquisition, bankruptcy): return up to the
    last known close, then cash at 0% (`trunc_h` = True);
  - windows containing a spin-off / special-dividend day (`ret` NaN, wrong in free
    data) get no label;
  - every trading day gets labels (overlapping windows: purge them in CV).

Open-to-open return of one day, from the panel's close-to-close total return:
    open(t+1)/open(t) = (1 + ret(t+1)) * [open/close](t+1) / [open/close](t)
It only uses ratios inside a day, so splits and ticker / source switches cancel.

`suspect_open` flags windows whose start or end open equals the close exactly
(~0.4-1.4% of days, probably a missing open filled with the close by the source).

Output: cache/labels.parquet
Usage: python data_pipeline/features/labels.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CACHE = ROOT / "cache"
HORIZONS = (5, 21)
START = pd.Timestamp("2010-01-01")  # before, the panel only holds history of later members (biased)
PANEL_COLUMNS = ["date", "company_id", "open", "close", "ret"]


def member_mask(panel: pd.DataFrame, membership: pd.DataFrame) -> np.ndarray:
    """True where the company is an index member on that date (start <= date < end)."""
    mask = np.zeros(len(panel), dtype=bool)
    rows = panel.groupby("company_id").indices
    for s in membership.itertuples():
        idx = rows.get(s.company_id)
        if idx is None:
            continue
        d = panel["date"].values[idx]
        inside = d >= np.datetime64(s.start)
        if pd.notna(s.end):
            inside &= d < np.datetime64(s.end)
        mask[idx[inside]] = True
    return mask


def build(panel: pd.DataFrame, membership: pd.DataFrame, horizons=HORIZONS, start=START) -> pd.DataFrame:
    p = panel[PANEL_COLUMNS].sort_values(["company_id", "date"]).reset_index(drop=True)
    days = np.sort(p["date"].unique())
    di = np.searchsorted(days, p["date"].values)
    comp = p["company_id"].astype("category").cat.codes.to_numpy().astype(np.int64)
    rowkey = comp * (len(days) + 1) + di  # sorted, unique per (company, day)
    same_next = np.r_[comp[1:] == comp[:-1], False]
    first = np.r_[True, comp[1:] != comp[:-1]]

    oc = (p["open"] / p["close"]).to_numpy()
    nxt = lambda a: np.r_[a[1:], np.nan]
    gross = (1 + nxt(p["ret"].to_numpy())) * nxt(oc) / oc  # open(row) -> open(next row)
    lg = np.where(same_next & np.isfinite(gross) & (gross > 0), np.log(gross), 0.0)
    log_open = np.cumsum(lg) - lg  # log of a chained open-price index...
    starts = np.flatnonzero(first)
    log_open -= np.repeat(log_open[starts], np.diff(np.r_[starts, len(p)]))  # ...restarted per company
    bad = np.cumsum(p["ret"].isna().to_numpy() & ~first)  # corp-action days seen so far
    log_last_close = log_open - np.log(oc)  # value at the close of the same row
    last_row = np.repeat(np.r_[starts[1:], len(p)] - 1, np.diff(np.r_[starts, len(p)]))
    open_is_close = (p["open"] == p["close"]).to_numpy()

    out = p[["date", "company_id"]].copy()
    out["member"] = member_mask(p, membership)
    # start of the window: next row of the same company must be the next trading day
    s = np.arange(len(p)) + 1
    has_start = same_next & (np.r_[di[1:], -1] == di + 1)
    s = np.where(has_start, s, 0)
    suspect = np.zeros(len(p), dtype=bool)
    for h in horizons:
        target = di + 1 + h
        in_calendar = target <= len(days) - 1
        tkey = comp * (len(days) + 1) + target
        e = np.searchsorted(rowkey, tkey, side="left")  # first row on or after the target day
        e = np.minimum(e, len(p) - 1)
        after = (comp[e] == comp) & (rowkey[e] >= tkey)  # series reaches the target day (or later, after a hole)
        trunc = in_calendar & ~after
        e = np.where(after, e, last_row)
        end_val = np.where(after, log_open[e], log_last_close[e])
        r = np.exp(end_val - log_open[s]) - 1
        ok = (out["member"] & (out["date"] >= start)).to_numpy() & has_start & in_calendar & (bad[e] == bad[s])
        r = np.where(ok, r, np.nan)
        out[f"ret_{h}"] = r.astype(np.float32)
        out[f"trunc_{h}"] = trunc & ok
        suspect |= ok & (open_is_close[s] | (open_is_close[e] & after))

        g = out.loc[ok, ["date", f"ret_{h}"]]
        ex = g[f"ret_{h}"] - g.groupby("date")[f"ret_{h}"].transform("mean")
        pct = ex.groupby(g["date"]).rank(pct=True)
        out[f"ex_{h}"] = ex.astype(np.float32)
        out[f"y_{h}"] = pd.cut(pct, [0, 1 / 3, 2 / 3, 1], labels=False).astype("Int8")
    out["suspect_open"] = suspect
    return out[out["member"] & (out["date"] >= start)].drop(columns="member").reset_index(drop=True)


if __name__ == "__main__":
    panel = pd.read_parquet(CACHE / "price_panel.parquet", columns=PANEL_COLUMNS)
    membership = pd.read_parquet(CACHE / "sp500_membership.parquet")
    labels = build(panel, membership)
    labels.to_parquet(CACHE / "labels.parquet")
    print(f"{len(labels):,} member-days, {labels['company_id'].nunique()} companies, "
          f"{labels['date'].min().date()} -> {labels['date'].max().date()}")
    for h in HORIZONS:
        y = labels[f"y_{h}"]
        print(f"h={h:>2}: labelled {y.notna().mean():.2%}, truncated {labels[f'trunc_{h}'].sum()}, "
              f"classes {y.value_counts(normalize=True).sort_index().round(3).to_dict()}")
    print(f"suspect_open: {labels['suspect_open'].mean():.2%}")
