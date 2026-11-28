"""Labels: cross-sectional terciles of forward open-to-open returns.

Timing: the decision on day t uses data up to the close of t; orders
fill at the open of t+1. The label of (company, t) for horizon h is the
total return from the open of t+1 to the open of t+1+h (trading days of
the market calendar), minus the mean of the index members labelled that
day, cut into terciles per date:
    0 = worst third, 1 = middle, 2 = best third.

Rules (decided 2026-11-28):
  - only index members on day t with a price on t and on t+1 get a
    label;
  - series ending inside the window (acquisition, bankruptcy): return
    up to the last known close, then cash at 0% (`trunc_h` = True);
  - windows containing a spin-off / special-dividend day (`ret` NaN,
    wrong in free data) get no label;
  - every trading day gets labels (overlapping windows: purge them in
    cross-validation).

Open-to-open return of one day, from the panel's close-to-close total
return:
    open(t+1)/open(t) = (1 + ret(t+1)) * oc(t+1) / oc(t)
where oc = open / close. It only uses ratios inside a day, so splits and
ticker / source switches cancel.

`suspect_open` flags windows whose start or end open equals the close
exactly (~0.4-1.4% of days, probably a missing open filled with the
close by the source).

Output: cache/labels.parquet
Usage: python data_pipeline/features/labels.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path(__file__).resolve().parents[1] / "cache"
HORIZONS = (5, 21)
# Before 2010 the panel only holds later members' history (biased).
START = pd.Timestamp("2010-01-01")
PANEL_COLUMNS = ["date", "company_id", "open", "close", "ret"]
TERCILE_EDGES = [0, 1 / 3, 2 / 3, 1]


def member_mask(panel: pd.DataFrame,
                membership: pd.DataFrame) -> np.ndarray:
    """True where the company is an index member on that date.

    A membership spell covers start <= date < end (end NaT = still in).
    """
    mask = np.zeros(len(panel), dtype=bool)
    rows = panel.groupby("company_id").indices
    dates = panel["date"].to_numpy()
    for spell in membership.itertuples():
        idx = rows.get(spell.company_id)
        if idx is None:
            continue
        inside = dates[idx] >= np.datetime64(spell.start)
        if pd.notna(spell.end):
            inside &= dates[idx] < np.datetime64(spell.end)
        mask[idx[inside]] = True
    return mask


def _shift_up(values: np.ndarray) -> np.ndarray:
    """Value of the next row (NaN for the last one)."""
    return np.r_[values[1:], np.nan]


def _log_open_index(p: pd.DataFrame, same_next: np.ndarray,
                    first: np.ndarray) -> np.ndarray:
    """Log of a chained open-price index, starting at 0 per company."""
    oc = (p["open"] / p["close"]).to_numpy()
    ret = p["ret"].to_numpy()
    gross = (1 + _shift_up(ret)) * _shift_up(oc) / oc  # open -> next open
    valid = same_next & np.isfinite(gross) & (gross > 0)
    step = np.log(gross, out=np.zeros_like(gross), where=valid)
    log_open = np.cumsum(step) - step
    starts = np.flatnonzero(first)
    sizes = np.diff(np.r_[starts, len(p)])
    return log_open - np.repeat(log_open[starts], sizes)


def _tercile(ret: pd.Series,
             dates: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Excess return over the date mean, and its tercile (0, 1, 2)."""
    excess = ret - ret.groupby(dates).transform("mean")
    pct = excess.groupby(dates).rank(pct=True)
    tercile = pd.cut(pct, TERCILE_EDGES, labels=False).astype("Int8")
    return excess.astype(np.float32), tercile


def build(panel: pd.DataFrame, membership: pd.DataFrame,
          horizons=HORIZONS, start=START) -> pd.DataFrame:
    """One row per member-day from `start`, with labels per horizon."""
    p = (panel[PANEL_COLUMNS].sort_values(["company_id", "date"])
         .reset_index(drop=True))
    n = len(p)
    days = np.sort(p["date"].unique())
    day = np.searchsorted(days, p["date"].to_numpy())
    comp = (p["company_id"].astype("category").cat.codes
            .to_numpy().astype(np.int64))
    row_key = comp * (len(days) + 1) + day  # sorted, unique
    same_next = np.r_[comp[1:] == comp[:-1], False]
    first = np.r_[True, comp[1:] != comp[:-1]]
    starts = np.flatnonzero(first)
    last_row = np.repeat(np.r_[starts[1:], n] - 1,
                         np.diff(np.r_[starts, n]))

    log_open = _log_open_index(p, same_next, first)
    oc = (p["open"] / p["close"]).to_numpy()
    log_close = log_open - np.log(oc)  # index value at the same close
    corp_actions = np.cumsum(p["ret"].isna().to_numpy() & ~first)
    open_is_close = (p["open"] == p["close"]).to_numpy()

    keep = member_mask(p, membership) & (p["date"] >= start).to_numpy()
    # The window opens on the next row: it must be the next trading day.
    has_start = same_next & (np.r_[day[1:], -1] == day + 1)
    start_row = np.where(has_start, np.arange(n) + 1, 0)

    out = p[["date", "company_id"]].copy()
    suspect = np.zeros(n, dtype=bool)
    for h in horizons:
        target = day + 1 + h
        in_calendar = target <= len(days) - 1
        target_key = comp * (len(days) + 1) + target
        # First row on or after the target day (after it: a hole).
        end_row = np.minimum(np.searchsorted(row_key, target_key), n - 1)
        reached = (comp[end_row] == comp) & (row_key[end_row] >= target_key)
        end_row = np.where(reached, end_row, last_row)
        end_value = np.where(reached, log_open[end_row], log_close[end_row])
        ok = (keep & has_start & in_calendar
              & (corp_actions[end_row] == corp_actions[start_row]))
        ret = np.exp(end_value - log_open[start_row]) - 1

        out[f"ret_{h}"] = np.where(ok, ret, np.nan).astype(np.float32)
        out[f"trunc_{h}"] = ok & ~reached
        suspect |= ok & (open_is_close[start_row]
                         | (open_is_close[end_row] & reached))
        labelled = out.loc[ok]
        out[f"ex_{h}"], out[f"y_{h}"] = _tercile(labelled[f"ret_{h}"],
                                                 labelled["date"])
    out["suspect_open"] = suspect
    return out[keep].reset_index(drop=True)


def main():
    panel = pd.read_parquet(CACHE / "price_panel.parquet",
                            columns=PANEL_COLUMNS)
    membership = pd.read_parquet(CACHE / "sp500_membership.parquet")
    labels = build(panel, membership)
    labels.to_parquet(CACHE / "labels.parquet")

    first_day = labels["date"].min().date()
    last_day = labels["date"].max().date()
    print(f"{len(labels):,} member-days, "
          f"{labels['company_id'].nunique()} companies, "
          f"{first_day} -> {last_day}")
    for h in HORIZONS:
        y = labels[f"y_{h}"]
        shares = y.value_counts(normalize=True).sort_index().round(3)
        print(f"h={h:>2}: labelled {y.notna().mean():.2%}, "
              f"truncated {labels[f'trunc_{h}'].sum()}, "
              f"classes {shares.to_dict()}")
    print(f"suspect_open: {labels['suspect_open'].mean():.2%}")


if __name__ == "__main__":
    main()
