"""Assemble one daily price series per company from the validated ticker periods.

Each company's history is a sequence of runs (ticker, source, first_ok, last_ok) from
validate.py. Each panel row carries a status:
  validated  between two consecutive Vanguard snapshots where the ticker's price
             matched the actual price;
  extended   same series continued beyond a run, where Vanguard cannot check it:
             before the first / after the last held snapshot (history before index
             entry, days around exit), or up to a neighbouring snapshot where the
             series simply has no quote. Never across a snapshot where the price
             was present but wrong (the series then belongs to another company).
Extensions stop at the first hole in the quotes (> MAX_GAP_DAYS) or extreme jump.

Rows are kept from LOOKBACK before each membership spell (features need history
before index entry) to FORWARD after its end (labels need future returns), from
PANEL_START on.

`ret` is the daily total return, computed inside each source series (NaN on
spin-off / special-dividend days, flagged `corp_action`, which free data get wrong); at a switch
between series it is close_unadj(t) / close_unadj(t-1) - 1 (validated 1:1 shares).
Price levels (open..close, adj_close) may jump at such switches: use `ret` to chain.

Output: cache/price_panel.parquet
Usage: python data_pipeline/prices/panel.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prices.download import load  # noqa: E402
from prices.validate import unadjusted_close  # noqa: E402

CACHE = ROOT / "cache"
PANEL_START = pd.Timestamp("2010-01-01")
# ~1 trading year of history before index entry
LOOKBACK = pd.Timedelta(days=400)
FORWARD = pd.Timedelta(days=45)  # ~21 trading days after exit, for labels
MAX_GAP_DAYS = 10  # calendar days without quote = hole
# |log raw close change| beyond x3 / ÷3 (non-split day)
MAX_JUMP = np.log(3)
COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "dividends",
    "splits",
]


def _windows(spells: pd.DataFrame, today: pd.Timestamp) -> list[tuple]:
    """Merged [start - LOOKBACK, end + FORWARD] windows of a company's spells from PANEL_START."""
    out = []
    for s in spells.sort_values("start").itertuples():
        end = s.end if pd.notna(s.end) else today
        if end < PANEL_START:
            continue
        lo, hi = max(s.start, PANEL_START) - LOOKBACK, end + FORWARD
        if out and lo <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def _corporate_action(px: pd.DataFrame) -> pd.Series:
    """Days whose total return free providers get wrong: spin-offs and special dividends.

    Yahoo encodes a spin-off as a fractional "split" (DHR 2016: 1.319, adjusted series
    +61%), Tiingo ignores the distributed shares (CSC 2015: -39%). Plain splits with
    simple ratios (2:1, 3:2, 1:4...) are reliable and not flagged."""
    r = px["splits"].where(px["splits"] > 0)
    simple = pd.Series(False, index=px.index)
    for x in (r, 1 / r):
        for den in (1, 2, 3, 4, 5, 8, 10, 20):
            simple |= ((x * den) - (x * den).round()).abs() < 0.01
    odd_split = r.notna() & ~simple
    special_div = px["dividends"] > 0.1 * px["close"].shift()
    return odd_split | special_div


def _series(ticker: str, source: str, cache: dict) -> pd.DataFrame:
    if (ticker, source) not in cache:
        px = load(ticker, source)
        if not px.empty:
            px = px[COLUMNS].copy()
            px["close_unadj"] = unadjusted_close(px)
            px["corp_action"] = _corporate_action(px)
            # unreliable that day
            px["ret_src"] = (
                px["adj_close"].pct_change().mask(px["corp_action"])
            )
            px["prev_date"] = px.index.to_series().shift()
            raw_move = np.log(px["close_unadj"]).diff().abs()
            px["break"] = (
                px["prev_date"].notna()
                & ((px.index - px["prev_date"]).dt.days > MAX_GAP_DAYS)
            ) | ((raw_move > MAX_JUMP) & (px["splits"] == 0))
        cache[ticker, source] = px
    return cache[ticker, source]


def _extension_limits(px: pd.DataFrame, run, prev_check, next_check) -> tuple:
    """How far a run's series may be continued before first_ok and after last_ok."""

    # neighbouring held snapshot: no snapshot, no quote, or ok
    def allowed(check) -> bool:
        return check is None or not check["has_px"] or check["ok"]

    lo, hi = run.first_ok, run.last_ok
    if allowed(prev_check):
        before = px.loc[: run.first_ok]
        breaks = before.index[before["break"]]
        # a break row starts a clean piece
        lo = breaks.max() if len(breaks) else before.index.min()
        if prev_check is not None:
            lo = max(lo, prev_check["date"] + pd.Timedelta(days=1))
    if allowed(next_check):
        after = px.loc[run.last_ok :]
        breaks = after.index[after["break"] & (after.index > run.last_ok)]
        hi = (
            breaks.min() - pd.Timedelta(days=1)
            if len(breaks)
            else after.index.max()
        )
        if next_check is not None:
            hi = min(hi, next_check["date"] - pd.Timedelta(days=1))
    return lo, hi


def build() -> pd.DataFrame:
    periods = pd.read_parquet(CACHE / "company_tickers.parquet")
    checks = pd.read_parquet(CACHE / "price_checks.parquet")
    spells = pd.read_parquet(CACHE / "sp500_membership.parquet")
    today = pd.Timestamp.today().normalize()

    keys = ["company_id", "ticker", "source", "date"]
    agg = (
        checks.groupby(keys)
        .agg(ok=("ok", "any"), has_px=("market", lambda s: s.notna().any()))
        .reset_index()
        .set_index(keys)
        .sort_index()
    )
    held = checks.groupby("company_id")["date"].apply(
        lambda s: sorted(s.unique())
    )

    frames = []
    for comp, runs in periods.groupby("company_id"):
        windows = _windows(spells[spells["company_id"] == comp], today)
        if not windows:
            continue
        # per company: keeps memory low (series are rarely shared)
        cache: dict = {}
        dates = held.get(comp, [])
        pieces = []
        for run in runs.itertuples():
            px = _series(run.ticker, run.source, cache)
            if px.empty:
                continue
            prev_d = [d for d in dates if d < run.first_ok]
            next_d = [d for d in dates if d > run.last_ok]

            def check(d):
                if d is None:
                    return None
                k = (comp, run.ticker, run.source, d)
                row = (
                    agg.loc[k]
                    if k in agg.index
                    else pd.Series({"ok": False, "has_px": False})
                )
                return {
                    "date": d,
                    "ok": bool(row["ok"]),
                    "has_px": bool(row["has_px"]),
                }

            lo, hi = _extension_limits(
                px,
                run,
                check(prev_d[-1] if prev_d else None),
                check(next_d[0] if next_d else None),
            )
            part = px.loc[
                max(lo, windows[0][0]) : min(hi, windows[-1][1])
            ].copy()
            core = (part.index >= run.first_ok) & (part.index <= run.last_ok)
            part["status"] = np.where(core, "validated", "extended")
            part["priority"] = np.where(core, 0, 1)
            part["dist"] = np.where(
                core,
                0,
                np.minimum(
                    abs((part.index - run.first_ok).days),
                    abs((part.index - run.last_ok).days),
                ),
            )
            part["ticker"], part["source"] = run.ticker, run.source
            pieces.append(part)
        if not pieces:
            continue
        df = pd.concat(pieces)
        df.index.name = "date"
        # One row per day: validated beats extended, then the closest run.
        df = (
            df.reset_index()
            .sort_values(["date", "priority", "dist"])
            .drop_duplicates("date")
            .set_index("date")
        )
        inside = np.zeros(len(df), dtype=bool)
        for lo, hi in windows:
            inside |= (df.index >= lo) & (df.index <= hi)
        df = df[inside]
        if df.empty:
            continue
        # Total return: from the source series when its previous quote is the panel's previous row.
        panel_prev = df.index.to_series().shift()
        same_series = (df["ticker"] == df["ticker"].shift()) & (
            df["source"] == df["source"].shift()
        )
        switch = (
            ~same_series
            & panel_prev.notna()
            & ((df.index - panel_prev).dt.days <= MAX_GAP_DAYS)
        )
        df["ret"] = np.where(
            df["prev_date"] == panel_prev, df["ret_src"], np.nan
        )
        df.loc[switch, "ret"] = (
            df["close_unadj"] / df["close_unadj"].shift() - 1
        )
        df["company_id"] = comp
        frames.append(
            df.drop(
                columns=["ret_src", "prev_date", "break", "priority", "dist"]
            )
        )
    panel = pd.concat(frames).reset_index()
    return panel[
        [
            "date",
            "company_id",
            "ticker",
            "source",
            "status",
            *COLUMNS,
            "close_unadj",
            "ret",
            "corp_action",
        ]
    ]


def coverage_report(panel: pd.DataFrame) -> None:
    """Share of index member-days (from PANEL_START) with a price, by status and year."""
    spells = pd.read_parquet(CACHE / "sp500_membership.parquet")
    calendar = pd.DatetimeIndex(sorted(panel["date"].unique()))
    calendar = calendar[calendar >= PANEL_START]
    rows = []
    for s in spells.itertuples():
        end = (
            s.end if pd.notna(s.end) else calendar.max() + pd.Timedelta(days=1)
        )
        d = calendar[(calendar >= s.start) & (calendar < end)]
        rows.append(pd.DataFrame({"date": d, "company_id": s.company_id}))
    member_days = pd.concat(rows).drop_duplicates()
    m = member_days.merge(
        panel[["date", "company_id", "status"]],
        on=["date", "company_id"],
        how="left",
    )
    m["status"] = m["status"].fillna("missing")
    tab = pd.crosstab(m["date"].dt.year, m["status"], normalize="index")
    tab["members/day"] = (
        m.groupby(m["date"].dt.year).size()
        / m.groupby(m["date"].dt.year)["date"].nunique()
    )
    print("Index member-days with a price (share per year):")
    print(
        tab.to_string(
            formatters={
                c: "{:.1%}".format for c in tab.columns if c != "members/day"
            }
            | {"members/day": "{:.0f}".format}
        )
    )


if __name__ == "__main__":
    panel = build()
    panel.to_parquet(CACHE / "price_panel.parquet")
    print(
        f"{len(panel):,} rows, {panel['company_id'].nunique()} companies, "
        f"{panel['date'].min().date()} -> {panel['date'].max().date()}"
    )
    print(panel["status"].value_counts(normalize=True).round(3).to_dict())
    print(f"missing returns: {panel['ret'].isna().mean():.2%}")
    coverage_report(panel)
