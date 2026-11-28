"""Tests of features/labels.py on synthetic prices.

Run: python data_pipeline/tests/test_labels.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.labels import build  # noqa: E402

DAYS = pd.bdate_range("2020-01-01", periods=40)
TOL = 1e-6


def company(cid, growth=0.01, split_day=None, nan_ret_day=None, keep=None):
    """Synthetic prices: open grows daily, close = open * 1.005.

    growth: daily growth of the open.
    split_day: 2:1 split on that day (raw prices halve).
    nan_ret_day: corporate-action day (ret NaN).
    keep: rows to keep (to cut the series or leave holes).
    """
    t = np.arange(len(DAYS))
    open_ = 100 * (1 + growth) ** t
    if split_day is not None:
        open_ = np.where(t >= split_day, open_ / 2, open_)
    close = open_ * 1.005
    ret = pd.Series(close).pct_change().to_numpy().copy()
    if split_day is not None:
        ret[split_day] = growth  # total return is unaffected by the split
    if nan_ret_day is not None:
        ret[nan_ret_day] = np.nan
    df = pd.DataFrame(
        {
            "date": DAYS,
            "company_id": cid,
            "open": open_,
            "close": close,
            "ret": ret,
        }
    )
    return df.iloc[keep] if keep is not None else df


def members(*cids, start=DAYS[0]):
    return pd.DataFrame(
        {"company_id": list(cids), "start": start, "end": pd.NaT}
    )


def row(out, cid, t):
    return out[(out["company_id"] == cid) & (out["date"] == DAYS[t])]


def label(out, cid, t, h):
    found = row(out, cid, t)
    return found[f"ret_{h}"].iloc[0] if len(found) else None


def test_open_to_open_and_split():
    panel = pd.concat([company("A"), company("B", split_day=10)])
    out = build(panel, members("A", "B"))
    for t in (0, 5, 8, 12):
        assert np.isclose(label(out, "A", t, 5), 1.01**5 - 1, atol=TOL)
        assert np.isclose(label(out, "B", t, 5), 1.01**5 - 1, atol=TOL)
    assert np.isclose(label(out, "A", 3, 21), 1.01**21 - 1, atol=TOL)


def test_end_of_calendar_has_no_label():
    out = build(company("A"), members("A"))
    n = len(DAYS)
    assert not np.isnan(label(out, "A", n - 7, 5))  # open n-6 -> open n-1
    assert np.isnan(label(out, "A", n - 6, 5))
    assert np.isnan(label(out, "A", n - 1, 5))


def test_truncated_series_uses_last_close():
    c = company("C", keep=slice(0, 13))  # last row = day 12
    out = build(pd.concat([company("A"), c]), members("A", "C"))
    expected = c["close"].iloc[12] / c["open"].iloc[11] - 1
    assert np.isclose(label(out, "C", 10, 5), expected, atol=TOL)
    assert row(out, "C", 10)["trunc_5"].iloc[0]
    assert not row(out, "C", 5)["trunc_5"].iloc[0]


def test_corp_action_day_drops_windows():
    panel = pd.concat([company("A"), company("D", nan_ret_day=8)])
    out = build(panel, members("A", "D"))
    for t in range(2, 7):  # windows open t+1 -> open t+6 cross day 8
        assert np.isnan(label(out, "D", t, 5)), t
    assert not np.isnan(label(out, "D", 1, 5))
    assert not np.isnan(label(out, "D", 7, 5))  # starts at the open of 8


def test_needs_price_on_next_day_and_membership():
    e = company("E", keep=[i for i in range(len(DAYS)) if i != 15])
    panel = pd.concat([company("A"), e])
    out = build(panel, members("A", "E", start=DAYS[3]))
    assert np.isnan(label(out, "E", 14, 5))  # no price on t+1
    assert label(out, "E", 2, 5) is None  # not a member yet
    assert not np.isnan(label(out, "E", 3, 5))


def test_start_date_cutoff():
    out = build(company("A"), members("A"), start=DAYS[10])
    assert out["date"].min() == DAYS[10]


def test_terciles():
    names = [f"S{i}" for i in range(9)]
    panel = pd.concat([company(c, 0.001 * i) for i, c in enumerate(names)])
    out = build(panel, members(*names))
    day = out[out["date"] == DAYS[0]].set_index("company_id")
    assert day["y_5"].tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert np.isclose(day["ex_5"].mean(), 0, atol=1e-7)


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
