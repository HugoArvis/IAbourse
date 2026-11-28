"""Tests of features/labels.py on synthetic prices. Run: python data_pipeline/tests/test_labels.py"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.labels import build  # noqa: E402

DAYS = pd.bdate_range("2020-01-01", periods=40)


def company(cid, growth=0.01, split_day=None, nan_ret_day=None, keep=None):
    """open grows by `growth` per day, close = open * 1.005; optional 2:1 split (raw prices halve)."""
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
    df = pd.DataFrame({"date": DAYS, "company_id": cid, "open": open_, "close": close, "ret": ret})
    return df.iloc[keep] if keep is not None else df


def members(*cids, start=DAYS[0]):
    return pd.DataFrame({"company_id": list(cids), "start": start, "end": pd.NaT})


def label(out, cid, t, h):
    row = out[(out.company_id == cid) & (out.date == DAYS[t])]
    return row[f"ret_{h}"].iloc[0] if len(row) else None


def test_open_to_open_and_split():
    out = build(pd.concat([company("A"), company("B", split_day=10)]), members("A", "B"))
    for t in (0, 5, 8, 12):
        assert np.isclose(label(out, "A", t, 5), 1.01 ** 5 - 1, atol=1e-6)
        assert np.isclose(label(out, "B", t, 5), 1.01 ** 5 - 1, atol=1e-6), t
    assert np.isclose(label(out, "A", 3, 21), 1.01 ** 21 - 1, atol=1e-6)


def test_end_of_calendar_has_no_label():
    out = build(company("A"), members("A"))
    n = len(DAYS)
    assert not np.isnan(label(out, "A", n - 7, 5))  # open(n-6) -> open(n-1)
    assert np.isnan(label(out, "A", n - 6, 5))
    assert np.isnan(label(out, "A", n - 1, 5))


def test_truncated_series_uses_last_close():
    c = company("C", keep=slice(0, 13))  # last row = day 12
    out = build(pd.concat([company("A"), c]), members("A", "C"))
    expected = c.close.iloc[12] / c.open.iloc[11] - 1
    assert np.isclose(label(out, "C", 10, 5), expected, atol=1e-6)
    assert out.loc[(out.company_id == "C") & (out.date == DAYS[10]), "trunc_5"].iloc[0]
    assert not out.loc[(out.company_id == "C") & (out.date == DAYS[5]), "trunc_5"].iloc[0]


def test_corp_action_day_drops_windows():
    out = build(pd.concat([company("A"), company("D", nan_ret_day=8)]), members("A", "D"))
    for t in range(2, 7):  # windows open(t+1) -> open(t+6) cross day 8
        assert np.isnan(label(out, "D", t, 5)), t
    assert not np.isnan(label(out, "D", 1, 5))
    assert not np.isnan(label(out, "D", 7, 5))  # starts at the open of day 8


def test_needs_price_on_next_day_and_membership():
    e = company("E", keep=[i for i in range(len(DAYS)) if i != 15])
    out = build(pd.concat([company("A"), e]), members("A", "E", start=DAYS[3]))
    assert np.isnan(label(out, "E", 14, 5))  # no price on t+1
    assert label(out, "E", 2, 5) is None     # not a member yet
    assert not np.isnan(label(out, "E", 3, 5))


def test_terciles():
    growth = [0.001 * i for i in range(9)]
    panel = pd.concat([company(f"S{i}", g) for i, g in enumerate(growth)])
    out = build(panel, members(*[f"S{i}" for i in range(9)]))
    day = out[out.date == DAYS[0]].set_index("company_id")
    assert day["y_5"].tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert np.isclose(day["ex_5"].mean(), 0, atol=1e-7)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
