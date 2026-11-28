"""Vanguard 500 Index Fund point-in-time holdings, from SEC filings.

The fund fully replicates the S&P 500, so its common-stock holdings are
an official, independent snapshot of index membership (and weights).

  - N-Q (2004-2018): HTML schedule of investments, names only.
  - NPORT-P (2019+): structured XML with CUSIP / ISIN / LEI.

Outputs:
  cache/vanguard500_snapshots.parquet  one row per date and holding
  cache/vanguard500_name_cik.parquet   holding name -> CIK
Usage: python data_pipeline/membership/vanguard_500.py
"""

import re
import sys
import xml.etree.ElementTree as ET
from io import StringIO
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sources import sec  # noqa: E402

VANGUARD_INDEX_FUNDS_CIK = 36405
# from https://www.sec.gov/files/company_tickers_mf.json (VFIAX, VOO)
SERIES_500_INDEX = "S000002839"
FUND_HEADER = re.compile(r"Vanguard\s+500\s+Index\s+Fund", re.I)
# Any other fund's schedule header ends the 500 Index section.
NEXT_FUND = re.compile(
    r"Vanguard\s+(?!500\b)[A-Z][\w\-\s&.]{2,60}?(Index\s+)?Fund\s*</", re.I
)


def _to_number(value) -> float | None:
    if pd.isna(value):
        return None
    s = re.sub(r"[$,\s]", "", str(value))
    s = s.replace("(", "-").replace(")", "")
    try:
        return float(s)
    except ValueError:
        return None


def _section(html: str) -> str:
    """Slice of the document with the fund's schedule of investments."""
    for m in FUND_HEADER.finditer(html):
        window = re.sub(
            r"<[^>]+>|&nbsp;|&#160;", " ", html[m.end() : m.end() + 4000]
        )
        if re.search(r"Shares", window) and re.search(
            r"Market\s+Value", window
        ):
            start = m.start()
            end_match = re.search(r"TOTAL\s+INVESTMENTS", html[start:], re.I)
            end = start + end_match.end() + 2000 if end_match else len(html)
            return html[max(0, start - 3000) : end]
    raise ValueError("500 Index Fund schedule not found")


def parse_nq(html: str) -> pd.DataFrame:
    """Common-stock rows of the 500 Index Fund.

    Columns: name, shares, market value in $000.
    """
    rows = []
    for table in pd.read_html(StringIO(_section(html))):
        for rec in table.itertuples(index=False):
            cells = [
                str(c).strip()
                for c in rec
                if not (isinstance(c, float) and pd.isna(c))
            ]
            # Name = first cell with real words; footnote markers (*, ^,
            # 1,2) may sit in their own cells.
            idx = next(
                (
                    i
                    for i, c in enumerate(cells)
                    if re.search(r"[A-Za-z]{2,}", c)
                ),
                None,
            )
            if idx is None:
                continue
            name = cells[idx]
            if re.match(r"\W*total\s+common\s+stocks", name, re.I):
                return _clean(rows)  # common stocks section is over
            nums = [
                n
                for n in (_to_number(c) for c in cells[idx + 1 :])
                if n is not None
            ]
            if len(nums) < 2 or re.match(r"\W*(total|\(cost)", name, re.I):
                continue
            # Merged cells can be duplicated by read_html: shares first,
            # market value last.
            rows.append({"name": name, "shares": nums[0], "value_k": nums[-1]})
    return _clean(rows)


def _clean(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    # leading markers (keep "3M")
    df["name"] = (
        df["name"]
        .str.replace(r"^[^A-Za-z0-9]+", "", regex=True)
        # trailing footnote
        .str.replace(r"\s*\(\d\)\s*$", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return df.drop_duplicates("name").reset_index(drop=True)


def nq_snapshots() -> pd.DataFrame:
    """All N-Q snapshots: one row per (date, holding)."""
    filings = sec.filings(VANGUARD_INDEX_FUNDS_CIK)
    frames = []
    for _, f in filings[filings["form"] == "N-Q"].iterrows():
        html = sec.get(
            sec.archive_url(
                VANGUARD_INDEX_FUNDS_CIK, f.accessionNumber, f.primaryDocument
            )
        )
        if not FUND_HEADER.search(html):
            continue
        try:
            df = parse_nq(html)
        except ValueError:
            continue
        df.insert(0, "date", f.reportDate)
        df["source"] = "N-Q"
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _series_accessions(series_id: str, form: str) -> list[str]:
    """Accession numbers of a fund series' filings (EDGAR atom feed)."""
    url = (
        "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"
        f"&CIK={series_id}&type={form}&dateb=&owner=include"
        "&count=100&output=atom"
    )
    return re.findall(r"<accession-number>([\d-]+)<", sec.get(url))


def parse_nport(xml: str) -> pd.DataFrame:
    """Common-stock holdings from an NPORT-P primary_doc.xml."""
    ns = {"n": "http://www.sec.gov/edgar/nport"}
    root = ET.fromstring(xml)
    date = pd.Timestamp(root.findtext(".//n:repPdDate", namespaces=ns))
    rows = []
    for sec_el in root.iterfind(".//n:invstOrSec", ns):
        if sec_el.findtext("n:assetCat", namespaces=ns) != "EC":
            continue
        isin = sec_el.find("n:identifiers/n:isin", ns)
        rows.append(
            {
                "date": date,
                "name": sec_el.findtext("n:name", namespaces=ns),
                "title": sec_el.findtext("n:title", namespaces=ns),
                "cusip": sec_el.findtext("n:cusip", namespaces=ns),
                "isin": isin.get("value") if isin is not None else None,
                "lei": sec_el.findtext("n:lei", namespaces=ns),
                "shares": float(sec_el.findtext("n:balance", namespaces=ns)),
                "value_k": (
                    float(sec_el.findtext("n:valUSD", namespaces=ns)) / 1000
                ),
            }
        )
    return pd.DataFrame(rows)


def nport_snapshots() -> pd.DataFrame:
    frames = []
    for acc in _series_accessions(SERIES_500_INDEX, "NPORT-P"):
        xml = sec.get(
            sec.archive_url(VANGUARD_INDEX_FUNDS_CIK, acc, "primary_doc.xml")
        )
        df = parse_nport(xml)
        df["source"] = "NPORT-P"
        frames.append(df)
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["date", "name"])
        .reset_index(drop=True)
    )


def snapshots(min_weight: float = 1e-5) -> pd.DataFrame:
    """All snapshots, with index weight.

    Drops residual positions (spin-off leftovers, warrants, just-removed
    names) whose weight is below `min_weight`.
    """
    df = pd.concat([nq_snapshots(), nport_snapshots()], ignore_index=True)
    df = df[df["value_k"] > 0]
    df["weight"] = df["value_k"] / df.groupby("date")["value_k"].transform(
        "sum"
    )
    return df[df["weight"] >= min_weight].reset_index(drop=True)


if __name__ == "__main__":
    from membership.entity_match import match_names

    snaps = snapshots()
    snaps.to_parquet(ROOT / "cache" / "vanguard500_snapshots.parquet")
    names = (
        snaps.groupby("name")["date"]
        .agg(first_seen="min", last_seen="max")
        .reset_index()
    )
    matched = match_names(names)
    matched.to_parquet(ROOT / "cache" / "vanguard500_name_cik.parquet")
    print(
        f"{snaps['date'].nunique()} snapshots, {len(names)} names, "
        f"{matched['cik'].notna().mean():.1%} matched to a CIK"
    )
