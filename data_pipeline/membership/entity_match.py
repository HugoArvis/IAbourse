"""Map company names to SEC CIK (stable company identifier).

Steps: normalise names -> exact lookup in SEC cik-lookup-data (current and former
names) -> fuzzy fallback. When several CIKs share a name, keep the one that files
annual reports (10-K / 20-F / 40-F), i.e. the operating company, not a trust or
a debt vehicle with a similar name.
"""

import json
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process

from sources import sec

RAW = Path(__file__).resolve().parents[1] / "cache" / "raw"
CIK_LOOKUP = RAW / "cik-lookup-data.txt"
ANNUAL_FORMS = {"10-K", "10-K405", "10-KSB", "20-F", "40-F"}

_SUFFIXES = (
    r"\b(THE|INCORPORATED|INC|CORPORATION|CORP|COMPANY|COMPANIES|COS|CO|LTD|LIMITED|PLC|"
    r"NV|N V|SA|AG|LP|L P|LLC|REIT|NEW|DEL|DE|MD|NY|PA|OH)\b"
)
_SHARE_CLASS = r"\b(CLASS|CL|SERIES|SER)\s+[A-Z]\b|\bSHARES?\b|\bCOMMON\b|\bSTOCK\b|\bADR\b|\bORD\b"


# Common names that differ from the legal name filed with the SEC.
ALIASES = {
    "FANNIE MAE": "FEDERAL NATIONAL MORTGAGE ASSOCIATION",
    "FREDDIE MAC": "FEDERAL HOME LOAN MORTGAGE",
}


def normalize(name: str) -> str:
    s = name.upper().replace("&", " AND ").replace("'", "")
    s = re.sub(r"\([^)]*\)", " ", s)  # "(Holding Co.)"
    # "/Delaware", EDGAR state tags /DE/
    s = re.sub(r"/.*$", " ", s)
    s = re.sub(r"\bHLDGS\b", "HOLDINGS", s)
    s = re.sub(r"\bGRP\b", "GROUP", s)
    s = re.sub(_SHARE_CLASS, " ", s)
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    s = re.sub(_SUFFIXES, " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # Join runs of initials: "D R HORTON" and "DR HORTON" -> "DR HORTON".
    s = re.sub(r"\b([A-Z])\s(?=[A-Z]\b)", r"\1", s)
    return ALIASES.get(s, s)


@lru_cache(maxsize=1)
def _lookup() -> pd.DataFrame:
    rows = []
    with open(CIK_LOOKUP, encoding="latin-1") as fh:
        for line in fh:
            parts = line.rstrip("\n").rsplit(":", 2)
            if len(parts) == 3 and parts[1].isdigit():
                rows.append((parts[0], int(parts[1])))
    df = pd.DataFrame(rows, columns=["edgar_name", "cik"])
    df["key"] = df["edgar_name"].map(normalize)
    return df[df["key"] != ""]


@lru_cache(maxsize=None)
def annual_report_dates(cik: int) -> tuple[pd.Timestamp, ...]:
    """Filing dates of the entity's annual reports (empty if none)."""
    try:
        data = json.loads(
            sec.get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
        )
    except Exception:
        return ()
    pages = [data["filings"]["recent"]]
    for page in data["filings"].get("files", []):
        pages.append(
            json.loads(
                sec.get("https://data.sec.gov/submissions/" + page["name"])
            )
        )
    dates = [
        pd.Timestamp(d)
        for p in pages
        for f, d in zip(p["form"], p["filingDate"])
        if f in ANNUAL_FORMS
    ]
    return tuple(sorted(dates))


def _pick(
    cands: list[tuple[int, str]],
    name: str,
    first_seen: pd.Timestamp,
    last_seen: pd.Timestamp,
) -> int | None:
    """Among candidate CIKs, the operating company active while the name was held.

    Activity window = [first annual report - 15 months, last annual report + 15 months].
    Among active candidates, prefer the one whose full EDGAR name (legal suffix
    included: Corp vs Ltd) is closest to `name`, then the longest overlap, then the
    number of annual reports filed inside the holding period, then in total.
    """
    margin = pd.DateOffset(months=15)
    simple = _simple(name)
    sims: dict[int, float] = {}
    for cik, edgar_name in cands:
        sims[cik] = max(
            sims.get(cik, 0.0), fuzz.ratio(simple, _simple(edgar_name))
        )
    active = []
    for cik, sim in sims.items():
        dates = annual_report_dates(cik)
        if not dates:
            continue
        start, end = dates[0] - margin, dates[-1] + margin
        overlap = (min(end, last_seen) - max(start, first_seen)).days
        if overlap < 0:
            continue
        inside = sum(
            first_seen - margin <= d <= last_seen + margin for d in dates
        )
        active.append((cik, sim, overlap, inside, len(dates)))
    if not active:
        return None
    # Only a clear name difference decides: candidates within 10 points of the best are tied.
    top_sim = max(a[1] for a in active)
    tied = [a for a in active if a[1] >= top_sim - 10]
    return max(tied, key=lambda a: (a[2], a[3], a[4]))[0]


def _simple(name: str) -> str:
    """Uppercase alphanumerics only, legal suffixes kept (used to compare full names)."""
    s = re.sub(r"\([^)]*\)|/.*$", " ", name.upper().replace("&", " AND "))
    s = re.sub(r"\bCORPORATION\b", "CORP", s)
    s = re.sub(r"\bINCORPORATED\b", "INC", s)
    s = re.sub(r"\bCOMPANY\b", "CO", s)
    s = re.sub(r"\bCOMPANIES\b", "COS", s)
    s = re.sub(r"^\s*THE\b|\bTHE\s*$", " ", s)
    return re.sub(r"[^A-Z0-9]", "", s)


def match_names(
    names: pd.DataFrame, fuzzy_threshold: int = 92
) -> pd.DataFrame:
    """Map names to CIK.

    `names` has columns name, first_seen, last_seen (period the name was observed).
    Returns those columns plus cik, method ('exact' / 'fuzzy' / None), matched_key.
    """
    lk = _lookup()
    by_key = (
        lk.groupby("key")[["cik", "edgar_name"]]
        .apply(lambda g: list(zip(g["cik"], g["edgar_name"])))
        .to_dict()
    )
    keys = list(by_key)
    out = []
    for row in names.itertuples(index=False):
        key = normalize(row.name)
        cik, method, matched = None, None, None
        if key in by_key:
            cik, method, matched = (
                _pick(by_key[key], row.name, row.first_seen, row.last_seen),
                "exact",
                key,
            )
        if cik is None and key:
            for cand, score, _ in process.extract(
                key, keys, scorer=fuzz.token_sort_ratio, limit=5
            ):
                if score < fuzzy_threshold:
                    break
                cik = _pick(
                    by_key[cand], row.name, row.first_seen, row.last_seen
                )
                if cik is not None:
                    method, matched = "fuzzy", cand
                    break
        out.append(
            {
                **row._asdict(),
                "key": key,
                "cik": cik,
                "method": method if cik is not None else None,
                "matched_key": matched,
            }
        )
    return pd.DataFrame(out)
