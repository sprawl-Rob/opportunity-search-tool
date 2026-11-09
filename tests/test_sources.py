"""Offline tests for parsing and filtering. Run: python -m pytest tests  (or python tests/test_sources.py)"""
import json
import os
import sys
import urllib.parse
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import sources as S

FUTURE = (S.today() + timedelta(days=20)).isoformat()
PAST = (S.today() - timedelta(days=2)).isoformat()
PROFILE = {"naics_codes": ["541511"], "keywords": ["software development"], "certifications": []}


def notice(**kw):
    """Shape taken from the example response in SAM.gov's Get Opportunities API docs."""
    base = {"noticeId": "abc123", "title": "Case Management System", "solicitationNumber": "W91-26-R-0001",
            "fullParentPathName": "DEPT OF DEFENSE.DEPT OF THE ARMY", "postedDate": "2026-11-01",
            "type": "Solicitation", "baseType": "Solicitation", "typeOfSetAside": None,
            "typeOfSetAsideDescription": None, "responseDeadLine": f"{FUTURE}T17:00:00-05:00",
            "naicsCode": "541511", "classificationCode": "DA01", "active": "Yes",
            "description": "https://api.sam.gov/prod/opportunities/v1/noticedesc?noticeid=abc123",
            "uiLink": "https://sam.gov/opp/abc123/view"}
    return {**base, **kw}


def run_sam(notices):
    calls = []

    def fake(url, payload=None):
        q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        calls.append(q)
        return {"opportunitiesData": notices if "title" not in q else []}
    original, S._json = S._json, fake
    os.environ["SAM_API_KEY"] = "test"
    try:
        out, raw = S.fetch_sam(PROFILE)
    finally:
        S._json = original
    return out, raw, calls


def test_normalizes_notice():
    out, raw, _ = run_sam([notice()])
    assert raw == 1 and len(out) == 1
    i = out[0]
    assert i["category"] == "contract" and i["deadline"] == FUTURE and i["reference"] == "W91-26-R-0001"
    assert i["agency"] == "DEPT OF DEFENSE › DEPT OF THE ARMY"


def test_small_business_set_aside_category():
    out, _, _ = run_sam([notice(typeOfSetAside="SBA", typeOfSetAsideDescription="Total Small Business")])
    assert out[0]["category"] == "small_business"


def test_hides_set_asides_company_is_not_eligible_for():
    out, _, _ = run_sam([notice(typeOfSetAside="8A"), notice(typeOfSetAside="SDVOSBC")])
    assert out == []


def test_shows_set_aside_when_certified():
    assert S.set_aside_eligible("WOSB", ["WOSB"]) and not S.set_aside_eligible("WOSB", ["8a"])


def test_filters_expired_inactive_and_off_target():
    out, _, _ = run_sam([notice(responseDeadLine=f"{PAST}T17:00:00-05:00"), notice(active="No"),
                         notice(naicsCode="236220", title="Roof repair")])
    assert out == []


def test_keyword_in_title_matches_other_naics():
    out, _, _ = run_sam([notice(naicsCode="541330", title="Software Development Support")])
    assert len(out) == 1


def test_crada_and_sbir_titles_classified():
    out, _, _ = run_sam([notice(title="CRADA opportunity for data platforms"), notice(title="SBIR Phase II software")])
    assert [i["category"] for i in out] == ["crada", "sbir"]


def test_one_broad_call_per_run_conserves_rate_limit():
    _, _, calls = run_sam([notice()])
    assert len(calls) == 3  # one broad search + two CRADA title searches


def test_no_key_means_not_configured():
    os.environ.pop("SAM_API_KEY", None)
    assert S.fetch_sam(PROFILE) == (None, 0)


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"{len(tests)} passed")
