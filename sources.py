"""
Structured opportunity fetchers. Each returns (items, raw_count), or (None, 0) when the
source isn't configured. Every item is a normalized dict:

  {title, agency, category, notice_type, reference, url, source, posted, deadline,
   set_aside, naics, summary}

category is one of: contract | sbir | crada | small_business

Discovery happens here and in deep search; the language model only scores.
"""

import html as _html
import json
import os
import re
import ssl
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone

try:
    import certifi
    SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CTX = ssl.create_default_context()

TIMEOUT = 40
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/126 Safari/537.36"

_errors: list[str] = []


def pop_errors() -> list[str]:
    errs, _errors[:] = _errors[:], []
    return errs


def _request(url: str, payload: dict | None = None) -> str:
    headers = {"User-Agent": UA, "Accept": "application/json"}
    data = None
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT, context=SSL_CTX) as r:
            return r.read().decode("utf8", "ignore")
    except urllib.error.HTTPError as e:
        # never include the URL: SAM.gov URLs carry the API key
        _errors.append(f"HTTP {e.code}: {e.read().decode('utf8', 'ignore')[:200]}")
        raise
    except Exception as e:
        _errors.append(type(e).__name__)
        raise


def _json(url: str, payload: dict | None = None):
    return json.loads(_request(url, payload))


def clean(s) -> str:
    t = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", str(s or ""))
    t = _html.unescape(re.sub(r"<[^>]+>", " ", t))
    return re.sub(r"\s+", " ", t).strip()


def today() -> date:
    return datetime.now(timezone.utc).date()


def not_expired(deadline: str | None) -> bool:
    return not deadline or deadline >= today().isoformat()


def keyword_regex(profile: dict) -> re.Pattern | None:
    words = [re.escape(k.strip()) for k in profile.get("keywords", []) if k.strip()]
    return re.compile(r"(?i)\b(" + "|".join(words) + r")\b") if words else None


# ---------------------------------------------------------------- SAM.gov
# https://open.gsa.gov/api/get-opportunities-public-api/
# Daily request limits depend on the account's role and can be very low, so one broad
# search per run is filtered locally rather than querying per NAICS code or keyword.
SAM_URL = "https://api.sam.gov/opportunities/v2/search"
SAM_NOTICE_TYPES = "o,k,p,r,s"  # solicitation, combined, presolicitation, sources sought, special

# set-aside code → the certification a company must hold (SBA/SBP need only small status)
SET_ASIDE_REQUIRES = {
    "8A": "8a", "8AN": "8a", "HZC": "hubzone", "HZS": "hubzone",
    "SDVOSBC": "sdvosb", "SDVOSBS": "sdvosb", "WOSB": "wosb", "WOSBSS": "wosb",
    "EDWOSB": "edwosb", "EDWOSBSS": "edwosb", "VSA": "vosb", "VSS": "vosb",
    "LAS": "local", "IEE": "iee", "ISBEE": "isbee", "BICiv": "bia", "IHE": "ihe",
}
SMALL_BUSINESS_CODES = {"SBA", "SBP"} | set(SET_ASIDE_REQUIRES) - {"LAS", "IEE", "ISBEE", "BICiv", "IHE"}
CRADA_RE = re.compile(r"(?i)\b(crada|cooperative research (and|&) development agreement)\b")
SBIR_RE = re.compile(r"(?i)\b(sbir|sttr|small business innovation research|small business technology transfer)\b")


class SamRateLimited(Exception):
    pass


def set_aside_eligible(code: str | None, certifications: list[str]) -> bool:
    need = SET_ASIDE_REQUIRES.get(code or "")
    return need is None or need in {c.strip().lower() for c in certifications}


def _sam_search(key: str, params: dict) -> dict:
    query = urllib.parse.urlencode({"api_key": key, **params})
    try:
        return _json(f"{SAM_URL}?{query}")
    except urllib.error.HTTPError as e:
        if e.code == 429:
            raise SamRateLimited() from e
        raise


def _sam_item(n: dict, category: str | None = None) -> dict:
    title = clean(n.get("title"))
    code = n.get("typeOfSetAside") or None
    if category is None:
        if CRADA_RE.search(title):
            category = "crada"
        elif SBIR_RE.search(title):
            category = "sbir"
        elif code in SMALL_BUSINESS_CODES:
            category = "small_business"
        else:
            category = "contract"
    notice_id = n.get("noticeId")
    return {
        "title": title,
        "agency": clean(n.get("fullParentPathName")).replace(".", " › ") or None,
        "category": category,
        "notice_type": n.get("type") or n.get("baseType"),
        "reference": n.get("solicitationNumber") or notice_id,
        "url": n.get("uiLink") or (f"https://sam.gov/opp/{notice_id}/view" if notice_id else None),
        "source": "SAM.gov",
        "posted": (n.get("postedDate") or "")[:10] or None,
        "deadline": (n.get("responseDeadLine") or "")[:10] or None,
        "set_aside": n.get("typeOfSetAsideDescription") or code,
        "naics": n.get("naicsCode"),
        "summary": None,
        "_description_url": n.get("description"),
        "_set_aside_code": code,
    }


def sam_item_matches(item: dict, profile: dict, kw: re.Pattern | None) -> bool:
    if not not_expired(item["deadline"]):
        return False
    if not set_aside_eligible(item["_set_aside_code"], profile.get("certifications", [])):
        return False
    return bool(item["naics"] in set(profile.get("naics_codes", []))
                or (kw and kw.search(item["title"])))


def fetch_sam(profile: dict):
    key = os.getenv("SAM_API_KEY")
    if not key:
        return None, 0
    end = today()
    start = end - timedelta(days=int(os.getenv("SAM_LOOKBACK_DAYS", "30")))
    fmt = "%m/%d/%Y"
    kw = keyword_regex(profile)
    out, raw = [], 0
    try:
        for page in range(int(os.getenv("SAM_MAX_PAGES", "2"))):
            data = _sam_search(key, {"postedFrom": start.strftime(fmt), "postedTo": end.strftime(fmt),
                                     "ptype": SAM_NOTICE_TYPES, "limit": 1000, "offset": page})
            notices = data.get("opportunitiesData") or []
            raw += len(notices)
            for n in notices:
                if n.get("active", "Yes") != "Yes":
                    continue
                item = _sam_item(n)
                if sam_item_matches(item, profile, kw):
                    out.append(item)
            if len(notices) < 1000:
                break
        # CRADA partner announcements, regardless of NAICS (rarely tagged usefully)
        crada_start = end - timedelta(days=min(364, int(os.getenv("CRADA_LOOKBACK_DAYS", "365"))))
        for term in ("CRADA", "Cooperative Research and Development Agreement"):
            data = _sam_search(key, {"postedFrom": crada_start.strftime(fmt), "postedTo": end.strftime(fmt),
                                     "title": term, "limit": 200, "offset": 0})
            notices = data.get("opportunitiesData") or []
            raw += len(notices)
            out += [i for i in (_sam_item(n, "crada") for n in notices if n.get("active", "Yes") == "Yes")
                    if not_expired(i["deadline"])]
    except SamRateLimited:
        _errors.append("SAM.gov daily request limit reached — keeping results fetched so far")
    _attach_sam_descriptions(out, key)
    return out, raw


def _attach_sam_descriptions(items: list[dict], key: str):
    limit = int(os.getenv("SAM_MAX_DESCRIPTIONS", "0"))
    todo = [i for i in items if i.get("_description_url")][:limit]
    for i in todo:
        sep = "&" if "?" in i["_description_url"] else "?"
        try:
            body = _request(f"{i['_description_url']}{sep}api_key={key}")
        except urllib.error.HTTPError as e:
            if e.code == 429:
                _errors.append("SAM.gov daily request limit reached while reading descriptions")
                return
            continue
        except Exception:
            continue
        try:
            body = json.loads(body).get("description", "")
        except (ValueError, AttributeError):
            pass
        i["summary"] = clean(body)[:2500] or None


# ---------------------------------------------------------------- DoD SBIR/STTR (DSIP)
DSIP_BASE = "https://www.dodsbirsttr.mil/topics/api/public/topics"
DSIP_PAGE = "https://www.dodsbirsttr.mil/topics-app/"
DSIP_OPEN_STATUSES = [591, 592]  # pre-release and open


def _epoch_day(ms) -> str | None:
    if not ms:
        return None
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).date().isoformat()


def _dsip_details(topic: dict) -> dict:
    try:
        return _json(f"{DSIP_BASE}/{topic['topicId']}/details")
    except Exception:
        return {}


def fetch_dsip(profile: dict):
    param = urllib.parse.quote(json.dumps({"topicReleaseStatus": DSIP_OPEN_STATUSES}, separators=(",", ":")))
    topics, page = [], 0
    while True:
        try:
            data = _json(f"{DSIP_BASE}/search?searchParam={param}&size=100&page={page}")
        except Exception:
            break
        batch = data.get("data") or []
        topics += batch
        page += 1
        if not batch or len(topics) >= (data.get("total") or 0) or page >= 10:
            break
    with ThreadPoolExecutor(max_workers=6) as ex:
        details = list(ex.map(_dsip_details, topics))
    kw = keyword_regex(profile)
    out = []
    for t, d in zip(topics, details):
        deadline = _epoch_day(t.get("topicEndDate"))
        if not not_expired(deadline):
            continue
        body = " ".join(clean(d.get(k)) for k in ("objective", "description"))
        terms = clean(d.get("keywords"))
        title = clean(t.get("topicTitle"))
        if kw and not kw.search(f"{title} {terms} {body}"):
            continue
        flags = []
        if d.get("itar"):
            flags.append("ITAR-restricted")
        if t.get("cmmcLevel"):
            flags.append(f"CMMC {t['cmmcLevel']}")
        prefix = f"[{'; '.join(flags)}] " if flags else ""
        out.append({
            "title": title, "agency": f"DoD — {t.get('component')}", "category": "sbir",
            "notice_type": f"{t.get('program')} topic ({t.get('topicStatus')})",
            "reference": t.get("topicCode"), "url": DSIP_PAGE, "source": "DoD SBIR/STTR",
            "posted": _epoch_day(t.get("topicStartDate")), "deadline": deadline,
            "set_aside": None, "naics": None, "summary": (prefix + body)[:2500] or None,
        })
    return out, len(topics)


# ---------------------------------------------------------------- Grants.gov (civilian SBIR/STTR)
GRANTS_SEARCH = "https://api.grants.gov/v1/api/search2"
GRANTS_FETCH = "https://api.grants.gov/v1/api/fetchOpportunity"


def _grants_date(s: str | None) -> str | None:
    try:
        return datetime.strptime(s, "%m/%d/%Y").date().isoformat()
    except (TypeError, ValueError):
        return None


def _grants_description(hit: dict) -> str | None:
    try:
        d = _json(GRANTS_FETCH, {"opportunityId": int(hit["id"])}).get("data", {})
    except Exception:
        return None
    text = (d.get("synopsis") or {}).get("synopsisDesc") or (d.get("forecast") or {}).get("forecastDesc")
    return clean(text)[:2500] or None


def fetch_grants(profile: dict):
    hits, seen, raw = [], set(), 0
    for term in ("SBIR", "STTR", "Small Business Innovation Research"):
        start = 0
        while start < 500:
            try:
                data = _json(GRANTS_SEARCH, {"keyword": term, "oppStatuses": "forecasted|posted",
                                             "rows": 100, "startRecordNum": start}).get("data", {})
            except Exception:
                break
            batch = data.get("oppHits") or []
            raw += len(batch)
            for h in batch:
                if h["id"] not in seen and SBIR_RE.search(f"{h.get('title')} {h.get('number')}"):
                    seen.add(h["id"])
                    hits.append(h)
            start += 100
            if len(batch) < 100:
                break
    hits = [h for h in hits if not_expired(_grants_date(h.get("closeDate")))]
    with ThreadPoolExecutor(max_workers=6) as ex:
        descriptions = list(ex.map(_grants_description, hits))
    return [{
        "title": clean(h.get("title")), "agency": h.get("agency"), "category": "sbir",
        "notice_type": f"Grant ({h.get('oppStatus')})", "reference": h.get("number"),
        "url": f"https://www.grants.gov/search-results-detail/{h['id']}", "source": "Grants.gov",
        "posted": _grants_date(h.get("openDate")), "deadline": _grants_date(h.get("closeDate")),
        "set_aside": None, "naics": None, "summary": d,
    } for h, d in zip(hits, descriptions)], raw


# ---------------------------------------------------------------- Federal Register (CRADA notices)
FR_URL = "https://www.federalregister.gov/api/v1/documents.json"


def fetch_federal_register(profile: dict):
    since = today() - timedelta(days=int(os.getenv("CRADA_LOOKBACK_DAYS", "365")))
    params = [("per_page", 100), ("order", "newest"),
              ("conditions[term]", '"cooperative research and development agreement"'),
              ("conditions[type][]", "NOTICE"),
              ("conditions[publication_date][gte]", since.isoformat())]
    params += [("fields[]", f) for f in ("title", "abstract", "agency_names", "html_url",
                                          "publication_date", "document_number")]
    data = _json(f"{FR_URL}?{urllib.parse.urlencode(params)}")
    results = data.get("results") or []
    out = []
    for r in results:
        title, abstract = clean(r.get("title")), clean(r.get("abstract"))
        if not CRADA_RE.search(f"{title} {abstract}"):
            continue  # merely cites the phrase in passing somewhere in the full text
        out.append({
            "title": title, "agency": " › ".join(r.get("agency_names") or []) or None, "category": "crada",
            "notice_type": "Federal Register notice", "reference": r.get("document_number"),
            "url": r.get("html_url"), "source": "Federal Register", "posted": r.get("publication_date"),
            "deadline": None, "set_aside": None, "naics": None, "summary": abstract[:2500] or None,
        })
    return out, len(results)


ALL_FETCHERS = [
    ("SAM.gov", fetch_sam),
    ("DoD SBIR/STTR", fetch_dsip),
    ("Grants.gov", fetch_grants),
    ("Federal Register", fetch_federal_register),
]


# ---------------------------------------------------------------- page text & link checks
def fetch_page_text(url: str, limit: int = 2500) -> str | None:
    """Best-effort text of a web page (used for deep-search finds with thin summaries)."""
    if not url:
        return None
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=20, context=SSL_CTX) as r:
            page = r.read().decode("utf8", "ignore")
    except Exception:
        return None
    main = re.search(r"(?is)<main[^>]*>(.*?)</main>", page)
    text = clean(main.group(1) if main else page)
    return text[:limit] if len(text) >= 300 else None
