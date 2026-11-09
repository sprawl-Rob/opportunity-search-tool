"""
Opportunity Search Tool — FastAPI backend.

Finds federal contracts, SBIR/STTR topics, CRADA opportunities, and small-business
procurements for one company. Structured government sources and multi-provider AI deep
search do the discovery; a language model scores each opportunity against the company
profile; SQLite tracks the capture pipeline.
"""

import hashlib
import json
import os
import re
import shutil
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import sources as S

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

DB_PATH = BASE_DIR / "opportunities.db"
PROFILE_PATH = BASE_DIR / "profile.json"
PROFILE_DEFAULT_PATH = BASE_DIR / "profile.default.json"
SETTINGS_PATH = BASE_DIR / "settings.json"
DIRECTORY_PATH = BASE_DIR / "crada_directory.json"

MAX_SEARCHES = int(os.getenv("MAX_WEB_SEARCHES", "15"))
MAX_FETCHES = int(os.getenv("MAX_PAGE_FETCHES", "12"))
GEMINI_MAX_SEARCHES = int(os.getenv("GEMINI_MAX_SEARCHES", "15"))
GEMINI_API_REVISION = "2026-05-20"
DEEP_SEARCH_TIMEOUT = int(os.getenv("DEEP_SEARCH_TIMEOUT", "900"))  # seconds, any provider

CATEGORIES = {"contract", "sbir", "crada", "small_business"}
ACTIVE = {"new", "tracking", "capture", "proposal"}
VALID_STATUSES = ACTIVE | {"submitted", "won", "lost", "no_bid", "closed"}

# ---------------------------------------------------------------- settings & providers
PROVIDER_KEYS = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "google": "GOOGLE_API_KEY"}
DEFAULT_MODELS = {  # used when the saved model's provider has no key
    "search": {"google": "gemini-4-argon", "openai": "gpt-6.1-sol", "anthropic": "claude-sonnet-5-5"},
    "scoring": {"anthropic": "claude-sonnet-5-5", "openai": "gpt-6.1-sol", "google": "gemini-4-argon"},
}
# Anthropic models that accept server-side fallback when a safety classifier declines
FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}


def provider_of(model: str) -> str:
    m = (model or "").lower()
    if m.startswith(("gpt", "o1", "o3", "o4", "chatgpt")):
        return "openai"
    if m.startswith("gemini"):
        return "google"
    return "anthropic"


def configured(provider: str) -> bool:
    return bool(os.getenv(PROVIDER_KEYS[provider]))


def _default_model(task: str) -> str | None:
    return next((m for p, m in DEFAULT_MODELS[task].items() if configured(p)), None)


def load_settings() -> dict:
    saved = {}
    if SETTINGS_PATH.exists():
        try:
            saved = json.loads(SETTINGS_PATH.read_text())
        except ValueError:
            saved = {}
    out = {}
    for task, key in (("search", "search_model"), ("scoring", "scoring_model")):
        model = saved.get(key)
        out[key] = model if model and configured(provider_of(model)) else _default_model(task)
    return out


def load_profile() -> dict:
    if not PROFILE_PATH.exists():
        shutil.copy(PROFILE_DEFAULT_PATH, PROFILE_PATH)
    return json.loads(PROFILE_PATH.read_text())


def _post_json(url: str, payload: dict, headers: dict, timeout: int = 120) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout, context=S.SSL_CTX) as r:
        return json.load(r)


def _get_json(url: str, headers: dict) -> dict:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=20, context=S.SSL_CTX) as r:
        return json.load(r)


def _strictify(schema: dict) -> dict:
    """Strict JSON-schema modes require additionalProperties:false and every key required."""
    s = json.loads(json.dumps(schema))

    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node["properties"].keys())
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
    walk(s)
    return s


def llm_structured(model: str, system: str, user: str, schema: dict, max_tokens: int = 8000) -> dict:
    """Provider-agnostic structured call. Returns the parsed object per `schema`."""
    p = provider_of(model)
    if p == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        kw = dict(model=model, max_tokens=max(max_tokens, 16000),
                  system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                  messages=[{"role": "user", "content": user}],
                  output_config={"format": {"type": "json_schema", "schema": _strictify(schema)}})
        if model in FALLBACK_MODELS:
            resp = client.beta.messages.create(betas=["server-side-fallback-2026-07-01"],
                                               extra_body={"fallbacks": "default"}, **kw)
        else:
            resp = client.messages.create(**kw)
        if resp.stop_reason == "refusal":
            raise RuntimeError("model declined the request (refusal)")
        if resp.stop_reason == "max_tokens":
            raise RuntimeError("output hit max_tokens before the JSON was complete")
        return json.loads("".join(b.text for b in resp.content if b.type == "text"))
    if p == "openai":
        out = _post_json("https://api.openai.com/v1/chat/completions",
                         {"model": model,
                          "messages": [{"role": "system", "content": system},
                                       {"role": "user", "content": user}],
                          "max_completion_tokens": max(max_tokens, 16000),
                          "response_format": {"type": "json_schema", "json_schema": {
                              "name": "emit", "strict": True, "schema": _strictify(schema)}}},
                         {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"}, timeout=300)
        return json.loads(out["choices"][0]["message"]["content"])
    out = _post_json(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        {"system_instruction": {"parts": [{"text": system}]},
         "contents": [{"role": "user", "parts": [{"text": user}]}],
         # thinking tokens count against this cap on Gemini 3.x and later
         "generationConfig": {"responseMimeType": "application/json", "responseSchema": schema,
                              "maxOutputTokens": max(max_tokens, 32000)}},
        {"x-goog-api-key": os.getenv("GOOGLE_API_KEY")}, timeout=300)
    return json.loads(out["candidates"][0]["content"]["parts"][0]["text"])


_model_cache = {"at": 0.0, "models": None}


def _openai_usable(mid: str) -> bool:
    if re.search(r"-\d{4}-\d{2}-\d{2}$|-\d{4}$", mid) or mid.startswith(("gpt-3.5", "gpt-4-", "gpt-4o")):
        return False
    return not any(x in mid for x in ("audio", "realtime", "image", "tts", "transcribe", "embed",
                                       "moderation", "codex", "search", "deep-research",
                                       "chat-latest", "instruct", "live"))


def list_provider_models(refresh: bool = False) -> dict:
    """Live model lists per configured provider; cached 1h (refresh=True bypasses)."""
    if not refresh and _model_cache["models"] and time.time() - _model_cache["at"] < 3600:
        return _model_cache["models"]
    out = {"anthropic": [], "openai": [], "google": []}
    if configured("anthropic"):
        try:
            d = _get_json("https://api.anthropic.com/v1/models?limit=50",
                          {"x-api-key": os.getenv("ANTHROPIC_API_KEY"), "anthropic-version": "2023-06-01"})
            out["anthropic"] = sorted(m["id"] for m in d.get("data", []) if m["id"].startswith("claude"))
        except Exception:
            out["anthropic"] = [DEFAULT_MODELS["scoring"]["anthropic"]]
    if configured("openai"):
        try:
            d = _get_json("https://api.openai.com/v1/models",
                          {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"})
            out["openai"] = sorted(m["id"] for m in d.get("data", [])
                                   if m["id"].startswith(("gpt", "o1", "o3", "o4")) and _openai_usable(m["id"]))
        except Exception:
            pass
    if configured("google"):
        try:
            d = _get_json("https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000",
                          {"x-goog-api-key": os.getenv("GOOGLE_API_KEY")})
            names = (m["name"].removeprefix("models/") for m in d.get("models", [])
                     if "generateContent" in m.get("supportedGenerationMethods", []))
            out["google"] = sorted(n for n in names if n.startswith("gemini") and not any(
                x in n for x in ("embedding", "aqa", "image", "tts", "audio", "computer-use", "transcribe",
                                 "robotics", "omni", "banana", "customtools", "live")))
        except Exception:
            pass
    _model_cache.update(at=time.time(), models=out)
    return out


# ---------------------------------------------------------------- database
@contextmanager
def db():
    # `with sqlite3.connect()` commits but never closes; the UI polls, so always close
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with db() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS opportunities (
            id TEXT PRIMARY KEY, title TEXT NOT NULL, agency TEXT, category TEXT, notice_type TEXT,
            reference TEXT, url TEXT, source TEXT, posted TEXT, deadline TEXT, set_aside TEXT,
            naics TEXT, summary TEXT, score INTEGER, rationale TEXT, scored_with TEXT,
            dismissed_score INTEGER, status TEXT DEFAULT 'new', notes TEXT,
            verified INTEGER DEFAULT 1, first_seen TEXT, last_seen TEXT);
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, started_at TEXT, finished_at TEXT,
            found INTEGER, new INTEGER, focus TEXT, error TEXT, sources TEXT);
        CREATE TABLE IF NOT EXISTS run_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER, ts TEXT, kind TEXT, text TEXT);
        CREATE TABLE IF NOT EXISTS opportunity_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT, opportunity_id TEXT, ts TEXT, note TEXT);
        """)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_event(run_id: int, kind: str, text: str):
    with db() as conn:
        conn.execute("INSERT INTO run_events (run_id, ts, kind, text) VALUES (?,?,?,?)",
                     (run_id, now(), kind, text))


def add_event(conn, oid: str, note: str):
    conn.execute("INSERT INTO opportunity_events (opportunity_id, ts, note) VALUES (?,?,?)", (oid, now(), note))


def _norm(s) -> str:
    return re.sub(r"[^a-z0-9 ]", "", str(s or "").lower()).strip()


def opportunity_id(item: dict) -> str:
    """Dedup key: the reference number when there is one (a presolicitation and its later
    solicitation share it), otherwise agency + title."""
    ref = re.sub(r"[^a-z0-9]", "", str(item.get("reference") or "").lower())
    key = ref if len(ref) >= 5 else f"{_norm(item.get('agency'))}|{_norm(item.get('title'))}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def extract_json(text: str) -> list[dict]:
    m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    raw = m.group(1) if m else None
    if raw is None:
        m = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
        raw = m.group(0) if m else None
    if raw is None:
        raise ValueError("No JSON array found in agent response")
    return json.loads(raw)


def salvage_json(text: str) -> list[dict]:
    """Recover complete objects from truncated or malformed output."""
    items = []
    for m in re.finditer(r"\{[^{}]*?\"title\"[^{}]*?\}", text, re.DOTALL):
        try:
            items.append(json.loads(m.group(0)))
        except json.JSONDecodeError:
            pass
    return items


def domain(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return ""


# ---------------------------------------------------------------- deep search
def profile_block(p: dict) -> str:
    join = lambda k: "; ".join(p.get(k) or []) or "(none listed)"
    certs = join("certifications")
    return f"""COMPANY: {p['company_name']}
{p['company_description']}
Capabilities: {join('capabilities')}
NAICS codes: {join('naics_codes')}
Keywords: {join('keywords')}
Socioeconomic certifications held: {certs}
Preferred agencies: {join('preferred_agencies')}
Contract size: {p.get('contract_size', '')}
Past performance: {join('past_performance')}
Notes: {p.get('scoring_notes', '')}"""


def known_opportunities_block(limit: int = 300) -> str:
    """Compact skip list so deep search spends its searches on things not yet in the tracker."""
    with db() as conn:
        rows = conn.execute("""SELECT title, agency, reference FROM opportunities
                               WHERE score >= 50 OR status != 'new'
                               ORDER BY last_seen DESC LIMIT ?""", (limit,)).fetchall()
    if not rows:
        return ""
    lines = [f"- {(r['agency'] or '')[:40]}: {r['title'][:70]} [{r['reference'] or ''}]" for r in rows]
    return "ALREADY KNOWN — do not spend searches finding these or report them:\n" + "\n".join(lines)


def build_prompt(p: dict, focus: str | None) -> tuple[str, str]:
    system = f"""You are a federal business-development researcher working for this company.

{profile_block(p)}

You supplement structured feeds that already cover SAM.gov, the DoD SBIR/STTR portal, Grants.gov,
and the Federal Register. Spend your searches on what they miss. Cover each lane with its own searches:
 (1) CONTRACTS — agency forecasts of upcoming procurements, agency and lab BAAs, Other Transaction
     Authority (OTA) consortia calls, and innovation-office solicitations (DIU, AFWERX, and similar).
 (2) SBIR/STTR — civilian agency topics and open solicitations on agency sites (NSF, DOE, NASA, DHS,
     DOT, ED, NOAA, USDA, and others) and sbir.gov.
 (3) CRADA — federal labs and agencies currently seeking CRADA partners in this company's fields.
     A CRADA is a research partnership, not a contract: the federal partner generally cannot pay the
     company.
 (4) SMALL BUSINESS — small-business procurement forecasts, subcontracting opportunities with large
     primes (SBA SubNet), and Small Business Office announcements.

STRICT ACCURACY RULES — these matter more than volume:
- Report ONLY opportunities you directly saw in a page or search result.
- The url must be the actual page you saw — never construct, guess, or "clean up" a URL.
- Only currently open or forecast opportunities. Leave out anything past its deadline.
- Do not report opportunities restricted to certifications this company does not hold.
- If you are not certain an opportunity exists, leave it out. Every link will be checked.

OUTPUT FORMAT
After searching, output ONLY a JSON array (inside a ```json fence) of the best matches (aim for 10-20):
{{"title": str, "agency": str, "category": "contract"|"sbir"|"crada"|"small_business",
  "notice_type": str, "reference": str|null, "url": str, "source": str, "posted": "YYYY-MM-DD"|null,
  "deadline": "YYYY-MM-DD"|null, "set_aside": str|null, "naics": str|null,
  "summary": str (2-3 sentences describing the work)}}"""
    user = "Run the opportunity search now and return the JSON array."
    known = known_opportunities_block()
    if known:
        user += "\n\n" + known
    if focus:
        user += f"\n\nAdditional focus for this run: {focus}"
    return system, user


LISTINGS_SCHEMA = {"type": "object", "properties": {"listings": {"type": "array", "items": {
    "type": "object", "properties": {k: {"type": "string"} for k in (
        "title", "agency", "category", "notice_type", "reference", "url", "source",
        "posted", "deadline", "set_aside", "naics", "summary")},
    "required": ["title", "url"]}}}, "required": ["listings"]}


def parse_listings(model: str, system: str, text: str) -> list[dict]:
    try:
        items = extract_json(text)
    except (ValueError, json.JSONDecodeError):
        items = salvage_json(text)
    if items or not text.strip():
        return items
    return llm_structured(model, system, "Record every REAL opportunity you directly observed "
                          "(real URLs only):\n\n" + text[:60000], LISTINGS_SCHEMA).get("listings", [])


def run_agent_anthropic(model: str, system: str, user: str, run_id: int):
    import anthropic
    client = anthropic.Anthropic()
    log_event(run_id, "info", f"Agent started — {model} with web search")
    tools = [{"type": "web_search_20250305", "name": "web_search", "max_uses": MAX_SEARCHES},
             {"type": "web_fetch_20250910", "name": "web_fetch", "max_uses": MAX_FETCHES,
              "max_content_tokens": 25000}]
    messages, text, sources, searches = [{"role": "user", "content": user}], "", set(), 0
    for _ in range(8):
        with client.beta.messages.stream(betas=["web-fetch-2025-09-10"], model=model, max_tokens=16000,
                                         messages=messages, tools=tools,
                                         system=[{"type": "text", "text": system,
                                                  "cache_control": {"type": "ephemeral"}}]) as stream:
            for event in stream:
                if getattr(event, "type", "") != "content_block_stop":
                    continue
                try:
                    block = stream.current_message_snapshot.content[event.index]
                except Exception:
                    continue
                btype = getattr(block, "type", "")
                if btype == "server_tool_use":
                    inp = getattr(block, "input", {}) or {}
                    if getattr(block, "name", "") == "web_search":
                        searches += 1
                        log_event(run_id, "query", f"Search {searches}: “{inp.get('query', '')}”")
                    elif getattr(block, "name", "") == "web_fetch":
                        sources.add(domain(inp.get("url", "")))
                        log_event(run_id, "fetch", f"Reading: {inp.get('url', '')}")
                elif btype == "web_search_tool_result" and isinstance(getattr(block, "content", None), list):
                    sources.update(domain(getattr(r, "url", "") or "") for r in block.content)
            final = stream.get_final_message()
        text += "".join(b.text for b in final.content if b.type == "text")
        if final.stop_reason != "pause_turn":
            break
        messages.append({"role": "assistant", "content": final.content})
    if final.stop_reason == "refusal":
        raise RuntimeError("model declined the search (refusal)")
    log_event(run_id, "info", f"{searches} searches run — extracting opportunities…")
    return parse_listings(model, system, text), sorted(s for s in sources if s)


def run_agent_openai(model: str, system: str, user: str, run_id: int):
    log_event(run_id, "info", f"Agent started — {model} with OpenAI web search")
    out = _post_json("https://api.openai.com/v1/responses",
                     {"model": model, "instructions": system, "input": user,
                      "tools": [{"type": "web_search"}], "max_output_tokens": 16000},
                     {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"}, timeout=600)
    text, sources, n = "", set(), 0
    for item in out.get("output", []):
        if item.get("type") == "web_search_call":
            q = (item.get("action") or {}).get("query") or ""
            if q:
                n += 1
                log_event(run_id, "query", f"Search {n}: “{q}”")
        elif item.get("type") == "message":
            for part in item.get("content", []):
                text += part.get("text", "")
                sources.update(domain(a["url"]) for a in part.get("annotations", []) if a.get("url"))
    log_event(run_id, "info", f"{n} searches run — extracting opportunities…")
    return parse_listings(model, system, text), sorted(s for s in sources if s)


def run_agent_google(model: str, system: str, user: str, run_id: int):
    """Gemini via the Interactions API: on generateContent, current models can silently
    ignore the search tool and answer from memory."""
    key = os.getenv("GOOGLE_API_KEY")
    log_event(run_id, "info", f"Agent started — {model} with Google Search")
    budget = ("\n\nSEARCH BUDGET: use at most {n} Google searches in total, then write your answer. "
              "Prefer searches that surface many opportunities at once.")
    body = {"model": model, "input": user + budget.format(n=GEMINI_MAX_SEARCHES),
            "system_instruction": system, "tools": [{"type": "google_search"}],
            "generation_config": {"max_output_tokens": 32000}, "store": False}
    out = None
    for attempt in (1, 2):
        try:
            out = _post_json("https://generativelanguage.googleapis.com/v1beta/interactions", body,
                             {"x-goog-api-key": key, "Api-Revision": GEMINI_API_REVISION}, timeout=900)
            break
        except urllib.error.HTTPError as e:
            msg = e.read().decode("utf8", "ignore")
            if e.code == 400 and "too many tool calls" in msg.lower() and attempt == 1:
                log_event(run_id, "info", "Gemini exceeded the search cap — retrying with a tighter budget")
                body["input"] = user + budget.format(n=GEMINI_MAX_SEARCHES // 2)
                continue
            raise RuntimeError(f"Gemini HTTP {e.code}: {msg[:200]}")
    if out is None:
        raise RuntimeError("Gemini exceeded its search cap twice")
    steps = out.get("steps") or []
    queries = [q for s in steps if s.get("type") == "google_search_call"
               for q in ((s.get("arguments") or {}).get("queries") or []) if q]
    for i, q in enumerate(queries, 1):
        log_event(run_id, "query", f"Search {i}: “{q}”")
    if not queries:
        log_event(run_id, "info", "Gemini did not search — discarding its unsearched answer")
        return [], []
    text, sources = "", set()
    for s in steps:
        if s.get("type") == "model_output":
            for c in s.get("content") or []:
                text += c.get("text", "") or ""
                sources.update(a["title"].lower() for a in c.get("annotations") or []
                               if a.get("type") == "url_citation" and a.get("title"))
    log_event(run_id, "info", f"{len(queries)} searches run — extracting opportunities…")
    return [_resolve_redirect(i) for i in parse_listings(model, system, text)], sorted(sources)


def _resolve_redirect(item: dict) -> dict:
    """Google cites through temporary redirect links; store the real URL instead."""
    url = item.get("url") or ""
    if "vertexaisearch.cloud.google.com" in url:
        try:
            req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": S.UA})
            with urllib.request.urlopen(req, timeout=15, context=S.SSL_CTX) as r:
                item["url"] = r.geturl()
        except urllib.error.HTTPError as e:
            if e.headers.get("Location"):
                item["url"] = e.headers["Location"]
        except Exception:
            pass
    return item


def run_agent(profile: dict, focus: str | None, run_id: int):
    model = load_settings()["search_model"]
    if not model:
        raise RuntimeError("no AI provider key is configured in .env")
    system, user = build_prompt(profile, focus)
    fn = {"openai": run_agent_openai, "google": run_agent_google}.get(provider_of(model), run_agent_anthropic)
    return fn(model, system, user, run_id)


def verify_urls(items: list[dict], run_id: int) -> list[dict]:
    """Link-check deep-search finds: drop dead links, flag what can't be confirmed."""
    def check(item):
        url = item.get("url")
        if not url or not url.startswith("http"):
            return None
        # sam.gov is a single-page app: any address returns 200, so a link check proves nothing
        if domain(url).endswith("sam.gov"):
            item["verified"] = 0
            return item
        req = urllib.request.Request(url, headers={"User-Agent": S.UA})
        try:
            with urllib.request.urlopen(req, timeout=15, context=S.SSL_CTX) as r:
                item["verified"] = 1 if r.status < 400 else 0
        except urllib.error.HTTPError as e:
            if e.code in (404, 410):
                return None
            item["verified"] = 0  # 403s are common on .mil sites: keep, but flag
        except Exception:
            item["verified"] = 0
        return item

    log_event(run_id, "info", f"Link-checking {len(items)} deep-search finds…")
    with ThreadPoolExecutor(max_workers=8) as ex:
        alive = [i for i in ex.map(check, items) if i is not None]
    if len(items) - len(alive):
        log_event(run_id, "info", f"Dropped {len(items) - len(alive)} dead link(s)")
    return alive


# ---------------------------------------------------------------- scoring
def recent_reactions(limit: int = 15) -> str:
    with db() as conn:
        pursued = conn.execute("""SELECT title, agency FROM opportunities
                                  WHERE status IN ('capture','proposal','submitted','won','lost')
                                  ORDER BY last_seen DESC LIMIT ?""", (limit,)).fetchall()
        passed = conn.execute("""SELECT title, agency, dismissed_score s FROM opportunities
                                 WHERE status='no_bid' AND dismissed_score >= 65
                                 ORDER BY last_seen DESC LIMIT ?""", (limit,)).fetchall()
    if not (pursued or passed):
        return ""
    lines = ["THE TEAM'S RECENT DECISIONS (stronger evidence than the profile text):"]
    lines += [f"- PURSUED: {r['title'][:80]} — {r['agency']}" for r in pursued]
    lines += [f"- NO-BID despite a score of {r['s']}: {r['title'][:80]} — {r['agency']}" for r in passed]
    return "\n".join(lines)


SCORE_SCHEMA = {"type": "object", "properties": {"scores": {"type": "array", "items": {
    "type": "object", "properties": {
        "i": {"type": "integer"}, "score": {"type": "integer"}, "exclude": {"type": "boolean"},
        "rationale": {"type": "string"}},
    "required": ["i", "score", "exclude", "rationale"]}}}, "required": ["scores"]}


def score_items(items: list[dict], profile: dict, focus: str | None, run_id: int):
    """Sets score and rationale on each item in place. Excluded items score 0 and stay visible
    to the 'all' view; a failed batch leaves its items unscored for the next run."""
    model = load_settings()["scoring_model"]
    if not model:
        raise RuntimeError("no AI provider key is configured in .env")
    system = f"""You assess federal opportunities for this company, as an experienced capture manager.

{profile_block(profile)}

Set "exclude": true for: work outside software/IT/data/R&D the company could plausibly do; goods or
supply purchases; set-asides requiring a certification it does not hold; anything already past deadline;
anything matching these excluded terms: {'; '.join(profile.get('exclude_keywords') or []) or '(none)'}.

Score 0-100 on fit AND realistic winnability for a company of this size. Judge each type on its own terms:
- contract: capability match, realistic size, whether a small firm could credibly prime or team.
- small_business: same, plus a real advantage from the set-aside. Sources-sought notices are valuable
  early capture opportunities even with no solicitation yet.
- sbir: technical match to the topic, and whether Phase I is achievable. Note ITAR or CMMC requirements
  in the rationale.
- crada: a research partnership, NOT revenue — the federal partner generally cannot fund the company.
  Score on strategic value: lab access, IP, credibility, and a path to later funded work. Never describe
  a CRADA as a contract win.

Bands — use them honestly; most opportunities are NOT strong fits:
  85-100 rare: strong capability match, right size, realistic to win.
  65-84  a real match worth a bid/no-bid discussion.
  40-64  adjacent, too large, or too little information to judge.
  below 40 poor fit.
If only a title is available, do not assume a fit: cap the score at 60.
{recent_reactions()}
{f'THIS RUN, EXTRA WEIGHT ON: {focus}' if focus else ''}

Return one entry per input in "scores" ("i" = the input index). Keep each rationale to one or two short sentences."""
    chunks = [items[i:i + 25] for i in range(0, len(items), 25)]

    def score_chunk(chunk):
        payload = [{"i": i, "title": it["title"], "agency": it.get("agency"), "type": it.get("category"),
                    "notice": it.get("notice_type"), "set_aside": it.get("set_aside"),
                    "naics": it.get("naics"), "deadline": it.get("deadline"),
                    "description": (it.get("summary") or "")[:1500] or None}
                   for i, it in enumerate(chunk)]
        return {r["i"]: r for r in llm_structured(model, system, json.dumps(payload), SCORE_SCHEMA)
                .get("scores", []) if isinstance(r.get("i"), int)}

    def safe(chunk):
        try:
            return score_chunk(chunk)
        except Exception as e:
            log_event(run_id, "error", f"Scoring batch failed ({str(e)[:150]}) — keeping unscored")
            return {}

    log_event(run_id, "info", f"Scoring {len(items)} opportunities with {model}…")
    with ThreadPoolExecutor(max_workers=4) as ex:
        for chunk, results in zip(chunks, ex.map(safe, chunks)):
            for i, it in enumerate(chunk):
                r = results.get(i)
                if r and r.get("exclude"):
                    it["score"], it["rationale"] = 0, "Excluded: " + (r.get("rationale") or "")
                elif r:
                    it["score"], it["rationale"] = r.get("score"), r.get("rationale")


def scoring_fingerprint(profile: dict) -> str:
    """Stored with each score: unchanged opportunities aren't re-billed on the next run."""
    raw = json.dumps(profile, sort_keys=True) + str(load_settings()["scoring_model"])
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


# ---------------------------------------------------------------- pipeline
_lock = threading.Lock()
_state = {"running": False, "run_id": None}


def do_search(focus: str | None, deep: bool):
    with db() as conn:
        run_id = conn.execute("INSERT INTO runs (started_at, focus) VALUES (?,?)", (now(), focus)).lastrowid
    _state["run_id"] = run_id
    try:
        profile = load_profile()
        merged: dict[str, dict] = {}
        origins: set[str] = set()
        log_event(run_id, "info", "Querying government sources…")
        for name, fn in S.ALL_FETCHERS:
            S.pop_errors()
            try:
                out, raw = fn(profile)
            except Exception as e:
                log_event(run_id, "source", f"{name}: failed ({type(e).__name__})")
                continue
            if out is None:
                log_event(run_id, "source", f"{name}: skipped — no API key in .env")
                continue
            errs = S.pop_errors()
            note = f" — {errs[-1][:160]}" if errs else ""
            log_event(run_id, "source", f"{name}: scanned {raw} → {len(out)} match{note}")
            if out:
                origins.add(name)
            for item in out:
                oid = opportunity_id(item)
                current = merged.get(oid)
                # a later notice replaces an earlier one; keep whichever has the longer summary
                if current and len(current.get("summary") or "") > len(item.get("summary") or ""):
                    item["summary"] = current["summary"]
                merged[oid] = item

        if deep:
            log_event(run_id, "info", "Launching deep web search…")
            try:
                # a hard limit: a stalled provider connection must not hold up the whole run
                pool = ThreadPoolExecutor(max_workers=1)
                try:
                    found, domains = pool.submit(run_agent, profile, focus, run_id).result(DEEP_SEARCH_TIMEOUT)
                except FuturesTimeout:
                    raise RuntimeError(f"timed out after {DEEP_SEARCH_TIMEOUT // 60} minutes")
                finally:
                    pool.shutdown(wait=False)
                found = verify_urls(found, run_id)
                added = 0
                for item in found:
                    # models sometimes return objects or "null" where text is expected
                    for k, v in list(item.items()):
                        item[k] = None if v in ("", "null", "N/A") else (
                            json.dumps(v) if isinstance(v, (dict, list)) else v)
                    if not item.get("title"):
                        continue
                    if item.get("category") not in CATEGORIES:
                        item["category"] = "contract"
                    if not S.not_expired(_iso_date(item.get("deadline"))):
                        continue
                    item["deadline"], item["posted"] = _iso_date(item.get("deadline")), _iso_date(item.get("posted"))
                    item["source"] = item.get("source") or domain(item.get("url", ""))
                    oid = opportunity_id(item)
                    if oid not in merged:
                        merged[oid] = item
                        added += 1
                origins.update(domains)
                log_event(run_id, "info", f"Deep search added {added} opportunities not found in the feeds")
            except Exception as e:
                log_event(run_id, "error", f"Deep search failed ({str(e)[:200]}) — continuing with feeds")

        excl = [k.strip() for k in profile.get("exclude_keywords") or [] if k.strip()]
        if excl:
            pattern = re.compile(r"(?i)\b(" + "|".join(map(re.escape, excl)) + r")\b")
            before = len(merged)
            merged = {k: v for k, v in merged.items() if not pattern.search(v.get("title") or "")}
            if before - len(merged):
                log_event(run_id, "info", f"Filtered {before - len(merged)} by excluded keywords")

        _attach_stored_and_page_text(merged, run_id)
        fingerprint = scoring_fingerprint(profile)
        with db() as conn:
            stored = {r["id"]: dict(r) for r in conn.execute(
                "SELECT id, score, rationale, scored_with FROM opportunities")}
        to_score = []
        for oid, item in merged.items():
            item["id"] = oid
            known = stored.get(oid)
            if known and known["scored_with"] == fingerprint and known["score"] is not None:
                item["score"], item["rationale"] = known["score"], known["rationale"]
            else:
                to_score.append(item)
        if to_score:
            score_items(to_score, profile, focus, run_id)
        else:
            log_event(run_id, "info", "Nothing new to score")

        new_count = 0
        ts = now()
        with db() as conn:
            for oid, it in merged.items():
                was_scored = it.get("score") is not None
                if oid in stored:
                    conn.execute(
                        """UPDATE opportunities SET last_seen=?, title=?, notice_type=COALESCE(?, notice_type),
                           deadline=COALESCE(?, deadline), set_aside=COALESCE(?, set_aside),
                           summary=CASE WHEN length(?) > length(COALESCE(summary,'')) THEN ? ELSE summary END,
                           score=COALESCE(?, score), rationale=COALESCE(?, rationale),
                           scored_with=CASE WHEN ? THEN ? ELSE scored_with END WHERE id=?""",
                        (ts, it["title"], it.get("notice_type"), it.get("deadline"), it.get("set_aside"),
                         it.get("summary") or "", it.get("summary"), it.get("score"), it.get("rationale"),
                         was_scored, fingerprint, oid))
                else:
                    new_count += 1
                    conn.execute(
                        """INSERT INTO opportunities (id,title,agency,category,notice_type,reference,url,source,
                           posted,deadline,set_aside,naics,summary,score,rationale,scored_with,verified,first_seen,last_seen)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (oid, it["title"], it.get("agency"), it.get("category"), it.get("notice_type"),
                         it.get("reference"), it.get("url"), it.get("source"), it.get("posted"), it.get("deadline"),
                         it.get("set_aside"), it.get("naics"), it.get("summary"), it.get("score"), it.get("rationale"),
                         fingerprint if was_scored else None, it.get("verified", 1), ts, ts))
            closed = conn.execute("""UPDATE opportunities SET status='closed'
                                     WHERE deadline < ? AND status IN ('new','tracking')""",
                                  (S.today().isoformat(),)).rowcount
            conn.execute("UPDATE runs SET finished_at=?, found=?, new=?, sources=? WHERE id=?",
                         (now(), len(merged), new_count, json.dumps(sorted(origins)), run_id))
        if closed:
            log_event(run_id, "info", f"Closed {closed} opportunities past their deadline")
        log_event(run_id, "done", f"Done — {len(merged)} opportunities ({new_count} new)")
    except Exception as e:
        with db() as conn:
            conn.execute("UPDATE runs SET finished_at=?, error=? WHERE id=?", (now(), str(e)[:500], run_id))
        log_event(run_id, "error", f"Search failed: {str(e)[:300]}")
    finally:
        _state["running"] = False
        _lock.release()


def _iso_date(value) -> str | None:
    m = re.match(r"\d{4}-\d{2}-\d{2}", str(value or ""))
    return m.group(0) if m else None


def _attach_stored_and_page_text(merged: dict[str, dict], run_id: int):
    with db() as conn:
        stored = {r["id"]: r["summary"] for r in conn.execute(
            "SELECT id, summary FROM opportunities WHERE length(summary) >= 300")}
    todo = []
    for oid, it in merged.items():
        known = stored.get(oid)
        if known and len(known) > len(it.get("summary") or ""):
            it["summary"] = known
        elif len(it.get("summary") or "") < 300 and it.get("url") and it.get("source") != "SAM.gov":
            todo.append(it)
    if not todo:
        return
    log_event(run_id, "info", f"Reading {len(todo)} opportunity pages…")
    with ThreadPoolExecutor(max_workers=8) as ex:
        texts = list(ex.map(lambda i: S.fetch_page_text(i["url"]), todo))
    for it, t in zip(todo, texts):
        if t:
            prefix = (it.get("summary") or "").strip()
            it["summary"] = f"{prefix}\n{t}" if prefix else t


# ---------------------------------------------------------------- API
app = FastAPI(title="Opportunity Search Tool")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
init_db()


class SearchRequest(BaseModel):
    focus: str | None = None
    deep: bool = True


class OpportunityUpdate(BaseModel):
    status: str | None = None
    notes: str | None = None


@app.get("/")
def index():
    # no-store: the UI is one local file; never let the browser serve a stale copy
    return FileResponse(BASE_DIR / "static" / "index.html", headers={"Cache-Control": "no-store"})


@app.post("/api/search")
def start_search(req: SearchRequest):
    if not _lock.acquire(blocking=False):
        raise HTTPException(409, "A search is already running")
    _state["running"] = True
    threading.Thread(target=do_search, args=(req.focus, req.deep), daemon=True).start()
    return {"started": True}


@app.get("/api/search/status")
def search_status(after: int = 0):
    run_id = _state["run_id"]
    if run_id is None:
        with db() as conn:
            row = conn.execute("SELECT MAX(id) m FROM runs").fetchone()
        run_id = row["m"]
    if run_id is None:
        return {"run_id": None, "running": False, "events": []}
    with db() as conn:
        events = [dict(r) for r in conn.execute(
            "SELECT * FROM run_events WHERE run_id=? AND id>? ORDER BY id", (run_id, after))]
    return {"run_id": run_id, "running": _state["running"], "events": events}


@app.get("/api/opportunities")
def list_opportunities(status: str | None = None, category: str | None = None,
                       min_score: int = 0, q: str | None = None):
    sql, args = "SELECT * FROM opportunities WHERE COALESCE(score, 100) >= ?", [min_score]
    if status == "active":
        sql += " AND status IN ('new','tracking','capture','proposal','submitted')"
    elif status:
        sql += " AND status = ?"
        args.append(status)
    if category:
        sql += " AND category = ?"
        args.append(category)
    if q:
        sql += " AND (title LIKE ? OR agency LIKE ? OR reference LIKE ?)"
        args += [f"%{q}%"] * 3
    sql += " ORDER BY COALESCE(score, -1) DESC, COALESCE(deadline, '9999') ASC"
    with db() as conn:
        rows = [dict(r) for r in conn.execute(sql, args)]
    return {"opportunities": rows, "count": len(rows)}


@app.patch("/api/opportunities/{oid}")
def update_opportunity(oid: str, upd: OpportunityUpdate):
    if upd.status and upd.status not in VALID_STATUSES:
        raise HTTPException(400, f"status must be one of {sorted(VALID_STATUSES)}")
    with db() as conn:
        row = conn.execute("SELECT status FROM opportunities WHERE id=?", (oid,)).fetchone()
        if not row:
            raise HTTPException(404, "opportunity not found")
        if upd.status and upd.status != row["status"]:
            conn.execute("UPDATE opportunities SET status=? WHERE id=?", (upd.status, oid))
            if upd.status == "no_bid":
                conn.execute("UPDATE opportunities SET dismissed_score=score WHERE id=?", (oid,))
            add_event(conn, oid, f"{row['status']} → {upd.status}")
        if upd.notes is not None:
            conn.execute("UPDATE opportunities SET notes=? WHERE id=?", (upd.notes, oid))
    return {"ok": True}


@app.get("/api/report")
def report():
    with db() as conn:
        rows = [dict(r) for r in conn.execute(
            """SELECT id, title, agency, category, reference, deadline, score, status, notes, url
               FROM opportunities WHERE status NOT IN ('new','closed')
               ORDER BY CASE status WHEN 'proposal' THEN 0 WHEN 'capture' THEN 1 WHEN 'submitted' THEN 2
                        WHEN 'tracking' THEN 3 WHEN 'won' THEN 4 WHEN 'lost' THEN 5 ELSE 6 END,
                        COALESCE(deadline, '9999')""")]
    return {"rows": rows}


@app.get("/api/crada_directory")
def crada_directory():
    """The curated agency offices, plus agencies seen announcing CRADAs in recent notices."""
    with db() as conn:
        seen = [dict(r) for r in conn.execute(
            """SELECT agency, COUNT(*) notices, MAX(posted) latest FROM opportunities
               WHERE category='crada' AND agency IS NOT NULL GROUP BY agency ORDER BY latest DESC""")]
    return {"offices": json.loads(DIRECTORY_PATH.read_text()), "active_agencies": seen}


@app.get("/api/profile")
def get_profile():
    return load_profile()


@app.put("/api/profile")
def put_profile(profile: dict):
    base = load_profile()
    if not str(profile.get("company_name", "")).strip():
        raise HTTPException(400, "company_name is required")
    list_keys = ("capabilities", "naics_codes", "keywords", "exclude_keywords", "certifications",
                 "preferred_agencies", "past_performance")
    for k in list_keys:
        if k in profile and not isinstance(profile[k], list):
            raise HTTPException(400, f"{k} must be a list")
    base.update({k: v for k, v in profile.items() if k in base})
    PROFILE_PATH.write_text(json.dumps(base, indent=2, ensure_ascii=False))
    return {"ok": True}


@app.get("/api/settings")
def get_settings(refresh: bool = False):
    models = list_provider_models(refresh)
    return {"settings": load_settings(),
            "sam_configured": bool(os.getenv("SAM_API_KEY")),
            "providers": {p: {"configured": configured(p), "models": models.get(p, [])} for p in PROVIDER_KEYS}}


@app.put("/api/settings")
def put_settings(settings: dict):
    clean = {}
    for key in ("search_model", "scoring_model"):
        model = settings.get(key)
        if model:
            if not configured(provider_of(model)):
                raise HTTPException(400, f"{provider_of(model)} API key not configured in .env")
            clean[key] = model
    SETTINGS_PATH.write_text(json.dumps(clean, indent=2))
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("PORT", "8010")))
