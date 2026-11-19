"""
Wikipedia, the one source Nova looks facts up in (Stage 3 of the brain port).

PRIVACY — CLAUDE.md invariant 3, exception added 2026-11-19 by Nicholas's
explicit choice (option C of three he was shown: A local-only, B lookups only
when he asks, C B plus overnight research on his past questions). Wikipedia
needs NO key and NO account, so nothing ties a request to him beyond the IP any
HTTP call carries. What leaves the machine:
  - on demand: the words of a general-knowledge question he asked out loud;
  - overnight (night_school.py): his own past general-knowledge questions that
    the language model answered from memory, and the page ids of facts already
    stored, to re-check them.
Never: anything in the first or second person, any name he has told Nova (see
world_facts.private_reason), memory, documents, or the conversation itself.
`facts.enabled: false` turns every request off.

Shape, like weather_engine: pure functions over HTTP with a hard timeout, and
NOTHING raises — a failed lookup returns {"ok": False} and Nova answers the way
she did before Stage 3 rather than inventing a source.

ONE request per question: the search runs as a generator inside the extracts
query, so the titles, lead sections, page ids, revision ids and URLs all come
back together (~450-650ms measured). Wikipedia rate-limits anonymous clients —
measured, about ten quick requests earned an HTTP 429 with Retry-After — so a
429 sets a "not before" time that every later call respects, and the overnight
job spaces its requests out.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional

log = logging.getLogger("nova.wiki")

API = "https://en.wikipedia.org/w/api.php"
# Wikimedia asks clients to identify themselves. Generic on purpose: no name,
# no email, no repository link — those would tie every request to him.
USER_AGENT = "NovaVoiceAssistant/1.0 (single-user local assistant; a few requests per day)"
_TIMEOUT = 3.0

_lock = threading.Lock()
_not_before = 0.0          # set from a 429's Retry-After


def backing_off(now: Optional[float] = None) -> bool:
    return (now or time.time()) < _not_before


def _get(params: dict, timeout: float = _TIMEOUT) -> dict:
    """One JSON GET. Never raises — {"ok": False, "error": ...} instead."""
    global _not_before
    if backing_off():
        return {"ok": False, "error": "rate limited", "retry_at": _not_before}
    q = {**params, "format": "json", "formatversion": "2"}
    url = f"{API}?{urllib.parse.urlencode(q)}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return {"ok": True, "data": json.loads(r.read().decode())}
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            try:
                wait = float(exc.headers.get("Retry-After") or 60)
            except (TypeError, ValueError):
                wait = 60.0
            with _lock:
                _not_before = max(_not_before, time.time() + min(max(wait, 5.0), 3600.0))
            log.warning(f"wikipedia: rate limited for {wait:.0f}s")
            return {"ok": False, "error": "rate limited", "retry_at": _not_before}
        log.warning(f"wikipedia: HTTP {exc.code}")
        return {"ok": False, "error": f"HTTP {exc.code}"}
    except Exception as exc:
        # Never log the query: it is his question.
        log.warning(f"wikipedia lookup failed: {type(exc).__name__}")
        return {"ok": False, "error": type(exc).__name__}


def _pages(data: dict) -> list[dict]:
    pages = ((data or {}).get("query") or {}).get("pages") or []
    out = []
    for p in pages:
        if p.get("missing") or not p.get("extract"):
            continue
        out.append({
            "pageid": int(p["pageid"]),
            "title": p.get("title", ""),
            "extract": p.get("extract", ""),
            "revid": int(p.get("lastrevid") or 0),
            "url": p.get("fullurl") or p.get("canonicalurl") or "",
            "index": int(p.get("index") or 0),     # search rank, 1 = best
        })
    return sorted(out, key=lambda p: p["index"] or 99)


def search(query: str, limit: int = 3, timeout: float = _TIMEOUT) -> dict:
    """Search and fetch lead sections in ONE request.
    {"ok": True, "pages": [{pageid, title, extract, revid, url, index}]}"""
    query = (query or "").strip()
    if not query:
        return {"ok": False, "error": "empty query"}
    r = _get({
        "action": "query", "generator": "search", "gsrsearch": query,
        "gsrlimit": limit, "gsrprop": "", "prop": "extracts|info",
        "exintro": 1, "explaintext": 1, "exlimit": limit, "inprop": "url",
        "redirects": 1,
    }, timeout=timeout)
    if not r["ok"]:
        return r
    return {"ok": True, "pages": _pages(r["data"])}


def pages(pageids: list[int], with_extract: bool = True, timeout: float = _TIMEOUT) -> dict:
    """Current state of known pages (the overnight audit). Up to 50 ids in one
    request. {"ok": True, "pages": {pageid: {revid, title, extract?, url}}}"""
    ids = [int(i) for i in pageids if i][:50]
    if not ids:
        return {"ok": True, "pages": {}}
    params = {"action": "query", "pageids": "|".join(map(str, ids)),
              "prop": "info" + ("|extracts" if with_extract else ""), "inprop": "url"}
    if with_extract:
        params.update({"exintro": 1, "explaintext": 1, "exlimit": len(ids)})
    r = _get(params, timeout=timeout)
    if not r["ok"]:
        return r
    out = {}
    for p in ((r["data"].get("query") or {}).get("pages") or []):
        if p.get("missing"):
            continue
        out[int(p["pageid"])] = {"revid": int(p.get("lastrevid") or 0),
                                 "title": p.get("title", ""),
                                 "extract": p.get("extract", ""),
                                 "url": p.get("fullurl", "")}
    return {"ok": True, "pages": out}


# ── sentences ────────────────────────────────────────────────────────────────
# Splitting on every period cut "The first director for the film, John D."
# out of the Jaws article — a fragment that names the wrong man. Initials,
# common abbreviations and decimals do not end a sentence.
_ABBREV = re.compile(r"(?:\b[A-Z]|\b(?:Mr|Mrs|Ms|Dr|St|Jr|Sr|Mt|Ft|vs|etc|approx|ca|c|No|Vol|Inc|Ltd|Co)"
                     r"|\b(?:e\.g|i\.e|U\.S|U\.K))\.$")
_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_PRONUNCIATION = re.compile(r"\s*\((?:[^()]*?(?:listen|pronounced|IPA|/)[^()]*)\)")


def sentences(extract: str) -> list[str]:
    text = re.sub(r"\s+", " ", (extract or "").replace("\n", " ")).strip()
    text = _PRONUNCIATION.sub("", text)
    merged: list[str] = []
    for part in _SPLIT.split(text):
        if merged and _ABBREV.search(merged[-1]):
            merged[-1] += " " + part           # "John D." does not end a sentence
        else:
            merged.append(part)
    out = []
    for s in merged:
        s = s.strip()
        if 20 <= len(s) <= 450:
            out.append(s)
    return out
