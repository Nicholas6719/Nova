"""
The world-fact store — realm 1 of JARVIS's brain roadmap, ported to Nova.

JARVIS (MiniPCJarvis docs/BRAIN_ROADMAP.md, sidecar/brain/facts.py) answers
facts that can never change from a store of web-verified answers in ~0.3s,
without waking its LLM. Three realms, Nicholas's rule there and here:
  1. timeless facts ("how tall is the Eiffel Tower")  -> this store;
  2. changeable facts ("who is the CEO of Apple")     -> never stored, never
     served from here;
  3. generation and conversation                       -> the LLM.
A fast wrong answer is worse than a slow right one, so every doubt is a "no".

WHAT IS DIFFERENT ON NOVA
- One source: Wikipedia (wiki.py), his choice — see the privacy note there.
- Nova's LLM has no tools, so nothing "researches" mid-answer. A general-
  knowledge question that reaches the LLM stage is looked up FIRST and the
  relevant sentences are handed to the model as reference (stage 8c). After
  the turn, off the live path, a one-sentence answer is synthesised from those
  same sentences at temperature 0 and offered to the store.
- SERVING IS GUARDED BY WORDS, NOT JUST SIMILARITY. Measured with Nova's
  embedder (bge-small): at JARVIS's 0.90 bar "who directed jaws 2" scored
  0.937 against "who directed jaws", "how wide is the mariana trench" 0.943
  against "how deep", and "who starred in jaws" 0.921 against "who directed".
  Each would have been answered with the other question's fact. A stored fact
  is served only when the question KIND matches (who / when / where / why /
  how many / other) AND the content words match exactly after a small synonym
  map (tall = high = height). A refusal costs one live lookup, never a wrong
  answer.
- TWO JUDGES FOR "TIMELESS", and only one of them can be trusted: a
  deterministic realm-2 screen (JARVIS's trigger words plus Nova's: present-
  tense "who is / holds / leads", "last", "won", counts that grow) AND the
  local model at temperature 0. JARVIS's classifier was a 20B. Measured on
  Nova's 3B it called 6 of 8 changeable facts timeless ("who is the CEO of
  Apple", Japan's population, the newest iPhone), so the model only ever
  VETOES; the screen is what keeps realm 2 out.
- Explanations ("why ...") are looked up but never stored: squeezed into one
  sentence by the 3B they lose the explanation.
- GROUNDED: every number in a stored answer must appear in the Wikipedia text
  it came from, and nearly every word must trace back to it — the same rule
  fact_reconciler applies to what he says about himself.
- PRIVACY GUARD before anything leaves the machine: nothing in the first or
  second person, nothing naming a person he has told Nova about.

Storage: `world_facts` and `world_lookups` in nova_memory.db (memory.py
already owns a table called `facts` — his personal facts — which is why this
one is not called that). Every write commits on a short-lived connection
(JARVIS's 2,600-message night was one uncommitted write).
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Optional

import numpy as np

log = logging.getLogger("nova.facts")

SERVE_THRESHOLD = 0.90       # cosine, AND the word guard below
RELEVANCE_FLOOR = 0.70       # best sentence vs the question. Measured on real Wikipedia
                             # text: right pages 0.795-0.898, another question's pages
                             # at most 0.564. JARVIS's 0.80 dropped "who directed jaws".
CONTEXT_SENTENCES = 5

# ── what a factual question looks like ──────────────────────────────────────
_WAKE = re.compile(r"^\s*(?:(?:hey|hi|ok|okay)[,\s]+)?nova\b[,.!?\s]*", re.I)
_LEAD = re.compile(r"^\s*(?:please\s+|(?:can|could|would)\s+you\s+(?:tell\s+me\s+)?|"
                   r"do\s+you\s+know\s+|tell\s+me\s+|i\s+wonder\s+|quick\s+question[,\s]+)+", re.I)
_OPEN = re.compile(r"^(?:what|what's|whats|who|who's|whom|whose|when|when's|where|where's|which|"
                   r"how|how's|why|why's|is|are|was|were|did|does|do|can|could)\b", re.I)
# The first or second person is about HIM or about NOVA: never a lookup, and
# never sent anywhere. ("what's my sister's name", "how are you", "who made you")
_PERSONAL = re.compile(r"\b(?:i|i'm|i've|i'd|i'll|me|my|mine|myself|we|we're|us|our|ours|"
                       # informal kinship is always HIS family ("when is mom's
                       # birthday" went to Wikipedia — audit, 2026-11-19)
                       r"mom|moms|mom's|mum|mum's|mommy|mama|dad|dads|dad's|daddy|papa|grandma|"
                       r"grandma's|grandpa|grandpa's|granny|nana|auntie|hubby|wifey|bestie|bro|sis|"
                       r"you|you're|you've|your|yours|yourself|nova)\b", re.I)
# Opinion, advice, creative work, recommendations: realm 3, the model's job.
_OPINION = re.compile(r"\b(?:should|shall|best|worst|favou?rite|recommend|suggest|opinion|"
                      r"think|feel|believe|better|worse|joke|story|poem|song\s+about|"
                      r"meaning\s+of\s+life|sense|okay|ok|fine|good\s+idea)\b", re.I)
# Things Wikipedia cannot know: today, prices, scores, the news. Not looked up
# at all (sending them would leak a question for nothing).
_LIVE = re.compile(r"\b(?:today|tonight|tomorrow|yesterday|right\s+now|this\s+(?:morning|"
                   r"afternoon|evening|week|weekend|month)|weather|forecast|temperature\s+"
                   r"outside|score|scores|standings|news|headlines|price|prices|stock|"
                   r"shares|open\s+(?:now|until|till|today)|traffic|time\s+is\s+it|"
                   r"what\s+time|what\s+day|date\s+today|playing)\b", re.I)

# Realm 2: changeable. May be LOOKED UP (Wikipedia is fresher than a 3B's
# training data), never STORED, never SERVED. JARVIS's list plus Nova's.
REALM2 = re.compile(
    r"\b(?:latest|newest|most\s+recent|recent(?:ly)?|current(?:ly)?|now|nowadays|still|"
    r"so\s+far|as\s+of|this\s+(?:year|season)|next|upcoming|coming\s+out|release\s+date|"
    r"worth|cost|costs|salary|net\s+worth|in\s+stock|available|version|update[sd]?|"
    r"record|records|richest|tallest|largest|biggest|fastest|oldest\s+living|"
    r"population|populations|how\s+many\s+(?:people|moons|members|employees|users|"
    r"subscribers|followers|countries|states|stores)|"
    r"ceo|president|prime\s+minister|leader|owner|coach|manager|champion|champions|"
    r"married|dating|alive|age\s+of|how\s+old\s+is|last|won|win|wins|winner|winners|"
    r"winning|holds?|ranks?|ranking|ranked|number\s+one|top|"
    # superlatives and growing totals that slipped through (audit): "the most
    # populous country", "highest grossing film", "longest river", GDP...
    r"most|least|highest|lowest|longest|shortest|deepest|smallest|heaviest|"
    r"lightest|grossing|populous|gdp|economy|inhabitants|residents|teams|"
    r"medals|oscars|grammys|awards|titles|trophies|"
    r"how\s+many\s+\w+(?:\s+\w+)?\s+(?:has|have)\s+\w+(?:\s+\w+)?\s+(?:\w+ed|won|made|"
    r"written|sold|scored|released))\b", re.I)
# Present tense about WHO is the classic changeable fact ("who is the CEO",
# "who holds the record"). Measured: the 3B called those timeless.
_PRESENT_WHO = re.compile(r"^(?:who|who's)\s+(?:is|are|has|have|holds|leads|runs|owns|plays|"
                          r"sings|coaches|manages|governs|rules|hosts|stars)\b|^who's\b", re.I)

_KIND = [
    ("count", re.compile(r"^how\s+(?:many|much)\b", re.I)),
    ("when", re.compile(r"^(?:when|when's|what\s+(?:year|date|day|century|decade)|"
                        r"which\s+(?:year|century))\b", re.I)),
    ("who", re.compile(r"^(?:who|who's|whom|whose)\b", re.I)),
    ("where", re.compile(r"^(?:where|where's|what\s+(?:country|city|continent))\b", re.I)),
    ("why", re.compile(r"^(?:why|why's)\b", re.I)),
    ("yesno", re.compile(r"^(?:is|are|was|were|did|does|do|can|could)\b", re.I)),
]

_STOP = {
    "what", "whats", "who", "whos", "whom", "whose", "when", "whens", "where", "wheres",
    "which", "how", "hows", "why", "whys", "is", "are", "was", "were", "be", "been",
    "did", "does", "do", "can", "could", "the", "a", "an", "of", "in", "on", "at",
    "to", "for", "by", "with", "and", "or", "it", "its", "s", "that", "this", "there",
    # NOT stopwords, though they look like scaffolding: "country", "city",
    # "year", "first"... name WHAT is asked. Dropped, "what city is the eiffel
    # tower in" was served "...in France" and "who was the wife of henry
    # viii" his FIRST wife (audit, 2026-11-19).
    "many", "much", "actually", "exactly", "really", "about",
    "approximately", "roughly", "away", "come", "came", "up", "get", "got",
    "also", "please", "tell", "know", "called", "named", "mean", "means", "meaning",
}
# A small synonym map, so a paraphrase the embedder already scores as the same
# question also passes the word guard. Unlisted words must match exactly.
_SYN = {
    "tall": "height", "high": "height", "height": "height", "heights": "height",
    "deep": "depth", "depth": "depth", "wide": "width", "width": "width",
    "broad": "width", "long": "length", "length": "length", "far": "distance",
    "distance": "distance", "big": "size", "large": "size", "size": "size",
    "heavy": "weight", "weigh": "weight", "weight": "weight", "weighs": "weight",
    "old": "age", "age": "age",
    "wrote": "write", "written": "write", "writes": "write", "writer": "write",
    "author": "write", "authored": "write",
    "director": "direct", "directed": "direct", "directs": "direct",
    "inventor": "invent", "invented": "invent", "invents": "invent", "invention": "invent",
    "painted": "paint", "painter": "paint", "composed": "compose", "composer": "compose",
    "discovered": "discover", "discoverer": "discover", "founded": "found",
    "founder": "found", "built": "build", "builder": "build", "designed": "design",
    "designer": "design", "created": "create", "creator": "create",
    "died": "die", "death": "die", "born": "birth", "birth": "birth",
    "fall": "fall", "fell": "fall", "sank": "sink", "sunk": "sink", "sink": "sink",
}


def _clean(text: str) -> str:
    t = _WAKE.sub("", (text or "").strip())
    for _ in range(3):
        t2 = _LEAD.sub("", t)
        if t2 == t:
            break
        t = t2
    return t.strip().rstrip("?.! ").strip()


def kind(text: str) -> str:
    t = _clean(text)
    for name, rx in _KIND:
        if rx.search(t):
            return name
    return "other"


def _stem(w: str) -> str:
    w = w.lower().replace("'", "")
    if w in _SYN:
        return _SYN[w]
    for suf in ("ings", "ing", "ers", "ies", "ed", "er", "es", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            w = w[: -len(suf)] + ("y" if suf == "ies" else "")
            break
    return _SYN.get(w, w)


def content_key(text: str) -> tuple[str, frozenset]:
    """(question kind, content stems). Two questions are the same fact only if
    both match — see the module docstring for the measured reason."""
    t = _clean(text).lower().replace("’", "'")
    t = re.sub(r"'s\b", "", t)
    words = re.findall(r"[a-z0-9]+", t)
    stems = frozenset(_stem(w) for w in words if w not in _STOP)
    return kind(text), stems


def is_factual_question(text: str) -> bool:
    """A general-knowledge question Wikipedia could answer, about neither him
    nor Nova, asking nothing live and no opinion."""
    t = _clean(text)
    if not t or len(t) < 8 or len(t.split()) > 18:
        return False
    if not _OPEN.search(t):
        return False
    if _PERSONAL.search(t) or _OPINION.search(t) or _LIVE.search(t):
        return False
    _, stems = content_key(t)
    return len(stems) >= 1


def changeable(text: str) -> bool:
    t = _clean(text)
    return bool(REALM2.search(t) or _PRESENT_WHO.search(t))


def private_reason(text: str, names: set[str]) -> Optional[str]:
    """Why this must not leave the machine, or None. The last check before any
    request: anything personal, or naming someone he has told Nova about."""
    t = _clean(text)
    if _PERSONAL.search(t):
        return "personal"
    if re.search(r"[\w.+-]+@[\w-]+\.\w+|https?://|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b", t):
        return "contact detail"
    # Possessives stripped on BOTH sides: "Sarah's" never equalled "sarah",
    # so a stored name went out the moment he said it possessively (audit).
    def bare(w: str) -> str:
        return re.sub(r"(?:'s|s')$", "", w.lower().replace("’", "'"))
    words = {bare(w) for w in re.findall(r"[A-Za-z][A-Za-z'’]*", t)}
    hit = words & {bare(n) for n in names if len(n) >= 3}
    if hit:
        return "a name from memory"
    return None


# ── grounding ────────────────────────────────────────────────────────────────
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
_GROUND_STOP = _STOP | {"has", "have", "had", "than", "from", "into", "over", "under",
                        "about", "around", "approximately", "roughly", "some", "made",
                        "known", "his", "her", "their", "they", "he",
                        "she", "were", "which", "who", "metres", "meters", "feet"}


_NUMWORD = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
            "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11",
            "twelve": "12", "twenty": "20", "hundred": "100", "thousand": "1000",
            "million": "1000000", "billion": "1000000000",
            "first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th",
            "fifth": "5th", "sixth": "6th", "seventh": "7th", "eighth": "8th",
            "ninth": "9th", "tenth": "10th", "once": "1x", "twice": "2x"}
_ORDINAL_DIGITS = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b")


def _numbers(text: str) -> set[str]:
    """Every number in the text, however it was written: "three funnels" and
    "third" count, so "had three funnels" is NOT grounded in "had four
    funnels" (it was — number words sat in the stop list; audit)."""
    t = (text or "").lower()
    out = {n.replace(",", "").rstrip(".") for n in _NUM.findall(t)}
    out |= {m.group(1) + "th" for m in _ORDINAL_DIGITS.finditer(t)}
    for w in re.findall(r"[a-z]+", t):
        if w in _NUMWORD:
            v = _NUMWORD[w]
            out.add(re.sub(r"(st|nd|rd)$", "th", v) if v[-2:] in ("st", "nd", "rd", "th") else v)
    return out


def grounded(answer: str, source: str) -> bool:
    """Every number in `answer` is in `source`, and at least 80% of its content
    words trace back to it (prefix match, so inflection still counts)."""
    if not answer or not source:
        return False
    if not _numbers(answer) <= _numbers(source):
        return False
    src = set(re.findall(r"[a-z0-9]+", source.lower()))
    words = [w for w in re.findall(r"[a-z]+", answer.lower())
             if len(w) > 2 and w not in _GROUND_STOP]
    if not words:
        return True
    ok = sum(1 for w in words
             if w in src or any(len(c) > 3 and (w.startswith(c[:4]) or c.startswith(w[:4]))
                                for c in src))
    return ok / len(words) >= 0.8


# ── prompts (temperature 0; anything not a clean verdict is a no) ────────────
SYNTH_SYSTEM = ("You answer a question using only the source sentences given. "
                "Reply with ONE short plain sentence that a voice assistant can say. "
                "No markdown. If the sources do not clearly answer it, reply exactly: UNKNOWN")
SYNTH_PROMPT = "Question: {q}\n\nSources:\n{src}\n\nAnswer in one sentence:"
TIMELESS_SYSTEM = "You classify facts. Reply with exactly one word: YES or NO."
TIMELESS_PROMPT = (
    "Question: {q}\nAnswer: {a}\n\n"
    "Will this answer still be true and unchanged twenty years from now, with no "
    "possibility of revision? Physical constants, completed history, definitions, "
    "and dimensions of finished structures are YES. Rankings, records, living "
    "people's roles, counts that grow, prices and ongoing series are NO.")
COMPARE_SYSTEM = "You check whether a stored fact is still supported. Reply with exactly one word."
COMPARE_PROMPT = (
    "Question: {q}\nStored answer: {a}\n\nCurrent text from the source:\n{src}\n\n"
    "Reply SAME if the text clearly supports the stored answer, CHANGED if it "
    "clearly contradicts it, UNCLEAR otherwise.")


def last_word(raw: str, choices: tuple[str, ...]) -> Optional[str]:
    words = re.findall(r"\b(" + "|".join(choices) + r")\b", (raw or "").upper())
    return words[-1] if words else None


# ── the store ────────────────────────────────────────────────────────────────
_SCHEMA = """
CREATE TABLE IF NOT EXISTS world_facts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    question    TEXT NOT NULL,
    kind        TEXT NOT NULL,
    stems       TEXT NOT NULL,          -- json list, the word guard
    answer      TEXT NOT NULL,
    sources     TEXT NOT NULL,          -- json [{pageid, title, url, revid}]
    origin      TEXT NOT NULL,          -- asked | night
    created_ts  REAL NOT NULL,
    verified_ts REAL NOT NULL,
    hits        INTEGER NOT NULL DEFAULT 0,
    strikes     INTEGER NOT NULL DEFAULT 0,
    status      TEXT NOT NULL DEFAULT 'active',   -- active | demoted
    note        TEXT,
    embedding   BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS world_lookups (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    ts       REAL NOT NULL,
    question TEXT NOT NULL,             -- exactly what was sent
    origin   TEXT NOT NULL,             -- asked | night | audit
    outcome  TEXT NOT NULL              -- found | nothing | failed | stored | ...
);
CREATE INDEX IF NOT EXISTS world_lookups_q ON world_lookups(question);
"""


class WorldFacts:
    def __init__(self, db_path: Path, config: Optional[dict] = None,
                 embedder: Optional[Callable] = None,
                 names: Optional[Callable[[], set]] = None,
                 search: Optional[Callable] = None) -> None:
        cfg = (config or {}).get("facts", {})
        self.enabled = bool(cfg.get("enabled", True))
        self.serve_threshold = float(cfg.get("serve_threshold", SERVE_THRESHOLD))
        self.relevance_floor = float(cfg.get("relevance_floor", RELEVANCE_FLOOR))
        # On the live path a slow Wikipedia must not hold up his answer.
        self.timeout_asked = float(cfg.get("lookup_timeout_s", 2.0))
        self.db_path = Path(db_path)
        self._embedder = embedder          # () -> fastembed TextEmbedding
        self._names = names or (lambda: set())
        self._search = search              # injectable for tests; default wiki.search
        self.last: Optional[dict] = None   # what the last answer rested on ("how do you know?")
        with self._db() as db:
            db.executescript(_SCHEMA)

    @contextmanager
    def _db(self):
        conn = sqlite3.connect(str(self.db_path), timeout=5.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ── embeddings ───────────────────────────────────────────────────────────
    def _model(self):
        if self._embedder is None:
            raise RuntimeError("no embedder")
        return self._embedder()

    def _vec(self, text: str) -> np.ndarray:
        from brain import _light
        v = np.array(list(self._model().embed([_light(_clean(text))])), dtype=np.float32)[0]
        return v / (np.linalg.norm(v) + 1e-9)

    def _rank(self, question: str, sents: list[tuple[str, str]]) -> list[tuple[float, str, str]]:
        """Score Wikipedia sentences against the question (bge query/passage)."""
        m = self._model()
        q = np.array(list(m.query_embed([_clean(question)])), dtype=np.float32)[0]
        q /= (np.linalg.norm(q) + 1e-9)
        p = np.array(list(m.passage_embed([f"{t}: {s}" for t, s in sents])), dtype=np.float32)
        p /= (np.linalg.norm(p, axis=1, keepdims=True) + 1e-9)
        scores = p @ q
        return sorted(((float(scores[i]), sents[i][0], sents[i][1]) for i in range(len(sents))),
                      reverse=True)

    # ── serve ────────────────────────────────────────────────────────────────
    def lookup(self, text: str) -> Optional[dict]:
        """A stored fact for this question, or None. Never for realm 2."""
        if not self.enabled or not is_factual_question(text) or changeable(text):
            return None
        k, stems = content_key(text)
        with self._db() as db:
            rows = db.execute("SELECT * FROM world_facts WHERE status='active' AND kind=?",
                              (k,)).fetchall()
        rows = [r for r in rows if frozenset(json.loads(r["stems"])) == stems]
        if not rows:
            return None
        try:
            qv = self._vec(text)
        except Exception as exc:
            log.debug(f"facts: no embedder ({exc})")
            return None
        best, score = None, 0.0
        for r in rows:
            v = np.frombuffer(r["embedding"], dtype=np.float32).copy()
            s = float(v @ qv)
            if s > score:
                best, score = r, s
        if best is None or score < self.serve_threshold:
            return None
        with self._db() as db:
            db.execute("UPDATE world_facts SET hits = hits + 1 WHERE id=?", (best["id"],))
        fact = self._row(best)
        fact["score"] = round(score, 3)
        self.last = {"how": "stored", "fact": fact, "ts": time.time()}
        log.info(f"[facts] served #{fact['id']} ({score:.3f})")
        return fact

    # ── look up (the live path) ──────────────────────────────────────────────
    def reference(self, text: str, origin: str = "asked") -> dict:
        """Look the question up on Wikipedia and pick the sentences that answer
        it. {"ok", "sentences": [(score, title, sentence)], "sources", "context",
        "reason"}. Never raises."""
        if not self.enabled:
            return {"ok": False, "reason": "disabled"}
        if not is_factual_question(text):
            return {"ok": False, "reason": "not a factual question"}
        why = private_reason(text, self._safe_names())
        if why:
            log.info(f"[facts] not looking that up: {why}")
            return {"ok": False, "reason": why}
        q = _clean(text)
        import wiki as _wiki
        if self._search is None and _wiki.backing_off():
            # Nothing is sent while Wikipedia has asked us to wait, so nothing
            # is logged as looked up either (night school used to record such
            # a question as done, and never send it).
            return {"ok": False, "reason": "rate limited"}
        if self._search is None:
            import wiki
            t = self.timeout_asked if origin == "asked" else 6.0
            search = lambda query: wiki.search(query, timeout=t)
        else:
            search = self._search
        r = search(q)
        if not r.get("ok"):
            self._log_lookup(q, origin, "failed")
            return {"ok": False, "reason": r.get("error", "failed")}
        pages = r.get("pages") or []
        import wiki
        sents = [(p["title"], s) for p in pages for s in wiki.sentences(p["extract"])]
        if not sents:
            self._log_lookup(q, origin, "nothing")
            return {"ok": False, "reason": "nothing found"}
        sents = self._shortlist(q, pages, sents)
        try:
            ranked = self._rank(q, sents)
        except Exception as exc:
            log.warning(f"[facts] could not rank: {exc}")
            return {"ok": False, "reason": "no embedder"}
        if ranked[0][0] < self.relevance_floor:
            self._log_lookup(q, origin, "irrelevant")
            return {"ok": False, "reason": "nothing relevant", "best": ranked[0][0]}
        cut = max(self.relevance_floor, ranked[0][0] - 0.10)
        top = [x for x in ranked[:CONTEXT_SENTENCES] if x[0] >= cut]
        used = {t for _, t, _ in top}
        sources = [{"pageid": p["pageid"], "title": p["title"], "url": p["url"],
                    "revid": p["revid"]} for p in pages if p["title"] in used]
        context = "\n".join(f"{s} ({t})" for _, t, s in top)
        self._log_lookup(q, origin, "found")
        out = {"ok": True, "question": q, "sentences": top, "sources": sources,
               "context": context, "best": top[0][0]}
        if origin == "asked":
            self.last = {"how": "looked_up", "sources": sources, "ts": time.time(),
                         "question": q}
        return out

    @staticmethod
    def _shortlist(q: str, pages: list[dict], sents: list[tuple[str, str]],
                   cap: int = 14) -> list[tuple[str, str]]:
        """Embedding every sentence of three articles cost 220-640ms on the
        live path (measured). Keep each page's opening two sentences — the
        lead usually says what the thing IS — and any sentence that shares a
        content word with the question; the embedder ranks only those."""
        _, stems = content_key(q)
        first: dict[str, int] = {}
        keep, rest = [], []
        for t, s in sents:
            first[t] = first.get(t, 0) + 1
            words = {_stem(w) for w in re.findall(r"[a-z0-9]+", s.lower())}
            if first[t] <= 2 or (stems & words):
                keep.append((t, s))
            else:
                rest.append((t, s))
        if len(keep) > cap:
            scored = sorted(keep, key=lambda ts: -len(stems & {_stem(w) for w in
                            re.findall(r"[a-z0-9]+", ts[1].lower())}))
            keep = scored[:cap]
        return keep or sents[:cap]

    def _safe_names(self) -> set:
        try:
            return set(self._names() or ())
        except Exception:
            return set()

    def _log_lookup(self, q: str, origin: str, outcome: str) -> None:
        try:
            with self._db() as db:
                db.execute("INSERT INTO world_lookups (ts, question, origin, outcome) "
                           "VALUES (?,?,?,?)", (time.time(), q, origin, outcome))
        except Exception as exc:
            log.warning(f"[facts] lookup not logged: {exc}")

    def looked_up(self, q: str) -> bool:
        with self._db() as db:
            return db.execute("SELECT 1 FROM world_lookups WHERE lower(question)=?",
                              (_clean(q).lower(),)).fetchone() is not None

    def lookups(self, since: float = 0.0, origin: Optional[str] = None) -> list[dict]:
        sql = "SELECT * FROM world_lookups WHERE ts >= ?"
        args: list = [since]
        if origin:
            sql += " AND origin=?"
            args.append(origin)
        with self._db() as db:
            return [dict(r) for r in db.execute(sql + " ORDER BY id", args)]

    # ── store (off the live path; `generate` runs on the nova-llm thread) ────
    def consider(self, ref: dict, generate: Callable[..., str], origin: str = "asked") -> Optional[int]:
        """Offer a looked-up answer to the store. Returns the new fact id, or
        None with the reason logged. `generate(system, user, temperature,
        max_tokens)` is the local model."""
        if not ref or not ref.get("ok"):
            return None
        q = ref["question"]
        if changeable(q):
            log.info(f"[facts] not storing (changeable): {q!r}")
            return None
        if kind(q) == "why":
            # An explanation is not a one-sentence fact. Measured: squeezed
            # into one sentence, "why is the sky blue" became "due to a
            # phenomenon called sky radiation" — grounded, and no answer.
            log.info(f"[facts] not storing (an explanation): {q!r}")
            return None
        src_text = "\n".join(s for _, _, s in ref["sentences"])
        try:
            raw = generate(SYNTH_SYSTEM, SYNTH_PROMPT.format(q=q, src=src_text), 0.0, 80)
        except Exception as exc:
            log.warning(f"[facts] synthesis failed: {exc}")
            return None
        answer = (raw or "").strip().splitlines()[0].strip().strip('"') if (raw or "").strip() else ""
        if not answer or "UNKNOWN" in answer.upper() or len(answer) > 300:
            log.info(f"[facts] not storing (no clear answer): {q!r}")
            return None
        if changeable(answer):
            log.info(f"[facts] not storing (answer is changeable): {answer!r}")
            return None
        if not grounded(answer, src_text):
            log.info(f"[facts] not storing (ungrounded): {answer!r}")
            return None
        try:
            verdict = last_word(generate(TIMELESS_SYSTEM, TIMELESS_PROMPT.format(q=q, a=answer),
                                         0.0, 8), ("YES", "NO"))
        except Exception as exc:
            log.warning(f"[facts] timeless check failed: {exc}")
            return None
        if verdict != "YES":
            log.info(f"[facts] not storing (not timeless): {q!r} -> {answer!r}")
            return None
        return self.store(q, answer, ref["sources"], origin)

    def store(self, q: str, answer: str, sources: list[dict], origin: str) -> Optional[int]:
        k, stems = content_key(q)
        if not stems or not sources:
            return None
        if self._duplicate(k, stems):
            log.info(f"[facts] already known: {q!r}")
            return None
        if self._disputed(k, stems):
            log.info(f"[facts] not storing (he said it was wrong): {q!r}")
            return None
        vec = self._vec(q)
        now = time.time()
        with self._db() as db:
            cur = db.execute(
                "INSERT INTO world_facts (question, kind, stems, answer, sources, origin, "
                "created_ts, verified_ts, embedding) VALUES (?,?,?,?,?,?,?,?,?)",
                (_clean(q), k, json.dumps(sorted(stems)), answer, json.dumps(sources[:3]),
                 origin, now, now, vec.astype(np.float32).tobytes()))
            fid = cur.lastrowid
        log.info(f"[facts] stored #{fid}: {q!r} -> {answer!r}")
        return fid

    def dispute(self, q: str) -> int:
        """He said the answer was wrong. Demote any stored fact for this
        question, and remember the dispute so the answer is never stored
        later — the intake may still be running when he says it."""
        k, stems = content_key(q)
        n = 0
        with self._db() as db:
            rows = db.execute("SELECT id, stems FROM world_facts WHERE status='active' AND kind=?",
                              (k,)).fetchall()
        for r in rows:
            if frozenset(json.loads(r["stems"])) == stems:
                self.demote(r["id"], "he said it was wrong")
                n += 1
        self._log_lookup(_clean(q), "asked", "disputed")
        return n

    def _disputed(self, k: str, stems: frozenset) -> bool:
        with self._db() as db:
            rows = db.execute("SELECT question FROM world_lookups WHERE outcome='disputed'").fetchall()
        return any(content_key(r["question"]) == (k, stems) for r in rows)

    def _duplicate(self, k: str, stems: frozenset) -> bool:
        with self._db() as db:
            rows = db.execute("SELECT stems FROM world_facts WHERE status='active' AND kind=?",
                              (k,)).fetchall()
        return any(frozenset(json.loads(r["stems"])) == stems for r in rows)

    # ── audit (night school) ─────────────────────────────────────────────────
    def due_for_audit(self, limit: int = 20, older_than_s: float = 0.0) -> list[dict]:
        with self._db() as db:
            rows = db.execute(
                "SELECT * FROM world_facts WHERE status='active' AND verified_ts <= ? "
                "ORDER BY verified_ts ASC LIMIT ?", (time.time() - older_than_s, limit)).fetchall()
        return [self._row(r) for r in rows]

    def mark_verified(self, fid: int, sources: Optional[list] = None) -> None:
        with self._db() as db:
            if sources is not None:
                db.execute("UPDATE world_facts SET verified_ts=?, strikes=0, sources=? WHERE id=?",
                           (time.time(), json.dumps(sources), fid))
            else:
                db.execute("UPDATE world_facts SET verified_ts=?, strikes=0 WHERE id=?",
                           (time.time(), fid))

    def strike(self, fid: int, why: str = "unclear") -> int:
        """An unclear audit. Two strikes demote — distrust is the default."""
        with self._db() as db:
            db.execute("UPDATE world_facts SET strikes = strikes + 1 WHERE id=?", (fid,))
            n = db.execute("SELECT strikes FROM world_facts WHERE id=?", (fid,)).fetchone()
        n = int(n[0]) if n else 0
        if n >= 2:
            self.demote(fid, f"two unclear audits ({why})")
        return n

    def demote(self, fid: int, reason: str = "") -> None:
        with self._db() as db:
            db.execute("UPDATE world_facts SET status='demoted', note=? WHERE id=?",
                       (reason[:200], fid))
        log.info(f"[facts] #{fid} demoted: {reason}")

    def active(self) -> list[dict]:
        with self._db() as db:
            return [self._row(r) for r in db.execute(
                "SELECT * FROM world_facts WHERE status='active' ORDER BY id")]

    def get(self, fid: int) -> Optional[dict]:
        with self._db() as db:
            r = db.execute("SELECT * FROM world_facts WHERE id=?", (fid,)).fetchone()
        return self._row(r) if r else None

    @staticmethod
    def _row(r) -> dict:
        d = dict(r)
        d.pop("embedding", None)
        d["sources"] = json.loads(d.get("sources") or "[]")
        d["stems"] = json.loads(d.get("stems") or "[]")
        return d


# ── spoken receipts ──────────────────────────────────────────────────────────
# Anchored at both ends ("are you sure it's going to rain tomorrow" is a
# weather question, not a request for a receipt — audit, 2026-11-19).
_END = r"\s*(?:nova)?[\s.!?]*$"
_HOW_KNOW = re.compile(r"^\s*(?:nova[,\s]+)?(?:how\s+do\s+you\s+know\s+(?:that|this)|"
                       r"where\s+did\s+you\s+(?:get|hear|read|find)\s+(?:that|this)|"
                       r"what(?:'s|\s+is)\s+(?:your|the)\s+source|says\s+who|"
                       r"are\s+you\s+sure(?:\s+about\s+that)?)" + _END, re.I)
_WRONG = re.compile(r"^\s*(?:nova[,\s]+)?(?:no[,\s]+)?(?:that(?:'s|\s+is)\s+(?:wrong|not\s+right|"
                    r"incorrect|not\s+true|false)|that\s+fact\s+is\s+wrong|you(?:'re|\s+are)\s+wrong)"
                    + _END, re.I)


def wants_source(text: str) -> bool:
    return bool(_HOW_KNOW.search(text or ""))


def says_wrong(text: str) -> bool:
    return bool(_WRONG.search(text or ""))


def _spoken_day(ts: float) -> str:
    import datetime
    d = datetime.datetime.fromtimestamp(ts)
    today = datetime.date.today()
    if d.date() == today:
        return "today"
    if (today - d.date()).days == 1:
        return "yesterday"
    return d.strftime("%B ") + str(d.day)


def receipt(last: Optional[dict]) -> Optional[str]:
    """Nova's answer to "how do you know that?" — from the record, never made up.
    Sources are never mentioned unless he asks (JARVIS's rule, kept)."""
    if not last:
        return None
    if last["how"] == "stored":
        f = last["fact"]
        title = (f["sources"][0]["title"] if f["sources"] else "Wikipedia")
        return (f"I looked it up on Wikipedia, in the article on {title}, and last "
                f"checked it {_spoken_day(f['verified_ts'])}.")
    if last["how"] == "looked_up":
        titles = [s["title"] for s in last.get("sources") or []]
        if not titles:
            return None
        where = titles[0] if len(titles) == 1 else ", ".join(titles[:-1]) + " and " + titles[-1]
        return f"I looked it up just now on Wikipedia, in the article on {where}."
    return None
