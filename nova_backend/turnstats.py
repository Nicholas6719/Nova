"""
What answered each turn, and how fast — the evidence the next round of work
starts from.

JARVIS's brain roadmap began with exactly this (BRAIN_ROADMAP.md stage 1,
"instrument"): a week of real turns showing which ones burned LLM time, which
is where "229 of 284 turns were reflexes" came from. Nova had no such record:
the only trace of a turn was the conversations table, which says what was said
and nothing about which stage answered or how long the decision took. So
Stage 5 of the brain port had to reconstruct evidence by replaying old text
through new code. From now on it is recorded as it happens.

One row per spoken or typed turn, in nova_memory.db (local, like the
conversation history it sits beside): the path that answered (reflex, regex
stage, math, watch, a taught phrase, a "did you mean", the LLM...), the skill
and its confidence when the brain decided, the time to DECIDE and the time to
the first word. Nothing leaves the machine.

It also answers "why did you do that?" — Nova explaining her last decision
from this record rather than inventing a reason.
"""
from __future__ import annotations

import logging
import re
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

log = logging.getLogger("nova.turnstats")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS turn_stats (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        REAL NOT NULL,
    text      TEXT NOT NULL,
    path      TEXT NOT NULL,     -- reflex | regex:<stage> | math | watch | phrase |
                                 -- lesson | ask | conversation | unsupported | llm |
                                 -- system | view | pending | explain
    skill     TEXT,
    confidence REAL,
    matched   TEXT,              -- the example the brain matched (canonical)
    decide_ms REAL,              -- utterance in -> a stage claimed it
    first_ms  REAL               -- utterance in -> first word out
);
CREATE INDEX IF NOT EXISTS turn_stats_ts ON turn_stats(ts);
"""

_WHY_RE = re.compile(
    r"^\s*(?:nova[,\s]+)?(?:why\s+did\s+you\s+(?:do|say|pick|choose)\s+(?:that|this)|"
    r"why\s+(?:that|would\s+you\s+do\s+that)|how\s+did\s+you\s+(?:decide|know|get)\s+that|"
    r"what\s+made\s+you\s+(?:do|say)\s+that|what\s+did\s+you\s+just\s+do)\b", re.I)
_REPORT_RE = re.compile(
    r"^\s*(?:nova[,\s]+)?(?:how\s+(?:often|much)\s+do\s+you\s+(?:answer|handle)\s+(?:things\s+)?"
    r"(?:by\s+)?yourself|how\s+(?:is|are)\s+(?:your\s+brain|you\s+learning)|"
    r"what\s+have\s+you\s+learned(?:\s+(?:lately|recently|this\s+week))?|brain\s+report)\b", re.I)

_STAGE_WORDS = {
    "calendar": "your calendar", "files": "files", "weather": "the weather",
    "market": "the markets", "screen": "the screen", "memory": "memory",
    "fastpath": "the basics like the time", "tools": "your Mac's controls",
    "actuation": "typing and clicking", "view": "my screens",
}


class TurnStats:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
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

    def record(self, row: dict) -> None:
        try:
            with self._db() as db:
                db.execute(
                    "INSERT INTO turn_stats (ts, text, path, skill, confidence, matched, "
                    "decide_ms, first_ms) VALUES (?,?,?,?,?,?,?,?)",
                    (time.time(), row.get("text", ""), row.get("path", "?"),
                     row.get("skill"), row.get("confidence"), row.get("matched"),
                     row.get("decide_ms"), row.get("first_ms")))
        except Exception as exc:
            # Bookkeeping never costs him a turn (JARVIS "Staying alive").
            log.warning(f"turn stats not recorded: {exc}")

    def summary(self, days: float = 7.0) -> dict:
        since = time.time() - days * 86400
        with self._db() as db:
            rows = db.execute("SELECT path, decide_ms, first_ms FROM turn_stats "
                              "WHERE ts >= ? AND path != 'explain'", (since,)).fetchall()
        total = len(rows)
        own = sum(1 for r in rows if not r["path"] in ("llm", "conversation"))
        reflex = sum(1 for r in rows if r["path"] == "reflex")
        llm = total - own

        def median(xs):
            xs = sorted(x for x in xs if x is not None)
            return xs[len(xs) // 2] if xs else None
        return {"total": total, "own": own, "reflex": reflex, "llm": llm,
                "own_first_ms": median(r["first_ms"] for r in rows
                                       if r["path"] not in ("llm", "conversation")),
                "llm_first_ms": median(r["first_ms"] for r in rows
                                       if r["path"] in ("llm", "conversation"))}


def wants_explanation(text: str) -> bool:
    return bool(_WHY_RE.search(text or ""))


def wants_report(text: str) -> bool:
    return bool(_REPORT_RE.search(text or ""))


def explain(last: Optional[dict], brain=None) -> str:
    """Nova's account of her last decision, from the record — never invented."""
    if not last:
        return "I haven't done anything yet this session."
    p = last.get("path", "")
    said = (last.get("text") or "").strip().rstrip(".?!")
    ms = last.get("decide_ms")
    fast = f" It took me {int(ms)} milliseconds to decide." if ms is not None and ms < 1000 else ""
    if p == "reflex":
        label = last.get("skill") or "that"
        if brain is not None and last.get("skill") in getattr(brain, "by_name", {}):
            label = brain.by_name[last["skill"]].label or label
        matched = (last.get("matched") or "").replace(" N ", " a number ").replace(" N", " a number")
        conf = last.get("confidence")
        pct = f", {int(round(conf * 100))} percent sure" if conf is not None else ""
        return (f"You said {said}. I took that as a request to {label}, because it was "
                f"closest to something I already know: {matched}{pct}.{fast}")
    if p.startswith("regex:"):
        what = _STAGE_WORDS.get(p.split(":", 1)[1], p.split(":", 1)[1])
        return (f"You said {said}. That matched one of my fixed rules for {what}, "
                f"so I followed it.{fast}")
    if p == "math":
        return "That was arithmetic, so I worked it out in code instead of guessing."
    if p == "watch":
        return "That was a request to keep watch on something, so I set a watch."
    if p == "phrase":
        return "That's a phrase you taught me, so I ran its steps."
    if p == "ask":
        return "I wasn't sure what you meant, so I asked rather than guess."
    if p == "unsupported":
        return ("That sounded like something you wanted done that I can't do yet, "
                "so I said so instead of pretending.")
    if p == "conversation":
        return "I was confident that was conversation, not a command, so I just talked."
    if p == "llm":
        return ("I didn't recognise that as something I can do myself, so I answered "
                "it in conversation.")
    if p == "view":
        return "That's how you move between my screens, so I switched screens."
    if p == "lesson":
        return "That was you teaching or correcting me, so I updated what I know."
    return "That was one of my built-in commands."


def report(stats: dict, learned: list) -> str:
    total = stats.get("total", 0)
    if not total:
        base = "I don't have a record of any turns yet this week."
    else:
        base = (f"Of your last {total} requests this week, I handled {stats['own']} myself"
                f" and used the language model for {stats['llm']}.")
        if stats.get("own_first_ms") is not None and stats.get("llm_first_ms") is not None:
            base += (f" My own answers started in about {int(stats['own_first_ms'])} "
                     f"milliseconds, the model's in about "
                     f"{stats['llm_first_ms'] / 1000:.1f} seconds.")
    recent = [x for x in learned if x.get("source") in ("learned", "user")]
    if recent:
        n = len(recent)
        base += (f" I've picked up {n} new way{'s' if n != 1 else ''} of asking for things"
                 f", most recently: {recent[0]['text'].replace(' N', ' a number')}.")
    return base
