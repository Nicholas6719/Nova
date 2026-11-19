"""
Standing watches — "tell me when ..." that are actually kept, across restarts.

Ported in idea from JARVIS (MiniPCJarvis sidecar/standing.py). The reason it
exists on Nova is the honesty filter's other half: the LLM used to say "I'll
let you know when it's charged" with nothing watching (honesty.py now replaces
that sentence with the truth). This is how the truth becomes "yes": a watch is
a ROW, checked by a loop, delivered once through the single door
(delivery.py, kind "watch" — he asked for it, so no ceiling holds it back).

LOCAL READINGS ONLY, deliberately: battery, CPU, memory. A watch on a stock
price or the weather would poll the network over and over without being asked
each time — that is his privacy line (CLAUDE.md invariant 3), so it is not
built until he says so.

Rules:
  - one-shot: a watch fires once and is marked fired; a restart never refires it;
  - a condition that is ALREADY true is answered now and not stored ("it's
    already at 85 percent") — a watch that fires the instant it is set is noise;
  - parsing is strict regex: "tell me about my battery" is a question, not a
    watch, and goes on to be answered as one.
"""
from __future__ import annotations

import logging
import re
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger("nova.standing")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS standing_watches (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    ts       REAL NOT NULL,
    metric   TEXT NOT NULL,          -- battery | battery_full | cpu | memory
    op       TEXT NOT NULL,          -- >= | <=
    value    REAL NOT NULL,
    said     TEXT NOT NULL,          -- his words, for "what are you watching"
    status   TEXT NOT NULL DEFAULT 'active',   -- active | fired | cancelled
    fired_ts REAL
);
"""

_ASK = r"^\s*(?:nova[,\s]+)?(?:please\s+)?(?:can\s+you\s+)?(?:tell|let|warn|alert|notify|ping)\s+me\s+(?:know\s+)?" \
       r"(?:when|if|once|as\s+soon\s+as|the\s+moment)\s+"
# Not a prefix of a bigger number: "above 1000" was read as 100 (audit).
_NUM = r"(\d{1,3}(?:\.\d+)?)(?![\d.])\s*(?:%|percent)?"
_PATTERNS: list[tuple[re.Pattern, Callable[[re.Match], tuple]]] = [
    (re.compile(_ASK + r"(?:my\s+|the\s+)?(?:battery|mac|laptop|computer)\s+(?:is\s+)?"
                r"(?:fully\s+charged|charged|full|done\s+charging|finished\s+charging|at\s+100)",
                re.I), lambda m: ("battery_full", ">=", 100.0)),
    # Words that SAY the direction fix it...
    (re.compile(_ASK + r"(?:my\s+|the\s+)?battery\s+(?:goes\s+(?:above|over|past)|is\s+(?:above|over)|"
                r"climbs\s+(?:above|over|past|to)|is\s+back\s+(?:up\s+)?to)\s+" + _NUM, re.I),
     lambda m: ("battery", ">=", float(m.group(1)))),
    (re.compile(_ASK + r"(?:my\s+|the\s+)?battery\s+(?:(?:drops|falls|goes|gets|is)\s+(?:below|under|"
                r"down\s+to)|(?:drops|falls)\s+to)\s+" + _NUM, re.I),
     lambda m: ("battery", "<=", float(m.group(1)))),
    # ...and words that do not ("is at", "hits", "reaches", "gets to", "goes
    # to") take it from where the battery is NOW. They were always read as
    # rising, so "tell me when my battery is at 20 percent" — the most natural
    # low-battery warning — answered "it's already at 80" (audit, 2026-11-19).
    (re.compile(_ASK + r"(?:my\s+|the\s+)?battery\s+(?:hits|reaches|gets\s+to|is\s+at|goes\s+to|"
                r"gets\s+down\s+to)\s+" + _NUM, re.I),
     lambda m: ("battery", "auto", float(m.group(1)))),
    (re.compile(_ASK + r"(?:my\s+|the\s+)?battery\s+(?:is\s+|gets\s+|is\s+getting\s+|runs\s+)?low", re.I),
     lambda m: ("battery", "<=", 20.0)),
    (re.compile(_ASK + r"(?:my\s+|the\s+)?(?:cpu|processor)(?:\s+usage)?\s+(?:drops|falls|goes|is|gets)\s+"
                r"(?:below|under|down\s+to)\s+" + _NUM, re.I),
     lambda m: ("cpu", "<=", float(m.group(1)))),
    (re.compile(_ASK + r"(?:my\s+|the\s+)?(?:cpu|processor)(?:\s+usage)?\s+(?:goes|gets|is|climbs|spikes)\s+"
                r"(?:above|over|past)\s+" + _NUM, re.I),
     lambda m: ("cpu", ">=", float(m.group(1)))),
    (re.compile(_ASK + r"(?:my\s+|the\s+)?(?:cpu|processor)(?:\s+usage)?\s+(?:hits|reaches|gets\s+to|"
                r"is\s+at|goes\s+to)\s+" + _NUM, re.I),
     lambda m: ("cpu", "auto", float(m.group(1)))),
    (re.compile(_ASK + r"(?:my\s+|the\s+)?(?:cpu|processor)\s+(?:calms\s+down|settles\s+down|is\s+idle)",
                re.I), lambda m: ("cpu", "<=", 30.0)),
    (re.compile(_ASK + r"(?:my\s+|the\s+)?(?:memory|ram)(?:\s+usage)?\s+(?:goes|gets|is|climbs)\s+"
                r"(?:above|over|past)\s+" + _NUM, re.I),
     lambda m: ("memory", ">=", float(m.group(1)))),
    (re.compile(_ASK + r"(?:my\s+|the\s+)?(?:memory|ram)(?:\s+usage)?\s+(?:drops|falls|goes|gets|is)\s+"
                r"(?:below|under|down\s+to)\s+" + _NUM, re.I),
     lambda m: ("memory", "<=", float(m.group(1)))),
    (re.compile(_ASK + r"(?:my\s+|the\s+)?(?:memory|ram)(?:\s+usage)?\s+(?:hits|reaches|gets\s+to|"
                r"is\s+at|goes\s+to)\s+" + _NUM, re.I),
     lambda m: ("memory", "auto", float(m.group(1)))),
]
# End-anchored: "what are you watching on netflix" is not about watches.
_LIST_RE = re.compile(r"^\s*(?:nova[,\s]+)?what\s+(?:are\s+you\s+watching(?:\s+for\s+me)?|watches\s+(?:do\s+i\s+have|"
                      r"are\s+(?:set|running|on)))(?:\s+(?:right\s+)?now)?\s*(?:nova)?[\s.!?]*$", re.I)
_CANCEL_RE = re.compile(
    r"^\s*(?:nova[,\s]+)?(?:stop|cancel|forget|delete|remove|clear)\s+(?:watching\s+(?:my\s+|the\s+)?"
    r"(?P<a>battery|cpu|processor|memory|ram)|(?:the\s+|my\s+)?(?P<b>battery|cpu|processor|memory|ram)\s+"
    r"(?:watch|alert)|(?P<all>all\s+(?:the\s+|my\s+)?(?:watches|alerts)|(?:the\s+|my\s+)?watches))\b", re.I)


def parse(text: str) -> Optional[dict]:
    """A watch request -> {"metric", "op", "value"}; None for anything else."""
    t = (text or "").strip()
    for rx, build in _PATTERNS:
        m = rx.search(t)
        if m:
            metric, op, value = build(m)
            if not (0 <= value <= 100):
                return None
            return {"metric": metric, "op": op, "value": value}
    return None


def claims(text: str) -> bool:
    """Would this stage take the sentence? Pure — for routing logs and tests."""
    t = text or ""
    return bool(parse(t) or _LIST_RE.search(t) or _CANCEL_RE.search(t))


def describe(w: dict, future: bool = True) -> str:
    """Spoken: 'your battery is fully charged', 'your CPU goes above 90 percent'."""
    v = int(w["value"])
    m, op = w["metric"], w["op"]
    if m == "battery_full":
        return "your battery is fully charged"
    noun = {"battery": "your battery", "cpu": "your CPU", "memory": "your memory use"}[m]
    if m == "battery":
        return f"{noun} {'reaches' if op == '>=' else 'drops to'} {v} percent"
    return f"{noun} {'goes above' if op == '>=' else 'drops below'} {v} percent"


class StandingWatches:
    def __init__(self, db_path: Path, read: Callable[[str], Optional[float]],
                 deliver: Callable[[str, str, str], object],
                 config: Optional[dict] = None) -> None:
        cfg = (config or {}).get("standing", {})
        self.enabled = bool(cfg.get("enabled", True))
        self.poll_s = float(cfg.get("poll_seconds", 30))
        self.db_path = Path(db_path)
        self._read = read            # metric -> current value (0-100) or None
        self._deliver = deliver      # (kind, text, key)
        self._thread: Optional[threading.Thread] = None
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

    # ── voice ────────────────────────────────────────────────────────────────
    def handle(self, text: str) -> Optional[str]:
        """The spoken reply if this is a watch request / list / cancel, else None."""
        if _LIST_RE.search(text or ""):
            ws = self.active()
            if not ws:
                return "I'm not watching anything for you right now."
            return "I'll tell you when " + self._join([describe(w) for w in ws]) + "."
        m = _CANCEL_RE.search(text or "")
        if m:
            if m.group("all"):
                n = self.cancel(None)
            else:
                metric = (m.group("a") or m.group("b")).lower()
                metric = {"processor": "cpu", "ram": "memory"}.get(metric, metric)
                n = self.cancel(metric)
            if not n:
                return "I wasn't watching that."
            return "Done, I've stopped watching that." if n == 1 else f"Done, I've stopped all {n} watches."
        w = parse(text)
        if w is None:
            return None
        now = self._reading_for(w)
        if w["op"] == "auto":
            w["op"] = self.direction(w, now)
        if now is not None and self._met(w, now):
            if w["metric"] == "battery_full":
                return "Your battery is already fully charged."
            return f"It's already at {int(now)} percent."
        if any((x["metric"], x["op"], x["value"]) == (w["metric"], w["op"], w["value"])
               for x in self.active()):
            return f"I'm already watching for that. I'll tell you when {describe(w)}."
        self.add(w, text)
        if now is None:
            return (f"I'll tell you when {describe(w)}, though I can't read it right "
                    "now, so I'll keep trying.")
        return f"Okay. I'll tell you when {describe(w)}."

    # ── storage ──────────────────────────────────────────────────────────────
    def add(self, w: dict, said: str) -> int:
        with self._db() as db:
            cur = db.execute("INSERT INTO standing_watches (ts, metric, op, value, said) "
                             "VALUES (?,?,?,?,?)",
                             (time.time(), w["metric"], w["op"], w["value"], said.strip()))
            wid = cur.lastrowid
        log.info(f"[watch] #{wid} set: {describe(w)}")
        return wid

    def active(self) -> list[dict]:
        with self._db() as db:
            return [dict(r) for r in db.execute(
                "SELECT * FROM standing_watches WHERE status='active' ORDER BY id")]

    def cancel(self, metric: Optional[str]) -> int:
        with self._db() as db:
            if metric is None:
                cur = db.execute("UPDATE standing_watches SET status='cancelled' "
                                 "WHERE status='active'")
            elif metric == "battery":
                cur = db.execute("UPDATE standing_watches SET status='cancelled' "
                                 "WHERE status='active' AND metric IN ('battery','battery_full')")
            else:
                cur = db.execute("UPDATE standing_watches SET status='cancelled' "
                                 "WHERE status='active' AND metric=?", (metric,))
            return cur.rowcount

    # ── the loop ─────────────────────────────────────────────────────────────
    def check_once(self) -> int:
        """Fire every active watch whose condition now holds. Returns how many."""
        fired = 0
        cache: dict = {}
        for w in self.active():
            if w["metric"] not in cache:
                cache[w["metric"]] = self._reading_for(w)
            now = cache[w["metric"]]
            if now is None or not self._met(w, now):
                continue
            # Marked fired BEFORE it is said (claim before speaking): a crash
            # between the two costs one message, never a repeat.
            with self._db() as db:
                cur = db.execute("UPDATE standing_watches SET status='fired', fired_ts=? "
                                 "WHERE id=? AND status='active'", (time.time(), w["id"]))
                if cur.rowcount != 1:
                    continue
            line = self._fired_line(w, now)
            self._deliver("watch", line, f"watch:{w['id']}")
            fired += 1
        return fired

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return

        def _loop():
            while True:
                time.sleep(self.poll_s)
                try:
                    self.check_once()
                except Exception as exc:
                    log.warning(f"[watch] check failed: {exc}")
        self._thread = threading.Thread(target=_loop, name="nova-standing", daemon=True)
        self._thread.start()

    # ── helpers ──────────────────────────────────────────────────────────────
    def _reading_for(self, w: dict) -> Optional[float]:
        metric = "battery" if w["metric"] == "battery_full" else w["metric"]
        try:
            v = self._read(metric)
            return None if v is None else float(v)
        except Exception as exc:
            log.debug(f"[watch] could not read {metric}: {exc}")
            return None

    @staticmethod
    def direction(w: dict, now: Optional[float]) -> str:
        """"Tell me when my battery is at 20": falling from 80, rising from 5.
        With no reading, the likely meaning for that metric."""
        if now is not None and now != w["value"]:
            return ">=" if now < w["value"] else "<="
        if w["metric"] == "battery":
            return "<=" if w["value"] <= 50 else ">="
        return ">=" if w["value"] >= 50 else "<="

    @staticmethod
    def _met(w: dict, now: float) -> bool:
        return now >= w["value"] if w["op"] == ">=" else now <= w["value"]

    @staticmethod
    def _fired_line(w: dict, now: float) -> str:
        if w["metric"] == "battery_full":
            return "Your battery is fully charged."
        noun = {"battery": "Your battery", "cpu": "Your CPU", "memory": "Your memory use"}[w["metric"]]
        return f"{noun} is at {int(now)} percent now."

    @staticmethod
    def _join(items: list[str]) -> str:
        return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]
