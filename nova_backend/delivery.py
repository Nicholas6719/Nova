"""
The single door for anything Nova says that he did not just ask for.

Ported in idea from JARVIS (MiniPCJarvis sidecar/delivery.py + notify_policy.py),
whose rule 5 reads: unprompted messages have a ceiling — claim before speaking,
de-duplicate everything, hard hourly cap. JARVIS learned it from a single
uncommitted SQLite write that let a reminder be re-announced every ten seconds:
about 2,600 messages in one night.

Nova had three ways to speak first, and none of them shared a rule:
  - the proactive monitor (an event about to start, a reminder just overdue),
    whose "never twice" was an in-memory set — so a restart inside the lead
    window announced the same meeting again, although its docstring said not;
  - timers and "remind me in N minutes", calling _announce straight from a
    timer thread;
  - standing watches ("tell me when my battery is charged"), new in Stage 4.
Now all of them come through deliver().

THE RULES
  1. CLAIM BEFORE SPEAKING. A key is inserted into `deliveries` (UNIQUE) and
     committed before a word is said. A key that is already there is never
     said again — across threads and across restarts.
  2. HARD CEILINGS for unprompted kinds: per hour and per day, counted from
     the table, so a restart does not reset them.
  3. QUIET IS NOT SILENT (JARVIS rule 7). Over a ceiling, or inside quiet
     hours, an unprompted item is SHOWN on screen instead of spoken — never
     dropped. A thing he asked for (a timer, a watch) is not subject to the
     ceiling at all: suppressing what he requested is a different failure.
  4. EVERY WRITE COMMITS on a short-lived connection; nothing raises into the
     caller — a delivery that fails to record still gets said once, in the
     log, and never takes a timer thread down.
"""
from __future__ import annotations

import datetime
import logging
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger("nova.delivery")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS deliveries (
    key    TEXT PRIMARY KEY,      -- identity of the THING, not the sentence
    kind   TEXT NOT NULL,
    text   TEXT NOT NULL,
    ts     REAL NOT NULL,
    status TEXT NOT NULL           -- claimed | spoken | shown
);
CREATE INDEX IF NOT EXISTS deliveries_ts ON deliveries(ts);
"""

# notify policy, per kind. `requested`: he asked for it (exempt from the
# ceilings, still de-duplicated). Everything else counts against the budget.
POLICY = {
    "timer":      {"requested": True},
    "watch":      {"requested": True},
    "event_soon": {"requested": False},
    "reminder":   {"requested": False},
    "proactive":  {"requested": False},
}


class Delivery:
    def __init__(self, db_path: Path, config: Optional[dict],
                 speak: Callable[[str], None],
                 show: Optional[Callable[[str], None]] = None) -> None:
        cfg = (config or {}).get("delivery", {})
        self.db_path = Path(db_path)
        self.per_hour = int(cfg.get("max_per_hour", 4))
        self.per_day = int(cfg.get("max_per_day", 12))
        self.quiet = cfg.get("quiet_hours")        # e.g. [23, 7], or None/off
        self._speak = speak
        self._show = show
        self._lock = threading.Lock()
        with self._db() as db:
            db.executescript(_SCHEMA)

    @contextmanager
    def _db(self):
        conn = sqlite3.connect(str(self.db_path), timeout=5.0)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ── the door ─────────────────────────────────────────────────────────────
    def claim(self, key: str, kind: str, text: str) -> bool:
        """Reserve `key`. False if it was already claimed — ever."""
        try:
            with self._lock, self._db() as db:
                cur = db.execute(
                    "INSERT OR IGNORE INTO deliveries (key, kind, text, ts, status) "
                    "VALUES (?,?,?,?, 'claimed')", (key, kind, text, time.time()))
                return cur.rowcount == 1
        except Exception as exc:
            log.warning(f"delivery: could not claim {key!r} ({exc}); saying it anyway")
            return True

    def was_delivered(self, key: str) -> bool:
        try:
            with self._db() as db:
                return db.execute("SELECT 1 FROM deliveries WHERE key=?",
                                  (key,)).fetchone() is not None
        except Exception:
            return False

    def deliver(self, kind: str, text: str, key: Optional[str] = None,
                claimed: bool = False) -> str:
        """Say (or show) one unprompted thing. Returns what happened:
        'spoken', 'shown', or 'duplicate'."""
        text = (text or "").strip()
        if not text:
            return "duplicate"
        key = key or f"{kind}:{uuid.uuid4().hex}"
        if not claimed and not self.claim(key, kind, text):
            log.info(f"[delivery] not repeating {key!r}")
            return "duplicate"
        how = self._route(kind)
        self._mark(key, how)
        log.info(f"[delivery] {how} ({kind}): {text}")
        try:
            if how == "spoken":
                if self._show is not None:
                    self._safe(self._show, text)
                self._speak(text)
            else:
                if self._show is not None:
                    self._safe(self._show, text)
        except Exception as exc:
            log.warning(f"delivery: {how} failed for {key!r}: {exc}")
        return how

    # ── policy ───────────────────────────────────────────────────────────────
    def _route(self, kind: str) -> str:
        if POLICY.get(kind, {"requested": False})["requested"]:
            return "spoken"
        reason = self.hold_reason()
        if reason:
            log.info(f"[delivery] holding to the screen: {reason}")
            return "shown"
        return "spoken"

    def hold_reason(self, now: Optional[float] = None) -> Optional[str]:
        now = now if now is not None else time.time()
        if self.in_quiet_hours(now):
            return "quiet hours"
        unprompted = tuple(k for k, p in POLICY.items() if not p["requested"])
        marks = ",".join("?" * len(unprompted))
        try:
            with self._db() as db:
                hour = db.execute(
                    f"SELECT COUNT(*) FROM deliveries WHERE status='spoken' "
                    f"AND kind IN ({marks}) AND ts >= ?", (*unprompted, now - 3600)).fetchone()[0]
                day = db.execute(
                    f"SELECT COUNT(*) FROM deliveries WHERE status='spoken' "
                    f"AND kind IN ({marks}) AND ts >= ?", (*unprompted, now - 86400)).fetchone()[0]
        except Exception:
            return None
        if hour >= self.per_hour:
            return f"{hour} spoken in the last hour (ceiling {self.per_hour})"
        if day >= self.per_day:
            return f"{day} spoken today (ceiling {self.per_day})"
        return None

    def in_quiet_hours(self, now: Optional[float] = None) -> bool:
        q = self.quiet
        if not q or len(q) != 2:
            return False
        start, end = int(q[0]), int(q[1])
        h = datetime.datetime.fromtimestamp(now or time.time()).hour
        return (start <= h or h < end) if start > end else (start <= h < end)

    def _mark(self, key: str, status: str) -> None:
        try:
            with self._db() as db:
                db.execute("UPDATE deliveries SET status=?, ts=? WHERE key=?",
                           (status, time.time(), key))
        except Exception as exc:
            log.warning(f"delivery: could not record {key!r} as {status}: {exc}")

    @staticmethod
    def _safe(fn, text):
        try:
            fn(text)
        except Exception as exc:
            log.warning(f"delivery: screen notice failed: {exc}")
