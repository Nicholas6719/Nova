"""
Nova Memory v2 — SQLite-backed persistent memory.

The core of Nova's personalization. Nova learns about Nicholas passively over
time and reasons from what it knows, without guessing or fabricating.

Design (see the memory-engine design notes):
  - Facts are STRUCTURED: (category, key) is a canonical UNIQUE identity, so a
    correction supersedes the old value instead of piling up contradictions
    (the #1 flaw in the Jarvis design we improved on). Values render to natural
    English at injection time.
  - Extraction is HYBRID: a cheap per-turn regex fast-path proposes obvious
    facts; a wake-mode LLM reconciliation pass (in nova.py, off the live path)
    decides insert / update / delete / ignore against what's already stored.
  - Injection is CAPPED and RANKED — not every fact every turn — so a small
    model's context window isn't flooded.
  - The memory layer NEVER speaks. Confirmations ("got it") are a conversational
    behavior driven by the system prompt, not by this module.

Thread safety: one lock around all DB access; WAL + busy_timeout so the
background wake-mode worker and the main loop never corrupt each other.

Storage: ~/Library/Application Support/Nova/nova_memory.db (NOVA_DATA_DIR).
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger("nova.memory")


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


# How many facts to inject into the prompt at most (anti-bloat on a small model).
MAX_INJECTED_FACTS = 40


class NovaMemory:
    def __init__(self, db_path: Path) -> None:
        self.db_path  = db_path
        self._lock    = threading.Lock()
        self._conn_obj: Optional[sqlite3.Connection] = None
        self._last_identity: Optional[tuple[str, str]] = None  # (category, key)
        self._init_db()
        log.info(f"Memory ready: {db_path}")

    # ── Connection ──────────────────────────────────────────────────────────────────
    def _conn(self) -> sqlite3.Connection:
        """One shared connection, guarded by self._lock. WAL + busy_timeout make
        concurrent access from the wake-mode worker and main loop safe."""
        if self._conn_obj is None:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("PRAGMA foreign_keys=ON")
            self._conn_obj = conn
        return self._conn_obj

    # ── Schema ────────────────────────────────────────────────────────────────────
    def _init_db(self) -> None:
        with self._lock:
            conn = self._conn()
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS facts (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    category   TEXT NOT NULL,          -- e.g. 'allergy', 'preference', 'identity'
                    key        TEXT NOT NULL,          -- canonical, e.g. 'peanuts', 'favorite_color'
                    value      TEXT NOT NULL,          -- structured value, e.g. 'red'
                    confidence TEXT NOT NULL DEFAULT 'high',   -- 'high' | 'tentative'
                    source     TEXT NOT NULL DEFAULT 'llm',    -- 'explicit' | 'regex' | 'llm'
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(category, key)
                );

                CREATE TABLE IF NOT EXISTS conversations (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    role       TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content    TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
            """)
            conn.commit()

    # ═════════════════════════════════════════════════════════════════════════════
    # Facts — structured, canonical (category, key) → value
    # ═════════════════════════════════════════════════════════════════════════════
    def upsert_fact(
        self,
        category: str,
        key: str,
        value: str,
        confidence: str = "high",
        source: str = "llm",
    ) -> None:
        """Insert or supersede a fact. Because (category, key) is UNIQUE, a new
        value for the same canonical key REPLACES the old one — corrections just
        work, no contradiction pile-up."""
        category = _norm(category)
        key      = _norm(key)
        value    = value.strip()
        if not category or not key or not value:
            return
        with self._lock:
            conn = self._conn()
            conn.execute(
                """INSERT INTO facts (category, key, value, confidence, source, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(category, key) DO UPDATE SET
                       value      = excluded.value,
                       confidence = excluded.confidence,
                       source     = excluded.source,
                       updated_at = excluded.updated_at""",
                (category, key, value, confidence, source, _now(), _now()),
            )
            conn.commit()
        self._last_identity = (category, key)
        log.debug(f"Fact upserted: {category}/{key} = {value} ({source})")

    def delete_fact(self, category: str, key: str) -> bool:
        """Remove a fact by canonical identity. Returns True if something was
        deleted (used for negation handling: 'I cancelled Netflix')."""
        category = _norm(category)
        key      = _norm(key)
        with self._lock:
            conn = self._conn()
            cur = conn.execute(
                "DELETE FROM facts WHERE category = ? AND key = ?", (category, key)
            )
            conn.commit()
            deleted = cur.rowcount > 0
        if deleted:
            log.debug(f"Fact deleted: {category}/{key}")
        return deleted

    def get_fact(self, category: str, key: str) -> Optional[str]:
        category = _norm(category)
        key      = _norm(key)
        with self._lock:
            conn = self._conn()
            row = conn.execute(
                "SELECT value FROM facts WHERE category = ? AND key = ?", (category, key)
            ).fetchone()
        return row["value"] if row else None

    def all_facts(self) -> list[dict]:
        """All facts as structured dicts, most-recently-updated first."""
        with self._lock:
            conn = self._conn()
            rows = conn.execute(
                """SELECT category, key, value, confidence, source, updated_at
                   FROM facts ORDER BY updated_at DESC"""
            ).fetchall()
        return [dict(r) for r in rows]

    def facts_in_category(self, category: str) -> list[dict]:
        category = _norm(category)
        with self._lock:
            conn = self._conn()
            rows = conn.execute(
                """SELECT category, key, value FROM facts
                   WHERE category = ? ORDER BY updated_at DESC""",
                (category,),
            ).fetchall()
        return [dict(r) for r in rows]

    def last_identity(self) -> Optional[tuple[str, str]]:
        """(category, key) of the most recent upsert — for 'actually, it's X'."""
        return self._last_identity

    # ═════════════════════════════════════════════════════════════════════════════
    # Conversation turns
    # ═════════════════════════════════════════════════════════════════════════════
    def add_turn(self, role: str, content: str) -> None:
        with self._lock:
            conn = self._conn()
            conn.execute(
                "INSERT INTO conversations (role, content, created_at) VALUES (?, ?, ?)",
                (role, content.strip(), _now()),
            )
            conn.commit()

    def get_recent_turns(self, n: int = 20) -> list[dict]:
        """The n most recent turns in chronological order (oldest first)."""
        with self._lock:
            conn = self._conn()
            rows = conn.execute(
                "SELECT role, content FROM conversations ORDER BY id DESC LIMIT ?",
                (n,),
            ).fetchall()
        turns = [{"role": r["role"], "content": r["content"]} for r in rows]
        turns.reverse()
        return turns

    # ═════════════════════════════════════════════════════════════════════════════
    # LLM context builder — capped, rendered to natural English
    # ═════════════════════════════════════════════════════════════════════════════
    def get_context_for_llm(self, limit: int = MAX_INJECTED_FACTS) -> str:
        """Build the memory block injected into the system prompt.

        Capped at `limit` most-recently-updated facts (anti-bloat), rendered as
        plain English lines grouped by category. Only stored facts are injected —
        never raw transcript dumps — so the model has nothing to confabulate from.
        """
        with self._lock:
            conn = self._conn()
            rows = conn.execute(
                """SELECT category, key, value, confidence
                   FROM facts ORDER BY updated_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        if not rows:
            return ""

        lines = []
        for r in rows:
            hedge = "" if r["confidence"] == "high" else " (you're not fully sure of this)"
            lines.append(f"  - {_render_fact(r['category'], r['key'], r['value'])}{hedge}")
        return "What you know about him:\n" + "\n".join(lines)

    def facts_for_readback(self) -> list[str]:
        """Natural-English fact sentences for an explicit 'what do you know about
        me' answer. Excludes the user's own name."""
        out = []
        for r in self.all_facts():
            if r["category"] == "identity" and r["key"] == "name":
                continue
            out.append(_render_fact(r["category"], r["key"], r["value"]))
        return out


# ── Helpers ───────────────────────────────────────────────────────────────────────
def _norm(s: str) -> str:
    return s.lower().strip().replace(" ", "_")


def _render_fact(category: str, key: str, value: str) -> str:
    """Render a structured fact as a natural English clause for prompt injection.

    Kept deliberately simple and category-driven; the LLM turns these into
    conversational phrasing when it speaks. Unknown categories fall back to a
    readable 'key: value' form.
    """
    k = key.replace("_", " ")
    if category == "allergy":
        return f"He is allergic to {value}."
    if category == "identity":
        return f"His {k} is {value}."
    if category == "preference":
        # key is the thing, value is the stance, e.g. key='coffee' value='likes'
        return f"He {value} {k}."
    if category == "favorite":
        return f"His favorite {k} is {value}."
    if category == "routine":
        # Value is a full clause, e.g. "goes to the gym every morning at 5:30".
        return f"He {value}." if not value.lower().startswith("he ") else f"{value}."
    if category == "location":
        return f"His {k} is {value}."
    if category == "possession":
        return f"His {k} is {value}."
    return f"{k}: {value}"
