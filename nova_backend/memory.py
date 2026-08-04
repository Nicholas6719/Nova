"""
Nova Memory — SQLite-backed persistent memory.

This is the core of Nova's personalization. Every fact Nicholas shares,
every episode Nova observes, every conversation turn — all stored locally
in ~/Library/Application Support/Nova/nova_memory.db.

Tables:
  facts          — explicit key/value personal facts ("my name is Nick")
  episodes       — semantic memories ("Nicholas prefers dark mode at night")
  conversations  — full turn history (recent N kept, older summarized)
  user_profile   — inferred attributes with confidence scores

Thread safety: NSLock-style threading.Lock around all DB operations.
Summarization: runs on background thread, never blocks the pipeline.
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


class NovaMemory:
    def __init__(self, db_path: Path) -> None:
        self.db_path  = db_path
        self._lock    = threading.Lock()
        self._last_key: Optional[str] = None
        self._init_db()
        log.info(f"Memory ready: {db_path}")

    # ── Schema ────────────────────────────────────────────────────────────────────
    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS facts (
                    key        TEXT PRIMARY KEY,
                    value      TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS episodes (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    summary    TEXT NOT NULL,
                    context    TEXT DEFAULT '',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS conversations (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    role       TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content    TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS user_profile (
                    attribute  TEXT PRIMARY KEY,
                    value      TEXT NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    updated_at TEXT NOT NULL
                );
            """)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")   # better concurrent read performance
        return conn

    # ═════════════════════════════════════════════════════════════════════════════
    # Facts — explicit, user-stated key/value pairs
    # ═════════════════════════════════════════════════════════════════════════════
    def save_fact(self, key: str, value: str) -> None:
        key = _normalize_key(key)
        with self._lock, self._conn() as conn:
            conn.execute(
                """INSERT INTO facts (key, value, updated_at) VALUES (?, ?, ?)
                   ON CONFLICT(key) DO UPDATE SET value = excluded.value,
                                                  updated_at = excluded.updated_at""",
                (key, value.strip(), _now()),
            )
        self._last_key = key
        log.debug(f"Fact saved: {key} = {value}")

    def recall_fact(self, key: str) -> Optional[str]:
        key = _normalize_key(key)
        with self._lock, self._conn() as conn:
            row = conn.execute(
                "SELECT value FROM facts WHERE key = ?", (key,)
            ).fetchone()
        self._last_key = key
        return row["value"] if row else None

    def delete_fact(self, key: str) -> None:
        key = _normalize_key(key)
        with self._lock, self._conn() as conn:
            conn.execute("DELETE FROM facts WHERE key = ?", (key,))
        log.debug(f"Fact deleted: {key}")

    def all_facts(self) -> dict[str, str]:
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                "SELECT key, value FROM facts ORDER BY key"
            ).fetchall()
        return {r["key"]: r["value"] for r in rows}

    def last_key(self) -> Optional[str]:
        """Return the key from the most recent save/recall (for 'actually it's X' corrections)."""
        return self._last_key

    # ═════════════════════════════════════════════════════════════════════════════
    # Episodes — semantic memories observed by Nova
    # ═════════════════════════════════════════════════════════════════════════════
    def save_episode(self, summary: str, context: str = "") -> None:
        """
        Store an inferred or summarized memory.
        E.g. "Nicholas mentioned he prefers working late at night."
        """
        with self._lock, self._conn() as conn:
            conn.execute(
                "INSERT INTO episodes (summary, context, created_at) VALUES (?, ?, ?)",
                (summary.strip(), context.strip(), _now()),
            )
        log.debug(f"Episode saved: {summary[:60]}")

    def get_recent_episodes(self, n: int = 5) -> list[dict]:
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                "SELECT summary, context, created_at FROM episodes ORDER BY id DESC LIMIT ?",
                (n,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ═════════════════════════════════════════════════════════════════════════════
    # Conversation turns
    # ═════════════════════════════════════════════════════════════════════════════
    def add_turn(self, role: str, content: str) -> None:
        with self._lock, self._conn() as conn:
            conn.execute(
                "INSERT INTO conversations (role, content, created_at) VALUES (?, ?, ?)",
                (role, content.strip(), _now()),
            )

    def get_recent_turns(self, n: int = 20) -> list[dict]:
        """Return the n most recent turns in chronological order (oldest first)."""
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                "SELECT role, content FROM conversations ORDER BY id DESC LIMIT ?",
                (n,),
            ).fetchall()
        turns = [{"role": r["role"], "content": r["content"]} for r in rows]
        turns.reverse()
        return turns

    # ═════════════════════════════════════════════════════════════════════════════
    # User profile — inferred attributes (confidence-weighted)
    # ═════════════════════════════════════════════════════════════════════════════
    def update_profile(self, attribute: str, value: str, confidence: float = 1.0) -> None:
        key = _normalize_key(attribute)
        with self._lock, self._conn() as conn:
            conn.execute(
                """INSERT INTO user_profile (attribute, value, confidence, updated_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(attribute) DO UPDATE SET
                     value = excluded.value,
                     confidence = excluded.confidence,
                     updated_at = excluded.updated_at""",
                (key, value, confidence, _now()),
            )

    def get_profile(self) -> dict[str, str]:
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                "SELECT attribute, value FROM user_profile WHERE confidence >= 0.7 ORDER BY attribute"
            ).fetchall()
        return {r["attribute"]: r["value"] for r in rows}

    # ═════════════════════════════════════════════════════════════════════════════
    # LLM context builder
    # ═════════════════════════════════════════════════════════════════════════════
    def get_context_for_llm(self) -> str:
        """
        Build a concise memory summary to inject into the system prompt.
        Called on every LLM turn — should be fast.
        """
        facts    = self.all_facts()
        profile  = self.get_profile()
        episodes = self.get_recent_episodes(n=5)

        parts = []

        if facts:
            lines = "\n".join(
                f"  {k.replace('_', ' ')}: {v}" for k, v in facts.items()
            )
            parts.append(f"Explicit facts:\n{lines}")

        if profile:
            lines = "\n".join(
                f"  {k.replace('_', ' ')}: {v}" for k, v in profile.items()
            )
            parts.append(f"Inferred preferences:\n{lines}")

        if episodes:
            lines = "\n".join(f"  - {ep['summary']}" for ep in episodes)
            parts.append(f"Recent memories:\n{lines}")

        return "\n\n".join(parts)

    # ═════════════════════════════════════════════════════════════════════════════
    # Background summarization
    # ═════════════════════════════════════════════════════════════════════════════
    def summarize_old_turns(self, keep_recent: int = 20) -> None:
        """
        Called from a background daemon thread after each LLM response.
        Never blocks the pipeline.

        Strategy: if total turns > keep_recent * 2, compress the oldest
        batch into an episode summary and delete them.
        """
        with self._lock, self._conn() as conn:
            total = conn.execute(
                "SELECT COUNT(*) AS n FROM conversations"
            ).fetchone()["n"]

        if total <= keep_recent * 2:
            return

        cutoff = total - keep_recent
        with self._lock, self._conn() as conn:
            old_rows = conn.execute(
                "SELECT id, role, content FROM conversations ORDER BY id ASC LIMIT ?",
                (cutoff,),
            ).fetchall()

        if not old_rows:
            return

        # Simple concatenated summary (LLM summarization can be added later)
        turns = [f"{r['role']}: {r['content']}" for r in old_rows]
        summary = " | ".join(turns[:15])
        self.save_episode(
            f"Earlier conversation ({len(old_rows)} turns): {summary[:400]}"
        )

        ids = [r["id"] for r in old_rows]
        placeholders = ",".join("?" * len(ids))
        with self._lock, self._conn() as conn:
            conn.execute(
                f"DELETE FROM conversations WHERE id IN ({placeholders})", ids
            )

        log.debug(f"Summarized and pruned {len(old_rows)} old turns.")


# ── Helpers ───────────────────────────────────────────────────────────────────────
def _normalize_key(key: str) -> str:
    return key.lower().strip().replace(" ", "_")
