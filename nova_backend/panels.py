"""
Panel payloads — the structured half of Nova's answers.

Every deterministic handler already computes real structure and then flattens it
into one spoken sentence. The weather engine has a temperature; the calendar has
events with times; the file manager has candidates with paths. All of that is
built and then thrown away.

This is how that structure reaches the screen. Nova SPEAKS the sentence and
SHOWS the data, and the two are different channels with different rules:

  - the spoken half stays under the listener rules (no markdown, no lists, no
    paths read aloud, invariant 10)
  - the panel can be as rich as it likes, because it is read with eyes

Critically, the LLM never touches a panel. Every number on screen comes from
the same payload the engine already templated, which is what keeps the panel
free of invented figures.

Block kinds are deliberately few. A small vocabulary the Swift side can render
generically beats a bespoke layout per feature, and it means a new panel is a
backend change only.
"""

from __future__ import annotations

from typing import Any, Optional


# ── Block builders ────────────────────────────────────────────────────────────

def stat(value: str, label: str = "", detail: str = "",
         accent: str = "") -> dict:
    """One big number. The thing you want to read from across the room."""
    return {"kind": "stat", "value": str(value), "label": label,
            "detail": detail, "accent": accent}


def rows(pairs: list[tuple[str, str]], title: str = "") -> dict:
    """Label/value pairs. Humidity 87%, Wind 6 mph."""
    return {"kind": "rows", "title": title,
            "rows": [{"label": str(a), "value": str(b)} for a, b in pairs]}


def items(entries: list[dict], title: str = "") -> dict:
    """A list of things: events, files, facts, headlines.

    Each entry may carry title / detail / meta / accent. Missing keys are
    simply not rendered, so a caller never has to pad.
    """
    clean = []
    for e in entries:
        row = {k: str(v) for k, v in e.items()
               if k in ("title", "detail", "meta", "accent") and v not in (None, "")}
        if row:
            clean.append(row)
    return {"kind": "items", "title": title, "items": clean}


def text(body: str, title: str = "") -> dict:
    """A paragraph. Used where there is genuinely prose to show."""
    return {"kind": "text", "title": title, "text": str(body)}


def note(body: str) -> dict:
    """A quiet aside — why something is empty, or what is missing."""
    return {"kind": "note", "text": str(body)}


def panel(title: str, blocks: list[Optional[dict]],
          subtitle: str = "") -> dict:
    """Assemble a panel, dropping any block a caller left as None.

    Callers build blocks conditionally ("now playing" only when something is
    playing), and letting them pass None keeps that readable at the call site
    instead of a pile of appends.
    """
    return {
        "title": title,
        "subtitle": subtitle,
        "blocks": [b for b in blocks if b],
    }
