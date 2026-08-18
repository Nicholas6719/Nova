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

import time
from typing import Any, Callable, Optional


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


def metrics(readings: list[dict], title: str = "") -> dict:
    """A ROW of instrumentation — CPU, memory, battery — not a card.

    Nicholas asked for this specifically as a line rather than a fourth box:
    it is glanceable furniture, not an answer, and a box gives it the same
    weight as his calendar. Each reading carries a label, a templated value,
    and optionally a 0..1 `pct` so the level is readable without reading the
    number, plus a `flag` (the charging bolt) and an `alert` for the one case
    that should catch his eye.

    Like every other block, the values are templated by the engine. The model
    never phrases a number that ends up here.
    """
    clean = []
    for r in readings:
        row = {"label": str(r.get("label", "")), "value": str(r.get("value", ""))}
        pct = r.get("pct")
        if isinstance(pct, (int, float)):
            row["pct"] = max(0.0, min(1.0, float(pct)))
        if r.get("flag"):
            row["flag"] = str(r["flag"])
        if r.get("alert"):
            row["alert"] = True
        if row["label"] or row["value"]:
            clean.append(row)
    return {"kind": "metrics", "title": title, "metrics": clean}


def steps(entries: list[dict], title: str = "", detail: str = "") -> dict:
    """What Nova is doing, WHILE she is doing it.

    Every deterministic handler already walks a sequence — open the browser,
    run the search, read the page, summarise — and until now the only evidence
    of any of it was a sentence at the end. This is that sequence, streamed as
    it happens, which is the difference between waiting and watching.

    Each entry is a label plus a state: "done", "running", or "pending".
    `meta` carries the elapsed time. A handler pushes the same block repeatedly
    with states advanced; the screen is the latest push, never an append log,
    so a re-render can never duplicate a step.
    """
    clean = []
    for e in entries:
        state = str(e.get("state", "pending")).lower()
        if state not in ("done", "running", "pending", "failed"):
            state = "pending"
        clean.append({"label": str(e.get("label", "")), "state": state,
                      "meta": str(e.get("meta", ""))})
    return {"kind": "steps", "title": title, "detail": detail, "steps": clean}


def at(block: Optional[dict], slot: str, card: str = "") -> Optional[dict]:
    """Stamp a block with where on the home screen it belongs.

    Slots are named (L1..L3, R1..R3, and "status" for the bottom row) so he can
    move a card by voice and it stays where he put it. This is a stamp rather
    than a new payload shape on purpose: a client that knows nothing about
    slots still renders the blocks in list order, so the panel degrades to
    exactly what it was before slots existed.
    """
    if block is None:
        return None
    block["slot"] = slot
    if card:
        block["card"] = card
    return block


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


# ── Showing the work ──────────────────────────────────────────────────────────

class Progress:
    """A step list a handler updates as it works, pushed to the screen live.

    Until now every deterministic handler walked a sequence — open the browser,
    run the search, read the page — and the only evidence any of it happened
    was one sentence at the end. This is that sequence, on screen, advancing.
    The difference between waiting and watching.

    The caller declares the whole sequence up front, so the steps still to come
    are visible from the first frame — a list that grows one line at a time
    tells him nothing about how much is left.

    Nothing here raises. A screen that fails to update must never take down the
    work it was describing, so every push is wrapped: `sink` failing costs the
    animation and nothing else.
    """

    def __init__(self, sink: Optional[Callable[[dict], None]],
                 title: str, labels: list[str], detail: str = "") -> None:
        self._sink = sink
        self._title = title
        self._detail = detail
        self._labels = list(labels)
        self._states = ["pending"] * len(labels)
        self._metas = [""] * len(labels)
        self._at = -1
        self._started = 0.0
        self._extra: list[dict] = []

    # ── Driving it ────────────────────────────────────────────────────────────
    def start(self) -> None:
        """Show the list with the first step running."""
        self._advance_to(0)

    def advance(self) -> None:
        """Finish the current step and start the next."""
        self._advance_to(self._at + 1)

    def fail(self, why: str = "") -> None:
        """Mark the current step failed and stop. The list stays on screen —
        a step that went red is more useful than a screen that went blank."""
        if 0 <= self._at < len(self._states):
            self._states[self._at] = "failed"
            self._metas[self._at] = self._elapsed()
        if why:
            self._extra.append(note(why))
        self._push()

    def show(self, *blocks: Optional[dict]) -> None:
        """Attach findings under the steps as they arrive."""
        self._extra.extend(b for b in blocks if b)
        self._push()

    def finish(self, *blocks: Optional[dict]) -> None:
        """Complete every remaining step and settle."""
        if 0 <= self._at < len(self._states) and self._states[self._at] == "running":
            self._states[self._at] = "done"
            self._metas[self._at] = self._elapsed()
        for i, st in enumerate(self._states):
            if st == "pending":
                self._states[i] = "done"
        self._extra.extend(b for b in blocks if b)
        self._push()

    # ── Internals ─────────────────────────────────────────────────────────────
    def _advance_to(self, index: int) -> None:
        if 0 <= self._at < len(self._states) and self._states[self._at] == "running":
            self._states[self._at] = "done"
            self._metas[self._at] = self._elapsed()
        self._at = index
        if 0 <= index < len(self._states):
            self._states[index] = "running"
            self._started = time.time()
        self._push()

    def _elapsed(self) -> str:
        if not self._started:
            return ""
        return f"{time.time() - self._started:.1f}s"

    def payload(self) -> dict:
        entries = [{"label": l, "state": st, "meta": m}
                   for l, st, m in zip(self._labels, self._states, self._metas)]
        return panel(title=self._title, subtitle="",
                     blocks=[steps(entries, detail=self._detail)] + self._extra)

    def _push(self) -> None:
        if self._sink is None:
            return
        try:
            self._sink(self.payload())
        except Exception:                 # the screen, never the work
            pass
