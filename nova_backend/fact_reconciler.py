"""
Fact reconciler — the LLM SLOW PATH of Nova's hybrid memory learning.

Runs once per conversation, in wake mode (after the user stops talking), on the
nova-llm worker thread. It reviews the conversation plus the regex fast-path
candidates and decides — as strict JSON — what to remember, update, or forget.

This is where the "self-actuation" lives: the model judges what is actually
important about Nicholas and reconciles it against what's already stored, so a
correction supersedes and a negation deletes, rather than piling up.

Design for a small (3B) model, per the Jarvis lessons:
  - Tiny, tightly-scoped prompt with a couple of few-shot examples.
  - STRICT JSON output; we parse defensively and ignore anything malformed.
  - Positive framing; post-parse validation is the real guarantee.
  - Never speaks; only writes to memory.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memory import NovaMemory

log = logging.getLogger("nova.memory.reconcile")

_ALLOWED_ACTIONS = {"insert", "update", "delete", "ignore"}

_SYSTEM = (
    "You extract durable personal facts about a user from a conversation, for a "
    "personal assistant's long-term memory. Only record things that are stable and "
    "worth remembering about the user (allergies, preferences, relationships, where "
    "they live or work, routines, important personal details). Ignore small talk, "
    "questions, one-off requests, and anything about the assistant. Never invent "
    "facts that were not actually stated. Reply with ONLY a JSON array."
)

_INSTRUCTION = """Return a JSON array of changes. Each item:
  {{"action": "insert|update|delete|ignore", "category": "...", "key": "...", "value": "..."}}

Rules:
- "category" is a short lowercase noun like allergy, preference, favorite, identity, location, possession, routine.
- "key" is the canonical thing this fact is ABOUT (e.g. "peanuts", "favorite_color", "employer", "gym"). Same key = same fact, so a correction is an "update" with the same category+key and the new value.
- "value" must capture the FULL detail the user gave, not just one word. If they mention a time, days, place, or amount, include all of it in the value (e.g. "every morning at 5:30", "Monday through Friday at 5:30").
- Use "delete" when the user says they no longer have/do something (e.g. cancelled a service, no longer likes something); give the category+key to remove.
- If nothing is worth storing, return [].
- Do not store the user's questions or requests.

Examples:
Conversation:
user: I'm allergic to peanuts. Also my favorite color is blue.
Output: [{{"action":"insert","category":"allergy","key":"peanuts","value":"peanuts"}},{{"action":"insert","category":"favorite","key":"favorite_color","value":"blue"}}]

Conversation:
user: I walk my dog every evening around 7pm.
Output: [{{"action":"insert","category":"routine","key":"dog_walk","value":"walks his dog every evening around 7pm"}}]

Conversation:
user: Actually my favorite color is red now.
Output: [{{"action":"update","category":"favorite","key":"favorite_color","value":"red"}}]

Conversation:
user: I cancelled my Netflix subscription.
Output: [{{"action":"delete","category":"subscription","key":"netflix","value":"netflix"}}]

Conversation:
user: What time is it? Can you play some music?
Output: []

Facts already known about the user:
{known}

Conversation to review:
{convo}

Output:"""


def build_prompt(convo_text: str, known_facts: str) -> str:
    return _INSTRUCTION.format(convo=convo_text, known=known_facts or "(none yet)")


def parse_decisions(raw: str) -> list[dict]:
    """Parse the model output into a list of validated decision dicts. Defensive:
    finds the first JSON array, ignores malformed entries, never raises."""
    if not raw:
        return []
    # Grab the first [...] block, tolerating any preamble the model adds.
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except Exception:
        return []
    if not isinstance(data, list):
        return []

    out = []
    for item in data:
        if not isinstance(item, dict):
            continue
        action = str(item.get("action", "")).lower().strip()
        category = str(item.get("category", "")).strip()
        key = str(item.get("key", "")).strip()
        value = str(item.get("value", "")).strip()
        if action not in _ALLOWED_ACTIONS or action == "ignore":
            continue
        if not category or not key:
            continue
        if action in ("insert", "update") and not value:
            continue
        out.append({"action": action, "category": category, "key": key, "value": value})
    return out


# Categories that describe the assistant itself, not the user — never store.
_REJECT_CATEGORIES = {"assistant", "ai", "nova", "system", "bot"}


# Filler words that may legitimately appear in a rendered fact without having
# been said verbatim ("goes to the gym" from "I go to the gym every morning").
_GROUNDING_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "his", "her", "their", "its", "he",
    "she", "they", "you", "your", "him", "them", "is", "are", "was", "were",
    "be", "been", "has", "have", "had", "does", "do", "did", "to", "of", "in",
    "on", "at", "for", "with", "from", "by", "as", "that", "this", "it",
    "every", "each", "some", "any", "all", "very", "really", "am", "pm",
})


def _is_grounded(value: str, convo_text: str) -> bool:
    """True if every substantive word in an extracted value actually traces back
    to what the user said.

    The small model sometimes FABRICATES detail: from "I go running every
    morning at 6" it produced "walks his dog every morning at 6" — inventing a
    dog. A fact Nova never heard is worse than no fact at all (see the
    never-guess principle), so anything ungrounded is dropped and simply
    re-learned later if the user mentions it again.

    Matching is prefix-based in both directions so ordinary inflection
    ("go" -> "goes", "run" -> "running") still counts as grounded.
    """
    convo = {w for w in re.findall(r"[a-z0-9]+", (convo_text or "").lower()) if w}
    if not convo:
        return True
    words = [w for w in re.findall(r"[a-z0-9]+", (value or "").lower())
             if len(w) > 2 and w not in _GROUNDING_STOPWORDS]
    for w in words:
        if any(w == c or w.startswith(c[:3]) or c.startswith(w[:3])
               for c in convo if len(c) >= 2):
            continue
        log.info(f"Rejecting ungrounded fact value {value!r}: {w!r} was never said")
        return False
    return True


def apply_decisions(memory: "NovaMemory", decisions: list[dict],
                    convo_text: str = "") -> int:
    """Apply validated decisions to memory. Returns how many were applied.

    A small model sometimes emits CONTRADICTORY decisions for the same key in one
    batch (insert + delete + update of routine/gym) — applying them in order lets
    a spurious delete wipe a just-inserted fact. So collapse per (category, key):
    if any insert/update exists for a key, that wins (last value) and any delete
    for that key is ignored; a delete only applies when it's the sole action for
    that key. insert and update are the same canonical upsert."""
    # Collapse to one decision per (category, key).
    upserts: dict[tuple, str] = {}
    deletes: set[tuple] = set()
    for d in decisions:
        cat, key = d["category"], d["key"]
        if cat.lower() in _REJECT_CATEGORIES:
            continue
        ident = (cat, key)
        if d["action"] in ("insert", "update"):
            upserts[ident] = d["value"]      # last insert/update value wins
            deletes.discard(ident)           # an upsert cancels any delete
        elif d["action"] == "delete" and ident not in upserts:
            deletes.add(ident)

    applied = 0
    for (cat, key), value in upserts.items():
        # Never store detail the user didn't actually say.
        if convo_text and not _is_grounded(value, convo_text):
            continue
        try:
            memory.upsert_fact(cat, key, value, source="llm")
            applied += 1
        except Exception:
            log.exception("Failed to upsert: %s/%s", cat, key)
    for (cat, key) in deletes:
        try:
            if memory.delete_fact(cat, key):
                applied += 1
        except Exception:
            log.exception("Failed to delete: %s/%s", cat, key)
    return applied


def reconcile(memory: "NovaMemory", llm, convo_text: str) -> int:
    """Full wake-mode pass: prompt the LLM over the conversation, parse, apply.
    Returns the number of memory changes made. Silent; caller runs it on the
    nova-llm worker thread."""
    if not convo_text.strip():
        return 0
    known = "\n".join(f"- {f['category']}/{f['key']} = {f['value']}" for f in memory.all_facts())
    prompt = build_prompt(convo_text, known)
    try:
        # temperature=0 for deterministic, parseable JSON (a 0.7-sampled run
        # intermittently produces malformed output on a small model).
        raw = llm.generate(
            system_prompt=_SYSTEM, history=[], user_message=prompt,
            temperature=0.0, max_tokens=400,
        )
    except Exception:
        log.exception("Reconciliation LLM call failed")
        return 0
    decisions = parse_decisions(raw)
    n = apply_decisions(memory, decisions, convo_text=convo_text)
    if n:
        log.info(f"Memory reconciliation applied {n} change(s).")
    return n
