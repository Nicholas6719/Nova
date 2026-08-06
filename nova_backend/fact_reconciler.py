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


def apply_decisions(memory: "NovaMemory", decisions: list[dict]) -> int:
    """Apply validated decisions to memory. Returns how many were applied.
    insert and update are the same operation (canonical upsert supersedes)."""
    applied = 0
    for d in decisions:
        try:
            if d["action"] in ("insert", "update"):
                memory.upsert_fact(d["category"], d["key"], d["value"], source="llm")
                applied += 1
            elif d["action"] == "delete":
                if memory.delete_fact(d["category"], d["key"]):
                    applied += 1
        except Exception:
            log.exception("Failed to apply memory decision: %s", d)
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
    n = apply_decisions(memory, decisions)
    if n:
        log.info(f"Memory reconciliation applied {n} change(s).")
    return n
