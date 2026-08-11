"""
Fact reconciler — the LLM SLOW PATH of Nova's hybrid memory learning.

Runs once per conversation, in wake mode (after the user stops talking), on the
nova-llm worker thread. It reviews the conversation and decides — as strict JSON — what to remember, update, or forget.

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


# Sentences that state a fact so plainly that no model judgment is required.
# Measured: handed "My favorite superhero is Spider-Man." the 3B returned []
# — it decided that was not worth remembering. Nova's whole promise is that it
# learns what Nicholas tells it about himself, so the unambiguous cases are
# extracted deterministically and the model only handles the rest.
_DIRECT_PATTERNS = [
    # "my favorite X is Y"
    (re.compile(r"\bmy\s+favou?rite\s+([\w ]{2,30}?)\s+(?:is|are|was)\s+(.+?)[.!?]*$", re.I),
     lambda m: ("favorite",
                "favorite_" + re.sub(r"\s+", "_", m.group(1).strip().lower()),
                m.group(2).strip())),
    # "I'm allergic to X"
    (re.compile(r"\bi(?:'?m| am)\s+allergic\s+to\s+(.+?)[.!?]*$", re.I),
     lambda m: ("allergy", m.group(1).strip().lower(), m.group(1).strip())),
    # "I live in X"
    (re.compile(r"\bi\s+live\s+in\s+(.+?)[.!?]*$", re.I),
     lambda m: ("location", "home", m.group(1).strip())),
    # "I work at/for X"
    (re.compile(r"\bi\s+work\s+(?:at|for)\s+(.+?)[.!?]*$", re.I),
     lambda m: ("location", "employer", m.group(1).strip())),
    # "my name is X"
    (re.compile(r"\bmy\s+name\s+is\s+(.+?)[.!?]*$", re.I),
     lambda m: ("identity", "name", m.group(1).strip())),
    # "my birthday is X"
    (re.compile(r"\bmy\s+(birthday|anniversary)\s+is\s+(.+?)[.!?]*$", re.I),
     lambda m: ("identity", m.group(1).lower(), m.group(2).strip())),
    # "I have a dog named X" / "my dog's name is X"
    (re.compile(r"\bi\s+have\s+an?\s+([\w ]{2,20}?)\s+named\s+(.+?)[.!?]*$", re.I),
     lambda m: ("relationship",
                re.sub(r"\s+", "_", m.group(1).strip().lower()),
                m.group(2).strip())),
]


def extract_direct_facts(convo_text: str) -> list[dict]:
    """High-precision facts pulled straight from what the user said.

    Only fires on sentences whose meaning is not in doubt, so it can run
    without the model and cannot invent anything: every value is a literal
    span of the user's own words.
    """
    out: list[dict] = []
    seen: set = set()
    for line in (convo_text or "").splitlines():
        line = re.sub(r"^\s*user:\s*", "", line).strip()
        if not line:
            continue
        # A trailing command clause is not part of the fact.
        line = re.sub(r"[,.]?\s*(?:that'?s all|go to sleep|thanks|thank you)\s*[.!]?$",
                      "", line, flags=re.I).strip()
        for sentence in re.split(r"(?<=[.!?])\s+", line):
            for pattern, build in _DIRECT_PATTERNS:
                m = pattern.search(sentence)
                if not m:
                    continue
                cat, key, value = build(m)
                value = value.strip().rstrip(".!?,")
                if not value or len(value) > 120:
                    continue
                ident = (cat, key)
                if ident in seen:
                    continue
                seen.add(ident)
                out.append({"action": "insert", "category": cat,
                            "key": key, "value": value})
                break
    return out


def build_prompt(convo_text: str, known_facts: str) -> str:
    return _INSTRUCTION.format(convo=convo_text, known=known_facts or "(none yet)")


# The 3B emits the right FIELDS with the wrong PUNCTUATION distressingly often:
#   ["action":"insert","category":"routine","key":"guitar_lesson", ...]   ← no braces
#   ["action":"insert","category":"allergy","key":"shellfish", ...}       ← mismatched
# Both are invalid JSON, so a strict parse dropped the fact SILENTLY. Measured:
# 3 of 4 fact-bearing sentences were lost this way, including an allergy. Since
# the schema is fixed and tiny, we salvage by scanning the fields in order and
# starting a new record at each "action" — punctuation is irrelevant to meaning.
_FIELD_RE = re.compile(
    r'"(action|category|key|value)"\s*:\s*"((?:[^"\\]|\\.)*)"'
)


def _salvage_objects(text: str) -> list[dict]:
    """Recover decision records from malformed JSON by reading the fields."""
    out: list[dict] = []
    current: dict = {}
    for field, value in _FIELD_RE.findall(text):
        # A new "action" starts a new record.
        if field == "action" and current:
            out.append(current)
            current = {}
        current[field] = value.replace('\\"', '"')
    if current:
        out.append(current)
    return out


def parse_decisions(raw: str) -> list[dict]:
    """Parse the model output into a list of validated decision dicts. Defensive:
    finds the first JSON array, ignores malformed entries, never raises."""
    if not raw:
        return []
    # Grab the first [...] block, tolerating any preamble the model adds.
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    blob = match.group(0) if match else raw

    data: list = []
    try:
        loaded = json.loads(blob)
        if isinstance(loaded, list):
            data = loaded
        elif isinstance(loaded, dict):
            data = [loaded]
    except Exception:
        data = []

    if not data:
        # Strict JSON failed (or produced nothing) — salvage the fields.
        data = _salvage_objects(blob)
        if data:
            log.info(f"reconciler: salvaged {len(data)} record(s) from malformed JSON")

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


# Grammatical categories are not facts. The 3B, handed a stream of commands,
# produced ('adjective', 'adjective', 'open,brave,brave,brave,brave') — it had
# nothing personal to extract, so it started labelling parts of speech. A fact
# must be ABOUT Nicholas, not about the words he used.
_JUNK_CATEGORIES = {
    "adjective", "adjectives", "noun", "nouns", "verb", "verbs", "adverb",
    "pronoun", "word", "words", "phrase", "sentence", "command", "commands",
    "query", "request", "action", "instruction", "text", "misc", "other",
    "unknown", "none", "n/a",
}

# Words that only ever appear because he was giving an ORDER, never because he
# was telling Nova something about himself.
_COMMAND_WORDS = {
    "open", "close", "quit", "launch", "start", "run", "search", "google",
    "play", "pause", "skip", "stop", "set", "show", "type", "send", "move",
    "copy", "rename", "delete", "scroll", "click", "mute", "unmute",
}


def _is_junk(category: str, key: str, value: str) -> bool:
    """Reject facts that describe the utterance rather than the user."""
    cat = (category or "").strip().lower()
    k = (key or "").strip().lower()
    v = (value or "").strip().lower()
    if not v:
        return True
    if cat in _JUNK_CATEGORIES or k in _JUNK_CATEGORIES:
        return True
    # NOTE: value == key is NOT junk. "allergy/peanuts/peanuts" is a legitimate
    # shape and appears in this module's own few-shot examples; memory.py owns
    # the degenerate-value policy. Only a value echoing the CATEGORY is empty.
    if v == cat:
        return True
    # A bare comma/space list built out of command words is a parse artifact,
    # not something he told Nova about himself.
    tokens = [t for t in re.split(r"[,\s]+", v) if t]
    if tokens and all(t in _COMMAND_WORDS for t in tokens):
        return True
    # Repetition of a single token ("brave,brave,brave") is never a fact.
    if len(tokens) >= 3 and len(set(tokens)) == 1:
        return True
    return False


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
        if _is_junk(cat, key, value):
            log.info(f"reconciler: rejected junk fact {cat}/{key}={value!r}")
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

    # Deterministic facts come FIRST so they win the per-key collapse below.
    # The model is allowed to add to them, never to veto them: it returned []
    # for "My favorite superhero is Spider-Man", which is exactly the kind of
    # plain statement Nova exists to remember.
    direct = extract_direct_facts(convo_text)
    if direct:
        known_keys = {(d["category"], d["key"]) for d in decisions}
        added = [d for d in direct if (d["category"], d["key"]) not in known_keys]
        if added:
            log.info(f"reconciler: {len(added)} fact(s) taken directly from what "
                     "he said")
        decisions = added + decisions
    n = apply_decisions(memory, decisions, convo_text=convo_text)
    if n:
        log.info(f"Memory reconciliation applied {n} change(s).")
    return n
