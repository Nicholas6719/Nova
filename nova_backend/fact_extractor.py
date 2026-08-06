"""
Fact extractor — the regex FAST PATH of Nova's hybrid memory learning.

Per turn, this cheaply scans the user's utterance for obvious self-declarations
(allergies, likes/dislikes, "my X is Y", where they live/work) and proposes
structured (category, key, value) candidates. It is a SAFETY NET, not the
decision-maker: candidates are handed to the wake-mode LLM reconciliation pass,
which decides insert / update / delete / ignore against what's already stored.

Why regex here: it's free, deterministic, has zero dependence on the 3B model,
and degrades gracefully (a miss just means the LLM pass catches it later). What
it must NOT do is fire on hypotheticals, commands, questions, or things said
about Nova — hence the reject list.

Nothing here writes to the DB or speaks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class FactCandidate:
    category: str
    key: str
    value: str
    source: str = "regex"


# Utterances containing any of these are too uncertain / not self-declarations.
_REJECTS = (
    "i would like", "i'd like", "i want you to", "i need you to",
    "i think", "i feel like", "i'm not sure", "i am not sure",
    "i guess", "i suppose", "i wonder", "maybe", "might",
    "nova", "you are", "you're", "can you", "could you", "would you", "please",
    "if i", "should i", "do i", "what if",
)

# Ordered (regex, builder) templates. First match wins per utterance.
# Each builder returns a FactCandidate or None.


def _clean(s: str) -> str:
    return s.strip().strip(".,!?").strip()


def _extract(text: str) -> list[FactCandidate]:
    t = text.strip()
    low = t.lower()

    if not t or t.endswith("?"):
        return []
    if any(r in low for r in _REJECTS):
        return []

    out: list[FactCandidate] = []

    # Allergies — "I'm allergic to X" / "I am allergic to X"
    m = re.search(r"\bi(?:'m| am) allergic to (.+)", t, re.IGNORECASE)
    if m:
        val = _clean(m.group(1))
        if val:
            out.append(FactCandidate("allergy", _key(val), val.lower()))
            return out

    # Negation — "I cancelled/canceled/quit/stopped/got rid of X" / "I don't X anymore"
    m = re.search(r"\bi (?:cancell?ed|quit|stopped|got rid of|deleted|dropped) (.+)", t, re.IGNORECASE)
    if m:
        val = _clean(m.group(1))
        if val:
            # A negation candidate — reconciliation will delete the matching fact.
            out.append(FactCandidate("_negation", _key(val), val.lower()))
            return out

    # Favorites — "my favorite X is Y"
    m = re.search(r"\bmy favorite (\w+(?:\s+\w+)?) is (.+)", t, re.IGNORECASE)
    if m:
        key, val = _clean(m.group(1)), _clean(m.group(2))
        if key and val:
            out.append(FactCandidate("favorite", _key(key), val))
            return out

    # Preferences — "I like/love/enjoy/hate/dislike/prefer X"
    m = re.search(r"\bi (like|love|enjoy|hate|dislike|prefer) (.+)", t, re.IGNORECASE)
    if m:
        stance, obj = m.group(1).lower(), _clean(m.group(2))
        if obj and obj.lower() not in ("you", "it", "that", "this", "them"):
            stance_word = {
                "like": "likes", "love": "loves", "enjoy": "enjoys",
                "hate": "hates", "dislike": "dislikes", "prefer": "prefers",
            }[stance]
            out.append(FactCandidate("preference", _key(obj), stance_word))
            return out

    # Location — "I live in X" / "I'm from X" / "I grew up in X"
    m = re.search(r"\bi live in (.+)", t, re.IGNORECASE)
    if m:
        val = _clean(m.group(1))
        if val:
            out.append(FactCandidate("location", "home", val))
            return out
    m = re.search(r"\bi(?:'m| am) from (.+)", t, re.IGNORECASE)
    if m:
        val = _clean(m.group(1))
        if val:
            out.append(FactCandidate("location", "hometown", val))
            return out

    # Work — "I work at X" / "I work as a X"
    m = re.search(r"\bi work at (.+)", t, re.IGNORECASE)
    if m:
        val = _clean(m.group(1))
        if val:
            out.append(FactCandidate("identity", "employer", val))
            return out
    m = re.search(r"\bi work as (?:an? )?(.+)", t, re.IGNORECASE)
    if m:
        val = _clean(m.group(1))
        if val:
            out.append(FactCandidate("identity", "occupation", val))
            return out

    # Named possessions/people — "my dog/cat/wife/car is named X" / "... is called X"
    m = re.search(r"\bmy (\w+) is (?:named|called) (.+)", t, re.IGNORECASE)
    if m:
        thing, name = _clean(m.group(1)), _clean(m.group(2))
        if thing and name:
            out.append(FactCandidate("possession", f"{_key(thing)}_name", name))
            return out

    return out


def extract_candidates(text: str) -> list[FactCandidate]:
    """Public entry: return regex-proposed fact candidates for one utterance.
    Never raises — extraction failures just yield no candidates."""
    try:
        return _extract(text)
    except Exception:
        return []


def _key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower().strip()).strip("_")[:40]
