"""
Is this sentence SHAPED like a request?

CLAUDE.md's rule for the tools matchers — command words that are also ordinary
English must be shaped like commands — turned out to be every read stage's
rule. Measured statements that were answered as requests:
  - "my next meeting is at three" read his calendar (the brain, Stage 1)
  - "I asked Nova what the weather was, it kept it on the screen..." fetched a
    forecast (his real turn, the weather regex)
  - "my battery died yesterday" read his battery; "Apple stock is way too
    expensive" fetched a quote
A request opens with a question word or a verb, or ends in a question mark; a
statement about the same subject opens with "my", "I", "the", a name...

And a question about what a thing IS is not a reading of it: "what does a CPU
actually do" is not a CPU reading, "what is a stock split" is not a quote
(JARVIS has the same rule as a canon rewrite, for the same reason).

One definition, shared by the brain's skills (skills.py) and the regex read
stages (weather, market, the tools readings), so the two routers cannot
disagree about what a request looks like.
"""
from __future__ import annotations

import re

_ASK_OPENERS = {
    "what", "what's", "whats", "when", "when's", "where", "where's", "which", "who",
    "who's", "how", "how's", "hows", "do", "does", "did", "am", "is", "are", "was",
    "any", "anything", "read", "show", "tell", "list", "check", "give", "get",
    "pull", "will", "would", "should", "have", "has", "got", "open", "find",
    "look", "let", "lets", "let's", "describe", "summarize", "say", "go", "take",
    "bring", "turn", "make", "set", "put", "start", "play", "pause", "skip",
    "stop", "resume", "mute", "unmute", "dim", "brighten", "lower", "raise",
    "crank", "throw", "close", "quit", "launch", "search", "google", "navigate",
    "cancel", "clear", "hide", "restore", "shuffle", "repeat", "next", "previous",
    "volume", "louder", "quieter", "battery", "cpu", "weather", "forecast",
    "hello", "hi", "hey", "good", "morning", "afternoon", "evening", "sorry",
    "one", "come", "remember", "forget", "actually", "or", "and", "so", "ok",
    "okay", "now", "also", "just", "please", "nova", "can", "could", "need",
    "full", "max", "reload", "refresh", "new", "minimize", "lock", "screenshot",
    "capture", "grab", "locate", "move", "copy", "rename", "drop", "drag",
    "stick", "shove", "file", "delete", "trash", "duplicate", "change", "use",
}


_WAKE = re.compile(r"^\s*(?:(?:hey|hi|ok|okay|so)[,\s]+)?nova\b[,.!?\s]*(?=\S)", re.I)
_POLITE = re.compile(r"^\s*(?:please|can you|could you|would you|will you)\b[,\s]*", re.I)


def _first_word(text: str) -> str:
    t = _WAKE.sub("", (text or "").strip())
    for _ in range(3):
        t2 = _POLITE.sub("", t)
        if t2 == t:
            break
        t = t2
    m = re.match(r"[a-z']+", t.lower())
    return m.group(0) if m else ""


def asks(text: str) -> bool:
    t = (text or "").strip()
    if t.endswith("?"):
        return True
    return _first_word(t) in _ASK_OPENERS


_DEFINITION = re.compile(
    r"^\s*(?:so\s+)?what(?:'s|\s+is|\s+are)\s+(?:a|an)\s+[a-z]"
    r"|\b(?:stand|stands)\s+for\b|\bshort\s+for\b"
    r"|\bwhat\s+(?:does|do)\s+(?:a|an|the)?\s*[\w ]{1,30}?\s+(?:actually\s+|even\s+)?(?:do|mean)\b"
    r"|\bhow\s+does\s+(?:a|an|the)\s+[\w ]{1,30}?\s+work\b",
    re.I)


def is_definition(text: str) -> bool:
    return bool(_DEFINITION.search(text or ""))


# "Tell me WHEN my battery is charged" asks for the future, not a reading now.
# It used to be answered with today's level (the battery regex) — and with the
# watch engine unavailable, the brain did the same. Never a reading, anywhere.
_LATER = re.compile(r"^\s*(?:nova[,\s]+)?(?:please\s+)?(?:can\s+you\s+)?(?:tell|let|warn|alert|notify|ping)"
                    r"\s+me\s+(?:know\s+)?(?:when|if|once|as\s+soon\s+as|the\s+moment|whenever)\b", re.I)


def asks_for_later(text: str) -> bool:
    return bool(_LATER.search(text or ""))


def reads(text: str) -> bool:
    """Shaped like a request to READ something now, and not a definition."""
    return asks(text) and not is_definition(text) and not asks_for_later(text)
