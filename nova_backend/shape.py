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
    "louder", "quieter", "hello", "hi", "hey", "sorry", "come", "remember", "forget",
    "please", "nova", "can", "could", "need", "reload", "refresh", "minimize",
    "lock", "screenshot", "capture", "grab", "locate", "move", "copy", "rename",
    "drop", "drag", "stick", "shove", "file", "delete", "trash", "duplicate",
    "change", "use",
}
# Nouns and fillers used to sit in the opener list ("battery", "weather",
# "good", "so", "actually"...), so "battery is at like five percent lol" read
# his battery and "good weather today huh" fetched a forecast (audit,
# 2026-11-19). Fillers are now skipped before the opener is read, and a noun
# phrase counts as a request only when it is TERSE and has no verb of its own
# ("battery percentage", "apple stock price", "boston weather") — which main
# answered and the opener list alone had started refusing.
_FILLERS = {"so", "and", "now", "also", "just", "actually", "ok", "okay", "or", "oh",
            "well", "um", "uh", "right", "alright", "hmm", "yeah"}
_FINITE = re.compile(
    r"\b(?:is|are|was|were|am|be|been|being|has|have|had|do|does|did|will|would|can|"
    r"could|should|shall|may|might|must|went|got|said|made|took|came|saw|ran|broke|"
    r"sucks|feels|looks|seems|lasts|keeps|kept|left|thought|told|felt|know|think|"
    r"need|want|like|love|hate|guess|mean|\w+'s|\w+'re|\w+n't|\w{3,}ed)\b", re.I)


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


def _opener(text: str) -> tuple[str, str]:
    """(first real word, the text from it on), fillers skipped."""
    t = _WAKE.sub("", (text or "").strip())
    for _ in range(3):
        t2 = _POLITE.sub("", t)
        if t2 == t:
            break
        t = t2
    words = re.findall(r"[a-z']+", t.lower())
    i = 0
    while i < len(words) - 1 and words[i] in _FILLERS:
        i += 1
    return (words[i] if words else ""), " ".join(words[i:])


def asks(text: str) -> bool:
    t = (text or "").strip()
    if t.endswith("?"):
        return True
    first, rest = _opener(t)
    if first in _ASK_OPENERS:
        return True
    # Terse and verbless: a spoken command with the verb left off.
    words = rest.split()
    return 0 < len(words) <= 6 and not _FINITE.search(rest) \
        and first not in ("i", "i'm", "my", "we", "our", "you", "your", "he", "she", "they")


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
