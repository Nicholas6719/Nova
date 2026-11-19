"""
Nova may not promise what nothing will do — or claim what nothing did.

Ported from JARVIS (MiniPCJarvis sidecar/honesty.py), where the model said "I'll
keep an eye on it" with nothing watching, and "I'll alert you right away when a
new call comes in" with no such rule. JARVIS keeps such a sentence when a tool
ran in that turn. Nova's LLM runs NO tools — every action Nova takes is a
deterministic handler, decided before the LLM is ever reached — so on Nova's
LLM path every such sentence is unbacked, without exception.

Nova adds a third family JARVIS did not need: claims of something already
DONE. CLAUDE.md records it: told to open a folder it could not open, the 3B
replied "I've opened the coding projects folder for you." Stage 8b refuses
action-SHAPED requests before they reach the model, but a conversational turn
can still end in "I've set a reminder for you".

Applied per sentence inside _stream_response, before the sentence is spoken,
so the streaming overlap (invariant 6) is untouched: the first unbacked
sentence is replaced by the truth, any later one in the same reply is dropped.

Not caught on purpose: "I'll remember that". The wake-mode fact reconciler
genuinely may store what he said, so that one is not a lie.
"""
from __future__ import annotations

import re

_I_WILL = r"\bI(?:'|’)?(?:ll| will| am going to|'m going to|’m going to)\s+"
_SOFT = r"(?:(?:also |just |certainly |definitely |happily |gladly |be sure to |make sure to )?)"

WATCHING = re.compile(
    _I_WILL + _SOFT +
    r"(?:keep (?:an eye|watch|you posted|you updated|you informed|track|monitoring|listening)|"
    r"let you know|notify you|alert you|tell you (?:when|if|as soon|the moment)|inform you|"
    r"get back to you|monitor|watch (?:for|out|it|that|this)|update you|report back|"
    r"check back|check in (?:with you|later)|remind you|ping you|follow up|"
    r"flag (?:it|that|anything)|be watching|be listening)\b", re.I)

WORKING = re.compile(
    _I_WILL + r"(?:start|begin) (?:by|with)\b|"
    r"\bonce (?:that(?:'|’)?s|that is|it(?:'|’)?s|i(?:'|’)?m) (?:done|finished|complete),? I(?:'|’)?ll\b|"
    r"\b(?:then|next|after that),? I(?:'|’)?ll (?:check|scan|review|go through|look|examine|analy[sz]e|run)\b|"
    + _I_WILL + r"(?:work on (?:it|that|this)|look into (?:it|that|this)|investigate|"
    r"dig into|research (?:it|that|this)|get (?:on|right on) (?:it|that))\b", re.I)

DONE = re.compile(
    r"\bI(?:'|’)?ve\s+(?:just\s+|already\s+|gone ahead and\s+)?|"
    r"\bI\s+have\s+(?:just\s+|already\s+|gone ahead and\s+)?", re.I)
_DONE_VERBS = re.compile(
    r"^(?:set|added|created|scheduled|opened|launched|saved|sent|booked|"
    r"moved|copied|deleted|renamed|played|paused|muted|turned (?:on|off|up|down)|"
    r"started|closed|ordered|emailed|texted|messaged|called|put (?:it|that|a reminder))\b",
    re.I)

CANT_WATCH = ("I can't keep watch on that by myself. Ask me again whenever you "
              "like, or tell me to set a reminder.")
CANT_WORK = "That's more than I can do on my own. I can't work on things in the background."
DIDNT = "To be clear, I haven't actually done anything there."


def unbacked(sentence: str) -> str | None:
    """The honest line to say INSTEAD of this sentence, or None if it neither
    promises nor claims an action. Only for Nova's LLM path."""
    s = sentence or ""
    if WATCHING.search(s):
        return CANT_WATCH
    if WORKING.search(s):
        return CANT_WORK
    for m in DONE.finditer(s):
        if _DONE_VERBS.match(s[m.end():]):
            return DIDNT
    return None


class Filter:
    """One reply's worth of state: the first unbacked sentence becomes the
    truth, later ones are dropped (one correction, not a chorus)."""

    def __init__(self) -> None:
        self.replaced: list[tuple[str, str]] = []
        self.kept: list[str] = []

    def __call__(self, sentence: str) -> str:
        honest = unbacked(sentence)
        if honest is None:
            self.kept.append(sentence)
            return sentence
        first = not self.replaced
        self.replaced.append((sentence, honest))
        if first:
            self.kept.append(honest)
            return honest
        return ""

    @property
    def text(self) -> str:
        return " ".join(x for x in self.kept if x).strip()
