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
    # "let you know" / "tell you" promise something only with a condition or
    # a time: "I'll let you know what I think" is an opinion, not a watch.
    r"let you know(?=\s*(?:[.!?,]|$|when\b|if\b|once\b|as soon|the moment|later|soon|"
    r"right away|immediately|tomorrow|tonight|by\b))|notify you|alert you|"
    r"tell you (?:when|as soon|the moment|if (?!I\b))|inform you|"
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
    r"closed|ordered|emailed|texted|messaged|taken care of|put (?:it|that|a reminder)|"
    # "started" and "called" only with an object an action would have:
    # "I've started to think you're right" is not a claim (audit, 2026-11-19).
    r"started (?:a|the|your)\s+(?:timer|music|song|playlist|download|recording|app|meeting)|"
    r"called (?:him|her|them|your|mom|dad))\b",
    re.I)

# Promises of an action in the future. Nova's LLM can do none of them; the
# handlers can, when he asks directly. Measured missing (audit): "I'll set a
# reminder for 3pm", "Let me set a timer for you", "I'll play some jazz".
FUTURE = re.compile(
    r"\b(?:I(?:'|’)?ll|I will|let me|I(?:'|’)?m going to|I am going to)\s+" + _SOFT +
    r"(?:go ahead and\s+)?"
    r"(?:set|add|schedule|create|book|open|launch|send|email|text|message|call|play|"
    r"pause|mute|close|delete|move|copy|rename|save|order|search for|look (?:it|that) up|"
    r"put (?:it|that|a reminder|an event|this)\s+(?:on|in|into)|turn (?:on|off|up|down)|"
    r"write (?:it|that) down|jot (?:it|that) down|start (?:a|the|your)\s+(?:timer|music|playlist))\b|"
    r"\bI(?:'|’)?m (?:setting|adding|scheduling|creating|booking|opening|sending|playing|"
    r"saving|calling|texting|emailing) (?:a|an|the|that|it|your|up)\b", re.I)
# Claims of a finished action that do not start with "I've".
CLAIMED = re.compile(
    r"\b(?:reminder|alarm|timer|event|meeting|appointment|message|email|text)\s+"
    r"(?:is\s+|has been\s+)?(?:set|added|scheduled|created|booked|sent)\b(?!\s+(?:up\s+)?by\s+you)|"
    r"\bconsider it done\b|\ball set\b(?=.*\b(?:reminder|alarm|timer|calendar)\b)", re.I)

CANT_WATCH = ("I can't keep watch on that by myself. Ask me again whenever you "
              "like, or tell me to set a reminder.")
CANT_WORK = "That's more than I can do on my own. I can't work on things in the background."
CANT_DO = ("I can't do that from here. Ask me for it directly and I'll do it if it's "
           "something I can do.")
DIDNT = "To be clear, I haven't actually done anything there."


def unbacked(sentence: str) -> str | None:
    """The honest line to say INSTEAD of this sentence, or None if it neither
    promises nor claims an action. Only for Nova's LLM path."""
    s = sentence or ""
    if WATCHING.search(s):
        return CANT_WATCH
    if WORKING.search(s):
        return CANT_WORK
    if FUTURE.search(s):
        return CANT_DO
    if CLAIMED.search(s):
        return DIDNT
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
