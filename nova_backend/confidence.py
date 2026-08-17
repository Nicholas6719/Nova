"""
Confidence gating — how sure Nova has to be before she acts.

Nova's UI has no transcript any more. That was the channel where a mishear was
visible: he would see "went to 25% of 10%" and know why the answer was strange.
Without it, every mis-transcription is a SILENT failure.

So confidence has to gate action instead. Whisper reports `avg_logprob` per
segment — how confident the decoder was. It is NOT a calibrated probability and
this module does not pretend otherwise; it is a score that separates right from
wrong well enough to threshold, which was measured rather than assumed.

MEASURED, 270 clips (18 commands x 3 voices x 5 noise levels, -2 to 40 dB SNR):

    avg_logprob when the transcript was usable : median -0.137
    avg_logprob when it was not                : median -0.768

    threshold   acted on a WRONG transcript   asked again unnecessarily
      -0.30            11.5%                        25.8%
      -0.50            24.6%                         5.3%     <- the knee
      -0.85            55.7%                          0%

There is no threshold that is right for everything, which is the whole design:
THE BAR SCALES WITH CONSEQUENCE. Being wrong about the time costs two seconds.
Being wrong about "send that to Sarah" costs something he cannot take back.

    LOW     chat, time, weather, "what's on my screen"   -> act, almost always
    MEDIUM  file moves, calendar writes, navigation      -> act at the knee
    HIGH    sending, deleting, buying, anything outbound -> read it back first

The honest limit, stated plainly: this reduces confident errors, it does not
eliminate them. Whisper can be confidently wrong. The read-back on high
consequence actions is the real safety net; confidence only decides when to
bother him with one.
"""

from __future__ import annotations

import re
from typing import Optional

# ── Thresholds, from the sweep above ──────────────────────────────────────────
# Everything above ACT_COMFORTABLE is acted on silently.
# Between a tier's floor and ACT_COMFORTABLE, Nova acts but shows `unsure` —
# so if she got it wrong he can see why, which is what the transcript used to do.
# Below the floor she does not act at all.
ACT_COMFORTABLE = -0.30

FLOOR_LOW = -0.85      # 0% unnecessary re-asks; wrong is cheap here
FLOOR_MEDIUM = -0.50   # the knee: 75% of mishears caught, 5% re-asks
FLOOR_HIGH = -0.30     # strictest; a read-back is required regardless

LOW, MEDIUM, HIGH = "low", "medium", "high"

_FLOORS = {LOW: FLOOR_LOW, MEDIUM: FLOOR_MEDIUM, HIGH: FLOOR_HIGH}


# ── What is this utterance about to do? ───────────────────────────────────────
# Deterministic regex, and deliberately over-inclusive: mistaking a harmless
# request for a consequential one costs one clarifying question, while the
# reverse costs a sent message.

# Reaches another person, or cannot be undone.
_HIGH_RE = re.compile(
    r"\b(?:send|sends|sending|sent|text|texts|texting|email|emails|emailing|"
    r"message|messages|messaging|reply|replies|replying|respond|forward|post|"
    r"publish|share|call|calls|calling|dial|"
    r"delete|deletes|deleting|erase|erasing|remove|removing|trash|wipe|"
    r"buy|buys|buying|purchase|order|orders|pay|paying|checkout|"
    r"shut\s*down|restart|reboot|log\s*out|sign\s*out|uninstall)\b",
    re.I,
)

# Changes something on his machine, but recoverably.
_MEDIUM_RE = re.compile(
    r"\b(?:move|moves|moving|copy|copies|copying|rename|renames|renaming|"
    r"create|creates|creating|add|adds|adding|set|sets|setting|make|makes|"
    r"remind|reminds|schedule|schedules|save|saves|write|writes|type|types|"
    r"install|download|open|opens|launch|play|plays|pause|skip)\b",
    re.I,
)


# A verb INSIDE a reminder or a note is content, not a command. "Remind me to
# call mom at six" creates a reminder; it does not call anyone, and treating it
# as outbound would make Nova read back every reminder he ever sets. Same shape
# as the calendar guard in file_intents.
_WRAPPED_RE = re.compile(
    r"\b(?:remind\s+me\s+to|reminder\s+to|set\s+a\s+reminder|"
    r"add\s+a\s+reminder|make\s+a\s+note|note\s+to\s+self|"
    r"remember\s+to|add\s+.*\bto\s+my\s+(?:list|calendar))\b",
    re.I,
)


def consequence(text: str) -> str:
    """How much it costs to act on a mis-heard version of this."""
    if not text:
        return HIGH          # nothing heard is never worth acting on
    # Checked BEFORE the high-consequence verbs: writing down "call mom" is a
    # calendar write, not a call.
    if _WRAPPED_RE.search(text):
        return MEDIUM
    if _HIGH_RE.search(text):
        return HIGH
    if _MEDIUM_RE.search(text):
        return MEDIUM
    return LOW


# ── The decision ──────────────────────────────────────────────────────────────
ACT = "act"                  # confident: just do it
ACT_UNSURE = "act_unsure"    # do it, but show that she was not certain
CONFIRM = "confirm"          # read it back and wait
REJECT = "reject"            # too poor to use at all


class Decision:
    __slots__ = ("action", "tier", "score", "readback")

    def __init__(self, action: str, tier: str, score: float,
                 readback: str = "") -> None:
        self.action = action
        self.tier = tier
        self.score = score
        self.readback = readback

    @property
    def should_act(self) -> bool:
        return self.action in (ACT, ACT_UNSURE)

    @property
    def is_unsure(self) -> bool:
        return self.action == ACT_UNSURE

    def __repr__(self) -> str:
        return (f"<Decision {self.action} tier={self.tier} "
                f"score={self.score:.2f}>")


def decide(text: str, score: Optional[float]) -> Decision:
    """What Nova should do with a transcript of this quality.

    `score` is the utterance's avg_logprob, or None when confidence could not
    be computed — in which case Nova behaves exactly as she did before this
    module existed and simply acts. A missing signal must not make her deaf.
    """
    tier = consequence(text)
    if score is None:
        return Decision(ACT, tier, 0.0)

    floor = _FLOORS[tier]

    if score < floor:
        return Decision(REJECT, tier, score)

    # High consequence always reads back, however confident she is. Whisper can
    # be confidently wrong, and this is where that is expensive.
    if tier == HIGH:
        return Decision(CONFIRM, tier, score, readback=_readback(text))

    if score < ACT_COMFORTABLE:
        return Decision(ACT_UNSURE, tier, score)
    return Decision(ACT, tier, score)


def _readback(text: str) -> str:
    """'Did you say send that message?' — his words, not a paraphrase, so the
    thing he confirms is exactly the thing that was heard."""
    cleaned = text.strip().rstrip(".?! ")
    return f"Did you say {cleaned}?"


def ask_again(tier: str) -> str:
    """What Nova says when she will not act on what she heard. Never invents a
    guess at what he meant — the whole point is that she does not know."""
    if tier == HIGH:
        return "I didn't catch that clearly enough to act on it. Say it again?"
    return "Sorry, I didn't catch that."
