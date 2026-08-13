"""
Listener assertions — the rules a person would notice instantly when hearing
Nova speak.

Most of what embarrassed Nicholas was not a logic error. It was audible the
moment it came out of the speaker:

  * a full filesystem path read aloud, character by character
  * "Nicholas has already informed me about his favorite movie" — third person,
    while talking to him
  * "This is a critical deadline, so be sure to submit your reflection" — advice
    invented out of nothing
  * numbers that were never in the file being summarized

Each of those was found by a human ear, not by a check, because every check I
wrote was scoped to one feature. These run over EVERY spoken string in EVERY
suite instead, so a regression in one feature is caught by the tests of another.

Usage:
    from listener import check_spoken
    problems = check_spoken(response)          # [] when it is fit to speak
    problems = check_spoken(response, source_text=file_contents)   # + numbers
"""
from __future__ import annotations

import re

# ── Rules ─────────────────────────────────────────────────────────────────────

# CLAUDE.md invariant 10 — Nova speaks aloud, so markdown is read as noise.
_MARKDOWN_RE = re.compile(r"[*_`#]|^\s*[-•]\s|\s—\s|^\s*\d+[.)]\s", re.M)

# An absolute path spoken aloud is unusable and leaks the filesystem layout.
_PATH_RE = re.compile(r"/Users/|/Library/|/private/|/var/folders/|~/[A-Za-z]")

# Talking ABOUT him while talking TO him.
_THIRD_PERSON_RE = re.compile(
    r"\b(?:nicholas\s+(?:has|is|was|told|said|likes|prefers|wants|needs)"
    r"|he\s+(?:has|is|was|told|said|likes|prefers|wants|needs)"
    r"|his\s+(?:favorite|favourite|calendar|reminder|file|screen))\b",
    re.I,
)

# Unsolicited coaching. Nova reports; it does not counsel.
_ADVICE_RE = re.compile(
    r"^\s*(?:"
    r"(?:please\s+)?(?:make|be)\s+sure\b|don'?t\s+forget\b|remember\s+to\b"
    r"|you\s+(?:should|ought|need\s+to|may\s+want|might\s+want)\b"
    r"|it'?s\s+(?:important|critical|essential)\b"
    r"|this\s+is\s+(?:a\s+)?(?:critical|important|urgent)\b"
    r"|(?:i'?d|i\s+would)\s+(?:recommend|suggest)\b"
    r"|good\s+luck\b"
    r")",
    re.I,
)

# Claiming an action. Nova's deterministic handlers report real outcomes; if a
# claim like this reaches the speaker from the LLM path, it is a fabrication.
_ACTION_CLAIM_RE = re.compile(
    r"\bi(?:'ve| have)?\s+(?:just\s+)?"
    r"(?:opened|closed|sent|typed|clicked|moved|copied|renamed|deleted|played|"
    r"installed|downloaded|searched)\b"
    r"|\bconsider it done\b|\bit'?s open now\b",
    re.I,
)

# Raw library errors leak paths and mean nothing to a listener.
_JARGON_RE = re.compile(
    r"\b(?:traceback|exception|errno|stacktrace|nonetype|attributeerror|"
    r"keyerror|package not found|no such file or directory)\b", re.I,
)

# Spoken answers should be short. Long enough to be a paragraph is too long.
_MAX_SPOKEN_CHARS = 600


def check_spoken(text: str, *, source_text: str | None = None,
                 allow_action_claim: bool = False) -> list[str]:
    """Return a list of reasons this string is unfit to speak. Empty is good.

    `source_text` enables the invented-number check: any number Nova says must
    appear in the material it was summarizing. The 3B once invented a
    leftover-budget figure that was nowhere in the file.

    `allow_action_claim` is for the deterministic handlers that genuinely DID
    perform the action and verified it.
    """
    problems: list[str] = []
    if text is None:
        return ["response is None"]
    t = str(text)
    if not t.strip():
        return ["response is empty"]

    if _PATH_RE.search(t):
        problems.append("speaks an absolute filesystem path")
    if _MARKDOWN_RE.search(t):
        problems.append("contains markdown or an em dash (invariant 10)")
    if _THIRD_PERSON_RE.search(t):
        problems.append("refers to Nicholas in the third person")
    if _JARGON_RE.search(t):
        problems.append("leaks a raw error or library jargon")
    if len(t) > _MAX_SPOKEN_CHARS:
        problems.append(f"too long to speak ({len(t)} chars)")
    if not allow_action_claim and _ACTION_CLAIM_RE.search(t):
        problems.append("claims to have performed an action")

    for sentence in re.split(r"(?<=[.!?])\s+", t.strip()):
        if _ADVICE_RE.match(sentence):
            problems.append(f"gives unsolicited advice: {sentence[:60]!r}")
            break

    if source_text is not None:
        spoken_nums = set(re.findall(r"\d[\d,.]*", t))
        source_nums = set(re.findall(r"\d[\d,.]*", source_text))
        invented = {n for n in spoken_nums
                    if n not in source_nums
                    and n.replace(",", "") not in {s.replace(",", "")
                                                   for s in source_nums}}
        if invented:
            problems.append(f"invents numbers not in the source: {sorted(invented)}")

    return problems


def assert_spoken(text: str, label: str, check_fn, **kwargs) -> None:
    """Convenience: run check_spoken and report through a suite's check()."""
    problems = check_spoken(text, **kwargs)
    check_fn(not problems, f"fit to speak: {label}",
             "; ".join(problems) + f" | {str(text)[:120]}" if problems else "")
