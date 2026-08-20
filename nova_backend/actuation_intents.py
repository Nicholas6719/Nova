"""
Actuation NL dispatch — "type this", "click Send", "press enter".

The riskiest module in Nova, so the gates come first:

  1. WORK MODE ONLY. Every action here is refused unless Nova is working
     alongside him. A hard check in code, not a sentence in a prompt.
  2. ACCESSIBILITY PERMISSION, preflighted. Without it, UI scripting fails
     with an error that reads like a bug, so Nova says what is actually wrong.
  3. A target must be FOUND BY NAME before it is clicked. Nova never clicks a
     coordinate she could not verify — which is the thing RileyJarvis does and
     calls computer use.
  4. Destructive labels are CONFIRMED. And Return is confirmed when the
     frontmost app is one where Return sends.

Typing is deliberately free, because he asked for that: what is typed is
visible before it goes anywhere, and sending is the gated step. His example,
which this module is shaped around:

    "Nova say 'okay thank you see you soon'"   -> types it
    "Can I send it?"                            -> Nova asks
    "yes"                                       -> Nova sends
"""

from __future__ import annotations

import logging
import re
from typing import Callable, Optional

import actuation as A

log = logging.getLogger("nova.actuation.intents")

_LEAD = r"^(?:hey\s+)?(?:nova[,\s]+)?(?:please\s+)?(?:can you\s+)?"

# "type X" / "say X" / "write X" — the words to put in the field.
_TYPE_RE = re.compile(
    _LEAD + r"(?:type|write|enter|input)\s+(?:out\s+)?(?:this[:,]?\s+|that[:,]?\s+)?"
    r"(?P<q>[\"'“](?P<quoted>.+?)[\"'”]|.+)$",
    re.I | re.S,
)
# "type IN my message window" names a DESTINATION, not content. A real garbled
# transcript — "Type in my message window, yes I do, then send it" — would
# otherwise have Nova type that whole string into whatever had focus.
_TYPE_PLACE_RE = re.compile(
    _LEAD + r"(?:type|write|enter|input)\s+(?:in|into|on|onto|at)\b", re.I)
# "say X" only counts as typing when the words are QUOTED. Unquoted "say hello"
# is Nova being asked to speak, not to type.
_SAY_TYPE_RE = re.compile(
    _LEAD + r"say\s+[\"'“](?P<quoted>.+?)[\"'”]\s*$", re.I | re.S)

# A control label is a NAME: at most three words. Allowing arbitrary spaces
# made "click here was the old web" a click target, because everything after
# the verb became the label.
_CLICK_RE = re.compile(
    _LEAD + r"(?:click|press|tap|hit|push)\s+(?:on\s+)?(?:the\s+)?"
    r"(?P<label>[\w][\w'&-]*(?:\s+[\w'&-]+){0,2})"
    r"(?:\s+button)?\s*[.?!]?$",
    re.I,
)
# Words that mean the sentence is ABOUT clicking, not a command to click.
_CLICK_NOT_RE = re.compile(
    r"\b(?:was|were|is|are|used\s+to|would|could|should|never|always)\b", re.I)
_KEY_RE = re.compile(
    _LEAD + r"(?:press|hit)\s+(?:the\s+)?(?P<key>return|enter|tab|escape|esc|"
    r"space|delete|backspace)(?:\s+key)?\s*[.?!]?$",
    re.I,
)
_SCROLL_RE = re.compile(
    _LEAD + r"scroll\s+(?P<dir>up|down|left|right)(?:\s+a\s+bit|\s+more)?\s*[.?!]?$",
    re.I,
)
_SEND_RE = re.compile(_LEAD + r"send\s+it\s*[.?!]?$", re.I)


class NovaActuation:
    """Typing and clicking, gated on work mode."""

    def __init__(self, config: dict, assistant=None) -> None:
        self.config = config
        self.assistant = assistant
        self.enabled = bool(config.get("actuation", {}).get("enabled", True))
        # A click Nova has found and is waiting for a yes on. Same one-shot
        # shape as the file confirmation: anything that is not a clear yes
        # cancels it.
        self.pending: Optional[Callable[[], str]] = None

    # ── Detection ─────────────────────────────────────────────────────────────
    def detect_intent(self, text: str) -> Optional[str]:
        if not self.enabled or not text or not text.strip():
            return None
        # WORK MODE IS THE GATE, and it lives here rather than only in handle()
        # so actuation never SHADOWS a handler that already works. "scroll
        # down" drives the browser through tools when they are not working
        # together; claiming it here would have replaced that with "say work
        # with me first", which is a regression dressed as a safety feature.
        if not getattr(self.assistant, "work_mode", False):
            return None
        t = text.strip()
        if _SAY_TYPE_RE.match(t):
            return "type"
        if _KEY_RE.match(t):
            return "key"
        if _SCROLL_RE.match(t):
            return "scroll"
        if _SEND_RE.match(t):
            return "click"
        if _CLICK_RE.match(t) and not _CLICK_NOT_RE.search(t):
            return "click"
        if _TYPE_PLACE_RE.match(t):
            return None
        if _TYPE_RE.match(t) and re.match(_LEAD + r"(?:type|write|input)\b", t,
                                          re.I):
            return "type"
        return None

    # ── Handling ──────────────────────────────────────────────────────────────
    def handle(self, intent: str, text: str) -> str:
        # GATE 1: work mode. Nova does not drive his Mac while she is sitting
        # in the middle of the screen being a voice assistant.
        if not getattr(self.assistant, "work_mode", False):
            return ("I only type and click while we're working together. "
                    "Say work with me first.")

        # GATE 2: the permission, preflighted rather than discovered.
        if not A.has_accessibility():
            return ("I need Accessibility permission to do that. You can grant "
                    "it under Privacy and Security, Accessibility.")

        try:
            if intent == "type":
                return self._type(text)
            if intent == "key":
                return self._key(text)
            if intent == "scroll":
                return self._scroll(text)
            if intent == "click":
                return self._click(text)
        except Exception as exc:
            log.exception(f"actuation failed: {exc}")
            return "Something went wrong doing that."
        return "I'm not sure what to do there."

    # ── Typing: free, by design ───────────────────────────────────────────────
    def _type(self, text: str) -> str:
        m = _SAY_TYPE_RE.match(text.strip())
        if m:
            body = m.group("quoted")
        else:
            m2 = _TYPE_RE.match(text.strip())
            body = (m2.group("quoted") or m2.group("q")) if m2 else ""
        body = (body or "").strip()
        if not body:
            return "What should I type?"
        if not A.type_text(body):
            return "I couldn't type that."
        # Says what it did and STOPS. Sending is a separate, gated step — but
        # only OFFERED where there is something to send. `return_sends` already
        # knew the difference and this line never asked it: Nova offered to
        # send a paragraph typed into TextEdit, where Return is just a newline
        # and there is no recipient. An offer that makes no sense in the app he
        # is looking at is how a confirmation stops being read.
        if A.return_sends():
            return "Typed it. Want me to send it?"
        return "Typed it."

    # ── Keys: Return is the ambiguous one ─────────────────────────────────────
    def _key(self, text: str) -> str:
        m = _KEY_RE.match(text.strip())
        key = (m.group("key") if m else "").lower()
        if key in ("return", "enter") and A.return_sends():
            app = A.frontmost_app()
            # HIS example. In Messages, Return IS send, so this is not a
            # keystroke, it is an irreversible action.
            self.pending = lambda: (
                "Sent." if A.press_key("return") else "I couldn't press it.")
            return f"That would send it in {app}. Want me to?"
        if key == "enter":
            key = "return"
        elif key == "esc":
            key = "escape"
        elif key == "backspace":
            key = "delete"
        if not A.press_key(key):
            return f"I can't press {key}."
        return f"Pressed {key}."

    def _scroll(self, text: str) -> str:
        m = _SCROLL_RE.match(text.strip())
        direction = (m.group("dir") if m else "down").lower()
        if not A.scroll(direction):
            return "I couldn't scroll."
        return f"Scrolled {direction}."

    # ── Clicking: found by name, confirmed when it matters ────────────────────
    def _click(self, text: str) -> str:
        if _SEND_RE.match(text.strip()):
            label = "Send"
        else:
            m = _CLICK_RE.match(text.strip())
            label = (m.group("label") if m else "").strip()
            # The label group is greedy, so "the Send button" arrives as
            # "Send button". A control is named "Send"; "button" is how he
            # refers to it, not part of the name.
            label = re.sub(r"\s+(?:button|link|tab|menu|icon)$", "", label,
                           flags=re.I).strip()
        if not label:
            return "What should I click?"

        target = A.find(label)
        if target is None:
            # Never falls back to a guessed coordinate. Not finding it is the
            # honest answer, and the whole difference from clicking blind.
            return f"I can't find anything called {label} on screen."

        if A.needs_confirmation(target.label):
            self.pending = lambda t=target: (
                f"Clicked {t.label}." if A.click(t)
                else f"I couldn't click {t.label}.")
            return f"That would click {target.label}. Want me to?"

        if not A.click(target):
            return f"I couldn't click {target.label}."
        return f"Clicked {target.label}."

    # ── Confirmation ──────────────────────────────────────────────────────────
    def resolve_pending(self, text: str) -> Optional[str]:
        """Answer a "want me to?". Anything that is not a clear yes cancels —
        the same strictness as the power commands, for the same reason."""
        if self.pending is None:
            return None
        action = self.pending
        self.pending = None
        if re.match(r"^\s*(yes|yeah|yep|yup|sure|do it|go ahead|please|"
                    r"confirm|send it)\b", text.strip(), re.I):
            return action()
        return "Okay, I won't."
