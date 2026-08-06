"""
Nova Calendar & Reminders — natural-language dispatch layer.

Sits on top of the pure-function engine in ``calendar_reminders.py``. Turns a
free-form utterance into a calendar/reminders action and returns a single
spoken response string (Nova's ``_respond`` handles the TTS).

Design notes specific to Nova (vs the Jarvis original this was ported from):

  * NO background threads or locks. ``NovaCalendar.handle`` is invoked from
    ``_handle_turn_impl``, which already runs on the single ``nova-llm`` worker
    thread. That means the LLM extraction/summarize calls below run on the one
    thread that owns MLX — calling ``self.llm.generate(...)`` directly is
    thread-safe. Do NOT call it from any other thread.
  * Structured extraction uses ``temperature=0`` — the 3B model needs
    determinism for JSON, and every value is post-processed rather than trusted
    (string "null" sanitation, weekday override, explicit-rename gating).
  * Reads are summarized by the LLM for a natural voice, but every read has a
    deterministic template fallback so a bad generation never hallucinates a
    day or time.

Detection is STRICT regex only: an utterance without an unambiguous calendar
word falls through to None so normal chat is never hijacked.
"""

from __future__ import annotations

import datetime
import json
import logging
import re
from typing import Optional

import calendar_reminders as cal

log = logging.getLogger("nova.calendar.intents")


class NovaCalendar:
    def __init__(self, config: dict, llm, memory) -> None:
        self.config = config
        self.llm = llm
        self.memory = memory
        self.name = config["user"]["address_as"]
        cal_cfg = config.get("calendar", {})
        # Template confirmations are ~4-6s faster than an extra LLM round-trip
        # on the create path and can't drift a day/time. Reads still use the
        # LLM (that's the point of a natural read). Default: template.
        self._natural_confirmations = bool(cal_cfg.get("natural_confirmations", False))

        # A read may leave a soft follow-up offer ("want to hear what's coming
        # up?"). nova.py reads this after handle() and arms a one-shot offer
        # that only fires on an affirmative reply. Reset before every handle().
        self.pending_intent: Optional[str] = None
        # Date already reported by a day-scoped reminder read; the look-ahead
        # offer reads reminders due AFTER this day.
        self._reminder_lookahead_exclude: Optional[datetime.date] = None

        # System prompt for all silent calendar LLM calls. Includes the user's
        # name so the 3B stops inventing one ("Samantha"). Nova speaks aloud,
        # so: no markdown, no lists, no em dashes (CLAUDE.md invariant #10).
        self._cal_system = (
            "You are Nova, a sharp, composed AI assistant. Speak naturally and "
            "conversationally, the way a person would say it out loud. Never use "
            "markdown, bullet points, numbered lists, or em dashes. Keep replies "
            f"brief. The user's name is {self.name}; address them as {self.name} "
            "and never use any other name."
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Intent detection (strict regex; returns intent name or None)
    # ═══════════════════════════════════════════════════════════════════════
    def detect_intent(self, text: str) -> Optional[str]:
        """Return one of: read_today, read_upcoming, read_reminders,
        create_event, create_reminder, complete_reminder, delete_reminder,
        update_reminder, delete_event — or None.

        Every pattern MUST include an unambiguous calendar word; anything
        without a strong signal returns None and falls through to chat."""
        t = (text or "").lower().strip()
        if not t:
            return None

        # ── Read reminders ───────────────────────────────────────────────
        # Broad READ phrasings only (kept clear of the delete/complete/update
        # verbs, which are matched further down). "what reminders do I have"
        # and "do I have any reminders" were both missed before.
        if re.search(r"\bwhat\s+(?:are|were)\s+(?:all\s+(?:of\s+)?)?my\s+reminders\b", t) or \
           re.search(r"\bwhat\s+reminders\s+(?:do|have)\s+i\b", t) or \
           re.search(r"\b(?:do|have)\s+i\s+have\s+any\s+reminders\b", t) or \
           re.search(r"\bany\s+reminders\b", t) or \
           re.search(r"\b(?:show|read|list|tell|give|check)\s+(?:me\s+)?(?:all\s+)?(?:my\s+)?reminders\b", t) or \
           re.search(r"\bwhat(?:'?s|\s+is)\s+on\s+my\s+reminders?\s+list\b", t) or \
           re.search(r"\bmy\s+reminders\s+(?:for\s+)?(?:today|this\s+week)\b", t) or \
           re.search(r"\breminders\s+(?:for\s+)?today\b", t) or \
           re.search(r"\bwhat\s+do\s+i\s+(?:need|have)\s+to\s+do\s+(?:today|this\s+week)\b", t):
            return "read_reminders"

        # ── Read upcoming (rest of the week) ─────────────────────────────
        # Checked BEFORE read_today so "calendar this week" doesn't get caught
        # by the bare "what's on my calendar" catch-all in read_today. REQUIRE
        # "this week"/"the week" (not bare "week") so "calendar week starts on
        # monday" doesn't false-fire.
        if re.search(r"\bwhat(?:'?s|\s+is)\s+coming\s+up\s+on\s+(?:my\s+)?(?:calendar|schedule)\b", t) or \
           re.search(r"\b(?:my\s+)?calendar\s+(?:for\s+)?(?:this|the)\s+week\b", t) or \
           re.search(r"\b(?:my\s+)?schedule\s+(?:for\s+)?(?:this|the)\s+week\b", t) or \
           re.search(r"\bwhat\s+do\s+i\s+have\s+(?:for\s+)?(?:this|the)\s+week\b", t) or \
           re.search(r"\bwhat(?:'?s|\s+is)\s+on\s+(?:my\s+)?agenda\b", t):
            return "read_upcoming"

        # ── Read today's calendar (explicit today, or a bare calendar ask) ──
        if re.search(r"\b(?:my\s+)?calendar\s+(?:for\s+)?today\b", t) or \
           re.search(r"\b(?:my\s+)?schedule\s+(?:for\s+)?today\b", t) or \
           re.search(r"\bwhat(?:'?s|\s+is)\s+my\s+schedule\s+today\b", t) or \
           re.search(r"\bwhat\s+do\s+i\s+have\s+(?:on\s+)?today\b", t) or \
           re.search(r"\banything\s+on\s+(?:my\s+)?calendar\b", t) or \
           re.search(r"\bwhat(?:'?s|\s+is)\s+on\s+(?:my\s+)?calendar\b", t) or \
           re.search(r"\bwhat(?:'?s|\s+is)\s+my\s+schedule\b", t):
            return "read_today"

        # ── Create reminder ──────────────────────────────────────────────
        # "reminder" is an unambiguous calendar word, so set/create/add/make +
        # "reminder" matches regardless of what follows ("set a reminder FOR
        # five to call mom" was missed when we required "set a reminder TO").
        # For the bare "remind me" verb we still require "to" so "remind me IN
        # five minutes" (a timer) doesn't route here.
        if re.search(r"\b(?:set|create|add|make|new)\s+(?:a\s+|an\s+|another\s+)?reminder\b", t) or \
           re.search(r"\bremind\s+me\s+to\b", t):
            return "create_reminder"

        # ── Complete a reminder (before delete: distinct verb) ───────────
        if re.search(r"\b(?:complete|finish|check\s+off|mark\s+(?:as\s+)?(?:done|complete|completed|finished))\b[^.?!]*\breminder\b", t) or \
           re.search(r"\b(?:complete|finish|check\s+off)\s+(?:the\s+)?reminder\b", t) or \
           re.search(r"\b(?:mark|flag)\s+(?:the\s+)?.+?\s+(?:as\s+)?(?:done|completed|complete|finished)\b", t) or \
           re.search(r"\breminder\s+(?:is\s+)?(?:done|completed|complete)\b", t):
            return "complete_reminder"

        # ── Delete a reminder ────────────────────────────────────────────
        if re.search(r"\b(?:delete|remove|cancel|drop|get\s+rid\s+of|throw\s+out)\s+(?:the\s+)?[^.?!]*?\breminder\b", t) or \
           re.search(r"\breminder\s+(?:for\s+[^.?!]*?\s+)?(?:is\s+)?(?:gone|cancelled|canceled|no\s+longer\s+needed)\b", t):
            return "delete_reminder"

        # ── Update a reminder (reschedule / rename / change notes) ────────
        # Non-capturing group wrapper is REQUIRED: this is an A|B|C alternation,
        # and the top-level `|` has the lowest precedence. Concatenated raw onto
        # a longer pattern, the bare `at\s+\d` alternative would match "at 3" in
        # ANY utterance (e.g. "schedule a meeting at 3" mis-routes to update).
        _DATE_OR_TIME_SUFFIX = (
            r"(?:\b(?:\d{1,2}(?:st|nd|rd|th)?|"
            r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
            r"tomorrow|today|tonight|this\s+\w+|next\s+\w+)"
            r"|at\s+\d|\d{1,2}\s*(?:am|pm|a\.m|p\.m))"
        )
        if re.search(r"\b(?:reschedule|move|change|update|edit|rename)\s+(?:the\s+)?[^.?!]*?\breminder\b", t) or \
           re.search(r"\b(?:change|update|move)\s+(?:the\s+)?reminder\b", t) or \
           re.search(r"\b(?:reschedule|move)\b[^.?!]{2,60}?\bto\b[^.?!]{0,30}?" + _DATE_OR_TIME_SUFFIX, t) or \
           re.search(r"\brename\s+(?:the\s+)?[^.?!]{2,60}?\s+to\s+[^.?!]{2,}", t):
            return "update_reminder"

        # ── Delete a calendar event ──────────────────────────────────────
        if re.search(r"\b(?:delete|remove|cancel|drop)\s+(?:the\s+)?[^.?!]*?\b(?:event|meeting|appointment|shift)\b", t) or \
           re.search(r"\b(?:cancel|remove|delete)\s+[^.?!]*?\s+from\s+(?:my\s+)?calendar\b", t) or \
           re.search(r"\b(?:take|get)\s+[^.?!]*?\s+off\s+(?:my\s+)?calendar\b", t):
            return "delete_event"

        # ── Create calendar event (strict) ───────────────────────────────
        if re.search(r"\b(?:add|schedule|put|create|book)\b[^.?!]{0,40}\b(?:to|on|in|for)\s+(?:my\s+)?calendar\b", t) or \
           re.search(r"\bput\s+(?:this|that|it)\s+on\s+my\s+calendar\b", t) or \
           re.search(r"\bcreate\s+(?:a\s+)?(?:new\s+)?(?:calendar\s+)?event\b", t) or \
           re.search(r"\badd\s+(?:a\s+)?(?:new\s+)?(?:calendar\s+)?event\b", t) or \
           re.search(r"\b(?:schedule|add|book|create)\s+[^.?!]{0,30}\b(?:meeting|appointment)\b", t) or \
           re.search(r"\bnew\s+(?:calendar\s+)?event\b", t):
            return "create_event"

        # ── Shift-style create_event ─────────────────────────────────────
        # Requires BOTH a working/shift phrase AND an unambiguous time signal
        # (explicit digits). The time requirement keeps "I'm working from home
        # today" (no digits) from matching.
        _TIME_SIGNAL = (
            r"\b(?:from\s+\d|at\s+\d|\d{1,2}\s*(?:to|until|-|till)\s*\d|"
            r"\d{1,2}\s*(?:am|pm|a\.m|p\.m|o'?clock))"
        )
        _SHIFT_PHRASE = (
            r"\b(?:i(?:'?m|\s+am)\s+working|"
            r"i(?:'?ll|\s+will)\s+be\s+working|"
            r"i\s+work\b|"
            r"i\s+have\s+(?:a\s+)?shift|"
            r"i(?:'?ve|\s+have)\s+got\s+(?:a\s+)?shift|"
            r"i(?:'?m|\s+am)\s+on\s+shift|"
            r"working\s+(?:a\s+)?shift)\b"
        )
        if re.search(_SHIFT_PHRASE, t) and re.search(_TIME_SIGNAL, t):
            return "create_event"

        return None

    # ═══════════════════════════════════════════════════════════════════════
    # Dispatch
    # ═══════════════════════════════════════════════════════════════════════
    def handle(self, intent: str, text: str) -> str:
        """Execute a calendar intent and return the spoken response string.

        Any RuntimeError from the engine (permission/timeout) is turned into a
        speakable message so a calendar failure never crashes the pipeline."""
        self.pending_intent = None  # cleared every turn; a read may re-arm it
        try:
            if intent == "read_today":
                return self._read_today()
            if intent == "read_upcoming":
                return self._read_upcoming()
            if intent == "read_rest_of_week":
                return self._read_rest_of_week()
            if intent == "read_reminders":
                return self._read_reminders(text)
            if intent == "read_reminders_lookahead":
                return self._read_reminders_lookahead()
            if intent == "create_event":
                return self._create_event(text)
            if intent == "create_reminder":
                return self._create_reminder(text)
            if intent == "complete_reminder":
                return self._complete_reminder(text)
            if intent == "delete_reminder":
                return self._delete_reminder(text)
            if intent == "update_reminder":
                return self._update_reminder(text)
            if intent == "delete_event":
                return self._delete_event(text)
        except RuntimeError as e:
            log.error(f"calendar engine error in intent={intent!r}: {e}")
            return (
                f"I wasn't able to access your calendar, {self.name}. "
                "You may need to grant permission in System Settings."
            )
        except Exception:
            log.exception(f"calendar handler error in intent={intent!r}")
            return f"Something went wrong with that calendar request, {self.name}."
        return f"I'm not sure how to handle that, {self.name}."

    # ═══════════════════════════════════════════════════════════════════════
    # Silent LLM helper (runs on the nova-llm thread — see module docstring)
    # ═══════════════════════════════════════════════════════════════════════
    def _llm_silent(self, user_prompt: str, max_tokens: int = 220,
                    temperature: float = 0.6) -> str:
        """One-shot generation with no history side effects."""
        try:
            return self.llm.generate(
                system_prompt=self._cal_system,
                history=[],
                user_message=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            ).strip()
        except Exception as e:
            log.error(f"_llm_silent error: {e}")
            return ""

    # ── Structured extraction ────────────────────────────────────────────
    def _extract_event_json(self, utterance: str) -> Optional[dict]:
        """Pull structured event/reminder fields from a free-form utterance."""
        today_date = datetime.date.today()
        today_iso = today_date.isoformat()
        today_name = today_date.strftime("%A")
        prompt = (
            "Extract calendar event details from this request. Respond in "
            "JSON only, no explanation, no markdown. Use JSON null (the bare "
            "keyword, not the string \"null\") for any field that isn't "
            "specified.\n"
            "\n"
            "Fields:\n"
            "- title: a short, descriptive event name. For a work shift use "
            '"Work". For a meeting use "Meeting". NEVER use "Nova" as a title. '
            "Never include day names (Monday, Saturday, etc.) in the title.\n"
            "- date: YYYY-MM-DD, or the word today/tomorrow/yesterday, or a "
            "weekday name. If the user explicitly names a weekday, USE THAT "
            'WEEKDAY, do NOT substitute "today".\n'
            "- start_time: HH:MM in 24-hour format. If the user says a bare "
            "hour with no AM/PM (e.g. 'five', 'at eight'), pick the time of day "
            "a person most likely means: hours 1 through 7 default to PM "
            "(afternoon/evening), 8 through 11 to AM, unless the request "
            "clearly says otherwise.\n"
            "- end_time: HH:MM in 24-hour format, or null\n"
            "- location: string, or null\n"
            "- notes: string, or null\n"
            "- is_reminder: true if this is a reminder, false if a calendar event\n"
            "\n"
            f"Today is {today_iso} ({today_name}). "
            f"User said: '{utterance}'"
        )
        raw = self._llm_silent(prompt, max_tokens=220, temperature=0.0)
        return self._parse_json(raw, label="event")

    def _extract_update_json(self, utterance: str) -> Optional[dict]:
        """Extraction prompt specifically for UPDATE operations — only fields
        the user wants to CHANGE, never the identifier used to find the item."""
        today_date = datetime.date.today()
        today_iso = today_date.isoformat()
        today_name = today_date.strftime("%A")
        prompt = (
            "The user wants to UPDATE an existing reminder. Extract ONLY the "
            "NEW field values they want to set. Fields the user is using to "
            "IDENTIFY the existing item (its current title or partial name) "
            "must NOT appear in your output.\n\n"
            "Respond in JSON only, no explanation, no markdown. Use JSON null "
            "(not the string 'null') for any field NOT being changed.\n\n"
            "Fields:\n"
            "- new_title: ONLY if the user explicitly asked to rename the item "
            "(e.g. 'rename X to Y'). If they used the old name just to identify "
            "the item, leave this null. Default: null.\n"
            "- new_date: YYYY-MM-DD or a weekday name or 'today'/'tomorrow', "
            "ONLY if rescheduling. Default: null.\n"
            "- new_time: HH:MM in 24-hour format, ONLY if a new time was given. "
            "Default: null.\n"
            "- new_notes: new notes content, ONLY if changing notes. Default: null.\n\n"
            "Examples:\n"
            "  'reschedule the grocery reminder to tomorrow at 6 PM' -> "
            "{\"new_title\": null, \"new_date\": \"tomorrow\", "
            "\"new_time\": \"18:00\", \"new_notes\": null}\n"
            "  'move the GPT subscription reminder to Sunday at 9 AM' -> "
            "{\"new_title\": null, \"new_date\": \"sunday\", "
            "\"new_time\": \"09:00\", \"new_notes\": null}\n"
            "  'rename the groceries reminder to weekly shopping' -> "
            "{\"new_title\": \"weekly shopping\", \"new_date\": null, "
            "\"new_time\": null, \"new_notes\": null}\n\n"
            f"Today is {today_iso} ({today_name}). "
            f"User said: '{utterance}'"
        )
        raw = self._llm_silent(prompt, max_tokens=180, temperature=0.0)
        return self._parse_json(raw, label="update")

    @staticmethod
    def _parse_json(raw: str, label: str) -> Optional[dict]:
        if not raw:
            return None
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        raw = re.sub(r"\s*```\s*$", "", raw).strip()
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if m:
            raw = m.group(0)
        try:
            return json.loads(raw)
        except Exception as e:
            log.warning(f"{label}-JSON parse error: {e}  — raw: {raw!r}")
            return None

    # ── Date / time resolution ───────────────────────────────────────────
    def _resolve_relative_date(self, date_str: Optional[str]) -> datetime.date:
        """Resolve today/tomorrow/ISO/<weekday> to an actual date."""
        today = datetime.date.today()
        if not date_str:
            return today
        s = str(date_str).lower().strip()
        if s in ("today", "now"):
            return today
        if s == "tomorrow":
            return today + datetime.timedelta(days=1)
        if s == "yesterday":
            return today - datetime.timedelta(days=1)
        try:
            return datetime.date.fromisoformat(s)
        except Exception:
            pass
        weekdays = {
            "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
            "friday": 4, "saturday": 5, "sunday": 6,
        }
        m = re.search(r"(next\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)", s)
        if m:
            target = weekdays[m.group(2)]
            delta = (target - today.weekday()) % 7
            if delta == 0:
                delta = 7 if m.group(1) else 0
            return today + datetime.timedelta(days=delta)
        return today

    @staticmethod
    def _parse_time(t: Optional[str]) -> Optional[tuple]:
        """Parse a time string into (hour, minute) 24-hour. Accepts '09:00',
        '9:00', '9:00 AM', '9 AM', '9am', '9 pm', '21:00'."""
        if not t:
            return None
        s = str(t).strip().lower()
        s_stripped_dot = s.replace(".", "")
        is_pm = s_stripped_dot.endswith("pm") or s_stripped_dot.endswith(" pm")
        is_am = s_stripped_dot.endswith("am") or s_stripped_dot.endswith(" am")
        s = re.sub(r"\s*[ap]\.?m\.?\s*$", "", s).strip()
        m = re.match(r"^(\d{1,2})(?:[.:](\d{2}))?$", s)
        if not m:
            return None
        try:
            h = int(m.group(1))
            mi = int(m.group(2)) if m.group(2) else 0
            if not (0 <= h <= 23 and 0 <= mi <= 59):
                return None
            if is_pm and h < 12:
                h += 12
            elif is_am and h == 12:
                h = 0
            return (h, mi)
        except Exception:
            return None

    # Words that pin the time of day; if any is present we trust the extracted
    # hour and never apply the bare-hour PM default below.
    _MERIDIEM_RE = re.compile(r"\b(a\.?m\.?|p\.?m\.?|noon|midnight|morning)\b", re.I)

    def _default_bare_hour(self, hour: int, utterance: str) -> int:
        """Deterministic AM/PM default for a bare spoken hour. When the user
        gave no AM/PM (and didn't say morning/noon/midnight), an hour of 1-7 is
        far more likely PM for a personal reminder/event ('remind me at five'
        means 5 PM). Idempotent: an already-PM hour (>=12) is untouched."""
        if 1 <= hour <= 7 and not self._MERIDIEM_RE.search(utterance or ""):
            return hour + 12
        return hour

    # ── Deterministic reminder parsing (no LLM) ──────────────────────────
    _NUM_WORDS = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
        "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    }
    _WEEKDAY_RE = r"monday|tuesday|wednesday|thursday|friday|saturday|sunday"

    def _strip_reminder_prefix(self, utterance: str) -> str:
        """Remove a leading wake word + the create-reminder command verb, so
        what's left is the task (+ any date/time). 'Nova, set a reminder to
        call mom at five' -> 'call mom at five'."""
        t = (utterance or "").strip()
        t = re.sub(r"^\s*(?:hey\s+)?nova\b[\s,]*", "", t, flags=re.I)
        t = re.sub(
            r"^\s*(?:please\s+|can\s+you\s+|could\s+you\s+|i\s+want\s+(?:you\s+)?to\s+|"
            r"i\s+need\s+(?:you\s+)?to\s+)?",
            "", t, flags=re.I,
        )
        # "set/create/add/make (a) reminder (to|for|about|that|saying|:)"
        t2 = re.sub(
            r"^\s*(?:set|create|add|make|new)\s+(?:a\s+|an\s+|another\s+)?reminders?\s*"
            r"(?:to|for|about|that|saying|:)?\s*",
            "", t, flags=re.I,
        )
        if t2 != t:
            return t2.strip()
        # "remind me (to|about|that)"
        return re.sub(r"^\s*remind\s+me\s+(?:to|about|that)\s+", "", t, flags=re.I).strip()

    # Time-phrase building blocks, reused for both parsing and title-stripping.
    _T_DIGIT_AT = r"\b(?:at|by|around|@)\s*(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)?"
    _T_DIGIT_MER = r"\b(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)"  # requires am/pm

    def _find_time(self, text: str):
        """Return (hour, minute) parsed from a time phrase, or None."""
        low = (text or "").lower()
        if re.search(r"\b(?:noon|midday)\b", low):
            return (12, 0)
        if re.search(r"\bmidnight\b", low):
            return (0, 0)
        m = re.search(self._T_DIGIT_AT, low) or re.search(self._T_DIGIT_MER, low)
        if m:
            h, mi = int(m.group(1)), int(m.group(2) or 0)
            mer = (m.group(3) or "").replace(".", "")
            if mer == "pm" and h < 12:
                h += 12
            elif mer == "am" and h == 12:
                h = 0
            elif not mer:
                h = self._default_bare_hour(h, text)
            if 0 <= h <= 23 and 0 <= mi <= 59:
                return (h, mi)
        numw = "|".join(self._NUM_WORDS)
        m = re.search(r"\b(?:at|by|around)\s+(" + numw + r")(?:\s+o'?clock)?\s*(a\.?m\.?|p\.?m\.?)?\b", low) \
            or re.search(r"\b(" + numw + r")\s+(a\.?m\.?|p\.?m\.?)\b", low)
        if m:
            h = self._NUM_WORDS[m.group(1)]
            mer = (m.group(2) or "").replace(".", "")
            if mer == "pm" and h < 12:
                h += 12
            elif mer == "am" and h == 12:
                h = 0
            elif not mer:
                h = self._default_bare_hour(h, text)
            return (h, 0)
        return None

    def _strip_when_phrases(self, text: str) -> str:
        """Remove date and time phrases so what remains is the reminder title."""
        numw = "|".join(self._NUM_WORDS)
        t = " " + (text or "") + " "
        t = re.sub(self._T_DIGIT_AT, " ", t, flags=re.I)
        t = re.sub(self._T_DIGIT_MER, " ", t, flags=re.I)
        t = re.sub(r"\b(?:at|by|around)\s+(?:" + numw + r")(?:\s+o'?clock)?\s*(?:a\.?m\.?|p\.?m\.?)?\b", " ", t, flags=re.I)
        t = re.sub(r"\b(?:" + numw + r")\s+(?:a\.?m\.?|p\.?m\.?)\b", " ", t, flags=re.I)
        t = re.sub(r"\b(?:noon|midday|midnight)\b", " ", t, flags=re.I)
        t = re.sub(r"\b(?:next\s+)?(?:today|tonight|tomorrow|" + self._WEEKDAY_RE + r")\b", " ", t, flags=re.I)
        t = re.sub(r"\bthis\s+(?:morning|afternoon|evening)\b", " ", t, flags=re.I)
        t = re.sub(r"\s+", " ", t).strip(" ,.!?;:-")
        # Drop a dangling connective left behind by the removed when-phrase.
        t = re.sub(r"^(?:to|for|on|at|about)\s+", "", t, flags=re.I)
        t = re.sub(r"\s+(?:to|for|on|at|by|around|about)$", "", t, flags=re.I)
        return t.strip(" ,.!?;:-").strip()

    def _parse_reminder(self, utterance: str):
        """Deterministically extract (title, due_datetime|None) from a
        create-reminder utterance. title is None if nothing usable remains."""
        body = self._strip_reminder_prefix(utterance)

        date_obj = None
        dm = re.search(r"\b(next\s+)?(today|tonight|tomorrow|" + self._WEEKDAY_RE + r")\b", body, re.I)
        if dm:
            date_obj = self._resolve_relative_date(
                (("next " if dm.group(1) else "") + dm.group(2)).lower()
            )

        hm = self._find_time(body)

        due_dt: Optional[datetime.datetime] = None
        if hm is not None:
            d = date_obj or datetime.date.today()
            due_dt = datetime.datetime.combine(d, datetime.time(hm[0], hm[1]))
        elif date_obj is not None:
            due_dt = datetime.datetime.combine(date_obj, datetime.time(9, 0))

        title = self._strip_when_phrases(body) or None
        return title, due_dt

    @staticmethod
    def _titlecase(s: str) -> str:
        """Capitalize the first letter of each word for a tidy reminder title
        ('call mom' -> 'Call Mom'), preserving the rest of each word so
        apostrophes and existing capitals survive ('mom's' -> 'Mom's')."""
        if not s:
            return s
        return " ".join(w[:1].upper() + w[1:] if w else w for w in s.split(" "))

    @staticmethod
    def _clean_optional(value):
        """Normalize the LLM's string "null"/"none"/etc. to a real None."""
        if value is None:
            return None
        s = str(value).strip()
        if s.lower() in ("null", "none", "n/a", "na", "nil", ""):
            return None
        return s

    # ═══════════════════════════════════════════════════════════════════════
    # Read handlers
    # ═══════════════════════════════════════════════════════════════════════
    def _read_today(self) -> str:
        """Read TODAY's events, then softly offer the rest of the week. The
        default is always today; the user opts into 'what's coming up'."""
        events = cal.get_today_events()
        today_txt = self._summarize_today(events)

        rest = self._rest_of_week_events()
        if rest:
            self.pending_intent = "read_rest_of_week"
            n = len(rest)
            today_txt += (
                f" You've also got {n} thing{'s' if n != 1 else ''} coming up "
                "later this week. Want me to run through those?"
            )
        return today_txt

    def _summarize_today(self, events: list) -> str:
        if not events:
            return f"Your calendar is clear for today, {self.name}."
        lines = self._today_lines(events)
        prompt = (
            "These are the events on the user's calendar for TODAY. Summarize "
            "them naturally and conversationally, the way a sharp assistant "
            "would say it out loud, not a list. They are all today, so give the "
            "time of each (for example 'at 7 AM'); do NOT state the date or day "
            "of week. Do not invent events. Keep it to one to three sentences.\n\n"
            + "\n".join(lines)
        )
        text = self._llm_silent(prompt, max_tokens=200)
        return text or self._template_today(events)

    def _read_upcoming(self) -> str:
        events = cal.get_upcoming_events()
        if not events:
            return f"Nothing on the books for the rest of the week, {self.name}."
        return self._summarize_week(events)

    def _read_rest_of_week(self) -> str:
        rest = self._rest_of_week_events()
        if not rest:
            return f"Nothing else on your calendar this week, {self.name}."
        return self._summarize_week(rest)

    def _summarize_week(self, events: list) -> str:
        lines = self._week_lines(events)
        prompt = (
            "These are upcoming events on the user's calendar. Summarize them "
            "naturally and conversationally, not a list.\n\n"
            "CRITICAL RULES:\n"
            "Read the day of week and time for each event EXACTLY as given in "
            "the input. Do not change AM to PM. Do not invent events. Keep it "
            "to two to five sentences.\n\n"
            + "\n".join(lines)
        )
        text = self._llm_silent(prompt, max_tokens=260)
        return text or self._template_week(events)

    def _rest_of_week_events(self) -> list:
        """Events from tomorrow through the end of this coming Saturday
        (get_upcoming_events covers today..Saturday; drop today's)."""
        today = datetime.date.today()
        out = []
        try:
            upcoming = cal.get_upcoming_events()
        except Exception:
            return []
        for e in upcoming:
            dt = self._record_dt(e)
            if dt is not None and dt.date() > today:
                out.append(e)
        return out

    def _read_reminders(self, text: str = "") -> str:
        reminders = cal.get_all_reminders()
        if not reminders:
            return f"You have no open reminders, {self.name}."

        scope, scope_date = self._reminder_scope(text)
        if scope == "day":
            return self._read_reminders_for_day(reminders, scope_date)
        if scope == "week":
            today = datetime.date.today()
            sat = today + datetime.timedelta(days=(5 - today.weekday()) % 7)
            wk = [r for r in reminders
                  if (dt := self._reminder_dt(r)) and today <= dt.date() <= sat]
            if not wk:
                return f"You have nothing due this week, {self.name}."
            return self._summarize_reminders(wk, with_dates=True, when_label="this week")
        # "all": every open reminder, with dates, no offer.
        return self._summarize_reminders(reminders, with_dates=True)

    def _reminder_scope(self, text: str):
        """Classify a reminder-read request: ('day', date) / ('week', None) /
        ('all', None). Default is 'all' when no day/week word is present."""
        low = (text or "").lower()
        if re.search(r"\ball\s+(?:my\s+|of\s+my\s+)?reminders\b", low) or "everything" in low:
            return ("all", None)
        today = datetime.date.today()
        if re.search(r"\btomorrow\b", low):
            return ("day", today + datetime.timedelta(days=1))
        if re.search(r"\b(?:today|tonight|this\s+(?:morning|afternoon|evening))\b", low):
            return ("day", today)
        m = re.search(r"\b(next\s+)?(" + self._WEEKDAY_RE + r")\b", low)
        if m:
            return ("day", self._resolve_relative_date(
                (("next " if m.group(1) else "") + m.group(2))))
        if re.search(r"\b(?:this|the)\s+week\b", low):
            return ("week", None)
        return ("all", None)

    @staticmethod
    def _reminder_dt(r: dict) -> Optional[datetime.datetime]:
        s = r.get("due_iso", "")
        if not s:
            return None
        try:
            return datetime.datetime.fromisoformat(s)
        except Exception:
            return None

    def _read_reminders_for_day(self, reminders: list, day: datetime.date) -> str:
        same, later, undated = [], [], []
        for r in reminders:
            dt = self._reminder_dt(r)
            if dt is None:
                undated.append(r)
            elif dt.date() == day:
                same.append(r)
            elif dt.date() > day:
                later.append(r)
            # earlier than `day` (overdue) is not volunteered here.

        label = self._day_word(day)
        if same:
            base = self._summarize_reminders(same, with_dates=False, when_label=label)
        else:
            base = f"You have no reminders for {label}, {self.name}."

        # Soft look-ahead offer for anything beyond the asked-about day.
        rest = later + undated
        if rest:
            self.pending_intent = "read_reminders_lookahead"
            self._reminder_lookahead_exclude = day
            n = len(rest)
            base += (f" You've also got {n} other reminder{'s' if n != 1 else ''} "
                     "coming up. Want to hear those?")
        return base

    def _read_reminders_lookahead(self) -> str:
        day = getattr(self, "_reminder_lookahead_exclude", None)
        reminders = cal.get_all_reminders()
        rest = []
        for r in reminders:
            dt = self._reminder_dt(r)
            if day is None or dt is None or dt.date() > day:
                rest.append(r)
        if not rest:
            return f"Nothing else on your list, {self.name}."
        return self._summarize_reminders(rest, with_dates=True)

    def _day_word(self, day: datetime.date) -> str:
        """A short spoken label for a date: today / tomorrow / the weekday."""
        today = datetime.date.today()
        if day == today:
            return "today"
        if day == today + datetime.timedelta(days=1):
            return "tomorrow"
        return day.strftime("%A")

    def _summarize_reminders(self, reminders: list, with_dates: bool,
                             when_label: Optional[str] = None) -> str:
        lines = []
        for i, r in enumerate(reminders, start=1):
            title = r.get("title", "")
            dt = self._reminder_dt(r)
            if with_dates:
                when = r.get("due") or (cal.format_datetime_for_speech(dt) if dt else "")
                lines.append(f"{i}. {title}" + (f" (due {when})" if when else " (no due date)"))
            else:
                when = cal.format_time_for_speech(dt) if dt else ""
                lines.append(f"{i}. {title}" + (f" (at {when})" if when else ""))
        if with_dates:
            rule = ("Read each due date and time EXACTLY as written; do not "
                    "change AM to PM, do not change the hour.")
        else:
            rule = ("These are all for the same day, so give only the TIME of "
                    "each (for example 'at 3 PM'); do NOT state the date or day "
                    "of week.")
        header = "These are the user's open reminders"
        if when_label:
            header += f" for {when_label}"
        prompt = (
            header + ", earliest first. Summarize them naturally and "
            "conversationally, not a list.\n\n"
            "CRITICAL RULES:\n"
            "Mention them in the given order. Do not invent reminders. " + rule +
            " Keep it brief, one to four sentences.\n\n" + "\n".join(lines)
        )
        text = self._llm_silent(prompt, max_tokens=220)
        return text or self._template_reminders(reminders, with_dates, when_label)

    # ── Event formatting (clean times/days, so summaries don't garble) ───
    @staticmethod
    def _record_dt(record: dict) -> Optional[datetime.datetime]:
        """Parse a record's 'start' string (the EventKit/AppleScript format)."""
        try:
            return datetime.datetime.strptime(
                record.get("start", ""), "%A, %B %d, %Y at %I:%M:%S %p"
            )
        except Exception:
            return None

    def _today_lines(self, events: list) -> list:
        lines = []
        for e in events:
            dt = self._record_dt(e)
            when = cal.format_time_for_speech(dt) if dt else (e.get("start") or "")
            seg = f"- {e.get('title', '')}"
            if when:
                seg += f" at {when}"
            if e.get("location"):
                seg += f" ({e['location']})"
            lines.append(seg)
        return lines

    def _week_lines(self, events: list) -> list:
        lines = []
        for e in events:
            dt = self._record_dt(e)
            when = cal.format_datetime_for_speech(dt) if dt else (e.get("start") or "")
            seg = f"- {e.get('title', '')}"
            if when:
                seg += f" on {when}"
            if e.get("location"):
                seg += f" ({e['location']})"
            lines.append(seg)
        return lines

    def _template_today(self, events: list) -> str:
        """Deterministic fallback — never hallucinates days/times."""
        n = len(events)
        parts = [f"You have {n} event{'s' if n != 1 else ''} today, {self.name}."]
        for e in events:
            dt = self._record_dt(e)
            seg = e.get("title", "Untitled")
            if dt:
                seg += f" at {cal.format_time_for_speech(dt)}"
            if e.get("location"):
                seg += f" at {e['location']}"
            parts.append(seg + ".")
        return " ".join(parts)

    def _template_week(self, events: list) -> str:
        n = len(events)
        parts = [f"You have {n} event{'s' if n != 1 else ''} coming up, {self.name}."]
        for e in events:
            dt = self._record_dt(e)
            seg = e.get("title", "Untitled")
            if dt:
                seg += f" on {cal.format_datetime_for_speech(dt)}"
            parts.append(seg + ".")
        return " ".join(parts)

    def _template_reminders(self, reminders: list, with_dates: bool = True,
                            when_label: Optional[str] = None) -> str:
        """Deterministic fallback. Day-scoped reads give times only; the
        all/week reads give full due dates."""
        n = len(reminders)
        head = f"You have {n} reminder{'s' if n != 1 else ''}"
        if when_label:
            head += f" for {when_label}"
        parts = [head + f", {self.name}."]
        for r in reminders:
            dt = self._reminder_dt(r)
            title = r.get("title", "")
            if with_dates:
                when = r.get("due") or (cal.format_datetime_for_speech(dt) if dt else "")
                parts.append(f"{title} is due {when}." if when else f"{title}.")
            else:
                when = cal.format_time_for_speech(dt) if dt else ""
                parts.append(f"{title} at {when}." if when else f"{title}.")
        return " ".join(parts)

    # ═══════════════════════════════════════════════════════════════════════
    # Create handlers
    # ═══════════════════════════════════════════════════════════════════════
    def _create_event(self, user_input: str) -> str:
        data_json = self._extract_event_json(user_input)
        if not data_json:
            return f"I didn't quite catch that, {self.name}. Could you say it again a bit more clearly?"

        # DO NOT trust the LLM's is_reminder flag — the regex router already
        # decided this is a calendar event (shift phrase + time, or a
        # calendar/event/meeting keyword). Letting the LLM flip it created
        # reminders for "I'm working tomorrow from 7 to 5".

        title = (data_json.get("title") or "New event").strip()
        date_field = data_json.get("date") or "today"
        start_time = data_json.get("start_time")
        end_time = data_json.get("end_time")
        location = self._clean_optional(data_json.get("location"))
        notes = self._clean_optional(data_json.get("notes"))

        # Title sanitation. Strip a leading "Nova" or day name; reject pure
        # day-name titles like "Saturday".
        _title_cleaned = re.sub(
            r"^\s*(?:nova|hey\s+nova|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s*[,:\-]?\s*",
            "", title, flags=re.IGNORECASE,
        ).strip()
        if _title_cleaned:
            title = _title_cleaned
        if title.lower() in (
            "monday", "tuesday", "wednesday", "thursday",
            "friday", "saturday", "sunday", "nova", "",
        ):
            low = user_input.lower()
            if re.search(r"\b(?:working|shift|work)\b", low):
                title = "Work"
            elif re.search(r"\b(?:meeting|1:1|stand\s*up|standup)\b", low):
                title = "Meeting"
            elif re.search(r"\b(?:dentist|doctor|appointment)\b", low):
                title = "Appointment"
            else:
                title = "Event"

        # Date correction. The user's explicit weekday ALWAYS wins over the
        # LLM's extraction — STT transcribes weekday names reliably and the
        # LLM sometimes hallucinates a different day.
        _weekday_match = re.search(
            r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
            user_input.lower(),
        )
        _has_today_phrase = bool(re.search(
            r"\b(?:today|right\s+now|this\s+morning|this\s+afternoon|"
            r"this\s+evening|tonight)\b",
            user_input.lower(),
        ))
        if _weekday_match and not _has_today_phrase:
            user_weekday = _weekday_match.group(1)
            llm_date_lower = str(date_field).lower().strip()
            llm_already_matches = user_weekday in llm_date_lower
            if not llm_already_matches:
                try:
                    _llm_dt = datetime.date.fromisoformat(llm_date_lower)
                    if _llm_dt.strftime("%A").lower() == user_weekday:
                        llm_already_matches = True
                except ValueError:
                    pass
            if not llm_already_matches:
                log.info(f"user said {user_weekday!r} but LLM extracted "
                         f"{date_field!r} — overriding to {user_weekday!r}")
                date_field = user_weekday

        resolved_date = self._resolve_relative_date(str(date_field))
        st = self._parse_time(start_time) or (9, 0)
        st = (self._default_bare_hour(st[0], user_input), st[1])
        start_dt = datetime.datetime.combine(resolved_date, datetime.time(st[0], st[1]))
        end_dt = None
        et = self._parse_time(end_time)
        if et is not None:
            end_dt = datetime.datetime.combine(resolved_date, datetime.time(et[0], et[1]))

        cal_name = cal.classify_calendar(user_input)

        assumed_end = end_dt is None
        if end_dt is None:
            end_dt = start_dt + datetime.timedelta(hours=1)

        try:
            cal.create_calendar_event(
                title=title, calendar_name=cal_name,
                start_datetime=start_dt, end_datetime=end_dt,
                location=location, notes=notes,
            )
        except Exception as e:
            log.error(f"create_calendar_event error: {e}")
            return (f"I wasn't able to create that event, {self.name}. "
                    "You may need to grant calendar permission in System Settings.")

        when = cal.format_datetime_for_speech(start_dt)
        end_str = cal.format_time_for_speech(end_dt)
        loc_phrase = f" at {location}" if location else ""

        if self._natural_confirmations:
            end_clause = (f" The end time was assumed as a one-hour default ({end_str})."
                          if assumed_end else f" It ends at {end_str}.")
            confirm_prompt = (
                "Confirm this calendar action naturally and conversationally, "
                "not scripted. If an end time was assumed as a one-hour default, "
                "mention it briefly and offer to change it. Action: Created an "
                f"event titled '{title}' on {when}{loc_phrase} in the {cal_name} "
                f"calendar.{end_clause} Keep it to two to three sentences max."
            )
            text = self._llm_silent(confirm_prompt, max_tokens=160)
            if text:
                return text

        if assumed_end:
            return (f"Done, {self.name}. I've added {title} on {when}{loc_phrase} "
                    f"to your {cal_name} calendar. I assumed a one-hour duration "
                    f"ending at {end_str}, let me know if you'd like to change it.")
        return (f"Done, {self.name}. Your {title} is scheduled for {when}"
                f"{loc_phrase}, ending at {end_str}, on your {cal_name} calendar.")

    def _create_reminder(self, user_input: str) -> str:
        # Deterministic first: strip the command prefix and pull the title +
        # due date/time with regex. The 3B, fed the event-extraction prompt,
        # hallucinated titles like "Work" from that prompt's own examples;
        # regex keeps the title = exactly what the user said to do.
        title, due_dt = self._parse_reminder(user_input)

        # Fallback to the LLM only if we couldn't isolate a title (unusual
        # phrasing). Reject the known hallucinated fillers.
        if not title:
            data_json = self._extract_event_json(user_input) or {}
            cand = self._clean_optional(data_json.get("title"))
            if cand and cand.lower() not in ("work", "meeting", "reminder", "event", "nova"):
                title = cand
            if due_dt is None:
                d = self._clean_optional(data_json.get("date"))
                st_raw = self._clean_optional(data_json.get("start_time"))
                if d or st_raw:
                    rd = self._resolve_relative_date(str(d or "today"))
                    st = self._parse_time(st_raw) or (9, 0)
                    st = (self._default_bare_hour(st[0], user_input), st[1])
                    due_dt = datetime.datetime.combine(rd, datetime.time(st[0], st[1]))
        if not title:
            return f"What should the reminder say, {self.name}?"
        title = self._titlecase(title)  # 'call mom' -> 'Call Mom'

        try:
            cal.create_reminder(title=title, due_datetime=due_dt, notes=None)
        except Exception as e:
            log.error(f"create_reminder error: {e}")
            return (f"I wasn't able to create that reminder, {self.name}. "
                    "You may need to grant reminders permission in System Settings.")

        if self._natural_confirmations:
            if due_dt:
                when = cal.format_datetime_for_speech(due_dt)
                action_desc = f"Created a reminder '{title}' due {when}."
            else:
                action_desc = f"Created a reminder '{title}' with no specific due date."
            confirm_prompt = (
                "Confirm this reminder action naturally and conversationally, "
                f"not scripted. Action: {action_desc} Keep it to one to two "
                "sentences max."
            )
            text = self._llm_silent(confirm_prompt, max_tokens=140)
            if text:
                return text

        if due_dt:
            when = cal.format_datetime_for_speech(due_dt)
            return f"Done, {self.name}. Reminder set: {title}, due {when}."
        return f"Done, {self.name}. I've added a reminder to {title}."

    # ═══════════════════════════════════════════════════════════════════════
    # Edit / delete / complete handlers
    # ═══════════════════════════════════════════════════════════════════════
    _TITLE_STRIP_VERBS = (
        # STT mishears the wake word inside utterances; strip those variants.
        "nova", "novah", "nofa", "no va",
        "can you", "could you", "please", "would you",
        "complete", "completed", "finish", "finished", "mark",
        "check off", "check",
        "delete", "deleting", "remove", "removing", "cancel", "cancelling",
        "canceled", "cancelled", "drop", "get rid of", "throw out", "dismiss",
        "reschedule", "move", "change", "update", "edit", "rename",
        "the", "my", "a", "an", "that", "this",
        "for me", "as done", "as complete", "as completed", "as finished",
        "reminder", "event", "meeting", "appointment", "shift",
        "from my calendar", "off my calendar", "on my calendar",
    )

    def _extract_target_hint(self, user_input: str) -> str:
        """Return the 'payload' of an edit/delete/complete command with the
        verb words and item-type words stripped away.
        "complete the Amazon smartwatch reminder for today at 10 a.m." ->
        "Amazon smartwatch"."""
        t = " " + user_input.lower().strip() + " "
        for phrase in self._TITLE_STRIP_VERBS:
            t = re.sub(r"\s" + re.escape(phrase) + r"\s", " ", t)
        # Strip trailing time-phrase qualifiers AND everything after them.
        t = re.sub(
            r"\s+(?:for\s+)?(?:today|tomorrow|tonight|"
            r"this\s+(?:morning|afternoon|evening|week)|next\s+\w+)\b.*",
            " ", t,
        )
        # Strip bare weekday names without eating surrounding text (delete
        # handlers extract the weekday separately as a date_hint).
        t = re.sub(
            r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
            " ", t,
        )
        t = re.sub(r"\s+at\s+\d[\w:.\s]*", " ", t)
        t = re.sub(r"\s+from\s+\d[\w:.\s]*", " ", t)
        t = re.sub(r"[^\w\s'-]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def _complete_reminder(self, user_input: str) -> str:
        hint = self._extract_target_hint(user_input)
        if not hint:
            return f"Which reminder would you like me to complete, {self.name}?"
        success, msg = cal.complete_reminder(hint)
        if success:
            return f"Done, {self.name}. I've completed the {msg} reminder."
        return f"I couldn't find a reminder matching {hint!r}, {self.name}."

    def _delete_reminder(self, user_input: str) -> str:
        hint = self._extract_target_hint(user_input)
        if not hint:
            return f"Which reminder would you like me to delete, {self.name}?"
        success, msg = cal.delete_reminder(hint)
        if success:
            return f"Deleted the {msg} reminder, {self.name}."
        return f"I couldn't find a reminder matching {hint!r}, {self.name}."

    def _delete_event(self, user_input: str) -> str:
        hint = self._extract_target_hint(user_input)
        if not hint:
            return f"Which event would you like me to delete, {self.name}?"

        # Narrow the search window if the user named a day (regex, not LLM —
        # fast, deterministic, always respects the explicit day).
        date_hint: Optional[datetime.date] = None
        _day_match = re.search(
            r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
            r"tomorrow|today|tonight)\b",
            user_input.lower(),
        )
        if _day_match:
            try:
                date_hint = self._resolve_relative_date(_day_match.group(1))
            except Exception:
                date_hint = None

        success, msg = cal.delete_calendar_event(hint, date_hint=date_hint)
        if success:
            return f"Deleted the {msg} event from your calendar, {self.name}."
        return f"I couldn't find an event matching {hint!r}, {self.name}."

    def _update_reminder(self, user_input: str) -> str:
        # ── Rename fast-path ──────────────────────────────────────────────
        # "rename X to Y" has an unambiguous structure. The 3B LLM packs both
        # halves into new_title when Y syntactically continues X; regex is
        # deterministic and never hallucinates.
        _rename = re.match(
            r"^\s*(?:please\s+|can\s+you\s+|could\s+you\s+)?"
            r"rename\s+(?:the\s+)?(?P<old>.+?)\s+to\s+(?P<new>.+?)\s*[.!?]?\s*$",
            user_input.strip(), re.IGNORECASE,
        )
        if _rename:
            old_hint = _rename.group("old").strip()
            new_title = _rename.group("new").strip().rstrip(".,!?")
            old_hint = re.sub(r"\s+reminder\s*$", "", old_hint, flags=re.IGNORECASE).strip()
            success, msg = cal.update_reminder(old_hint, new_title=new_title)
            if success:
                return f"Renamed the {msg} reminder to {new_title}, {self.name}."
            return f"I couldn't find a reminder matching {old_hint!r}, {self.name}."

        # ── General update path (reschedule, change notes) ────────────────
        hint = self._extract_target_hint(user_input)
        if not hint:
            return f"Which reminder would you like me to update, {self.name}?"

        data_json = self._extract_update_json(user_input) or {}

        def _pick(*names):
            for n in names:
                v = data_json.get(n)
                if v is not None:
                    cleaned = self._clean_optional(v)
                    if cleaned is not None:
                        return cleaned
            return None

        new_title = _pick("new_title", "title", "rename_to", "name")
        new_date_field = _pick("new_date", "date", "due_date")
        new_time_field = _pick("new_time", "time", "due_time", "start_time", "at_time")
        new_notes = _pick("new_notes", "notes", "note", "body")

        # Only accept new_title if the utterance has an EXPLICIT rename verb —
        # otherwise the LLM invents a title during a reschedule.
        _has_rename_verb = bool(re.search(
            r"\b(?:rename|re[-\s]?title|call\s+it|change\s+(?:the\s+)?(?:name|title))\b",
            user_input.lower(),
        ))
        if new_title and not _has_rename_verb:
            new_title = None
        elif new_title and new_title.lower().strip() == hint.lower().strip():
            new_title = None

        new_due: Optional[datetime.datetime] = None
        if new_date_field or new_time_field:
            if new_date_field:
                resolved_date = self._resolve_relative_date(new_date_field)
            else:
                resolved_date = datetime.date.today()
            parsed_time = self._parse_time(new_time_field) or (9, 0)
            new_due = datetime.datetime.combine(
                resolved_date, datetime.time(parsed_time[0], parsed_time[1]))

        if new_title is None and new_due is None and new_notes is None:
            return (f"I heard you wanted to update a reminder, {self.name}, but "
                    "I didn't catch what to change. Try, for example, "
                    "'reschedule the groceries reminder to tomorrow at 6 PM'.")

        success, msg = cal.update_reminder(
            hint, new_title=new_title, new_due=new_due, new_notes=new_notes)
        if not success:
            return f"I couldn't find a reminder matching {hint!r}, {self.name}."

        parts = [f"Updated the {msg} reminder"]
        if new_title:
            parts.append(f", renamed to {new_title}")
        if new_due:
            parts.append(f", due {cal.format_datetime_for_speech(new_due)}")
        if new_notes:
            parts.append(", notes updated")
        return "".join(parts) + f", {self.name}."
