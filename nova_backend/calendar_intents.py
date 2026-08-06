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
    # System prompt for all silent calendar LLM calls. Nova speaks its output
    # aloud, so: no markdown, no lists, no em dashes (CLAUDE.md invariant #10).
    _NOVA_CAL_SYSTEM = (
        "You are Nova, a sharp, composed AI assistant. Speak naturally and "
        "conversationally, the way a person would say it out loud. Never use "
        "markdown, bullet points, numbered lists, or em dashes. Keep replies "
        "brief. Address the user by name when it fits."
    )

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
        if re.search(r"\bwhat\s+are\s+my\s+reminders\b", t) or \
           re.search(r"\bwhat\s+do\s+i\s+need\s+to\s+do\s+(?:today|this\s+week)\b", t) or \
           re.search(r"\bshow\s+(?:me\s+)?my\s+reminders\b", t) or \
           re.search(r"\bread\s+(?:me\s+)?my\s+reminders\b", t) or \
           re.search(r"\blist\s+(?:all\s+)?my\s+reminders\b", t):
            return "read_reminders"

        # ── Read today's calendar ────────────────────────────────────────
        if re.search(r"\bwhat(?:'?s|\s+is)\s+on\s+my\s+calendar\s+today\b", t) or \
           re.search(r"\b(?:my\s+)?calendar\s+(?:for\s+)?today\b", t) or \
           re.search(r"\b(?:my\s+)?schedule\s+for\s+today\b", t) or \
           re.search(r"\bwhat(?:'?s|\s+is)\s+my\s+schedule\s+today\b", t) or \
           re.search(r"\banything\s+on\s+(?:my\s+)?calendar\s+today\b", t):
            return "read_today"

        # ── Read upcoming (rest of the week) ─────────────────────────────
        # Deliberately REQUIRE "this week"/"the week" (not bare "week") so
        # "calendar week starts on monday" doesn't false-fire. Tolerate an
        # optional "for" between calendar/schedule and the week phrase.
        if re.search(r"\bwhat(?:'?s|\s+is)\s+coming\s+up\s+on\s+(?:my\s+)?(?:calendar|schedule)\b", t) or \
           re.search(r"\b(?:my\s+)?calendar\s+(?:for\s+)?(?:this|the)\s+week\b", t) or \
           re.search(r"\b(?:my\s+)?schedule\s+(?:for\s+)?(?:this|the)\s+week\b", t) or \
           re.search(r"\bwhat\s+do\s+i\s+have\s+(?:for\s+)?(?:this|the)\s+week\b", t) or \
           re.search(r"\bwhat(?:'?s|\s+is)\s+on\s+(?:my\s+)?agenda\b", t):
            return "read_upcoming"

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
        try:
            if intent == "read_today":
                return self._read_today()
            if intent == "read_upcoming":
                return self._read_upcoming()
            if intent == "read_reminders":
                return self._read_reminders()
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
                system_prompt=self._NOVA_CAL_SYSTEM,
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
        events = cal.get_today_events()
        if not events:
            return f"Your calendar is clear for today, {self.name}."
        lines = self._format_event_lines(events)
        prompt = (
            "These are the events on my calendar for today. Summarize them "
            "naturally and conversationally, not a list, just how a sharp "
            "assistant would say it out loud. Keep it to two to four sentences.\n\n"
            + "\n".join(lines)
        )
        text = self._llm_silent(prompt, max_tokens=220)
        return text or self._template_events(events, "today")

    def _read_upcoming(self) -> str:
        events = cal.get_upcoming_events()
        if not events:
            return f"Nothing on the books for the rest of the week, {self.name}."
        lines = self._format_event_lines(events)
        prompt = (
            "These are the events on my calendar between now and the end of "
            "this coming Saturday. Summarize them naturally and "
            "conversationally, not a list, just how a sharp assistant would "
            "say it out loud.\n\n"
            "CRITICAL RULES:\n"
            "Read the day of week for each event EXACTLY as given in the input. "
            "Read each time EXACTLY as written; do not change AM to PM. Do not "
            "invent events that aren't in the input. Keep it to three to five "
            "sentences.\n\n"
            + "\n".join(lines)
        )
        text = self._llm_silent(prompt, max_tokens=260)
        return text or self._template_events(events, "this week")

    def _read_reminders(self) -> str:
        reminders = cal.get_all_reminders()
        if not reminders:
            return f"You have no open reminders, {self.name}."
        lines = []
        for idx, r in enumerate(reminders, start=1):
            parts = [f"{idx}. {r.get('title', '')}"]
            if r.get("due"):
                parts.append(f"(due {r['due']})")
            lines.append(" ".join(parts))
        prompt = (
            "These are the user's open reminders, listed in chronological "
            "order, earliest due date first. Summarize them naturally and "
            "conversationally, not a list, just how a sharp assistant would "
            "say them out loud.\n\n"
            "CRITICAL RULES:\n"
            "Mention the reminders in the exact order given below. Read each "
            "due date and time EXACTLY as written; do not change AM to PM, do "
            "not change the hour, do not round. Keep it brief, two to four "
            "sentences.\n\n"
            + "\n".join(lines)
        )
        text = self._llm_silent(prompt, max_tokens=220)
        return text or self._template_reminders(reminders)

    def _format_event_lines(self, events: list) -> list:
        lines = []
        for e in events:
            parts = [f"- {e.get('title', '')}"]
            if e.get("start"):
                parts.append(f"starts {e['start']}")
            if e.get("end"):
                parts.append(f"ends {e['end']}")
            if e.get("location"):
                parts.append(f"at {e['location']}")
            if e.get("calendar"):
                parts.append(f"[{e['calendar']}]")
            lines.append(" ".join(parts))
        return lines

    def _template_events(self, events: list, span: str) -> str:
        """Deterministic fallback — never hallucinates days/times."""
        n = len(events)
        parts = [f"You have {n} event{'s' if n != 1 else ''} {span}, {self.name}."]
        for e in events:
            segment = e.get("title", "Untitled")
            if e.get("start"):
                segment += f" on {e['start']}"
            if e.get("location"):
                segment += f" at {e['location']}"
            parts.append(segment + ".")
        return " ".join(parts)

    def _template_reminders(self, reminders: list) -> str:
        if len(reminders) == 1:
            r = reminders[0]
            if r.get("due"):
                return f"You have one open reminder, {self.name}: {r['title']}, due {r['due']}."
            return f"You have one open reminder, {self.name}: {r['title']}."
        parts = [f"You have {len(reminders)} open reminders, {self.name}."]
        for r in reminders:
            if r.get("due"):
                parts.append(f"{r['title']} is due {r['due']}.")
            else:
                parts.append(f"{r['title']}.")
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
        data_json = self._extract_event_json(user_input)
        if not data_json:
            return f"I didn't quite catch that, {self.name}. Could you say it again?"

        title = (data_json.get("title") or "Reminder").strip()
        notes = self._clean_optional(data_json.get("notes"))
        date_field = self._clean_optional(data_json.get("date"))
        start_time = self._clean_optional(data_json.get("start_time"))

        due_dt: Optional[datetime.datetime] = None
        if date_field or start_time:
            resolved_date = self._resolve_relative_date(str(date_field or "today"))
            st = self._parse_time(start_time) or (9, 0)
            st = (self._default_bare_hour(st[0], user_input), st[1])
            due_dt = datetime.datetime.combine(resolved_date, datetime.time(st[0], st[1]))

        try:
            cal.create_reminder(title=title, due_datetime=due_dt, notes=notes)
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
