"""
Natural-language dispatch over weather_engine.

Follows the pattern the calendar and file intents settled on:
  - detection is STRICT REGEX. No weather word, no weather answer. A loose
    matcher here would steal ordinary conversation, the way the memory intent
    once stole "what's my calendar".
  - the answer is a DETERMINISTIC template. The 3B is never asked to phrase a
    temperature: it has invented figures before, and a wrong number sounds
    exactly like a right one to someone listening.
  - a failure says so plainly instead of guessing.

Runs on the nova-llm worker thread like the other intent handlers, but touches
no model at all.
"""

from __future__ import annotations

import datetime
import logging
import re
from typing import Optional

log = logging.getLogger("nova.weather")

# A weather word must appear. "How is it out there" is not weather.
_WEATHER_RE = re.compile(
    r"\b(weather|forecast|temperature|how (?:hot|cold|warm)|"
    r"rain(?:ing|y)?|snow(?:ing|y)?|sunny|humid|humidity|windy?|"
    r"umbrella|jacket|coat)\b",
    re.IGNORECASE,
)
# ...but not when he is asking Nova to *do* something with a file or app.
_NOT_WEATHER_RE = re.compile(
    r"\b(open|play|search for|google|remind|schedule|delete|move|rename)\b",
    re.IGNORECASE,
)

_TOMORROW_RE = re.compile(r"\btomorrow\b", re.IGNORECASE)
_TODAY_RE    = re.compile(r"\b(today|this (?:morning|afternoon|evening)|tonight)\b",
                          re.IGNORECASE)
_WEEK_RE     = re.compile(r"\b(this week|next few days|rest of the week|"
                          r"coming days|next couple of days)\b", re.IGNORECASE)
# "weather in Boston", "forecast for New York"
_PLACE_RE = re.compile(
    r"\b(?:in|for|at)\s+([A-Z][A-Za-z.\-']+(?:\s+[A-Z][A-Za-z.\-']+){0,2})\s*\??$"
)
# Words that follow "in/for" but are a time, not a place.
_NOT_A_PLACE = {"today", "tomorrow", "tonight", "the", "a", "an", "this",
                "morning", "afternoon", "evening", "week", "celsius",
                "fahrenheit", "here", "now"}


def _weekday_in(days: int) -> str:
    """"Thursday" for a day this many days out. Panel-only: the spoken path
    says "tomorrow" and "the day after" and never counts past that."""
    return (datetime.date.today() + datetime.timedelta(days=days)).strftime("%A")


class NovaWeather:
    """Weather questions. `detect_intent` returns None for anything else."""

    def __init__(self, config: dict) -> None:
        self.config = config
        # (view_name, payload) for the panel, picked up by nova.py after the
        # spoken answer. Same one-shot pattern as calendar.pending_intent.
        # The panel shows the SAME templated numbers the voice says — nothing
        # here is phrased by a model, on either channel.
        self.last_panel: Optional[tuple] = None

    # ── Detection ─────────────────────────────────────────────────────────────
    def detect_intent(self, text: str) -> Optional[str]:
        if not text or not text.strip():
            return None
        if not self.config.get("weather", {}).get("enabled", True):
            return None
        if _NOT_WEATHER_RE.search(text):
            return None
        if not _WEATHER_RE.search(text):
            return None
        if _WEEK_RE.search(text):
            return "week"
        if _TOMORROW_RE.search(text):
            return "tomorrow"
        return "now"

    # ── Handling ──────────────────────────────────────────────────────────────
    def handle(self, text: str, intent: str) -> str:
        import weather_engine as we

        place_name = self._named_place(text)
        if place_name:
            geo = we.geocode(place_name)
            if not geo["ok"]:
                return f"I couldn't find {place_name}."
            lat, lon, label = geo["lat"], geo["lon"], geo["name"]
        else:
            coord = self._here()
            if coord is None:
                return ("I can't get your location right now, so I can't check "
                        "the weather here. You can ask me for a specific place, "
                        "like the weather in Boston.")
            lat, lon = coord
            label = None

        data = we.fetch(lat, lon, days=7 if intent == "week" else 3)
        if not data.get("ok"):
            return "I couldn't reach the weather service just now."

        self.last_panel = ("weather", self._panel(we, data, label))

        if intent == "tomorrow":
            return we.say_day(data, 1, "tomorrow", label)
        if intent == "week":
            return self._say_week(we, data, label)
        return we.say_current(data, label)

    # ── Panel ─────────────────────────────────────────────────────────────────
    def _panel(self, we, data: dict, label: Optional[str]) -> dict:
        """The screen gets the whole week; the voice gets three days.

        This is the point of having both channels. `_say_week` deliberately
        stops at three days because a seven-day rundown spoken aloud is a
        monologue — but seven days on a panel is just a row he can scan.
        """
        import panels as P

        cur = data.get("current") or {}
        days = data.get("days") or []
        temp = cur.get("temp")

        def deg(v) -> str:
            return f"{round(v)}°" if isinstance(v, (int, float)) else "—"

        detail_rows = [
            ("Feels like", deg(cur.get("feels_like"))),
            ("Humidity", f"{round(cur['humidity'])}%"
                if isinstance(cur.get("humidity"), (int, float)) else "—"),
            ("Wind", f"{round(cur['wind'])} mph"
                if isinstance(cur.get("wind"), (int, float)) else "—"),
        ]

        week = []
        for i, d in enumerate(days[:7]):
            when = "Today" if i == 0 else ("Tomorrow" if i == 1
                                           else _weekday_in(i))
            rain = d.get("rain_chance")
            week.append({
                "title": when,
                "detail": we.describe_code(d.get("code")).capitalize(),
                "meta": f"{deg(d.get('high'))} / {deg(d.get('low'))}"
                        + (f"   {round(rain)}% rain"
                           if isinstance(rain, (int, float)) and rain else ""),
            })

        return P.panel(
            title="Weather",
            subtitle=label or "Here",
            blocks=[
                P.stat(deg(temp), label="Now",
                       detail=we.describe_code(cur.get("code")).capitalize()),
                P.rows(detail_rows),
                P.items(week, title="Next few days") if week else None,
            ],
        )

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _here(self):
        """Current coordinate, asking the app for a fresh fix if needed."""
        try:
            import maps_engine
            return maps_engine.current_location()
        except Exception as exc:
            log.warning(f"location lookup failed: {exc}")
            return None

    def _named_place(self, text: str) -> Optional[str]:
        m = _PLACE_RE.search(text.strip())
        if not m:
            return None
        place = m.group(1).strip()
        if place.lower() in _NOT_A_PLACE:
            return None
        if any(w.lower() in _NOT_A_PLACE for w in place.split()):
            return None
        return place

    def _say_week(self, we, data: dict, label: Optional[str]) -> str:
        """Three days is enough to say out loud. A seven-day rundown spoken
        aloud is a monologue, and Nova's whole length budget exists because of
        that."""
        days = data.get("days") or []
        if not days:
            return "I couldn't get the forecast for the next few days."
        parts = [we.say_current(data, label)]
        for i, name in ((1, "tomorrow"), (2, "the day after")):
            if i < len(days):
                parts.append(we.say_day(data, i, name))
        return " ".join(parts)
