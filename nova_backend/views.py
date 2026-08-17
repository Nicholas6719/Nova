"""
View navigation — "go home", "show me the menu", "go to finance".

Nova's UI has no sidebar and nothing to click: destinations are reached BY
VOICE. This module is the router for that, and it follows the same shape the
calendar, weather and file intents settled on:

  - detection is STRICT REGEX, anchored at the START of the utterance and
    ending there. "go home" navigates; "I go home every friday" is
    conversation. A loose matcher here would be the memory-regex mistake all
    over again, except worse, because navigation words are ordinary English.
  - the destination must be a name in VIEWS. "go to finance" navigates,
    "I want to go to italy someday" falls through to the LLM. An unknown
    destination is never guessed at.
  - a view that is not built yet SAYS SO. It never shows an empty panel and
    pretends.

The spoken reply is deliberately tiny. The panel is what answers; the voice
just acknowledges. Nothing here touches the LLM.
"""

from __future__ import annotations

import datetime
import logging
import re
from typing import Optional

log = logging.getLogger("nova.views")


def _greeting() -> str:
    hour = datetime.datetime.now().hour
    if hour < 12:
        return "Good morning"
    return "Good afternoon" if hour < 18 else "Good evening"


def _today_line() -> str:
    return datetime.datetime.now().strftime("%A, %B %-d")


class View:
    """One navigable destination.

    A view with a non-empty `planned` has no panel yet: Nova says so out loud
    rather than showing an empty screen. Payloads are built by
    `NovaViews._payload`, not here.
    """

    def __init__(self, name: str, spoken: str, aliases: tuple[str, ...] = (),
                 planned: str = "") -> None:
        self.name = name
        self.spoken = spoken        # what Nova says on arriving
        self.aliases = aliases
        self.planned = planned      # non-empty => not built yet; this is the excuse

    @property
    def is_live(self) -> bool:
        return not self.planned


# ── The registry ──────────────────────────────────────────────────────────────
# Order matters only for the alias regex; longer aliases must be tried first so
# "finance screen" doesn't match the bare "finance" alias and leave "screen".
_VIEW_DEFS: tuple[tuple, ...] = (
    ("home",          "Home.",                    ("home",)),
    ("menu",          "Here's everything I can do.",
                                                  ("menu", "capabilities")),
    ("weather",       "Weather.",                 ("weather", "forecast")),
    ("calendar",      "Your calendar.",           ("calendar", "schedule", "agenda")),
    ("files",         "Files.",                   ("files", "file")),
    ("memory",        "Here's what I remember.",  ("memory", "memories",
                                                   "what you remember")),
    ("conversations", "Your past conversations.", ("conversations",
                                                   "past conversations",
                                                   "conversation history",
                                                   "history")),
    ("finance",       "Markets.",                 ("finance", "markets", "market",
                                                   "stocks", "portfolio")),
    ("health",        "Health.",                  ("health", "fitness")),
)

# Views whose panel does not exist yet, and what Nova says instead. Kept here so
# adding the panel is a one-line deletion.
_PLANNED = {
    "finance": "I don't have markets wired up yet.",
    "health":  "I can't see your health data yet. That needs the phone app.",
}


def _build_registry() -> dict[str, View]:
    out: dict[str, View] = {}
    for name, spoken, aliases in _VIEW_DEFS:
        out[name] = View(name, spoken, aliases, planned=_PLANNED.get(name, ""))
    return out


VIEWS: dict[str, View] = _build_registry()

# Every alias, longest first, as a regex alternation.
_ALIASES: list[tuple[str, str]] = sorted(
    ((alias, v.name) for v in VIEWS.values() for alias in v.aliases),
    key=lambda pair: -len(pair[0]),
)
_ALIAS_ALT = "|".join(re.escape(a) for a, _ in _ALIASES)
_ALIAS_TO_VIEW = dict(_ALIASES)

# Optional "nova," lead-in on any command.
_LEAD = r"^(?:hey\s+)?(?:nova[,\s]+)?"
# Optional trailing "screen"/"panel"/"page"/"view", and trailing punctuation.
_TAIL = r"(?:\s+(?:screen|panel|page|view|tab))?\s*[.?!]?$"

# 1. Home, as a command. Anchored both ends so a sentence ABOUT going home
#    ("I go home every friday", "remind me to call mom when I get home") can
#    never match — the utterance has to be the command and nothing else.
_HOME_RE = re.compile(
    _LEAD + r"(?:go|take me|bring me|head|get me)\s+(?:back\s+)?home" + _TAIL
    + r"|" + _LEAD + r"(?:return|back)\s+(?:to\s+)?home" + _TAIL
    + r"|" + _LEAD + r"(?:the\s+)?home\s+(?:screen|page|view)\s*[.?!]?$",
    re.IGNORECASE,
)

# 2. The menu, and the two ways he asks what Nova can do.
_MENU_RE = re.compile(
    _LEAD + r"(?:show|open|bring up|pull up|give me)\s+(?:me\s+)?(?:the\s+)?menu"
    + _TAIL
    + r"|" + _LEAD + r"what can you (?:do|show me)\s*[.?!]?$"
    + r"|" + _LEAD + r"(?:show me\s+)?what you can do\s*[.?!]?$",
    re.IGNORECASE,
)

# 3. A named destination. The name must be a known alias, and the phrasing must
#    be navigational — note there is no "my" here, which is what keeps
#    "show me my calendar" pointed at the calendar handler that actually reads
#    his events, rather than at a panel.
_GOTO_RE = re.compile(
    _LEAD + r"(?:go to|take me to|bring me to|switch to|open|show me|pull up|"
    r"jump to)\s+(?:the\s+)?(" + _ALIAS_ALT + r")" + _TAIL,
    re.IGNORECASE,
)

# 3b. Work mode: park in the corner and work alongside him. Anchored like the
#     rest — "let's work on this together" is conversation, not a command.
_WORK_RE = re.compile(
    _LEAD + r"(?:let's\s+)?work\s+(?:with\s+me|on\s+this|together)\s*[.?!]?$"
    + r"|" + _LEAD + r"(?:take\s+over|take\s+the\s+wheel)"
    r"(?:\s+(?:my\s+)?(?:computer|mac|screen))?\s*[.?!]?$"
    + r"|" + _LEAD + r"(?:go\s+to\s+)?(?:work\s+mode|puck\s+mode)\s*[.?!]?$"
    + r"|" + _LEAD + r"minimi[sz]e\s*[.?!]?$",
    re.IGNORECASE,
)

# 4. Bare "<name> screen" — "finance screen", "the memory panel".
_BARE_RE = re.compile(
    _LEAD + r"(?:the\s+)?(" + _ALIAS_ALT + r")\s+(?:screen|panel|page|view|tab)"
    r"\s*[.?!]?$",
    re.IGNORECASE,
)


class NovaViews:
    """Voice navigation between UI destinations.

    `detect_intent` returns a view name or None. It is deliberately the
    stingiest matcher in the pipeline: navigation words are common English, so
    anything that isn't unmistakably a command is somebody else's problem.
    """

    def __init__(self, config: dict, ws=None, assistant=None) -> None:
        self.config = config
        self.ws = ws
        # The running VoiceAssistant, so home can reach the same engines that
        # answer these questions out loud. Optional: the routing harness builds
        # views without one, and home degrades to what it can reach.
        self.assistant = assistant
        self.current: str = "home"

    # ── Detection ─────────────────────────────────────────────────────────────
    def detect_intent(self, text: str) -> Optional[str]:
        if not text or not text.strip():
            return None
        t = text.strip()

        if _HOME_RE.match(t):
            return "home"
        if _WORK_RE.match(t):
            return "work"
        if _MENU_RE.match(t):
            return "menu"
        for pattern in (_GOTO_RE, _BARE_RE):
            m = pattern.match(t)
            if m:
                return _ALIAS_TO_VIEW.get(m.group(1).lower())
        return None

    # ── Handling ──────────────────────────────────────────────────────────────
    def handle(self, view_name: str, text: str = "") -> str:
        # Work mode is a MODE, not a screen: Nova parks in the corner and the
        # conversation timeout lengthens. Going home is what ends it, which is
        # why "go home" is the one phrase that always brings her back.
        if view_name == "work":
            if self.assistant is not None:
                self.assistant.set_work_mode(True, reason="asked")
            return "Alright, I'm right here."

        view = VIEWS.get(view_name)
        if view is None:                      # unreachable via detect_intent
            return "I don't have a screen for that."

        if view.name == "home" and self.assistant is not None:
            self.assistant.set_work_mode(False, reason="went home")

        if not view.is_live:
            # Honest degradation: never show an empty panel and imply it works.
            return view.planned

        payload = self._payload(view)
        self.current = view.name
        if self.ws is not None:
            try:
                self.ws.send_view(view.name, payload)
            except Exception as exc:
                # The panel failing must never cost him the spoken answer.
                log.warning(f"could not send view {view.name}: {exc}")
        return view.spoken

    # ── Payloads ──────────────────────────────────────────────────────────────
    def _payload(self, view: View) -> dict:
        if view.name == "menu":
            return {"title": "Menu",
                    "subtitle": "Everything I can do",
                    "sections": self.menu_sections()}
        if view.name == "home":
            return self._home_payload()
        if view.name == "memory":
            return self._memory_payload()
        return {}

    # ── Memory ────────────────────────────────────────────────────────────────
    def _memory_payload(self) -> dict:
        """Everything Nova knows about him, grouped by category.

        This is the screen that makes passive learning honest: facts get
        written from ordinary conversation, and he should be able to see the
        whole set rather than discover them one surprise at a time. Nothing is
        summarised — these are the stored rows.
        """
        import panels as P

        memory = getattr(self.assistant, "memory", None)
        if memory is None:
            return P.panel(title="Memory",
                           blocks=[P.note("I can't reach my memory right now.")])
        try:
            facts = memory.all_facts()
        except Exception as exc:
            log.warning(f"memory panel unavailable ({exc})")
            return P.panel(title="Memory",
                           blocks=[P.note("I couldn't read my memory just now.")])

        if not facts:
            return P.panel(
                title="Memory",
                subtitle="Nothing stored yet",
                blocks=[P.note("I haven't learned anything about you yet. "
                               "Tell me something and I'll keep it.")])

        grouped: dict[str, list[dict]] = {}
        for f in facts:
            grouped.setdefault(f.get("category") or "other", []).append(f)

        blocks = []
        for category in sorted(grouped):
            blocks.append(P.items(
                [{"title": (f.get("key") or "").replace("_", " "),
                  "detail": f.get("value") or "",
                  # Where it came from matters here more than anywhere else:
                  # a fact he stated and a fact Nova inferred are different
                  # things, and he should be able to tell them apart.
                  "meta": (f.get("source") or "")}
                 for f in grouped[category]],
                title=category.replace("_", " ")))

        return P.panel(title="Memory",
                       subtitle=f"{len(facts)} thing{'s' if len(facts) != 1 else ''} I know",
                       blocks=blocks)

    # ── Home ──────────────────────────────────────────────────────────────────
    def _home_payload(self) -> dict:
        """Everything he wants at a glance, and nothing he doesn't.

        Deliberately NOT the mockup's dashboard: no quick actions, no recent
        activity, no CPU gauges. Now Playing appears only when something is
        actually playing, which is the rule he set.

        Every block is defensive. Home must render if the calendar is
        permission-blocked, the weather service is down and no music is open —
        a broken tile is not a reason to show him a broken screen.
        """
        import panels as P

        return P.panel(
            title=_greeting(),
            subtitle=_today_line(),
            blocks=[
                self._home_events(),
                self._home_weather(),
                self._home_music(),
            ],
        )

    def _home_events(self) -> Optional[dict]:
        import panels as P
        try:
            import calendar_reminders as cal
            from calendar_intents import build_events_panel
            events = cal.get_today_events()
        except Exception as exc:
            log.warning(f"home: calendar unavailable ({exc})")
            return None
        if not events:
            return P.note("Nothing on your calendar today.")
        inner = build_events_panel(events, "Today")
        block = inner["blocks"][0]
        block["title"] = "Today"
        return block

    def _home_weather(self) -> Optional[dict]:
        import panels as P
        try:
            import maps_engine
            import weather_engine as we
            coord = maps_engine.current_location()
            if coord is None:
                return None
            data = we.fetch(coord[0], coord[1], days=1)
            if not data.get("ok"):
                return None
            cur = data.get("current") or {}
            temp = cur.get("temp")
            if not isinstance(temp, (int, float)):
                return None
            return P.stat(f"{round(temp)}°", label="Weather",
                          detail=we.describe_code(cur.get("code")).capitalize())
        except Exception as exc:
            log.warning(f"home: weather unavailable ({exc})")
            return None

    def _home_music(self) -> Optional[dict]:
        """Only when something is actually playing — his rule."""
        import panels as P
        tools = getattr(self.assistant, "tools", None)
        if tools is None:
            return None
        try:
            playing = tools.current_track_for_panel()
        except Exception as exc:
            log.warning(f"home: music unavailable ({exc})")
            return None
        if not playing:
            return None
        title, artist = playing
        return P.items([{"title": title, "detail": artist}], title="Now playing")

    def menu_sections(self) -> list[dict]:
        """What Nova can do and where you can go. Generated from the registry so
        it can never drift from what actually exists."""
        destinations = [
            {"name": v.name, "say": f"go to {v.aliases[0]}",
             "available": v.is_live,
             "note": v.planned or ""}
            # Neither home nor menu belongs in its own list: he is already
            # looking at the menu, and home is the one destination he never
            # needs told about.
            for v in VIEWS.values() if v.name not in ("home", "menu")
        ]
        return [
            {
                "title": "Where you can go",
                "items": destinations,
            },
            {
                "title": "What you can ask me",
                "items": [
                    {"name": "Weather",  "say": "what's the weather tomorrow"},
                    {"name": "Calendar", "say": "what's on my calendar today"},
                    {"name": "Reminders", "say": "remind me to call mom at six"},
                    {"name": "Files",    "say": "find my resume"},
                    {"name": "Screen",   "say": "what's on my screen"},
                    {"name": "Music",    "say": "play something by Nirvana"},
                    {"name": "Memory",   "say": "remember that I like tea"},
                    {"name": "Apps",     "say": "open Xcode"},
                ],
            },
        ]
