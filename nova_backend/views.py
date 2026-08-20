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

import atexit
import datetime
import json
import logging
import os
import re
import threading
import time
from pathlib import Path
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
    # Weather and calendar are deliberately NOT navigable. They have real
    # handlers that fetch real data and build their own panels, and navigation
    # runs at stage 2c — three stages ahead of both. "Show me the weather" was
    # being claimed here and answered with the word "Weather." over an empty
    # panel, because _payload has no case for it and never did. A destination
    # whose screen is produced by a handler should be reached THROUGH that
    # handler, not around it.
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


# ── The home grid ─────────────────────────────────────────────────────────────
# Six named slots, three down each side of the orb. They exist so he can move a
# card by voice and have it STAY there — "move now playing to the bottom right"
# is only meaningful if there is a bottom right to move it to.
SLOTS = ("L1", "L2", "L3", "R1", "R2", "R3")

# Which card sits where, until he says otherwise. Markets top-left because that
# is where he said he reads them; the day on the right.
DEFAULT_SLOTS: dict[str, str] = {
    "market":   "L1",
    "playing":  "L2",
    "weather":  "R1",
    "upcoming": "R2",
}

_SLOT_ALIASES: dict[str, str] = {
    "top left": "L1", "upper left": "L1", "top left corner": "L1",
    "left top": "L1", "top of the left": "L1",
    "middle left": "L2", "center left": "L2", "centre left": "L2",
    "left middle": "L2", "middle of the left": "L2",
    "bottom left": "L3", "lower left": "L3", "bottom left corner": "L3",
    "left bottom": "L3", "bottom of the left": "L3",
    "top right": "R1", "upper right": "R1", "top right corner": "R1",
    "right top": "R1", "top of the right": "R1",
    "middle right": "R2", "center right": "R2", "centre right": "R2",
    "right middle": "R2", "middle of the right": "R2",
    "bottom right": "R3", "lower right": "R3", "bottom right corner": "R3",
    "right bottom": "R3", "bottom of the right": "R3",
    # Bare sides land at the top of that side, which is what "put it on the
    # left" means to a person.
    "left": "L1", "left side": "L1", "right": "R1", "right side": "R1",
    "left hand side": "L1", "right hand side": "R1",
    "lefthand side": "L1", "righthand side": "R1",
}

_CARD_ALIASES: dict[str, str] = {
    "markets": "market", "market": "market", "stocks": "market",
    "stock": "market", "finance": "market", "ticker": "market",
    "tickers": "market", "watchlist": "market",
    "now playing": "playing", "music": "playing", "spotify": "playing",
    "the music": "playing", "track": "playing", "song": "playing",
    "player": "playing",
    "weather": "weather", "forecast": "weather", "temperature": "weather",
    "upcoming": "upcoming", "calendar": "upcoming", "schedule": "upcoming",
    "agenda": "upcoming", "events": "upcoming", "reminders": "upcoming",
    "my day": "upcoming", "today": "upcoming",
    # Not movable — it is a row, not a card — but recognised so Nova can say
    # so instead of dropping the sentence on the model.
    "system": "system", "system info": "system", "system status": "system",
    "status": "system", "stats": "system", "status bar": "system",
}

_CARD_SPOKEN: dict[str, str] = {
    "market": "markets", "playing": "now playing",
    "weather": "weather", "upcoming": "upcoming",
}
_SLOT_SPOKEN: dict[str, str] = {
    "L1": "top left", "L2": "middle left", "L3": "bottom left",
    "R1": "top right", "R2": "middle right", "R3": "bottom right",
}

_CARD_ALT = "|".join(re.escape(a) for a in
                     sorted(_CARD_ALIASES, key=len, reverse=True))
_SLOT_ALT = "|".join(re.escape(a) for a in
                     sorted(_SLOT_ALIASES, key=len, reverse=True))

# 5. Moving a card. Strict on BOTH ends: the thing moved must be a known card
#    and the destination must be a known slot. That is what keeps "move the
#    file to Documents" out of here — file intents run at stage 5, well after
#    this, and would never see the utterance if this matched loosely.
_MOVE_RE = re.compile(
    _LEAD + r"(?:move|put|send|shift|drag|stick)\s+(?:the\s+|my\s+)?"
    r"(" + _CARD_ALT + r")\s*"
    r"(?:tab|card|panel|window|box|widget|thing)?\s+"
    # The preposition is optional: "put markets top left" is how he actually
    # says it. Safe to relax because BOTH ends still have to be known names.
    r"(?:(?:to|into|over\s+to|down\s+to|up\s+to|on|in)\s+)?(?:the\s+)?"
    r"(" + _SLOT_ALT + r")"
    r"(?:\s+(?:corner|side|slot|spot|position))?"
    # "...of the screen" is how he actually said it, and the end-anchor threw
    # the whole command away: "move the music widget to the right side of the
    # screen" was answered with "I can't do that one yet."
    r"(?:\s+(?:of|on)\s+(?:the|my)\s+(?:screen|window|display|home|panel))?"
    r"\s*[.?!]?$",
    re.IGNORECASE,
)


def _move_match(text: str):
    """Match a card move against the utterance, or any sentence in it.

    Whisper repeats him when he says something twice while waiting — a real
    transcript from his desk was "Move now playing to bottom right. Move now
    playing to the bottom." Anchoring against the whole string threw both
    halves away. Each sentence gets its own chance, latest first, so the last
    thing he said wins.
    """
    t = (text or "").strip()
    m = _MOVE_RE.match(t)
    if m:
        return m
    parts = [p.strip() for p in re.split(r"[.?!]+", t) if p.strip()]
    for part in reversed(parts):
        m = _MOVE_RE.match(part)
        if m:
            return m
    return None

# 6. Clearing home down to the orb, and putting it back. Anchored like
#    everything else: "clear my calendar" is not this, and never reaches it.
_CLEAR_RE = re.compile(
    _LEAD + r"(?:clear|hide|empty)\s+(?:the\s+|my\s+)?"
    r"(?:home|screen|home\s*screen|everything|cards|the\s+cards|panels)"
    r"\s*[.?!]?$",
    re.IGNORECASE,
)
_RESTORE_RE = re.compile(
    _LEAD + r"(?:restore|bring\s+back|put\s+back|unhide|give\s+me\s+back)\s+"
    r"(?:the\s+|my\s+)?"
    r"(?:home|screen|home\s*screen|everything|cards|the\s+cards|panels)"
    r"\s*[.?!]?$",
    re.IGNORECASE,
)


def _data_dir() -> Path:
    """Where the saved layout lives. Same directory as memory and credentials,
    set by the Swift BackendManager so it survives an app update."""
    env = os.environ.get("NOVA_DATA_DIR", "").strip()
    return Path(env) if env else (Path.home() / "Library" /
                                  "Application Support" / "Nova")


# ── Shutdown ──────────────────────────────────────────────────────────────────
# Tile refreshes run on short-lived background threads, and one of them —
# reminders — goes through EventKit, whose fetch delivers its completion block
# on ITS OWN dispatch queue. If the interpreter finalises while that block is
# still in flight, the block tries to take a GIL that is being torn down and
# Foundation kills the process outright: EXC_BREAKPOINT, SIGKILL, no traceback.
# Observed exactly that in the routing corpus, which exits the moment it
# finishes while a primed tile is still fetching.
#
# So exit waits for outstanding refreshes. Bounded, because a hung EventKit
# call must delay a quit, never prevent one.
_LIVE: set = set()
_LIVE_LOCK = threading.Lock()
_STOPPING = threading.Event()


def _track(thread: threading.Thread) -> None:
    with _LIVE_LOCK:
        _LIVE.add(thread)


def _untrack(thread: threading.Thread) -> None:
    with _LIVE_LOCK:
        _LIVE.discard(thread)


# Set once a tile has gone through EventKit, so the settle below is only paid
# by processes that actually took the risk.
_TOUCHED_EVENTKIT = threading.Event()


def _drain(budget: float = 6.0) -> None:
    """Let in-flight tile refreshes finish before the interpreter goes away."""
    _STOPPING.set()
    deadline = time.time() + budget
    while True:
        with _LIVE_LOCK:
            alive = [t for t in _LIVE if t.is_alive()]
        remaining = deadline - time.time()
        if not alive or remaining <= 0:
            break
        alive[0].join(min(remaining, 1.0))
    # Joining the threads is NOT sufficient on its own. EventKit releases the
    # completion block on its own queue a moment AFTER the fetch returns, and
    # that release runs PyObjC's dispose helper, which takes the GIL. If the
    # interpreter has begun finalising by then, Foundation kills the process —
    # SIGKILL, no traceback. Measured: 4 crashes in 20 runs of a suite that
    # builds a NovaViews and exits within half a second.
    #
    # There is nothing to wait ON: the release is not ours to observe. So this
    # is a short, bounded settle, paid only when a tile actually went through
    # EventKit, and only at exit where a fraction of a second costs nothing.
    if _TOUCHED_EVENTKIT.is_set():
        time.sleep(0.4)


atexit.register(_drain)


class _Tile:
    """One home card, refreshed BEHIND the screen instead of in front of him.

    Measured on his Mac: the music check is 352ms of AppleScript and the
    calendar read is 218ms of EventKit. Home is not a page he opens — it is
    re-rendered at startup, after every single answer when the panel dismisses,
    and on every tick of the ticker. Paying half a second of blocking calls
    each time to redraw furniture is the wrong trade.

    So `get()` never blocks. It hands back the last value and, if that value
    has gone stale, starts one refresh in the background. When the new value
    differs from the old one it calls back, which is what makes home LIVE:
    a song he started himself, an event that just began, a temperature that
    moved — they arrive on their own, without him asking for anything.

    A failed refresh keeps the previous value and re-stamps the clock, so a
    calendar Nova cannot read degrades to a slightly old card rather than a
    card that flickers away and back.
    """

    def __init__(self, name: str, build, ttl: float, on_change=None) -> None:
        self.name = name
        self._build = build
        self._ttl = ttl
        self._on_change = on_change
        self._lock = threading.Lock()
        self._value: Optional[dict] = None
        self._at = 0.0
        self._busy = False

    def get(self) -> Optional[dict]:
        now = time.time()
        with self._lock:
            spawn = (now - self._at) >= self._ttl and not self._busy
            if spawn:
                self._busy = True
            value = self._value
        if spawn:
            self._spawn(f"nova-tile-{self.name}")
        return value

    def prime(self) -> None:
        """Warm the tile at startup so the first home render is not empty."""
        with self._lock:
            if self._busy:
                return
            self._busy = True
        self._spawn(f"nova-prime-{self.name}")

    def _spawn(self, name: str) -> None:
        # Nothing new once shutdown has begun: a refresh started here would be
        # exactly the in-flight work _drain exists to avoid.
        if _STOPPING.is_set():
            with self._lock:
                self._busy = False
            return
        t = threading.Thread(target=self._refresh, name=name, daemon=True)
        _track(t)
        t.start()

    def _refresh(self) -> None:
        try:
            fresh = self._build()
        except Exception as exc:
            log.warning(f"home tile {self.name} failed ({exc})")
            fresh = None
            ok = False
        else:
            ok = True
        with self._lock:
            changed = ok and fresh != self._value
            if ok:
                self._value = fresh
            # Stamped even on failure, so a broken engine is retried on the
            # interval rather than on every single render.
            self._at = time.time()
            self._busy = False
        _untrack(threading.current_thread())
        if changed and self._on_change and not _STOPPING.is_set():
            try:
                self._on_change()
            except Exception as exc:
                log.warning(f"home tile {self.name} callback failed ({exc})")


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
        # "Clear home" strips the surface back to the orb. Held here rather
        # than pushed as a one-off payload so anything that re-renders home —
        # the panel dismissal timer, the music poller — respects it too.
        self.home_cleared = False
        # Where each card sits. Loaded from disk so a layout he set by voice
        # survives a restart; a corrupt or partial file falls back to the
        # defaults rather than leaving him with a blank home.
        self.slots: dict[str, str] = dict(DEFAULT_SLOTS)
        self._load_layout()
        # The status row shows at launch, leaves with the greeting, and comes
        # back when he asks about CPU / memory / battery. This is when that
        # recall expires.
        self._system_until = 0.0
        self._system_token = 0
        self._lock = threading.Lock()
        # Last payload actually sent. Home is re-pushed on a ticker so a song
        # he starts himself appears on its own — but pushing an IDENTICAL
        # payload every few seconds would have the app rebuilding a screen
        # that did not change, forever.
        self._last_sent: Optional[dict] = None
        # TTLs are how fast each thing genuinely moves. Music is the one he
        # would notice lagging, because he can hear it start.
        self._tiles: dict[str, _Tile] = {
            "market":   _Tile("market", self._build_market, 300.0, self.refresh_home),
            # Music is the one he can HEAR change, so it is checked often.
            # Affordable because current_track_for_panel short-circuits on
            # NSWorkspace when no player is running, which is nearly always.
            "playing":  _Tile("playing", self._build_music, 2.0, self.refresh_home),
            "weather":  _Tile("weather", self._build_weather, 600.0, self.refresh_home),
            "upcoming": _Tile("upcoming", self._build_upcoming, 120.0, self.refresh_home),
        }
        for tile in self._tiles.values():
            tile.prime()

    # ── Saved layout ──────────────────────────────────────────────────────────
    @property
    def _layout_path(self) -> Path:
        return _data_dir() / "home_layout.json"

    def _load_layout(self) -> None:
        """Read the saved card positions. Never raises, never half-applies.

        A move he made once should still be true next week, so this is on
        disk. It is validated key by key: an unknown card or a slot that no
        longer exists is dropped rather than trusted, because the file outlives
        the code that wrote it.
        """
        try:
            raw = json.loads(self._layout_path.read_text())
        except FileNotFoundError:
            return
        except Exception as exc:
            log.warning(f"home layout unreadable, using defaults ({exc})")
            return
        if not isinstance(raw, dict):
            return
        clean = {c: sl for c, sl in raw.items()
                 if c in DEFAULT_SLOTS and sl in SLOTS}
        # Two cards in one slot would hide one of them. Keep the first and let
        # the rest fall back to their defaults.
        seen: set[str] = set()
        for card, slot in list(clean.items()):
            if slot in seen:
                del clean[card]
            else:
                seen.add(slot)
        self.slots = {**DEFAULT_SLOTS, **clean}

    def _save_layout(self) -> None:
        """Best effort. Losing a saved position costs him one sentence to
        redo; a crash on write would cost him the command he just gave."""
        try:
            path = self._layout_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self.slots, indent=2))
        except Exception as exc:
            log.warning(f"could not save home layout: {exc}")

    # ── The status row ────────────────────────────────────────────────────────
    def recall_system(self) -> None:
        """Bring the status row back because he just asked about it.

        The number Nova said out loud should also be a number he can see. It
        is HELD here with no expiry: the countdown starts when she stops
        talking, not when the handler runs, because a timer armed here would
        already be several seconds down by the time he had heard the answer.
        `settle_system` is what starts it.
        """
        with self._lock:
            self._system_until = float("inf")
            self._system_token += 1
        self.refresh_home()

    def settle_system(self, seconds: float = 10.0) -> None:
        """Start the countdown to hide the row again. Called once Nova has
        finished speaking, so the ten seconds are ten seconds of him looking
        at it rather than ten seconds of her talking over it."""
        with self._lock:
            if self._system_until != float("inf"):
                return                       # nothing being held
            self._system_until = time.time() + seconds
            self._system_token += 1
            token = self._system_token

        def _retire() -> None:
            # Only the LATEST recall may retire the row — asking twice must not
            # have the first timer close the second one's window.
            with self._lock:
                if self._system_token != token:
                    return
                self._system_until = 0.0
            self.refresh_home()

        t = threading.Timer(seconds, _retire)
        t.daemon = True
        t.start()

    def _home_is_showing(self) -> bool:
        """Is home what the app is actually rendering right now?

        Asked of the WS SERVER rather than of `self.current`. That attribute is
        a second copy of the same fact, and other code writes it — nova.py
        stamps it when a handler emits a panel — so the two can drift. When
        they do, home goes permanently stale: the ticker sees a view that is
        not "home", declines to push, and Now Playing never appears again for
        the life of the process. Nothing in the app tells him why.

        The server holds what was last sent to the client, which is by
        definition what he is looking at.
        """
        shown = getattr(self.ws, "_view", None)
        if shown is None:                     # a harness WS without the field
            return self.current == "home"
        return shown == "home"

    def refresh_home(self) -> None:
        """Re-push home, but only if home is what he is looking at.

        The music poller and the status recall both call this from background
        threads. Neither may yank an answer off the screen to show him a card
        that changed behind it.
        """
        if self.ws is None or not self._home_is_showing():
            return
        try:
            payload = self._home_payload()
            if payload == self._last_sent:
                return                          # nothing moved; do not redraw
            self._last_sent = payload
            self.ws.send_view("home", payload)
        except Exception as exc:
            log.warning(f"could not refresh home: {exc}")

    # ── Detection ─────────────────────────────────────────────────────────────
    def detect_intent(self, text: str) -> Optional[str]:
        if not text or not text.strip():
            return None
        t = text.strip()

        # Before navigation: "bring back the home screen" would otherwise read
        # as a request to go home, which is nearly right but would not unhide
        # anything, and "clear home" has no navigation reading at all.
        if _CLEAR_RE.match(t):
            return "clear_home"
        if _RESTORE_RE.match(t):
            return "restore_home"
        if _move_match(t):
            return "move_card"

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

        if view_name in ("clear_home", "restore_home"):
            self.home_cleared = (view_name == "clear_home")
            # These are only meaningful ON home, so they also take him there —
            # "restore home" from the finance screen should show him home, not
            # silently rearrange a screen he cannot see.
            self.current = "home"
            if self.home_cleared:
                self._push("home")
                return "Cleared."
            # "Everything" includes the status row. It is normally tied to the
            # greeting, and by the time he says this the greeting is long gone —
            # so restore borrows the recall path and the row settles away again
            # on its own rather than becoming permanent furniture.
            self.recall_system()
            return "Here's everything again."

        if view_name == "move_card":
            return self._move_card(text)

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

    def _push(self, view_name: str) -> None:
        """Send a view's payload, swallowing transport failures.

        The panel failing must never cost him the spoken answer — that rule
        predates this method and this is just the one place it now lives.
        """
        if self.ws is None:
            return
        try:
            view = VIEWS.get(view_name)
            self.ws.send_view(view_name,
                              self._payload(view) if view else {})
        except Exception as exc:
            log.warning(f"could not send view {view_name}: {exc}")

    def _move_card(self, text: str) -> str:
        """"Move now playing to the bottom right."

        A slot holds one card. If the destination is taken the two cards SWAP,
        which is the only resolution that leaves every card still on screen —
        evicting the occupant would silently lose him a card he never mentioned.
        """
        m = _move_match(text)
        if not m:                                   # unreachable via detect
            return "I didn't catch where you wanted that."
        card = _CARD_ALIASES.get(m.group(1).lower(), "")
        slot = _SLOT_ALIASES.get(m.group(2).lower(), "")
        if not card or not slot:
            return "I didn't catch where you wanted that."
        if card == "system":
            # Honest rather than silently doing nothing: it is a row, not a
            # card, and it lives at the bottom by design.
            return "The status line stays along the bottom, it isn't a card."

        current = self.slots.get(card)
        if current == slot:
            return f"{_CARD_SPOKEN[card].capitalize()} is already there."

        occupant = next((c for c, sl in self.slots.items()
                         if sl == slot and c != card), None)
        self.slots[card] = slot
        if occupant and current:
            self.slots[occupant] = current
        self._save_layout()

        # Moving a card is also a request to look at home.
        self.home_cleared = False
        self.current = "home"
        self._push("home")

        where = _SLOT_SPOKEN.get(slot, slot)
        if occupant:
            return (f"Moved {_CARD_SPOKEN[card]} to the {where}, and "
                    f"{_CARD_SPOKEN[occupant]} took its place.")
        return f"Moved {_CARD_SPOKEN[card]} to the {where}."

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
        if view.name == "finance":
            return self._finance_payload()
        return {}

    def _finance_payload(self) -> dict:
        """The indices and his watchlist. Built by market_intents so the screen
        and the spoken answer come from the same templated numbers."""
        import panels as P
        market = getattr(self.assistant, "market", None)
        if market is None:
            return P.panel(title="Markets",
                           blocks=[P.note("Market data isn't available.")])
        try:
            return market.screen_payload()
        except Exception as exc:
            log.warning(f"finance panel unavailable ({exc})")
            return P.panel(title="Markets",
                           blocks=[P.note("I couldn't reach market data just now.")])

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
        activity. Cards appear on their own terms — Now Playing only while
        something is actually playing, the status row only at launch or when he
        asks — and each one carries the SLOT he last put it in, so a layout he
        arranged by voice survives a restart.

        Slots are a stamp on each block rather than a new payload shape, so a
        client that knows nothing about them still renders the list in order.

        Every block is defensive. Home must render if the calendar is
        permission-blocked, the weather service is down, the market is
        unreachable and no music is open — a broken tile is not a reason to
        show him a broken screen.
        """
        import panels as P

        # The greeting is a welcome, not a permanent header. It goes the moment
        # he starts talking — the WAKE WORD, not her reply — and it does not
        # come back that session. `_last_response` was the old signal and it
        # was a beat too late: it only becomes true after Nova has finished
        # answering, so the welcome was still on screen while she listened.
        spoken_yet = bool(getattr(self.assistant, "_conversation_started", False)
                          or getattr(self.assistant, "_last_response", ""))

        if self.home_cleared:
            # Just her. No cards, no row, and no greeting to imply otherwise.
            return P.panel(title="", subtitle=_today_line(), blocks=[])

        cards = []
        for card, tile in self._tiles.items():
            block = tile.get()                  # cached; never blocks
            if block is not None:
                cards.append(P.at(dict(block),
                                  self.slots.get(card, DEFAULT_SLOTS[card]), card))
        # Sorted by slot so the fallback list order matches what he sees.
        cards.sort(key=lambda b: b.get("slot", "Z"))

        return P.panel(
            title="" if spoken_yet else _greeting(),
            subtitle=_today_line(),
            blocks=cards + [self._home_system(spoken_yet)],
        )

    def _home_system(self, spoken_yet: bool) -> Optional[dict]:
        """CPU, memory and battery as a ROW along the bottom, not a card.

        His call, and the right one: this is glanceable furniture, and a box
        would give it the same weight as his calendar. It shows at launch,
        leaves with the greeting, and comes back when he asks about it.
        """
        import panels as P
        if not (not spoken_yet or time.time() < self._system_until):
            return None
        tools = getattr(self.assistant, "tools", None)
        if tools is None:
            return None
        readings = tools.status_row()          # cached; never blocks
        if not readings:
            return None
        return P.at(P.metrics(readings), "status", "system")

    def _build_market(self) -> Optional[dict]:
        """His watchlist. Cached by market_intents, so this never waits on the
        network — home redraws far too often to spend five HTTP calls on it."""
        market = getattr(self.assistant, "market", None)
        if market is None:
            return None
        return market.home_block()

    def _build_upcoming(self) -> Optional[dict]:
        """Calendar and reminders TOGETHER, which is what he actually means by
        what is coming up. Sorted by time, with reminders marked so the two are
        still tellable apart at a glance.

        Only today's reminders, plus anything already overdue. A reminder due
        next month is not "upcoming" on a home screen.
        """
        import panels as P
        rows: list[dict] = []

        try:
            import calendar_reminders as cal
            for e in cal.get_today_events():
                start = str(e.get("start") or "")
                when = ""
                if " at " in start:
                    when = start.split(" at ", 1)[1].strip()
                    when = re.sub(r":00\s*(AM|PM)$", r" \1", when, flags=re.I)
                rows.append({"sort": start, "title": e.get("title") or "Untitled",
                             "detail": when, "meta": e.get("location") or ""})
        except Exception as exc:
            log.warning(f"home: calendar unavailable ({exc})")

        try:
            import calendar_reminders as cal
            today = datetime.date.today()
            _TOUCHED_EVENTKIT.set()
            for r in cal.get_all_reminders():
                iso = r.get("due_iso") or ""
                if not iso:
                    continue                    # undated tasks are not "upcoming"
                try:
                    due = datetime.datetime.fromisoformat(iso)
                except ValueError:
                    continue
                if due.date() > today:
                    continue
                rows.append({"sort": iso, "title": r.get("title") or "Reminder",
                             "detail": (r.get("due") or "").split(" at ")[-1],
                             "accent": "reminder"})
        except Exception as exc:
            log.warning(f"home: reminders unavailable ({exc})")

        if not rows:
            return P.note("Nothing on your calendar today.")
        rows.sort(key=lambda r: r.get("sort") or "")
        for r in rows:
            r.pop("sort", None)
        return P.items(rows[:6], title="Upcoming")

    def _build_weather(self) -> Optional[dict]:
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

    def _build_music(self) -> Optional[dict]:
        """Only when something is actually playing — his rule. The poller in
        nova.py calls refresh_home when this changes, so the card also arrives
        on its own when HE starts the music."""
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
