"""
Nova Tools — macOS system control.

Fast-path: every response here is deterministic. Never touches the LLM.
macOS-only by design: uses osascript, pmset, vm_stat, df, top, screencapture, open.

Routing order inside match() (first hit wins — put SPECIFIC before GENERAL):
  1.  Power (shutdown / restart / sleep)   ← asks for confirmation, never acts directly
  2.  Lock screen
  3.  Timers + short in-process reminders  ← announce themselves when they fire
  4.  Do Not Disturb                        ← declines honestly, no reliable API
  5.  Brightness (query / set / up / down)
  6.  Volume (query / set / up / down / mute / unmute)
  7.  Battery
  8.  System stats (RAM / CPU / disk)
  9.  Running / frontmost apps
  10. Screenshot
  10b Browser control (sites / tabs / back / reload / scroll) — before maps
  11. System info (model + chip)
  11a Maps (how far / how long / navigate) — speaks the answer, offers directions
  11b Music control (Spotify / Apple Music) — before app launch
  12. Minimize / restore windows
  13. Finder folders          ← before app launch: "open downloads" is a folder
  14. Quit / close an app
  15. App launch              ← ~50 spoken aliases + dynamic install scan
  16. Web search

Two things flow OUT of this module besides the spoken string:
  * ``pending_confirm`` — set when an action is destructive enough to need a
    yes/no first (power). nova.py picks it up and arms its confirmation slot.
  * ``on_announce`` — callback used by timers to speak when they fire, since
    nothing is asking at that moment.

Adding a tool: write a `_handle_x`, add ONE match block above, keep it
deterministic, and never claim success you haven't verified.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import re
import subprocess
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger("nova.tools")

# Spoken number words we accept in durations ("give me two minutes").
_NUM_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40,
    "forty-five": 45, "fortyfive": 45, "sixty": 60, "ninety": 90,
}


# Finder locations Nova can open by spoken name.
_FOLDERS: dict[str, str] = {
    "downloads": "~/Downloads", "download": "~/Downloads",
    "desktop": "~/Desktop",
    "documents": "~/Documents", "document": "~/Documents", "docs": "~/Documents",
    "pictures": "~/Pictures", "picture": "~/Pictures", "photos folder": "~/Pictures",
    "movies": "~/Movies", "videos": "~/Movies",
    "music folder": "~/Music",
    "applications": "/Applications", "apps folder": "/Applications",
    "home": "~", "home folder": "~", "user folder": "~",
    "trash": "~/.Trash",
    "library": "~/Library",
    "icloud": "~/Library/Mobile Documents/com~apple~CloudDocs",
    "icloud drive": "~/Library/Mobile Documents/com~apple~CloudDocs",
}

# Spoken names → exact macOS .app names. Only needed where the spoken form
# DIFFERS from the real app name, or where Whisper reliably mishears it —
# anything else is resolved dynamically against what's actually installed.
_APP_ALIASES: dict[str, str] = {
    # browsers
    "brave": "Brave Browser", "brave browser": "Brave Browser",
    "chrome": "Google Chrome", "google chrome": "Google Chrome",
    # editors / dev
    "vs code": "Visual Studio Code", "vscode": "Visual Studio Code",
    "v s code": "Visual Studio Code", "code": "Visual Studio Code",
    "x code": "Xcode", "excode": "Xcode",
    # AI apps — Whisper mangles these constantly
    "clawed": "Claude", "cloud": "Claude", "claud": "Claude", "clod": "Claude",
    "chat gpt": "ChatGPT", "chatgbt": "ChatGPT", "chat g p t": "ChatGPT",
    "gpt": "ChatGPT", "lm studio": "LM Studio", "elem studio": "LM Studio",
    # Microsoft suite (installed as "Microsoft X")
    "word": "Microsoft Word", "excel": "Microsoft Excel",
    "powerpoint": "Microsoft PowerPoint", "power point": "Microsoft PowerPoint",
    "outlook": "Microsoft Outlook",
    # misc where spoken != bundle name
    "zoom": "zoom.us", "system preferences": "System Settings",
    "settings": "System Settings", "preferences": "System Settings",
    "imessage": "Messages", "text messages": "Messages",
    "apple music": "Music", "face time": "FaceTime",
    "app store": "App Store", "appstore": "App Store",
    "activity monitor": "Activity Monitor", "task manager": "Activity Monitor",
    "quicktime": "QuickTime Player", "quick time": "QuickTime Player",
    "voice memos": "VoiceMemos", "disc cord": "Discord",
    "spotifi": "Spotify", "text edit": "TextEdit",
}


class NovaTools:
    def __init__(self, config: dict,
                 on_announce: Optional[Callable[[str], None]] = None,
                 on_progress: Optional[Callable[[dict], None]] = None) -> None:
        self.config = config
        # Called by a timer/short-reminder when it fires. nova.py supplies a
        # callback that speaks safely (never on top of an in-flight response).
        self._on_announce = on_announce
        # Pushes a live step list to the screen while a handler works. None in
        # the routing harness, where Progress simply becomes a no-op.
        self._on_progress = on_progress
        # True when this turn actually manipulated his Mac (launched an app,
        # drove the browser). nova.py reads it to park Nova in the corner, so
        # she gets out of the way of whatever she just opened. One-shot.
        self.touched_mac = False
        # True when this turn answered a CPU / memory / battery question. The
        # home screen reads it to bring the status row back, so the number she
        # said out loud is also the number he can see. One-shot.
        self.showed_system = False
        # Status-row cache. Primed here rather than on first use so the very
        # first home render — the greeting one, the one he actually looks at —
        # already has numbers instead of filling in a moment later.
        self._status_lock = threading.Lock()
        self._status_cache: list[dict] = []
        self._status_at = 0.0
        self._status_busy = False
        threading.Thread(target=self._refresh_status,
                         name="nova-status-prime", daemon=True).start()
        # Set to a zero-arg callable when the user must confirm before we act.
        self.pending_confirm: Optional[Callable[[], str]] = None
        # SOFT follow-up ("want me to pull up directions?"). Unlike
        # pending_confirm, a non-answer just drops it and routes normally.
        self.pending_offer: Optional[Callable[[], str]] = None
        self._timers: dict[str, dict] = {}      # label -> {timer, kind, fires_at}
        self._timer_seq = 0
        self._lock = threading.Lock()
        # (player, original_volume) while the music is turned down for
        # listening, else None. See duck_music().
        self._ducked: Optional[tuple] = None

    # ═══════════════════════════════════════════════════════════════════════
    # Dispatch
    # ═══════════════════════════════════════════════════════════════════════
    def match(self, text: str) -> Optional[str]:
        """Return a response string if the text matches a tool intent, else None."""
        low = text.lower().strip()
        self.pending_confirm = None
        self.pending_offer = None

        # ── 1. Mac power — CONFIRM FIRST, never act on the first utterance ──
        m = re.search(r"\b(shut\s*down|power\s*off|turn\s+off|restart|reboot|sleep)\b", low)
        if m and re.search(r"\b(mac|macbook|computer|laptop|machine|system)\b", low):
            return self._request_power(m.group(1))

        # ── 2. Lock screen ──────────────────────────────────────────────────
        if re.search(r"\block\s+(?:the\s+)?(?:screen|mac|computer|laptop)\b", low) \
           or re.search(r"\block\s+it\s+up\b", low):
            return self._lock_screen()

        # ── 3. Timers / short in-process reminders ──────────────────────────
        if re.search(r"\b(cancel|stop|clear)\b.*\b(timer|alarm)s?\b", low):
            return self._cancel_timers()
        if re.search(r"\b(?:how\s+much\s+time|how\s+long).*\b(?:left|remaining)\b", low) \
           or re.search(r"\bcheck\s+(?:my\s+)?timers?\b", low):
            return self._timer_status()
        # "remind me IN <duration>" is an in-process alarm, NOT a Reminders.app
        # entry ("remind me TO ..." is handled by the calendar layer).
        m = re.search(r"\bremind\s+me\s+in\s+(.+)", low)
        if m:
            return self._start_timer(m.group(1), kind="reminder", source=low)
        if re.search(r"\b(?:set|start|give\s+me|make)\b.*\btimer\b", low) \
           or re.search(r"\btimer\s+for\b", low) \
           or re.search(r"\b(?:set|start)\s+(?:a\s+|an\s+)?(?:\w+[\s-]?)?(?:second|minute|hour)s?\b", low) \
           or re.search(r"\bgive\s+me\s+(.+?)\s+(?:seconds?|minutes?|hours?)\b", low):
            return self._start_timer(low, kind="timer", source=low)

        # ── 4. Do Not Disturb (honest decline) ──────────────────────────────
        if re.search(r"\b(do\s+not\s+disturb|dnd|focus\s+mode)\b", low):
            return ("I can't toggle Do Not Disturb reliably, there's no stable "
                    "macOS API for it. You can switch it from Control Center in "
                    "the menu bar.")

        # ── 5. Brightness ───────────────────────────────────────────────────
        # "brighter days ahead" is not a request to change the display. The
        # comparative words only count when something on the Mac is named.
        if "brightness" in low or re.search(
                r"\b(?:brighter|dimmer|brighten|dim)\b[^.?!]{0,20}"
                r"\b(?:screen|display|monitor|it)\b", low) or re.search(
                r"\b(?:screen|display|monitor)\b[^.?!]{0,20}"
                r"\b(?:brighter|dimmer|brighten|dim)\b", low):
            m = re.search(r"brightness\s+(?:to\s+)?(\d{1,3})\b", low) \
                or re.search(r"\bset\s+(?:the\s+)?brightness\s+(\d{1,3})\b", low)
            if m:
                return self._brightness_set(int(m.group(1)))
            if re.search(r"\b(up|increase|raise|brighter|brighten)\b", low):
                return self._brightness_step(+1)
            if re.search(r"\b(down|decrease|lower|dimmer|dim)\b", low):
                return self._brightness_step(-1)
            return self._brightness_query()

        # ── 6. Volume ───────────────────────────────────────────────────────
        # Explicit un/mute of SYSTEM AUDIO. Checked before the generic volume
        # block. `(un)?` matters: "mute" is a substring of "unmute", so a naive
        # `"mute" in low` test muted the Mac when asked to UNMUTE.
        m = re.search(r"\b(un)?mute\b", low)
        if m and re.search(r"\b(audio|sound|speakers?|volume|mac|computer)\b", low):
            return self._mute_audio(mute=not bool(m.group(1)))
        # "music volume" / "Spotify volume" means the PLAYER's own volume — let
        # it fall through to the music section rather than moving system audio.
        # "the volume of work is insane" is not a request. The word has to be
        # paired with something to DO to it, or a question about it.
        if "volume" in low and not re.search(r"\b(music|spotify|song|track)\b", low) \
           and re.search(r"\b(?:up|down|louder|quieter|softer|increase|raise|"
                         r"decrease|lower|max|full|mute|off|silent|set|what|"
                         r"how\s+loud|current|check|\d{1,3})\b", low):
            m = re.search(r"volume\s*(?:to|at)?\s*(\d{1,3})\b", low)
            if m:
                return self._volume_set(int(m.group(1)))
            if re.search(r"\b(what|how\s+loud|current|check)\b", low):
                return self._volume_query()
            if re.search(r"\b(up|louder|increase|raise)\b", low):
                return self._volume_adjust(+15)
            if re.search(r"\b(down|quieter|decrease|lower|softer)\b", low):
                return self._volume_adjust(-15)
            if re.search(r"\b(max|full|all the way up)\b", low):
                return self._volume_set(100)
            if re.search(r"\b(off|silent)\b", low):
                return self._volume_set(0)
            return self._volume_query()

        # ── 7. Battery ──────────────────────────────────────────────────────
        if any(p in low for p in ("battery", "how much charge", "power level", "how charged")):
            return self._battery_status()

        # ── 8. System stats ─────────────────────────────────────────────────
        # These three are about the MACHINE. Bare words claimed "my memory is
        # terrible these days", "in memory of my grandfather", "I need more
        # space in my closet" and "we should give the team some space" — all
        # answered with a system stat.
        if self._MEMORY_STAT_RE.search(low) and not re.search(r"\bremember\b", low):
            return self._memory_status()
        if re.search(r"\bcpu\b|\bprocessor\s+usage\b", low):
            return self._cpu_status()
        if self._DISK_STAT_RE.search(low):
            return self._disk_status()
        if re.search(r"\bsystem\s+stats?\b|\bhow.*\b(mac|computer)\b.*\bdoing\b", low):
            return self._all_stats()

        # ── 9. Running / frontmost apps ─────────────────────────────────────
        if re.search(r"\b(what|which)\b.*\b(app|application)\b.*\b(front|frontmost|active|focused|using)\b", low) \
           or re.search(r"\bfrontmost\s+app\b", low):
            return self._frontmost_app()
        if re.search(r"\b(what|which)\b.*\b(apps?|applications?|programs?)\b.*\b(running|open)\b", low) \
           or re.search(r"\blist\s+(?:my\s+)?(?:open\s+)?apps?\b", low) \
           or re.search(r"\bwhat(?:'?s| is)\s+running\b", low):
            return self._running_apps()

        # ── 10. Screenshot ──────────────────────────────────────────────────
        # Shaped like a request, not a substring. This matched a bare
        # "screenshot" ANYWHERE, so "shove the old screenshots in the archive
        # folder" took a fresh capture, dropped a PNG on his Desktop and
        # reported success — a loud, file-creating side effect for a sentence
        # that was asking Nova to tidy up.
        if self._SCREENSHOT_RE.search(low):
            return self._screenshot()

        # ── 11. System info ─────────────────────────────────────────────────
        if any(p in low for p in ("what mac", "what computer", "what machine", "system info")):
            return self._system_info()

        # ── 10b. Browser control (BEFORE maps: "navigate to youtube.com" is a
        #         website, while "navigate to Boston" is a drive) ────────────
        resp = self._match_browser(low)
        if resp is not None:
            return resp

        # ── 11a. Maps: distance / travel time / navigation ──────────────────
        resp = self._match_maps(low)
        if resp is not None:
            return resp

        # ── 11b. Music control (BEFORE app launch so "start the music" plays
        #         rather than opening an app called "the music") ─────────────
        resp = self._match_music(low)
        if resp is not None:
            return resp

        # ── 12. Minimize / restore windows ──────────────────────────────────
        m = re.search(r"\b(?:minimi[sz]e|hide)\s+(?:the\s+|my\s+)?(.*)", low)
        if m and not self._NOT_AN_APP_RE.search(m.group(1).strip()):
            return self._minimize(m.group(1).strip().rstrip("."), minimize=True)
        m = re.search(r"\b(?:unminimi[sz]e|restore|bring\s+back|un-?hide)\s+(?:the\s+|my\s+)?(.*)", low)
        if m and not self._NOT_AN_APP_RE.search(m.group(1).strip()):
            return self._minimize(m.group(1).strip().rstrip("."), minimize=False)

        # ── 13. Finder folders (BEFORE app launch: "open downloads" is a
        #        folder, not an app) ─────────────────────────────────────────
        m = re.search(r"\b(?:open|show|go\s+to|take\s+me\s+to|bring\s+up|pull\s+up)\s+"
                      r"(?:me\s+)?(?:my\s+|the\s+)?(.+)", low)
        if m:
            target = m.group(1).strip().rstrip(".")
            # Strip the trailing politeness FIRST, or "coding projects folder
            # for me" never loses the word "folder" and matches nothing.
            target = re.sub(r"\s+(?:for|please)\s+me$|\s+please$", "", target).strip()
            target = re.sub(r"\s+(?:folder|directory)$", "", target).strip()
            if target in _FOLDERS:
                return self._open_folder(target)
            # A REAL subfolder ("open the coding projects folder"). Without this
            # the request reached the LLM, which cheerfully said it had opened
            # a folder it never touched.
            found = self._find_folder(target)
            if found is not None:
                return self._open_found_folder(found)

        # ── 14. Quit / close an app ─────────────────────────────────────────
        m = re.search(r"\b(?:quit|close|exit|shut)\s+(?:down\s+)?(?:the\s+|my\s+)?(.+)", low)
        if m:
            resp = self._quit_app(m.group(1).strip().rstrip("."))
            if resp is not None:      # None => not a real app, keep routing
                return resp

        # ── 15. App launch ──────────────────────────────────────────────────
        # ANCHORED to the start of the utterance. An unanchored search matched a
        # launch verb anywhere in a sentence, so "I wanted to start getting into
        # them" (about comics) became "I couldn't find an app called getting
        # into them". Measured: 5 of 9 ordinary sentences were hijacked.
        m = re.match(r"\s*(?:hey\s+|please\s+)?"
                     r"(?:(?:can|could|would)\s+you\s+)?(?:please\s+)?"
                     r"(?:open|launch|start|run|fire\s+up|pull\s+up)\s+(.+)", low)
        if m and not self._NOT_AN_APP_RE.search(m.group(1).strip()):
            resp = self._open_app(m.group(1).strip().rstrip("."))
            if resp is not None:   # None => not a real app; keep routing
                return resp

        # ── 16. Web search ──────────────────────────────────────────────────
        # Anchored for the same reason as app launch: an unanchored "find" or
        # "look for" turned ordinary sentences into web searches.
        m = re.match(r"\s*(?:hey\s+|please\s+)?"
                     r"(?:(?:can|could|would)\s+you\s+)?(?:please\s+)?"
                     r"(?:search|look\s+up|google)\s+(?:for\s+)?(.+)", low)
        if m:
            return self._web_search(m.group(1).strip().rstrip("."))

        return None

    # ═══════════════════════════════════════════════════════════════════════
    # Power / lock  (destructive → confirmation required)
    # ═══════════════════════════════════════════════════════════════════════
    def _request_power(self, verb: str) -> str:
        verb = verb.replace(" ", "").lower()
        if verb in ("shutdown", "poweroff", "turnoff"):
            action, script = "shut down", 'tell application "System Events" to shut down'
        elif verb in ("restart", "reboot"):
            action, script = "restart", 'tell application "System Events" to restart'
        else:
            action, script = "sleep", 'tell application "System Events" to sleep'

        def _do() -> str:
            subprocess.Popen(["osascript", "-e", script])
            return f"Okay, {action}ping your Mac now." if action == "sleep" \
                else f"Okay, {action}ting your Mac now."

        self.pending_confirm = _do
        return f"Are you sure you want me to {action} your Mac?"

    def _lock_screen(self) -> str:
        r = subprocess.run(
            ["osascript", "-e", 'tell application "System Events" to keystroke "q" using {control down, command down}'],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            # Fallback: put the display to sleep, which also locks if required.
            subprocess.run(["pmset", "displaysleepnow"], check=False)
        return "Locking your screen."

    # ═══════════════════════════════════════════════════════════════════════
    # Timers and short in-process reminders
    # ═══════════════════════════════════════════════════════════════════════
    @staticmethod
    def parse_duration(text: str) -> Optional[int]:
        """Parse a spoken duration into SECONDS. Handles digits and number
        words, plus 'half an hour' / 'quarter of an hour'. None if absent."""
        t = text.lower()
        if re.search(r"\bhalf\s+an?\s+hour\b", t):
            return 1800
        if re.search(r"\bquarter\s+(?:of\s+)?an?\s+hour\b", t):
            return 900
        if re.search(r"\bhour\s+and\s+a\s+half\b", t):
            return 5400
        unit_re = r"(seconds?|secs?|minutes?|mins?|hours?|hrs?)"
        m = re.search(r"(\d+(?:\.\d+)?)\s*" + unit_re, t)
        if m:
            qty, unit = float(m.group(1)), m.group(2)
        else:
            words = "|".join(sorted(_NUM_WORDS, key=len, reverse=True))
            m = re.search(r"\b(" + words + r")\s+" + unit_re, t)
            if not m:
                return None
            qty, unit = float(_NUM_WORDS[m.group(1)]), m.group(2)
        if unit.startswith(("second", "sec")):
            secs = qty
        elif unit.startswith(("minute", "min")):
            secs = qty * 60
        else:
            secs = qty * 3600
        secs = int(round(secs))
        return secs if 1 <= secs <= 24 * 3600 else None

    @classmethod
    def _duration_adjective(cls, secs: int) -> str:
        """Singular ADJECTIVE form for phrases like 'your 5 minute timer'.
        (_spoken_duration gives the noun form: '5 minutes'.)"""
        return re.sub(r"s$", "", cls._spoken_duration(secs))

    @staticmethod
    def _spoken_duration(secs: int) -> str:
        if secs % 3600 == 0 and secs >= 3600:
            n = secs // 3600
            return f"{n} hour" + ("s" if n != 1 else "")
        if secs % 60 == 0 and secs >= 60:
            n = secs // 60
            return f"{n} minute" + ("s" if n != 1 else "")
        return f"{secs} second" + ("s" if secs != 1 else "")

    def _start_timer(self, phrase: str, kind: str, source: str) -> str:
        secs = self.parse_duration(phrase)
        if secs is None:
            return ("How long would you like? Try something like "
                    "'set a timer for five minutes'.")
        # For "remind me in 10 minutes to check the oven", keep the task text.
        task = None
        m = re.search(r"\bto\s+(.+)$", phrase)
        if m and kind == "reminder":
            task = m.group(1).strip(" .!?")

        with self._lock:
            self._timer_seq += 1
            label = f"{kind}-{self._timer_seq}"

        def _fire() -> None:
            with self._lock:
                self._timers.pop(label, None)
            if kind == "reminder" and task:
                msg = f"Reminder: {task}."
            elif kind == "reminder":
                msg = f"That's your {self._duration_adjective(secs)} reminder."
            else:
                msg = f"Your {self._duration_adjective(secs)} timer is up."
            log.info(f"{label} fired: {msg}")
            if self._on_announce:
                try:
                    self._on_announce(msg)
                except Exception:
                    log.exception("timer announce failed")

        t = threading.Timer(secs, _fire)
        t.daemon = True
        with self._lock:
            self._timers[label] = {"timer": t, "kind": kind, "fires_at": time.time() + secs}
        t.start()

        spoken = self._spoken_duration(secs)
        if kind == "reminder":
            return f"Okay, I'll remind you in {spoken}." if not task \
                else f"Okay, I'll remind you to {task} in {spoken}."
        return f"Timer set for {spoken}."

    def _cancel_timers(self) -> str:
        with self._lock:
            if not self._timers:
                return "You don't have any timers running."
            n = len(self._timers)
            for entry in self._timers.values():
                entry["timer"].cancel()
            self._timers.clear()
        return "Cancelled your timer." if n == 1 else f"Cancelled all {n} timers."

    def _timer_status(self) -> str:
        with self._lock:
            if not self._timers:
                return "You don't have any timers running."
            parts = []
            for entry in self._timers.values():
                left = max(0, int(round(entry["fires_at"] - time.time())))
                parts.append(f"{self._spoken_duration(left)} left on your {entry['kind']}")
        return ", and ".join(parts) + "."

    # ═══════════════════════════════════════════════════════════════════════
    # Brightness
    # ═══════════════════════════════════════════════════════════════════════
    @staticmethod
    def _brightness_read() -> Optional[float]:
        """Current display brightness 0..1 via CoreDisplay, or None."""
        try:
            cd = ctypes.CDLL("/System/Library/Frameworks/CoreDisplay.framework/CoreDisplay")
            cg = ctypes.CDLL(ctypes.util.find_library("CoreGraphics"))
            cd.CoreDisplay_Display_GetUserBrightness.restype = ctypes.c_double
            cd.CoreDisplay_Display_GetUserBrightness.argtypes = [ctypes.c_uint32]
            return float(cd.CoreDisplay_Display_GetUserBrightness(cg.CGMainDisplayID()))
        except Exception as e:
            log.warning(f"brightness read failed: {e}")
            return None

    def _brightness_query(self) -> str:
        lvl = self._brightness_read()
        if lvl is None:
            return "I couldn't read your display brightness."
        return f"Brightness is at {round(lvl * 100)} percent."

    def _brightness_apply(self, target: Optional[float], steps: int = 0) -> bool:
        """Try to change brightness. Returns True only if it VERIFIABLY moved.

        macOS gives no supported way to set built-in display brightness on
        Apple Silicon: CoreDisplay's setter silently no-ops, and synthetic
        F1/F2 key codes are dropped unless the app holds Accessibility
        permission (osascript still exits 0, so we must verify by re-reading).
        The optional `brightness` CLI (brew install brightness) does work.
        """
        before = self._brightness_read()
        if target is not None and _which("brightness"):
            subprocess.run(["brightness", str(round(target, 2))], capture_output=True)
        else:
            key = 144 if steps > 0 else 145
            for _ in range(max(1, abs(steps))):
                subprocess.run(
                    ["osascript", "-e", f'tell application "System Events" to key code {key}'],
                    capture_output=True,
                )
                time.sleep(0.05)
        time.sleep(0.35)
        after = self._brightness_read()
        if before is None or after is None:
            return False
        return abs(after - before) > 0.005

    _BRIGHTNESS_HELP = (
        " I can read it but not change it. macOS blocks brightness control "
        "unless Nova has Accessibility permission, in System Settings under "
        "Privacy and Security, or you install the brightness command line tool."
    )

    def _brightness_set(self, pct: int) -> str:
        pct = max(0, min(100, pct))
        if self._brightness_apply(target=pct / 100.0):
            return f"Brightness set to {pct} percent."
        lvl = self._brightness_read()
        now = f" It's still at {round(lvl * 100)} percent." if lvl is not None else ""
        return f"I wasn't able to change the brightness.{now}{self._BRIGHTNESS_HELP}"

    def _brightness_step(self, direction: int) -> str:
        if self._brightness_apply(target=None, steps=2 * direction):
            lvl = self._brightness_read()
            where = f" Now at {round(lvl * 100)} percent." if lvl is not None else ""
            return ("Brightness up." if direction > 0 else "Brightness down.") + where
        return f"I wasn't able to change the brightness.{self._BRIGHTNESS_HELP}"

    # ═══════════════════════════════════════════════════════════════════════
    # Volume
    # ═══════════════════════════════════════════════════════════════════════
    @staticmethod
    def _volume_read() -> tuple[Optional[int], bool]:
        r = subprocess.run(
            ["osascript", "-e",
             "set v to output volume of (get volume settings)\n"
             "set m to output muted of (get volume settings)\n"
             "return (v as text) & \",\" & (m as text)"],
            capture_output=True, text=True,
        )
        out = (r.stdout or "").strip()
        try:
            v, m = out.split(",")
            return int(v), m.strip().lower() == "true"
        except Exception:
            return None, False

    def _volume_query(self) -> str:
        vol, muted = self._volume_read()
        if vol is None:
            return "I couldn't read the volume."
        if muted:
            return f"Volume is at {vol} percent, but the audio is muted."
        return f"Volume is at {vol} percent."

    def _volume_set(self, level: int) -> str:
        level = max(0, min(100, level))
        subprocess.run(["osascript", "-e", f"set volume output volume {level}"], check=False)
        return "Volume off." if level == 0 else f"Volume set to {level}."

    def _volume_adjust(self, delta: int) -> str:
        cur, _ = self._volume_read()
        if cur is None:
            return "I couldn't read the volume."
        new = max(0, min(100, cur + delta))
        subprocess.run(["osascript", "-e", f"set volume output volume {new}"], check=False)
        return f"Volume {'up' if delta > 0 else 'down'} to {new}."

    def _mute_audio(self, mute: bool) -> str:
        subprocess.run(
            ["osascript", "-e", f"set volume output muted {'true' if mute else 'false'}"],
            check=False,
        )
        return "System audio muted." if mute else "System audio unmuted."

    # ═══════════════════════════════════════════════════════════════════════
    # Battery + system stats
    # ═══════════════════════════════════════════════════════════════════════
    # Structured readings. These exist because the same numbers now go two
    # places — Nova SAYS them and the home screen SHOWS them — and a spoken
    # sentence is a terrible thing to parse back into a percentage. The spoken
    # versions below are built FROM these, so there is exactly one place where
    # a reading is taken and one place where it could ever be wrong.
    #
    # Every one returns None rather than raising: an unreadable stat costs a
    # segment of the status row, never the row and never the answer.

    def battery_reading(self) -> Optional[dict]:
        """{'level': 84, 'charging': True, 'ac': True} or None."""
        try:
            out = subprocess.run(["pmset", "-g", "batt"],
                                 capture_output=True, text=True).stdout
            m = re.search(r"(\d+)%", out)
            if not m:
                return None
            level = int(m.group(1))
            ac = "AC Power" in out
            return {"level": level, "ac": ac, "charging": ac and level < 100}
        except Exception as e:
            log.warning(f"battery read failed: {e}")
            return None

    def memory_reading(self) -> Optional[dict]:
        """{'used_gb', 'total_gb', 'free_gb', 'pct'} or None."""
        try:
            # Checked rather than assumed: under load these occasionally come
            # back empty, and `int("")` turned a transient into a warning and a
            # missing segment every few seconds once the row ran on a timer.
            r = subprocess.run(["sysctl", "-n", "hw.memsize"],
                               capture_output=True, text=True)
            raw = (r.stdout or "").strip()
            if r.returncode != 0 or not raw.isdigit():
                return None
            total = int(raw)
            vm = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
            if not vm:
                return None
            page = int(re.search(r"page size of (\d+) bytes", vm).group(1))

            def pages(name):
                m = re.search(rf"{name}:\s+(\d+)\.", vm)
                return int(m.group(1)) if m else 0

            free = (pages("Pages free") + pages("Pages inactive")) * page
            used = total - free
            gb = 1024 ** 3
            return {"used_gb": used / gb, "total_gb": total / gb,
                    "free_gb": free / gb, "pct": round(used / total * 100)}
        except Exception as e:
            log.warning(f"memory read failed: {e}")
            return None

    def cpu_reading(self) -> Optional[dict]:
        """{'busy': 18, 'idle': 82} or None."""
        try:
            out = subprocess.run(["top", "-l", "1", "-n", "0"],
                                 capture_output=True, text=True).stdout
            m = re.search(r"CPU usage:\s+([\d.]+)%\s+user,\s+([\d.]+)%\s+sys,"
                          r"\s+([\d.]+)%\s+idle", out)
            if not m:
                return None
            user, sys_, idle = (float(m.group(i)) for i in (1, 2, 3))
            return {"busy": round(user + sys_), "idle": round(idle)}
        except Exception as e:
            log.warning(f"cpu read failed: {e}")
            return None

    def status_row(self, max_age: float = 5.0) -> list[dict]:
        """The bottom-left instrumentation line, ready for panels.metrics().

        NEVER BLOCKS. Measured, `top -l 1` alone is 363ms of a 343ms row, and
        home is re-rendered at startup, on every panel dismissal, and on every
        tick of the now-playing poller — paying a third of a second each time
        to redraw furniture is the wrong trade. So this returns the last
        reading and refreshes behind it. The row is glanceable context; five
        seconds of staleness is invisible, and a stalled pipeline is not.

        The spoken answer does NOT come through here: "what's my CPU" calls
        `cpu_reading` directly and gets a fresh number, because a question
        deserves a real answer even if it costs 363ms.

        Each reading that fails is simply absent. A row with two of three is
        still useful; a row that refuses to render because the battery could
        not be read is not.
        """
        now = time.time()
        with self._status_lock:
            fresh = (now - self._status_at) < max_age
            busy = self._status_busy
            cached = list(self._status_cache)
            if not fresh and not busy:
                self._status_busy = True
                spawn = True
            else:
                spawn = False
        if spawn:
            t = threading.Thread(target=self._refresh_status,
                                 name="nova-status", daemon=True)
            t.start()
        return cached

    def _refresh_status(self) -> None:
        """Take the readings and store them. Runs off the pipeline thread."""
        try:
            out: list[dict] = []
            cpu = self.cpu_reading()
            if cpu:
                out.append({"label": "CPU", "value": f"{cpu['busy']}%",
                            "pct": cpu["busy"] / 100.0,
                            # The only reading that should ever catch his eye.
                            "alert": cpu["busy"] >= 85})
            mem = self.memory_reading()
            if mem:
                out.append({"label": "Memory", "value": f"{mem['used_gb']:.1f} GB",
                            "pct": mem["pct"] / 100.0,
                            "alert": mem["pct"] >= 90})
            bat = self.battery_reading()
            if bat:
                out.append({"label": "Battery", "value": f"{bat['level']}%",
                            "pct": bat["level"] / 100.0,
                            "flag": "charging" if bat["charging"] else
                                    ("plugged" if bat["ac"] else ""),
                            "alert": bat["level"] <= 15 and not bat["ac"]})
        except Exception as e:                       # never take the row down
            log.warning(f"status refresh failed: {e}")
            out = None
        with self._status_lock:
            if out is not None:
                self._status_cache = out
            self._status_at = time.time()
            self._status_busy = False

    def _battery_status(self) -> str:
        self.showed_system = True
        b = self.battery_reading()
        if b is None:
            return "I couldn't read the battery level."
        level = b["level"]
        if b["ac"]:
            status = "and charging" if level < 100 else "and fully charged"
        elif level > 20:
            status = "on battery"
        else:
            status = "on battery, getting low"
        return f"Battery is at {level} percent, {status}."

    def _memory_status(self) -> str:
        self.showed_system = True
        m = self.memory_reading()
        if m is None:
            return "I couldn't read the memory usage."
        return (f"You're using about {m['used_gb']:.1f} of {m['total_gb']:.0f} "
                f"gigabytes of memory, roughly {m['pct']} percent, with "
                f"{m['free_gb']:.1f} gigabytes free.")

    def _cpu_status(self) -> str:
        self.showed_system = True
        c = self.cpu_reading()
        if c is None:
            return "I couldn't read the CPU usage."
        busy = c["busy"]
        mood = ("mostly idle" if busy < 25 else
                "working steadily" if busy < 70 else "under heavy load")
        return f"CPU is at about {busy} percent, {mood}. {c['idle']} percent idle."

    def _disk_status(self) -> str:
        try:
            out = subprocess.run(["df", "-k", "/System/Volumes/Data"],
                                 capture_output=True, text=True).stdout.strip().splitlines()
            parts = out[-1].split()
            total_gb = int(parts[1]) / (1024 ** 2)
            avail_gb = int(parts[3]) / (1024 ** 2)
            used_pct = round((total_gb - avail_gb) / total_gb * 100)
            return (f"You have {avail_gb:.0f} gigabytes free of {total_gb:.0f}, "
                    f"about {used_pct} percent used.")
        except Exception as e:
            log.warning(f"disk stat failed: {e}")
            return "I couldn't read the disk space."

    def _all_stats(self) -> str:
        return " ".join([self._cpu_status(), self._memory_status(), self._disk_status()])

    # ═══════════════════════════════════════════════════════════════════════
    # Running apps
    # ═══════════════════════════════════════════════════════════════════════
    def _frontmost_app(self) -> str:
        r = subprocess.run(
            ["osascript", "-e",
             'tell application "System Events" to get name of first application process whose frontmost is true'],
            capture_output=True, text=True,
        )
        name = (r.stdout or "").strip()
        return f"You're in {name} right now." if name else "I couldn't tell which app is in front."

    def _running_apps(self) -> str:
        r = subprocess.run(
            ["osascript", "-e",
             'tell application "System Events" to get name of every application process whose background only is false'],
            capture_output=True, text=True,
        )
        raw = (r.stdout or "").strip()
        if not raw:
            return "I couldn't read the list of running apps."
        apps = [a.strip() for a in raw.split(",") if a.strip()]
        if not apps:
            return "Nothing looks open right now."
        if len(apps) <= 6:
            return f"You have {len(apps)} apps open: " + ", ".join(apps) + "."
        return (f"You have {len(apps)} apps open, including "
                + ", ".join(apps[:6]) + ".")

    # ═══════════════════════════════════════════════════════════════════════
    # Screenshot / info / launch / search
    # ═══════════════════════════════════════════════════════════════════════
    def _screenshot(self) -> str:
        dest = Path.home() / "Desktop" / f"screenshot_{int(time.time())}.png"
        subprocess.run(["screencapture", "-x", str(dest)], check=False)
        return "Screenshot saved to your Desktop."

    def _system_info(self) -> str:
        result = subprocess.run(["system_profiler", "SPHardwareDataType"],
                                capture_output=True, text=True)
        m = re.search(r"Model Name:\s+(.+)", result.stdout)
        chip = re.search(r"Chip:\s+(.+)", result.stdout)
        if m:
            detail = f" with {chip.group(1).strip()}" if chip else ""
            return f"You're on a {m.group(1).strip()}{detail}."
        return "I couldn't determine your Mac model."

    # ── App name resolution ───────────────────────────────────────────────
    _app_index: Optional[dict] = None      # lowercase name -> exact .app name

    @classmethod
    def _installed_apps(cls, refresh: bool = False) -> dict:
        """Map of every installed app. Resolving against what's ACTUALLY
        installed beats a hardcoded list — anything installed later just works.

        Cached because the scan runs on every "open X", but `refresh=True`
        re-scans so an app installed WHILE Nova is running is still found
        without needing a restart (see _resolve_app's retry-on-miss)."""
        if cls._app_index is None or refresh:
            index = {}
            for d in ("/Applications", "/System/Applications",
                      "/System/Applications/Utilities",
                      str(Path.home() / "Applications")):
                try:
                    for p in Path(d).glob("*.app"):
                        index[p.stem.lower()] = p.stem
                except Exception:
                    continue
            # Nested bundles (e.g. /Applications/Utilities/*.app) — one level.
            try:
                for sub in Path("/Applications").glob("*/*.app"):
                    index.setdefault(sub.stem.lower(), sub.stem)
            except Exception:
                pass
            cls._app_index = index
        return cls._app_index

    def _resolve_app(self, raw: str) -> Optional[str]:
        """Spoken name -> exact app name. alias → exact → unique partial.

        On a miss the installed-app index is re-scanned once and the lookup
        retried, so an app installed since Nova started is still found."""
        clean = re.sub(r"[^\w\s.]", "", raw or "").strip().lower()
        clean = re.sub(r"^(?:the|a|an|my)\s+", "", clean)
        clean = re.sub(r"\s+(?:app|application)$", "", clean).strip()
        if not clean:
            return None
        if clean in _APP_ALIASES:
            return _APP_ALIASES[clean]

        for refresh in (False, True):        # second pass re-scans /Applications
            apps = self._installed_apps(refresh=refresh)
            if clean in apps:
                return apps[clean]
            # Unique partial ("visual studio" -> Visual Studio Code). Only accept
            # a single match, so we never open the wrong app on an ambiguous name.
            hits = {v for k, v in apps.items() if clean in k or k.startswith(clean)}
            if len(hits) == 1:
                return hits.pop()
            if len(hits) > 1:
                return None                  # ambiguous: don't guess
        return None

    @staticmethod
    def _on_screen_windows(app: str) -> Optional[int]:
        """Count real on-screen windows for an app, or None if we can't tell.

        Asks the window server rather than the app: it needs no scripting
        support and cannot launch anything as a side effect.
        """
        try:
            from Quartz import (CGWindowListCopyWindowInfo,
                                kCGWindowListOptionOnScreenOnly, kCGNullWindowID)
            raw = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly,
                                             kCGNullWindowID) or []
        except Exception:
            return None
        n = 0
        for w in raw:
            if str(w.get("kCGWindowOwnerName") or "") != app:
                continue
            if w.get("kCGWindowLayer", 99) != 0:
                continue
            b = w.get("kCGWindowBounds") or {}
            if int(b.get("Width", 0)) >= 120 and int(b.get("Height", 0)) >= 90:
                n += 1
        return n

    def _open_app(self, name: str) -> Optional[str]:
        """Launch an app and VERIFY the user can actually see it.

        `open -a` returning 0 only means the app bundle was found — not that it
        started, and not that anything appeared on screen. The gap is real:
        an app that is already running with zero windows (Chromium browsers do
        this after you close the last window without quitting) just gets
        activated, so Nova said "Opening Brave" while nothing happened.
        """
        resolved = self._resolve_app(name)
        target = resolved or name.title()
        was_running = self._app_running(target)
        # Opening an app is Nova acting on his machine, which is what puts her
        # in the corner so she is out of the way of the thing she just opened.
        # Set BEFORE the launch: if it half-works he still wants her parked.
        self.touched_mac = True

        # Unknown name => None, so the utterance keeps routing and ends up in
        # normal conversation instead of "I couldn't find an app called ...".
        # Only an explicitly resolved alias is worth an error message.
        result = subprocess.run(["open", "-a", target], capture_output=True, text=True)
        if result.returncode != 0:
            if resolved:
                return f"I couldn't open {target}."
            return None

        # It was found — now wait for the process to actually exist.
        started = False
        for _ in range(12):
            if self._app_running(target):
                started = True
                break
            time.sleep(0.5)
        if not started:
            return f"I tried to open {target} but it didn't start."

        # Running is not the same as visible. Give it a moment to draw, then
        # make sure a window is actually on screen.
        windows = None
        for _ in range(6):
            windows = self._on_screen_windows(target)
            if windows is None or windows > 0:
                break
            time.sleep(0.5)

        if windows == 0:
            # Bring it forward and ask for a window. `make new window` works for
            # browsers and Finder; apps without scripting support just error,
            # which is fine — we still report honestly below.
            self._osa(f'tell application "{target}" to activate')
            self._osa(f'tell application "{target}" to make new window')
            time.sleep(1.0)
            windows = self._on_screen_windows(target)
            if windows == 0:
                return (f"{target} is running but didn't put a window on screen. "
                        "It may be hidden or on another desktop.")

        if was_running:
            self._osa(f'tell application "{target}" to activate')
            return f"{target} was already open. Brought it to the front."
        return f"Opening {target}."

    def _quit_app(self, name: str) -> Optional[str]:
        """Quit an app by spoken name. Returns None when the name doesn't
        resolve to something real, so an unrelated 'close ...' phrase falls
        through to the rest of the pipeline instead of erroring."""
        resolved = self._resolve_app(name)
        if not resolved:
            return None
        running = subprocess.run(["pgrep", "-x", resolved], capture_output=True).returncode == 0
        if not running:
            # Some bundles run under a different process name; ask System Events.
            check = subprocess.run(
                ["osascript", "-e",
                 f'tell application "System Events" to (name of processes) contains "{resolved}"'],
                capture_output=True, text=True)
            running = check.stdout.strip() == "true"
        if not running:
            return f"{resolved} isn't running."

        # `quit app "X"` rather than `tell application "X" to quit`: measured,
        # the tell-form returns -128 "User canceled" against a browser with real
        # windows open (Brave's warn-before-quitting), while this form goes
        # through. Try it first, then fall back.
        for script in (f'quit app "{resolved}"',
                       f'tell application "{resolved}" to quit'):
            subprocess.run(["osascript", "-e", script], capture_output=True)
            for _ in range(8):
                time.sleep(0.5)
                if not self._app_running(resolved):
                    return f"Closing {resolved}."

        # NEVER claim success we haven't verified. An app can legitimately
        # refuse: unsaved changes, a modal dialog, warn-before-quitting.
        return (f"I asked {resolved} to quit but it's still open. It may be "
                "waiting on unsaved changes or a confirmation.")

    @staticmethod
    def _app_running(name: str) -> bool:
        if subprocess.run(["pgrep", "-x", name], capture_output=True).returncode == 0:
            return True
        check = subprocess.run(
            ["osascript", "-e",
             f'tell application "System Events" to (name of processes) contains "{name}"'],
            capture_output=True, text=True)
        return check.stdout.strip() == "true"

    # ═══════════════════════════════════════════════════════════════════════
    # Browser control
    # ═══════════════════════════════════════════════════════════════════════
    def _match_browser(self, low: str) -> Optional[str]:
        import browser_control as bc

        # ── page/tab state ──────────────────────────────────────────────
        if re.search(r"\b(what|which)\s+(page|site|website)\b.*\b(am\s+i|is\s+this)\b", low) \
           or re.search(r"\bwhere\s+am\s+i\b", low) \
           or re.search(r"\bwhat(?:'?s| is)\s+(?:this|the)\s+(?:page|site|website|url)\b", low) \
           or re.search(r"\bcurrent\s+(?:page|url|tab)\b", low):
            return bc.where_am_i()
        if re.search(r"\b(what|which|list|show)\b.*\btabs?\b.*\b(open|do\s+i\s+have)\b", low) \
           or re.search(r"\blist\s+(?:my\s+)?tabs\b", low) \
           or re.search(r"\bwhat\s+tabs\b", low):
            return bc.list_tabs()

        # ── tab management ──────────────────────────────────────────────
        m = re.search(r"\bswitch\s+to\s+(?:the\s+)?(.+?)\s+tab\b", low) \
            or re.search(r"\bgo\s+to\s+(?:the\s+)?(.+?)\s+tab\b", low)
        if m:
            return bc.switch_tab(m.group(1).strip())
        if re.search(r"\bclose\s+(?:all\s+)?(?:the\s+)?other\s+tabs\b", low):
            return bc.close_other_tabs()
        if re.search(r"\b(close|shut)\s+(?:this\s+|the\s+|current\s+)?tab\b", low):
            return bc.close_tab()
        if re.search(r"\b(?:open|new)\s+(?:a\s+)?new\s+tab\b", low) \
           or re.search(r"\bnew\s+tab\b", low):
            return bc.new_tab()

        # ── history / reload ────────────────────────────────────────────
        if re.search(r"\bgo\s+back\b", low) and not re.search(r"\b(song|track|music)\b", low):
            return bc.navigate_history("back")
        if re.search(r"\bgo\s+forward\b", low):
            return bc.navigate_history("forward")
        # The object was OPTIONAL, so a bare "refresh" anywhere reloaded his
        # browser — "refresh my memory on that" did it.
        if re.search(r"\b(?:reload|refresh)\s+(?:the\s+|this\s+)?"
                     r"(?:page|tab|site|website|browser)\b", low) \
           or re.fullmatch(r"\s*(?:hey\s+)?(?:nova[,\s]+)?(?:please\s+)?"
                           r"(?:reload|refresh)\s*[.?!]*\s*", low):
            return bc.reload_page()

        # ── scrolling ───────────────────────────────────────────────────
        # Anchored as a COMMAND. A bare \bscroll\b anywhere claimed ordinary
        # sentences — "scroll through my photos sometime" was answered with
        # "No browser is open right now." Same failure family as the launch
        # verbs matching mid-sentence.
        if re.match(r"^(?:hey\s+)?(?:nova[,\s]+)?(?:please\s+)?"
                    r"(?:can you\s+)?scroll\b(?:\s+(?:up|down|to|back|"
                    r"the\s+page|a\s+bit|more|further|down\s+more))?"
                    r"[\s.?!]*$", low):
            if re.search(r"\b(top|beginning)\b", low):
                return bc.scroll("top")
            if re.search(r"\b(bottom|end)\b", low):
                return bc.scroll("bottom")
            return bc.scroll("up" if re.search(r"\bup\b", low) else "down")

        # ── search WITHIN a site ────────────────────────────────────────
        # "search amazon for spider-man" means Amazon's own search, not a
        # Google search for the words "amazon for spider-man".
        site = query = None
        m = re.search(r"\b(?:search|look)\s+(?:on\s+|in\s+)?([\w .]+?)\s+for\s+(.+)", low)
        if m:
            site, query = m.group(1), m.group(2)
        else:
            m = re.search(r"\bsearch\s+for\s+(.+?)\s+on\s+([\w .]+)$", low)
            if m:
                query, site = m.group(1), m.group(2)
        if site and query:
            query = re.sub(r"[?.!,]+$", "", query.strip())
            hit = bc.site_search(site.strip(), query)
            if hit:
                url, label = hit
                bc.open_url(url, label)
                return f"Searching {label} for {query}."

        # ── search the web ──────────────────────────────────────────────
        # Anchored at the head of the utterance. As a bare search it grabbed
        # the "search" inside "job search", so "I'm resuming my job search next
        # month" ACTIVATED the browser and searched for "next month".
        m = self._WEB_SEARCH_RE.match(low)
        if m:
            q = re.sub(r"[?.!,]+$", "", m.group(1).strip())
            if q and q not in _FOLDERS:
                url, label = bc.resolve_target(q)
                resp = bc.open_url(url, label)
                # "Searching for X" reads better aloud than "Opening a search for X".
                if resp.startswith("Opening") and label.startswith("a search"):
                    return f"Searching for {q}."
                return resp

        # ── navigate to a site ──────────────────────────────────────────
        m = (re.search(r"\b(?:go\s+to|pull\s+up|bring\s+up|visit|browse\s+to)\s+(.+)", low)
             or re.search(r"\bnavigate\s+to\s+(.+)", low)
             or re.search(r"\bopen\s+(.+)", low))
        if m:
            target = re.sub(r"[?.!,]+$", "", m.group(1).strip())
            target = re.sub(r"^(?:the|a|an|my)\s+", "", target)
            plain = re.sub(r"\s+(?:folder|directory)$", "", target).strip()
            # Folders and real apps are NOT websites — let those routes win.
            if plain in _FOLDERS or self._resolve_app(target):
                return None
            key = re.sub(r"\s+(?:website|site|page)$", "", target).strip().lower()
            if key in bc.SITES or bc._DOMAIN_RE.match(target):
                url, label = bc.resolve_target(target)
                return bc.open_url(url, label)
            return None      # unknown word → let app-launch report honestly

        return None

    # ═══════════════════════════════════════════════════════════════════════
    # Maps: how far / how long / navigate
    # ═══════════════════════════════════════════════════════════════════════
    # "How long to the nearest CVS?" answers OUT LOUD without opening Maps, then
    # offers directions — the answer is the point, not the map. `pending_offer`
    # is picked up by nova.py exactly like the calendar follow-up.
    _MODE_WORDS = ((r"\bwalk(?:ing)?\b", "walking"),
                   (r"\b(transit|bus|train|subway|public\s+transport)\b", "transit"),
                   (r"\b(driv(?:e|ing)|car)\b", "driving"))

    def _travel_mode(self, low: str) -> str:
        for pat, mode in self._MODE_WORDS:
            if re.search(pat, low):
                return mode
        return "driving"

    @staticmethod
    def _clean_place(raw: str) -> str:
        s = re.sub(r"[?.!,]+$", "", (raw or "").strip())
        s = re.sub(r"^(the|a|an)\s+", "", s, flags=re.I)
        s = re.sub(r"\s+(from here|from my location|by car|on foot|"
                   r"driving|walking|by transit)$", "", s, flags=re.I)
        return s.strip()

    @staticmethod
    def _spoken_place(s: str) -> str:
        """Title-case a place for speech — routing lowercases the utterance, so
        without this Nova says "directions to boston"."""
        small = {"of", "the", "and", "at", "in", "on", "de", "la"}
        words = [w for w in (s or "").split() if w]
        return " ".join(
            w if any(c.isupper() for c in w)                 # keep CVS, McDonald's
            else (w if i and w.lower() in small else w.capitalize())
            for i, w in enumerate(words))

    def _match_maps(self, low: str) -> Optional[str]:
        import maps_engine

        # ── Distance / travel time, spoken (does NOT open Maps) ─────────
        # Checked BEFORE navigation: "how long would it TAKE ME TO get to the
        # nearest CVS" contains the words "take me to", which would otherwise
        # be swallowed by the navigation pattern below.
        m = (re.search(r"\bhow\s+long\s+(?:would\s+it\s+)?(?:take\s+)?(?:me\s+)?(?:to\s+)?"
                       r"(?:get\s+to|drive\s+to|walk\s+to|reach)\s+(.+)", low)
             or re.search(r"\bhow\s+(?:long|far)\s+(?:is\s+it\s+)?(?:to|from\s+here\s+to)\s+(.+)", low)
             or re.search(r"\bhow\s+far\s+(?:away\s+)?is\s+(.+)", low)
             or re.search(r"\b(?:where|how\s+close)\s+is\s+the\s+nearest\s+(.+)", low)
             or re.search(r"\bnearest\s+(.+?)(?:\s+from\s+here)?$", low))
        if m:
            return self._maps_eta(low, m.group(1))

        # ── Navigation: open Maps with the destination ──────────────────
        m = (re.search(r"\b(?:navigate|directions?)\s+to\s+(.+)", low)
             or re.search(r"\bgive\s+me\s+directions?\s+to\s+(.+)", low)
             or re.search(r"\bhow\s+do\s+i\s+get\s+to\s+(.+)", low)
             or re.search(r"\btake\s+me\s+to\s+(.+)", low))
        if m:
            place = self._clean_place(m.group(1))
            # "take me to my downloads folder" is a Finder request, not a drive.
            folder_key = re.sub(r"^my\s+", "", place)
            folder_key = re.sub(r"\s+(?:folder|directory)$", "", folder_key).strip()
            if not place or folder_key in _FOLDERS:
                return None
            mode = self._travel_mode(low)
            spoken = self._spoken_place(place)
            return (f"Opening directions to {spoken}." if maps_engine.open_directions(place, mode)
                    else f"I couldn't open directions to {spoken}.")

        return None

    def _maps_eta(self, low: str, raw_place: str) -> Optional[str]:
        """Speak how far/long away a place is, then offer directions."""
        import maps_engine
        place = self._clean_place(re.sub(r"^(?:the\s+)?nearest\s+", "", raw_place))
        if not place:
            return None
        mode = self._travel_mode(low)

        res = maps_engine.nearest(place, mode=mode)
        if not res.get("ok"):
            if res.get("error") == "no_location":
                # Be useful anyway: we can still open directions without a fix.
                sp = self._spoken_place(place)
                self.pending_offer = lambda p=place, s=sp, md=mode: (
                    f"Opening directions to {s}." if maps_engine.open_directions(p, md)
                    else f"I couldn't open directions to {s}.")
                # Only send him to System Settings when NovaOS is actually
                # listed there — i.e. after the app has asked and he declined.
                # It used to say this unconditionally, so he went looking for a
                # NovaOS entry that did not exist, because nothing had ever
                # requested location. See LocationProvider.swift.
                if maps_engine.location_was_denied():
                    return ("Location is turned off for me. You can switch it "
                            "on under Privacy and Security, then Location "
                            "Services, and pick NovaOS. Want me to open "
                            "directions instead?")
                return ("I don't have a location fix yet, so I can't measure "
                        "the distance. Want me to open directions instead?")
            if res.get("error") == "no results":
                return f"I couldn't find a {self._spoken_place(place)} nearby."
            return f"I couldn't work out how far {place} is."

        name = res.get("name") or place
        miles = res.get("miles")
        mins = res.get("minutes")
        how = {"walking": "walking", "transit": "by transit"}.get(mode, "")
        if mins is not None:
            lead = (f"The nearest {name} is about {mins} minute"
                    f"{'s' if mins != 1 else ''} away{(' ' + how) if how else ''}")
            if miles:
                lead += f", {miles} mile{'s' if miles != 1 else ''}"
        else:
            lead = f"The nearest {name} is about {miles} miles away"
        addr = res.get("address")
        lead += f", on {addr.split(',')[0]}." if addr else "."

        # Offer directions — answered by nova.py's follow-up handler.
        self.pending_offer = lambda p=name, md=mode: (
            f"Opening directions to {p}." if maps_engine.open_directions(p, md)
            else f"I couldn't open directions to {p}.")
        return lead + " Want me to pull up directions?"

    # ═══════════════════════════════════════════════════════════════════════
    # Music control (Spotify + Apple Music)
    # ═══════════════════════════════════════════════════════════════════════
    # The two apps differ in ways that matter, so everything goes through the
    # adapters below rather than raw AppleScript at each call site:
    #   duration  — Spotify reports MILLISECONDS, Music reports seconds
    #   shuffle   — Spotify `shuffling` (bool), Music `shuffle enabled` (bool)
    #   repeat    — Spotify `repeating` (bool), Music `song repeat` (off/one/all)
    #   previous  — Music also has `back track` (restart-then-previous)
    _PLAYERS = ("Spotify", "Music")     # preference order

    def any_player_running(self) -> bool:
        """Cheap pre-check for the now-playing poller.

        `_running_player` costs one subprocess per player per call, and the
        poller runs forever while the answer is almost always no. NSWorkspace
        answers from memory, and — unlike `tell application "Spotify"` — it
        cannot start anything by asking.

        Returns True when it genuinely cannot tell, so the real check still
        gets its say and a missing PyObjC can never make Now Playing vanish.
        """
        try:
            from AppKit import NSWorkspace
            names = {a.localizedName() for a
                     in NSWorkspace.sharedWorkspace().runningApplications()}
            return any(p in names for p in self._PLAYERS)
        except Exception:
            return True

    def _running_player(self, launch_if_none: bool = False) -> Optional[str]:
        """Which music app is running (Spotify preferred). Never auto-launches
        unless asked — `tell application "X"` would otherwise silently start it."""
        for app in self._PLAYERS:
            r = subprocess.run(
                ["osascript", "-e",
                 f'tell application "System Events" to (name of processes) contains "{app}"'],
                capture_output=True, text=True)
            if r.stdout.strip() == "true":
                return app
        if launch_if_none:
            subprocess.run(["open", "-a", "Spotify"], capture_output=True)
            for _ in range(20):             # wait for it to accept AppleScript
                time.sleep(0.5)
                r = subprocess.run(
                    ["osascript", "-e",
                     'tell application "System Events" to (name of processes) contains "Spotify"'],
                    capture_output=True, text=True)
                if r.stdout.strip() == "true":
                    time.sleep(1.0)
                    return "Spotify"
        return None

    @staticmethod
    def _osa(script: str) -> tuple[bool, str]:
        r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
        return r.returncode == 0, (r.stdout or "").strip()

    def _player_get(self, app: str, prop: str) -> Optional[str]:
        ok, out = self._osa(f'tell application "{app}" to get {prop}')
        return out if ok else None

    def _player_do(self, app: str, cmd: str) -> bool:
        ok, _ = self._osa(f'tell application "{app}" to {cmd}')
        return ok

    def _now_playing(self, app: str) -> Optional[tuple]:
        ok, out = self._osa(
            f'tell application "{app}" to return (name of current track) & "||" & '
            f'(artist of current track) & "||" & (player state as text)')
        if not ok or "||" not in out:
            return None
        name, artist, state = (out.split("||") + ["", "", ""])[:3]
        return name.strip(), artist.strip(), state.strip().lower()

    def current_track_for_panel(self) -> Optional[tuple]:
        """(title, artist) if music is ACTUALLY playing, else None.

        For the home screen, where Now Playing appears only while something is
        playing — a paused player is not something he wants staring at him.
        Never launches a player: `_running_player` without launch_if_none, so
        asking for home does not open Spotify.
        """
        # NSWorkspace first: 0.1ms against 289ms for the AppleScript path, and
        # home asks this every couple of seconds forever. Almost every one of
        # those asks happens with no player running at all, and that case is
        # now free.
        if not self.any_player_running():
            return None
        app = self._running_player(launch_if_none=False)
        if not app:
            return None
        np = self._now_playing(app)
        if not np or not np[0] or np[2] != "playing":
            return None
        return np[0], np[1]

    def _track_seconds(self, app: str) -> tuple:
        """(position, duration) in seconds — normalising Spotify's ms duration."""
        pos = self._player_get(app, "player position")
        dur = self._player_get(app, "duration of current track")
        try:
            pos = float(pos)
        except (TypeError, ValueError):
            pos = None
        try:
            dur = float(dur)
            if app == "Spotify":
                dur = dur / 1000.0
        except (TypeError, ValueError):
            dur = None
        return pos, dur

    @staticmethod
    def _mmss(secs: float) -> str:
        secs = int(round(secs))
        m, s = divmod(max(0, secs), 60)
        if m and s:
            return f"{m} minute{'s' if m != 1 else ''} and {s} second{'s' if s != 1 else ''}"
        if m:
            return f"{m} minute{'s' if m != 1 else ''}"
        return f"{s} second{'s' if s != 1 else ''}"

    def _no_player(self) -> str:
        return "Neither Spotify nor Apple Music is running."

    def _match_music(self, low: str) -> Optional[str]:
        """Music intents. Returns None when the phrase isn't about music, so
        the rest of the tool router still gets a shot at it."""
        MUSIC = r"(music|song|track|spotify|apple music|playback|tune)"

        # ── what's playing ──────────────────────────────────────────────
        if re.search(r"\bwhat(?:'?s| is)\s+(?:currently\s+)?playing\b", low) \
           or re.search(r"\bwhat\s+song\s+is\s+(?:this|playing)\b", low) \
           or re.search(r"\bwho\s+(?:sings|is\s+singing)\s+this\b", low) \
           or re.search(r"\bname\s+of\s+(?:this|the)\s+song\b", low):
            app = self._running_player()
            if not app:
                return self._no_player()
            np = self._now_playing(app)
            if not np or not np[0]:
                return f"Nothing is playing in {self._say(app)} right now."
            name, artist, state = np
            by = f" by {artist}" if artist else ""
            if state == "playing":
                return f"Playing {name}{by}."
            # NOT "Paused on X" — spoken aloud that sounds like Nova just
            # paused it, when this is only a status report.
            return f"{name}{by}, currently paused."

        # ── how much of the track is left ───────────────────────────────
        if re.search(r"\bhow\s+(?:much\s+)?(?:long|time)\b.*\b(left|remaining)\b.*" + MUSIC, low) \
           or re.search(r"\bhow\s+much\s+longer\b.*" + MUSIC, low) \
           or re.search(r"\btime\s+left\s+(?:in|on)\s+(?:this|the)\s+" + MUSIC, low):
            app = self._running_player()
            if not app:
                return self._no_player()
            pos, dur = self._track_seconds(app)
            if pos is None or dur is None:
                return "I couldn't read the track position."
            return f"{self._mmss(dur - pos)} left of {self._mmss(dur)}."

        # ── transport: next / previous / restart ────────────────────────
        # Shaped like a command. A bare \b(next|skip)\b matched "next month",
        # "next week", "what's next" and "skip the meeting" — every one of them
        # skipped his music. Third instance of this exact bug in this file,
        # after "resume" and "screenshot": a transport word that is also an
        # ordinary English word needs the utterance to be ABOUT it.
        if self._SKIP_RE.search(low) \
           and not re.search(r"\bskip\s+(?:ahead|forward|back)\b", low):
            app = self._running_player()
            if not app:
                return self._no_player()
            self._player_do(app, "next track")
            time.sleep(0.6)
            np = self._now_playing(app)
            return (f"Skipped to {np[0]}" + (f" by {np[1]}." if np[1] else ".")) if np and np[0] \
                else "Skipped to the next track."
        if re.search(r"\b(previous|last|go\s+back(?:\s+a)?)\b.*\b(song|track)\b", low) \
           or re.search(r"\bplay\s+(?:the\s+)?(previous|last)\b", low):
            app = self._running_player()
            if not app:
                return self._no_player()
            self._player_do(app, "previous track")
            time.sleep(0.6)
            np = self._now_playing(app)
            return (f"Back to {np[0]}" + (f" by {np[1]}." if np[1] else ".")) if np and np[0] \
                else "Went back a track."
        # "play it again" belongs here, not in the named search below — it means
        # replay what is on, and searching Spotify for a song called "it again"
        # is exactly the kind of literal-minded wrong answer to avoid.
        _restart_verb = re.search(
            r"\b(restart|start\s+over|from\s+the\s+(?:beginning|top)|replay)\b", low)
        _again = (re.search(r"\bplay\s+(?:it|that|this|the\s+song)\s+again\b", low)
                  or re.fullmatch(r"\s*(?:play\s+)?again\s*\.?\s*", low))
        if _again or (_restart_verb and (re.search(MUSIC, low)
                                         or re.search(r"\bthis\b", low))):
            app = self._running_player()
            if not app:
                return self._no_player()
            self._player_do(app, "set player position to 0")
            return "Starting the track over."

        # ── seek ────────────────────────────────────────────────────────
        m = re.search(r"\b(?:skip|jump|go|seek|fast[\s-]?forward)\s+(?:ahead\s+|forward\s+)?"
                      r"(\d+)\s*(seconds?|secs?|minutes?|mins?)", low)
        m_back = re.search(r"\b(?:skip|jump|go|rewind)\s+back(?:ward)?s?\s+"
                           r"(\d+)\s*(seconds?|secs?|minutes?|mins?)", low)
        if m or m_back:
            app = self._running_player()
            if not app:
                return self._no_player()
            src = m_back or m
            amt = int(src.group(1)) * (60 if src.group(2).startswith(("min", "minute")) else 1)
            if m_back:
                amt = -amt
            pos, dur = self._track_seconds(app)
            if pos is None:
                return "I couldn't read the track position."
            new = max(0, pos + amt)
            if dur is not None:
                new = min(new, dur - 1)
            self._player_do(app, f"set player position to {new:.1f}")
            return f"{'Skipped ahead' if amt > 0 else 'Went back'} {self._mmss(abs(amt))}."

        # ── shuffle / repeat ────────────────────────────────────────────
        # "repeat after me", "shuffle the deck of cards" and "the loop of the
        # rollercoaster" all changed his playback mode.
        m = self._SHUFFLE_RE.search(low)
        if m:
            app = self._running_player()
            if not app:
                return self._no_player()
            want_off = bool(re.search(r"\b(off|stop|disable|turn\s+off|no)\b", low))
            on = "false" if want_off else "true"
            word = "off" if want_off else "on"
            if m.group(1) == "shuffle":
                prop = "shuffling" if app == "Spotify" else "shuffle enabled"
                if self._player_do(app, f"set {prop} to {on}"):
                    return f"Shuffle {word}."
                return "I couldn't change shuffle."
            if app == "Spotify":
                ok = self._player_do(app, f"set repeating to {on}")
            else:
                ok = self._player_do(app, f'set song repeat to {"off" if want_off else "all"}')
            return f"Repeat {word}." if ok else "I couldn't change repeat."

        # ── music volume (the PLAYER's own volume, not system audio) ─────
        if re.search(r"\b(music|spotify|song)\b", low) and "volume" in low:
            app = self._running_player()
            if not app:
                return self._no_player()
            mv = re.search(r"(\d{1,3})", low)
            if mv:
                lvl = max(0, min(100, int(mv.group(1))))
                self._player_do(app, f"set sound volume to {lvl}")
                # Read back: Spotify quantises its volume scale (asking for 30
                # lands on 29), so report what it ACTUALLY is.
                time.sleep(0.2)
                actual = self._player_get(app, "sound volume")
                try:
                    lvl = int(float(actual))
                except (TypeError, ValueError):
                    pass
                return f"{self._say(app)} volume set to {lvl}."
            cur = self._player_get(app, "sound volume")
            try:
                cur_i = int(float(cur))
            except (TypeError, ValueError):
                return "I couldn't read the music volume."
            if re.search(r"\b(up|louder|increase|raise)\b", low):
                new = min(100, cur_i + 15)
            elif re.search(r"\b(down|quieter|lower|decrease|softer)\b", low):
                new = max(0, cur_i - 15)
            else:
                return f"{self._say(app)} volume is at {cur_i}."
            self._player_do(app, f"set sound volume to {new}")
            return f"{self._say(app)} volume {'up' if new > cur_i else 'down'} to {new}."

        # ── pause / resume / play ───────────────────────────────────────
        if re.search(r"\b(pause|hold)\b", low) and (re.search(MUSIC, low)
                                                    or re.fullmatch(r"\s*pause\s*\.?\s*", low)):
            app = self._running_player()
            if not app:
                return self._no_player()
            self._player_do(app, "pause")
            return "Paused."
        if re.search(r"\b(stop)\b.*" + MUSIC, low):
            app = self._running_player()
            if not app:
                return self._no_player()
            self._player_do(app, "pause")
            return "Stopped the music."
        # "Just put music on" — every way he might say it. This path needs NO
        # credentials and never has: Spotify resumes whatever context it was
        # last in, which is usually the playlist he was listening to. Six of
        # these phrasings used to fall through to "I can't do that one yet",
        # which is the opposite of true.
        _ANY_MUSIC = r"(?:music|songs?|tunes?|something|anything|playback)"
        if self._RESUME_RE.search(low) \
           or re.search(r"\b(play|start)\b.*" + MUSIC, low) \
           or re.fullmatch(r"\s*play\s*\.?\s*", low) \
           or re.search(r"\b(?:put|throw)\s+on\b.*" + _ANY_MUSIC, low) \
           or re.search(r"\b(?:get|turn)\s+(?:some\s+)?" + _ANY_MUSIC
                        + r"\s+(?:going|on)\b", low) \
           or re.fullmatch(r"\s*(?:play|put\s+on)\s+(?:some\s+)?"
                           r"(?:something|anything)\s*\.?\s*", low) \
           or re.fullmatch(r"\s*(?:some\s+)?music[,\s]*please\s*\.?\s*", low):
            # This is the one place we'll start a player: asking to play music
            # with nothing running clearly means "start some music".
            app = self._running_player(launch_if_none=True)
            if not app:
                return "I couldn't start Spotify."
            # VERIFY it actually started. Spotify's `play` sometimes no-ops
            # (e.g. resuming after a pause with no active device), and reporting
            # "Playing X" while it sits paused is a fake success.
            self._player_do(app, "play")
            time.sleep(0.7)
            if (self._player_get(app, "player state") or "").lower() != "playing":
                self._player_do(app, "playpause")     # nudge it
                time.sleep(0.7)
            state = (self._player_get(app, "player state") or "").lower()
            np = self._now_playing(app)
            if state != "playing":
                return (f"I couldn't get {self._say(app)} to start playing. "
                        "It may need a track or device selected.")
            if np and np[0]:
                return f"Playing {np[0]}" + (f" by {np[1]}." if np[1] else ".")
            return f"Playing in {self._say(app)}."

        # ── play SOMETHING BY NAME ──────────────────────────────────────
        # LAST, so every transport phrase above still wins. "play the next
        # song" must skip, not search for a song called "the next song".
        named = self._named_request(low)
        if named is not None:
            return self._play_named(*named)

        return None

    # ── Play something by name ────────────────────────────────────────────────
    # Spotify's desktop app exposes six AppleScript commands and none of them is
    # `search`; `play track` needs a URI. So a name has to be resolved before the
    # app can play it. See spotify_search.py for what that costs and what it
    # cannot reach (his own library and personal playlists need a user login).
    _NAMED_PLAY_RE = re.compile(
        r"^\s*(?:hey\s+)?(?:nova[,\s]+)?(?:please\s+)?"
        r"(?:(?:can|could|would|will)\s+you\s+)?(?:please\s+)?"
        r"(?:play|put\s+on|throw\s+on|start\s+playing)\s+"
        # The article needs a boundary. Without one it chewed into the next
        # word: "play something" parsed as a request for "thing", and
        # "play anything" as "nything".
        r"(?:me\s+)?(?:(?:some|a|an|the)\s+)?(.+?)\s*[.?!]*\s*$",
        re.IGNORECASE,
    )
    # An app name is a NAME. "your feelings", "faith in people", "to new ideas"
    # are phrases, and treating them as app names produced "I couldn't find an
    # app called your feelings" in the middle of a conversation. A leading
    # function word or an embedded preposition is the tell.
    _NOT_AN_APP_RE = re.compile(
        r"^(?:to|of|for|with|from|about|your|his|her|their|our|out|up|down|"
        r"in|on|at|into|onto|off|away|back|it|me|us|them|this|that)\b"
        r"|\s+(?:in|of|for|with|from|about|into|onto)\s+",
        re.IGNORECASE,
    )

    # System stats, asked about the MACHINE rather than mentioned in passing.
    _MEMORY_STAT_RE = re.compile(
        r"\bram\b"
        r"|\bmemory\s+(?:usage|use|used|free|available|pressure|left)\b"
        r"|\b(?:how\s+much|check|show|what(?:'?s| is))\b[^.?!]{0,24}\bmemory\b"
        r"|\bmemory\s+(?:am\s+i|is)\s+(?:i\s+)?using\b",
        re.IGNORECASE,
    )
    _DISK_STAT_RE = re.compile(
        r"\bdisk\b|\bstorage\b|\bhard\s+drive\s+space\b"
        r"|\b(?:free|disk|drive)\s+space\b"
        r"|\bspace\s+(?:left|free|remaining|available|do\s+i\s+have)\b"
        r"|\b(?:how\s+much)\b[^.?!]{0,20}\bspace\b",
        re.IGNORECASE,
    )
    # Playback mode, as a command rather than a word in a sentence.
    _SHUFFLE_RE = re.compile(
        r"^\s*(?:hey\s+)?(?:nova[,\s]+)?(?:please\s+)?"
        r"(?:(?:turn|switch)\s+(?:on|off)\s+)?"
        r"(shuffle|repeat|loop)\b"
        r"(?:\s+(?:mode|on|off|this|it|the\s+(?:song|track|album|playlist)))?"
        r"\s*[.?!]*\s*$"
        r"|\b(?:turn|switch)\s+(?:on|off)\s+(shuffle|repeat|loop)\b"
        r"|\b(shuffle|repeat|loop)\s+(?:this\s+|that\s+|the\s+|it\s+)?"
        r"(?:music|song|songs|track|tracks|album|playlist|playback)\b"
        r"|\b(?:music|playlist|album)\b[^.?!]{0,20}\b(shuffle|repeat|loop)\b",
        re.IGNORECASE,
    )
    # A request to skip a track, as opposed to any sentence containing "next".
    _SKIP_RE = re.compile(
        r"^\s*(?:hey\s+)?(?:nova[,\s]+)?(?:please\s+)?"
        r"(?:(?:can|could|would|will)\s+you\s+)?(?:please\s+)?"
        r"(?:next|skip)(?:\s+(?:this|that|it))?"
        r"(?:\s+(?:the\s+)?(?:song|track|one))?\s*[.?!]*\s*$"
        r"|\b(?:next|skip)\s+(?:the\s+|this\s+|that\s+)?(?:song|track)\b"
        r"|\bskip\s+(?:this|that|it)\b",
        re.IGNORECASE,
    )
    # A request to TAKE a screenshot, as opposed to any sentence containing the
    # word. See the call site for what the substring version did.
    _SCREENSHOT_RE = re.compile(
        r"^\s*(?:hey\s+)?(?:nova[,\s]+)?(?:please\s+)?"
        r"(?:(?:can|could|would|will)\s+you\s+)?(?:please\s+)?"
        r"(?:take|grab|get|capture|snap|make)\s+(?:me\s+)?"
        r"(?:a|an|the)?\s*(?:screen\s?shot|screen\s+capture|"
        r"(?:picture|shot)\s+of\s+(?:my\s+|the\s+)?screen)\b"
        r"|^\s*(?:hey\s+)?(?:nova[,\s]+)?(?:please\s+)?screen\s?shot"
        r"(?:\s+(?:this|that|it|my\s+screen|the\s+screen))?\s*[.?!]*\s*$"
        r"|\bcapture\s+(?:my|the)\s+screen\b|\bgrab\s+(?:my|the)\s+screen\b",
        re.IGNORECASE,
    )
    # A request to SEARCH, as opposed to any sentence containing the word.
    _WEB_SEARCH_RE = re.compile(
        r"^\s*(?:hey\s+)?(?:nova[,\s]+)?(?:please\s+)?"
        r"(?:(?:can|could|would|will)\s+you\s+)?(?:please\s+)?"
        r"(?:search|google|look\s+up)\s+(?:the\s+web\s+)?(?:for\s+)?(.+)",
        re.IGNORECASE,
    )
    # "Resume" is a TRANSPORT verb and also, far more often in this house, a
    # noun: his resume is the worked example in the briefing. A bare
    # \bresume\b meant every sentence containing the word started playing
    # music — "is my resume up to date" launched Spotify and played AC/DC.
    # So it has to be shaped like a command: the head of the utterance, or
    # followed by the thing being resumed.
    _RESUME_RE = re.compile(
        r"\b(?:unpause|keep\s+playing)\b"
        r"|^\s*(?:hey\s+)?(?:nova[,\s]+)?(?:please\s+)?"
        r"(?:(?:can|could|would|will)\s+you\s+)?(?:please\s+)?"
        r"resume\b(?:\s+(?:the\s+)?(?:music|song|track|playback|playing|it))?"
        r"\s*[.?!]*\s*$"
        r"|\bresume\s+(?:the\s+)?(?:music|song|track|playback|playing)\b",
        re.IGNORECASE,
    )
    # Phrases that are transport or state, never a thing to search for. If the
    # whole request reduces to one of these, it is not a named request.
    _NOT_A_NAME = re.compile(
        r"^(?:music|song|songs|track|tracks|something|anything|playback|tune|"
        r"tunes|it|that|this|more|again|next|previous|last|spotify|apple\s+music)$",
        re.IGNORECASE,
    )
    # "Play" is not always about music. Asking to play something on Netflix is a
    # request Nova genuinely cannot do, and it must keep reaching the honest
    # refusal instead of quietly searching Spotify for a film. This phrase is in
    # the adversarial corpus, and adding named playback broke it — which is what
    # that corpus is for.
    # "Play" heads a great many English idioms, and each one was becoming a
    # Spotify search: "play it cool" opened a search for "it cool".
    _PLAY_IDIOM_RE = re.compile(
        r"^\s*(?:hey\s+)?(?:nova[,\s]+)?(?:please\s+)?play\s+"
        r"(?:it\s+(?:cool|safe|by\s+ear|down|again\s+sam)"
        r"|devil'?s?\s+advocate|along|dumb|hardball|ball|dead|nice|house|"
        r"hooky|catch(?:\s*-?\s*up)?|favou?rites|games|the\s+(?:field|victim|"
        r"fool|part|role|odds|long\s+game)|both\s+sides|with\s+fire|"
        r"a\s+(?:part|role|joke)|second\s+fiddle|hard\s+to\s+get"
        # ...and actual games, which Nova cannot play either.
        r"|chess|cards|poker|golf|tennis|soccer|football|basketball|"
        r"a\s+game|the\s+game)\b",
        re.IGNORECASE,
    )
    _NOT_MUSIC_TARGET = re.compile(
        r"\b(netflix|hulu|disney|prime\s+video|max|peacock|youtube|tv|"
        r"television|movie|film|show|episode|series|trailer|game)\b",
        re.IGNORECASE,
    )

    def _named_request(self, low: str):
        """(query, prefer) for "play X", or None. `prefer` is a Spotify search
        type when he said which kind of thing he wanted."""
        if self._NOT_MUSIC_TARGET.search(low) or self._PLAY_IDIOM_RE.match(low):
            return None
        m = self._NAMED_PLAY_RE.match(low)
        if not m:
            return None
        q = m.group(1).strip()
        if not q or self._NOT_A_NAME.match(q):
            return None

        prefer = None
        # "play the album X" / "play some music by X" / "play the X playlist"
        for pat, kind in ((r"\balbum\b", "album"),
                          (r"\bplaylist\b", "playlist"),
                          (r"\bartist\b", "artist"),
                          (r"\bsong\b|\btrack\b", "track")):
            if re.search(pat, q, re.I):
                prefer = kind
                q = re.sub(pat, " ", q, flags=re.I)
                break
        # Strip the framing words that are about HOW to play, not WHAT.
        q = re.sub(r"\b(?:on|in)\s+(?:spotify|apple\s+music|itunes)\b", " ", q, re.I)
        q = re.sub(r"\bby\s+the\s+artist\b", "by", q, re.I)
        q = re.sub(r"^\s*(?:of|by)\s+", "", q).strip()
        q = re.sub(r"\s{2,}", " ", q).strip(" .,")
        if not q or self._NOT_A_NAME.match(q):
            return None
        return q, prefer

    # ── Ducking: turn the music down while Nova is listening ──────────────────
    # Nicholas played music and Nova stopped hearing him. That is the same
    # physics as barge-in, which is parked because cancelling Nova's own voice
    # needs acoustic echo cancellation — but music is different: NOVA OWNS THE
    # VOLUME KNOB, so it can simply turn the interference down instead of
    # trying to subtract it. Measured, with music mixed into real command audio:
    #
    #     silent        WER   0.0%   understood 6/6
    #     ducked 15%    WER   5.6%   understood 5/6
    #     full          WER   9.7%   understood 4/6
    #     loud          WER 709.5%   understood 3/6   <- Whisper transcribing
    #                                                    the music itself
    #
    # The PLAYER's volume is ducked, never the system's — system volume would
    # take Nova's own voice down with it, which is the opposite of helpful.
    def duck_music(self) -> None:
        """Lower the player while Nova listens, WITHOUT making him wait.

        Measured on a real Spotify: the AppleScript round trip is 534ms. It sat
        between the wake word firing and the microphone opening, and he only
        gets about 700ms of head start before recording begins — so ducking ate
        most of the window and the first word of his command was being clipped.
        That is the likeliest reason Nova mis-heard him over music.

        Off-thread instead. The volume drops a beat into the utterance rather
        than before it, which is almost always still inside the VAD's wait for
        speech onset, and the recording starts on time either way.
        """
        if self._ducked is not None:
            return                      # already ducked; no AppleScript at all
        threading.Thread(target=self._duck_music_now,
                         name="nova-duck", daemon=True).start()

    def _duck_music_now(self) -> None:
        try:
            app = self._running_player()          # never launches
            if not app:
                return
            if (self._player_get(app, "player state") or "").lower() != "playing":
                return                  # nothing audible to duck
            cur = self._player_get(app, "sound volume")
            level = int(float(cur))
            target = int(self.config.get("music", {}).get("duck_level", 20))
            if level <= target:
                return                  # already quiet enough to leave alone
            self._player_do(app, f"set sound volume to {target}")
            self._ducked = (app, level)
            log.info(f"ducked {app} {level} -> {target} while listening")
        except Exception as exc:
            log.debug(f"could not duck music: {exc}")

    def restore_music(self) -> None:
        """Put the volume back. Must run on EVERY path out of a conversation —
        leaving his music quiet because a turn raised would be its own bug."""
        state = self._ducked
        if not state:
            return
        app, level = state
        self._ducked = None             # cleared FIRST, so a failure cannot
        try:                            # wedge Nova into never ducking again
            self._player_do(app, f"set sound volume to {level}")
            log.info(f"restored {app} volume to {level}")
        except Exception as exc:
            log.debug(f"could not restore music volume: {exc}")

    def _open_spotify_search(self, query: str) -> str:
        """Open a search inside the desktop app. No credential needed — Spotify
        registers the `spotify:` URL scheme, so this works out of the box. It
        stops one step short of playing, and Nova says exactly that rather than
        implying it started something."""
        uri = "spotify:search:" + urllib.parse.quote(query)
        r = subprocess.run(["open", uri], capture_output=True)
        if r.returncode != 0:
            return f"I couldn't open a Spotify search for {query}."
        return (f"I've opened a search for {query} in Spotify. "
                "Pick one and I can take it from there.")

    def _play_named(self, query: str, prefer=None) -> str:
        import spotify_search

        if not spotify_search.is_configured():
            # NO NAGGING. Searching by name is a bonus that needs a credential;
            # everything else about music works without one. So do the useful
            # thing the desktop app CAN do unaided — open that search in
            # Spotify — and say so plainly. Telling him to go set up an API key
            # in the middle of asking for music is the wrong answer.
            return self._open_spotify_search(query)

        res = spotify_search.search(query, prefer=prefer)
        if not res.get("ok"):
            reason = res.get("reason")
            if reason == "not_found":
                return (f"I couldn't find anything called {query} on Spotify. "
                        "Your own playlists and saved songs aren't searchable "
                        "this way, only Spotify's catalogue.")
            if reason == "auth_failed":
                return "Spotify rejected my credentials, so I couldn't search."
            return self._open_spotify_search(query)

        app = self._running_player(launch_if_none=True)
        if app != "Spotify":
            # Apple Music cannot play a spotify: URI. Say so rather than
            # silently doing nothing.
            if app:
                return (f"I found {res['label']} on Spotify, but {self._say(app)} "
                        "is the player that's running.")
            return "I couldn't start Spotify."

        self._player_do(app, f'play track "{res["uri"]}"')
        # VERIFY, do not claim. Spotify's play track is a no-op when there is no
        # active device, and reporting a song that never started is exactly the
        # kind of fake success this project has been bitten by.
        time.sleep(0.9)
        state = (self._player_get(app, "player state") or "").lower()
        if state != "playing":
            self._player_do(app, "play")
            time.sleep(0.7)
            state = (self._player_get(app, "player state") or "").lower()
        if state != "playing":
            return (f"I found {res['label']}, but Spotify wouldn't start playing. "
                    "It may need an active device.")

        # Report what is ACTUALLY playing, read back from the player.
        np = self._now_playing(app)
        if np and np[0]:
            return f"Playing {np[0]}" + (f" by {np[1]}." if np[1] else ".")
        return f"Playing {res['label']}."

    @staticmethod
    def _say(app: str) -> str:
        return "Apple Music" if app == "Music" else app

    # ── Window management ─────────────────────────────────────────────────
    def _visible_apps(self) -> list:
        r = subprocess.run(
            ["osascript", "-e",
             'tell application "System Events" to get name of every application process '
             'whose background only is false and visible is true'],
            capture_output=True, text=True)
        return [a.strip() for a in (r.stdout or "").split(",") if a.strip()]

    def _set_miniaturized(self, app: str, value: bool) -> bool:
        """Minimize/restore an app's windows. Prefers app-level scripting
        (Automation permission) and falls back to System Events' accessibility
        attribute, which some apps need. Returns True only if it worked."""
        flag = "true" if value else "false"
        r = subprocess.run(
            ["osascript", "-e",
             f'tell application "{app}" to set miniaturized of every window to {flag}'],
            capture_output=True, text=True)
        if r.returncode == 0:
            return True
        r = subprocess.run(
            ["osascript", "-e",
             f'tell application "System Events" to tell process "{app}" '
             f'to set value of attribute "AXMinimized" of every window to {flag}'],
            capture_output=True, text=True)
        return r.returncode == 0

    def _minimize(self, target: str, minimize: bool) -> Optional[str]:
        verb = "Minimized" if minimize else "Restored"
        target = re.sub(r"\b(window|windows|app|application)s?\b", "", target).strip()

        # "minimize everything" / "minimize all"
        if not target or re.fullmatch(r"(all|everything|them all|it all)?", target):
            if target in ("all", "everything", "them all", "it all"):
                done = [a for a in self._visible_apps() if self._set_miniaturized(a, minimize)]
                if not done:
                    return f"I wasn't able to {'minimize' if minimize else 'restore'} anything."
                return f"{verb} {len(done)} app windows."
            # bare "minimize this window" → whatever is in front
            front = subprocess.run(
                ["osascript", "-e",
                 'tell application "System Events" to get name of first application '
                 'process whose frontmost is true'],
                capture_output=True, text=True).stdout.strip()
            if not front:
                return "I couldn't tell which window is in front."
            if self._set_miniaturized(front, minimize):
                return f"{verb} {front}."
            return f"I wasn't able to {'minimize' if minimize else 'restore'} {front}."

        # Named app. Answer honestly rather than falling through: letting an
        # unresolved "minimize <x>" reach the LLM produced a confabulated
        # "Flibbertigibbet minimized." — a success that never happened.
        app = self._resolve_app(target)
        if not app:
            return f"I couldn't find an app called {target}."
        if self._set_miniaturized(app, minimize):
            return f"{verb} {app}."
        return f"I wasn't able to {'minimize' if minimize else 'restore'} {app}."

    def _open_folder(self, key: str) -> str:
        path = Path(_FOLDERS[key]).expanduser()
        if not path.exists():
            return f"I couldn't find your {key} folder."
        subprocess.run(["open", str(path)], check=False)
        label = "Trash" if key == "trash" else key.replace(" folder", "").title()
        return f"Opening {label}."

    # Where a spoken folder name might actually live. Shallow on purpose: this
    # answers "open my X folder", not "search my whole disk".
    _FOLDER_SEARCH_ROOTS = ("~/Documents", "~/Desktop", "~/Downloads",
                            "~/Pictures", "~/Movies", "~/Music", "~")

    @classmethod
    def _find_folder(cls, spoken: str):
        """Find a real directory matching a spoken name, or None.

        Spoken names lose punctuation, so "coding projects" has to match
        "Coding_Projects" and "HTML Files".
        """
        want = re.sub(r"[^a-z0-9]", "", (spoken or "").lower())
        if not want or len(want) < 3:
            return None
        for root in cls._FOLDER_SEARCH_ROOTS:
            base = Path(root).expanduser()
            if not base.is_dir():
                continue
            try:
                for entry in base.iterdir():
                    if not entry.is_dir() or entry.name.startswith("."):
                        continue
                    if re.sub(r"[^a-z0-9]", "", entry.name.lower()) == want:
                        return entry
            except (PermissionError, OSError):
                continue
        return None

    def _open_found_folder(self, path) -> str:
        subprocess.run(["open", str(path)], check=False)
        time.sleep(0.6)
        return f"Opening {path.name.replace('_', ' ')}."

    def _web_search(self, query: str) -> str:
        """Search the web, and SHOW the work.

        This used to be three lines: build a URL, `open` it, say "Searching for
        X." Nova never looked at the page, so the one thing she could not tell
        him about a search was what it found.

        Two things changed. It goes through `browser_control` now, so the
        search lands in whichever browser is ALREADY running rather than in
        whatever macOS considers default — every other part of Nova works that
        way and this was the odd one out. And it reads the results back off the
        page she just opened, which costs no additional network call: the
        browser already fetched it.

        What it reads is deliberately narrow — titles and hostnames, nothing
        else, and none of it reaches the LLM. A web page is untrusted text, and
        this same Nova can type, click and move files. Body content in the
        prompt would make any page she visits able to talk to her. So the
        spoken answer is templated from titles here, in Python, the same way
        every other number and fact she speaks is.
        """
        import panels as P
        self.touched_mac = True
        url = f"https://www.google.com/search?q={urllib.parse.quote(query)}"

        try:
            import browser_control as B
        except Exception as exc:
            log.warning(f"browser_control unavailable ({exc})")
            subprocess.run(["open", url], check=False)
            return f"Searching for {query}."

        prog = P.Progress(self._on_progress, "Searching the web",
                          ["Opening the browser", "Loading the page",
                           "Reading the results"], detail=query)
        prog.start()

        opened = B.open_url(url, f"a search for {query}", new_tab=True)
        if opened.startswith("I couldn't"):
            prog.fail("The browser wouldn't open.")
            return f"I couldn't open a search for {query}."
        prog.advance()

        ok, results, why = B.read_results(on_ready=prog.advance)
        if not ok or not results:
            # Honest degradation: the search DID happen and it is on his
            # screen. Only the reading failed, and the difference matters.
            prog.fail(why)
            return (f"I searched for {query}. It's on your screen, but I "
                    f"couldn't read the results off the page.")

        prog.finish(P.items(
            [{"title": r["title"][:90], "meta": r["host"]} for r in results[:6]],
            title="What's on the page"))
        return self._speak_results(query, results)

    @staticmethod
    def _speak_results(query: str, results: list) -> str:
        """One sentence about what came back. Templated, never generated.

        Hostnames are spoken as their name rather than their domain — "from
        imdb", not "from imdb dot com" — because he is listening, not reading.
        """
        def site(host: str) -> str:
            parts = [p for p in (host or "").split(".") if p]
            return parts[-2] if len(parts) >= 2 else (parts[0] if parts else "")

        def clean(title: str) -> str:
            # A page title is untrusted text. It cannot do anything spoken
            # aloud, but it can be enormous, so it gets cut to a sentence.
            t = " ".join((title or "").split())
            return t[:80].rstrip(" -|·") if len(t) > 80 else t

        top = results[0]
        where = site(top.get("host", ""))
        line = f"Top result for {query} is {clean(top.get('title'))}"
        line += f", from {where}." if where else "."
        others = [site(r.get("host", "")) for r in results[1:3]]
        others = [o for o in others if o and o != where]
        if len(others) == 2:
            line += f" There's also {others[0]} and {others[1]}."
        elif len(others) == 1:
            line += f" There's also {others[0]}."
        return line


def _which(cmd: str) -> bool:
    from shutil import which
    return which(cmd) is not None
