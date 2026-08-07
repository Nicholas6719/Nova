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
  11. System info (model + chip)
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
                 on_announce: Optional[Callable[[str], None]] = None) -> None:
        self.config = config
        # Called by a timer/short-reminder when it fires. nova.py supplies a
        # callback that speaks safely (never on top of an in-flight response).
        self._on_announce = on_announce
        # Set to a zero-arg callable when the user must confirm before we act.
        self.pending_confirm: Optional[Callable[[], str]] = None
        self._timers: dict[str, dict] = {}      # label -> {timer, kind, fires_at}
        self._timer_seq = 0
        self._lock = threading.Lock()

    # ═══════════════════════════════════════════════════════════════════════
    # Dispatch
    # ═══════════════════════════════════════════════════════════════════════
    def match(self, text: str) -> Optional[str]:
        """Return a response string if the text matches a tool intent, else None."""
        low = text.lower().strip()
        self.pending_confirm = None

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
        if "brightness" in low or re.search(r"\b(brighter|dimmer|dim\s+the\s+screen)\b", low):
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
        if "volume" in low:
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
        if re.search(r"\b(ram|memory)\b", low) and not re.search(r"\bremember\b", low):
            return self._memory_status()
        if re.search(r"\bcpu\b|\bprocessor\s+usage\b", low):
            return self._cpu_status()
        if re.search(r"\b(disk|storage|hard\s+drive|space)\b", low):
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
        if any(p in low for p in ("take a screenshot", "screenshot", "capture the screen", "grab the screen")):
            return self._screenshot()

        # ── 11. System info ─────────────────────────────────────────────────
        if any(p in low for p in ("what mac", "what computer", "what machine", "system info")):
            return self._system_info()

        # ── 12. Minimize / restore windows ──────────────────────────────────
        m = re.search(r"\b(?:minimi[sz]e|hide)\s+(?:the\s+|my\s+)?(.*)", low)
        if m:
            return self._minimize(m.group(1).strip().rstrip("."), minimize=True)
        m = re.search(r"\b(?:unminimi[sz]e|restore|bring\s+back|un-?hide)\s+(?:the\s+|my\s+)?(.*)", low)
        if m:
            return self._minimize(m.group(1).strip().rstrip("."), minimize=False)

        # ── 13. Finder folders (BEFORE app launch: "open downloads" is a
        #        folder, not an app) ─────────────────────────────────────────
        m = re.search(r"\b(?:open|show|go\s+to|take\s+me\s+to|bring\s+up|pull\s+up)\s+"
                      r"(?:me\s+)?(?:my\s+|the\s+)?(.+)", low)
        if m:
            target = re.sub(r"\s+(?:folder|directory)$", "",
                            m.group(1).strip().rstrip(".")).strip()
            if target in _FOLDERS:
                return self._open_folder(target)

        # ── 14. Quit / close an app ─────────────────────────────────────────
        m = re.search(r"\b(?:quit|close|exit|shut)\s+(?:down\s+)?(?:the\s+|my\s+)?(.+)", low)
        if m:
            resp = self._quit_app(m.group(1).strip().rstrip("."))
            if resp is not None:      # None => not a real app, keep routing
                return resp

        # ── 15. App launch ──────────────────────────────────────────────────
        m = re.search(r"(?:open|launch|start|run)\s+(.+)", low)
        if m:
            return self._open_app(m.group(1).strip().rstrip("."))

        # ── 16. Web search ──────────────────────────────────────────────────
        m = re.search(r"(?:search|look up|google|look for|find)\s+(?:for\s+)?(.+)", low)
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
    def _battery_status(self) -> str:
        output = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
        m = re.search(r"(\d+)%", output)
        if not m:
            return "I couldn't read the battery level."
        level = int(m.group(1))
        if "AC Power" in output:
            status = "and charging" if level < 100 else "and fully charged"
        elif level > 20:
            status = "on battery"
        else:
            status = "on battery, getting low"
        return f"Battery is at {level} percent, {status}."

    def _memory_status(self) -> str:
        """Free/used RAM from vm_stat (page counts) + total from sysctl."""
        try:
            total = int(subprocess.run(["sysctl", "-n", "hw.memsize"],
                                       capture_output=True, text=True).stdout.strip())
            vm = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
            page = int(re.search(r"page size of (\d+) bytes", vm).group(1))
            def pages(name):
                m = re.search(rf"{name}:\s+(\d+)\.", vm)
                return int(m.group(1)) if m else 0
            free = (pages("Pages free") + pages("Pages inactive")) * page
            used = total - free
            gb = 1024 ** 3
            pct = round(used / total * 100)
            return (f"You're using about {used/gb:.1f} of {total/gb:.0f} gigabytes of memory, "
                    f"roughly {pct} percent, with {free/gb:.1f} gigabytes free.")
        except Exception as e:
            log.warning(f"memory stat failed: {e}")
            return "I couldn't read the memory usage."

    def _cpu_status(self) -> str:
        try:
            out = subprocess.run(["top", "-l", "1", "-n", "0"], capture_output=True, text=True).stdout
            m = re.search(r"CPU usage:\s+([\d.]+)%\s+user,\s+([\d.]+)%\s+sys,\s+([\d.]+)%\s+idle", out)
            if not m:
                return "I couldn't read the CPU usage."
            user, sys_, idle = (float(m.group(i)) for i in (1, 2, 3))
            busy = round(user + sys_)
            mood = "mostly idle" if busy < 25 else "working steadily" if busy < 70 else "under heavy load"
            return f"CPU is at about {busy} percent, {mood}. {round(idle)} percent idle."
        except Exception as e:
            log.warning(f"cpu stat failed: {e}")
            return "I couldn't read the CPU usage."

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

    def _open_app(self, name: str) -> str:
        resolved = self._resolve_app(name)
        target = resolved or name.title()
        result = subprocess.run(["open", "-a", target], capture_output=True, text=True)
        if result.returncode == 0:
            return f"Opening {target}."
        return f"I couldn't find an app called {name}."

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
        subprocess.run(["osascript", "-e", f'tell application "{resolved}" to quit'],
                       capture_output=True)
        return f"Closing {resolved}."

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

    def _web_search(self, query: str) -> str:
        url = f"https://www.google.com/search?q={urllib.parse.quote(query)}"
        subprocess.run(["open", url], check=False)
        return f"Searching for {query}."


def _which(cmd: str) -> bool:
    from shutil import which
    return which(cmd) is not None
