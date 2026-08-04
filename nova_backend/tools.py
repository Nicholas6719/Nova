"""
Nova Tools — macOS system tools router.

Fast-path: all responses are deterministic. Never touches the LLM.
macOS-only by design: uses osascript, pmset, mdfind, screencapture, open.

Tool routing order:
  1. App launch / open
  2. Volume (set / up / down / mute / unmute audio)
  3. Battery status
  4. Web search (opens in default browser)
  5. Screenshot
  6. System info

Adding a new tool: implement _handle_<name> and add a match block in match().
"""

from __future__ import annotations

import logging
import re
import subprocess
import urllib.parse
from pathlib import Path
from typing import Optional

log = logging.getLogger("nova.tools")


class NovaTools:
    def __init__(self, config: dict) -> None:
        self.config = config

    def match(self, text: str) -> Optional[str]:
        """Return a response string if the text matches a tool intent, else None."""
        low = text.lower().strip()

        # ── App launch ────────────────────────────────────────────────────────
        m = re.search(r"(?:open|launch|start|run)\s+(.+)", low)
        if m:
            return self._open_app(m.group(1).strip().rstrip("."))

        # ── Volume ────────────────────────────────────────────────────────────
        if "volume" in low:
            m = re.search(r"(?:set|turn|change)?\s*volume\s*to\s*(\d+)", low)
            if m:
                return self._volume_set(int(m.group(1)))
            if "up" in low or "louder" in low or "increase" in low:
                return self._volume_adjust(+15)
            if "down" in low or "quieter" in low or "decrease" in low or "lower" in low:
                return self._volume_adjust(-15)
            if "max" in low or "full" in low or "all the way up" in low:
                return self._volume_set(100)
            if "off" in low or "silent" in low:
                return self._volume_set(0)

        # ── Audio mute / unmute (system audio, distinct from Nova's own mute) ──
        if re.search(r"\b(mute|unmute)\b.*(audio|sound|speakers?)", low):
            return self._mute_audio("mute" in low)

        # ── Battery ───────────────────────────────────────────────────────────
        if any(p in low for p in ("battery", "how much charge", "power level", "how charged")):
            return self._battery_status()

        # ── Web search ────────────────────────────────────────────────────────
        m = re.search(r"(?:search|look up|google|look for|find)\s+(?:for\s+)?(.+)", low)
        if m:
            return self._web_search(m.group(1).strip().rstrip("."))

        # ── Screenshot ────────────────────────────────────────────────────────
        if any(p in low for p in ("take a screenshot", "screenshot", "capture the screen", "grab the screen")):
            return self._screenshot()

        # ── System info ───────────────────────────────────────────────────────
        if any(p in low for p in ("what mac", "what computer", "what machine", "system info")):
            return self._system_info()

        return None

    # ── App launch ────────────────────────────────────────────────────────────────
    def _open_app(self, name: str) -> str:
        # Try -a flag first (searches Applications folders automatically)
        result = subprocess.run(
            ["open", "-a", name.title()],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return f"Opening {name.title()}."

        # Try mdfind for apps not in standard locations
        found = subprocess.run(
            ["mdfind", "-onlyin", "/Applications",
             f"kMDItemContentType == 'com.apple.application-bundle' && kMDItemDisplayName == '{name}'cdw"],
            capture_output=True,
            text=True,
        ).stdout.strip().splitlines()

        if found:
            subprocess.run(["open", found[0]], check=False)
            return f"Opening {name.title()}."

        return f"I couldn't find an app called {name}."

    # ── Volume ────────────────────────────────────────────────────────────────────
    def _volume_set(self, level: int) -> str:
        level = max(0, min(100, level))
        subprocess.run(
            ["osascript", "-e", f"set volume output volume {level}"],
            check=False,
        )
        if level == 0:
            return "Volume off."
        return f"Volume set to {level}."

    def _volume_adjust(self, delta: int) -> str:
        script = f"""
tell application "System Events"
    set cur to output volume of (get volume settings)
    set nv to cur + {delta}
    if nv < 0 then set nv to 0
    if nv > 100 then set nv to 100
    set volume output volume nv
    return nv
end tell"""
        result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
        new_vol = result.stdout.strip()
        direction = "up" if delta > 0 else "down"
        return f"Volume {direction}{f' to {new_vol}' if new_vol.isdigit() else ''}."

    def _mute_audio(self, mute: bool) -> str:
        val = "true" if mute else "false"
        subprocess.run(["osascript", "-e", f"set volume output muted {val}"], check=False)
        return "System audio muted." if mute else "System audio unmuted."

    # ── Battery ───────────────────────────────────────────────────────────────────
    def _battery_status(self) -> str:
        result = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True)
        output = result.stdout

        m = re.search(r"(\d+)%", output)
        if not m:
            return "I couldn't read the battery level."

        level    = int(m.group(1))
        charging = "AC Power" in output
        if charging:
            status = "and charging"
        elif level > 20:
            status = "on battery"
        else:
            status = "on battery — getting low"

        return f"Battery is at {level} percent, {status}."

    # ── Web search ────────────────────────────────────────────────────────────────
    def _web_search(self, query: str) -> str:
        url = f"https://www.google.com/search?q={urllib.parse.quote(query)}"
        subprocess.run(["open", url], check=False)
        return f"Searching for {query}."

    # ── Screenshot ────────────────────────────────────────────────────────────────
    def _screenshot(self) -> str:
        import time
        dest = Path.home() / "Desktop" / f"screenshot_{int(time.time())}.png"
        subprocess.run(["screencapture", "-x", str(dest)], check=False)
        return "Screenshot saved to your Desktop."

    # ── System info ───────────────────────────────────────────────────────────────
    def _system_info(self) -> str:
        result = subprocess.run(
            ["system_profiler", "SPHardwareDataType"],
            capture_output=True,
            text=True,
        )
        m = re.search(r"Model Name:\s+(.+)", result.stdout)
        chip = re.search(r"Chip:\s+(.+)", result.stdout)
        if m:
            model = m.group(1).strip()
            detail = f" with {chip.group(1).strip()}" if chip else ""
            return f"You're on a {model}{detail}."
        return "I couldn't determine your Mac model."
