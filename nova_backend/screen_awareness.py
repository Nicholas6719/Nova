"""
Nova Screen Awareness — "what's on my screen?"

Answers from GROUND TRUTH, not from a guess. Three local sources, combined:

  1. The frontmost application      (NSWorkspace)
  2. Real on-screen window titles   (CGWindowListCopyWindowInfo)
  3. The text actually on screen    (Apple's Vision framework OCR)

Nova's LLM only ever SUMMARIZES those facts; it is never asked to look at an
image and imagine what might be there.

Why not a vision model
──────────────────────
The obvious build is a small local VLM (Moondream), which is what the Jarvis
project does. Measured on this machine against a real screenshot, the 0.5B
produced filler ("the user is likely a researcher") and the 2B invented
hardware that was not on screen ("a web browser, a keyboard, and a mouse").
Nova's first principle is never guess, never hallucinate, so that was
disqualifying. Moondream is also dependency-incompatible with Nova: it pins
huggingface-hub and tokenizers to versions mlx_lm cannot import, so it could
only ever run in a separate venv behind a subprocess.

Native OCR wins on every axis that matters here: it reads real text, needs no
model download, adds no runtime, and cannot fabricate. Its one weakness is
imagery — it reads a screen, it does not describe a photograph. Describing
actual pictures is a separate capability and is deliberately NOT claimed here.

The permission trap
───────────────────
`screencapture` does NOT fail when Screen Recording permission is missing. It
writes a perfectly valid PNG containing only the desktop wallpaper and the
menu bar. Nova would then confidently describe an empty desktop while the
user is staring at a full screen of windows — a false answer that looks like
a real one. So permission is ALWAYS preflighted via
CGPreflightScreenCaptureAccess before a capture is trusted, and Nova declines
honestly when it is missing. Never infer permission from "the PNG exists".

Per project_macos_permissions: a headless python run proves nothing here.
This interpreter already holds the grant; Nova.app is a separate TCC identity
and must be granted on its own.

Privacy contract
────────────────
  * The screenshot is written to a temp path, read once, and deleted in a
    `finally` — including on every error path.
  * Its CONTENTS are never logged (only line counts), never written to the
    memory database, never added to conversation history, and never leave the
    machine. Vision OCR is fully on-device.
  * Nothing here touches the network. There is no cloud fallback to fall into.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
import re
import subprocess
import tempfile
from typing import Callable, Optional

log = logging.getLogger("nova.screen")


# ═══════════════════════════════════════════════════════════════════════════
# Screen Recording permission
# ═══════════════════════════════════════════════════════════════════════════
def _core_graphics():
    lib = ctypes.util.find_library("CoreGraphics")
    return ctypes.CDLL(lib) if lib else None


def has_permission() -> bool:
    """True if this process may capture other applications' windows.

    Preflight only — never prompts. Returns False rather than raising when the
    symbol is unavailable, so an older macOS degrades to an honest decline.
    """
    cg = _core_graphics()
    if cg is None or not hasattr(cg, "CGPreflightScreenCaptureAccess"):
        return False
    try:
        cg.CGPreflightScreenCaptureAccess.restype = ctypes.c_bool
        return bool(cg.CGPreflightScreenCaptureAccess())
    except Exception as exc:
        log.warning(f"preflight failed: {exc}")
        return False


def request_permission() -> bool:
    """Ask macOS to show the Screen Recording prompt (once per app identity).

    macOS only shows this for an app that has not yet been answered; after a
    denial it does nothing and the user must go to System Settings. Returns
    the immediate result, which is normally False the first time — the grant
    does not take effect until the app is relaunched.
    """
    cg = _core_graphics()
    if cg is None or not hasattr(cg, "CGRequestScreenCaptureAccess"):
        return False
    try:
        cg.CGRequestScreenCaptureAccess.restype = ctypes.c_bool
        return bool(cg.CGRequestScreenCaptureAccess())
    except Exception as exc:
        log.warning(f"permission request failed: {exc}")
        return False


PERMISSION_HELP = (
    "I don't have permission to see the screen yet. You can grant it in "
    "System Settings, under Privacy and Security, then Screen Recording. "
    "Turn on Nova there and relaunch me, and I'll be able to read the screen."
)


# ═══════════════════════════════════════════════════════════════════════════
# Capture
# ═══════════════════════════════════════════════════════════════════════════
def capture_screen() -> Optional[str]:
    """Capture the main display silently. Returns a temp path, or None.

    `-x` suppresses the shutter sound. The caller MUST call discard() when
    done, even on failure.
    """
    fd, path = tempfile.mkstemp(prefix="nova_screen_", suffix=".png")
    os.close(fd)
    try:
        os.remove(path)  # so a failed capture can't be mistaken for a success
    except OSError:
        pass
    try:
        subprocess.run(["screencapture", "-x", path], check=False, timeout=15)
    except Exception as exc:
        log.warning(f"screencapture failed: {exc}")
        discard(path)
        return None
    if not os.path.isfile(path) or os.path.getsize(path) < 1024:
        discard(path)
        return None
    return path


def discard(path: Optional[str]) -> None:
    """Delete a captured screenshot. Always safe to call; never raises.

    Defensive: only removes files this module created, so a bad caller can
    never turn cleanup into data loss.
    """
    if not path:
        return
    try:
        if os.path.basename(path).startswith("nova_screen_") and os.path.isfile(path):
            os.remove(path)
    except Exception as exc:
        log.warning(f"screenshot cleanup failed: {exc}")


# ═══════════════════════════════════════════════════════════════════════════
# Window structure — deterministic, always correct, no model involved
# ═══════════════════════════════════════════════════════════════════════════
def frontmost_app() -> Optional[str]:
    """The app the user is actually working in.

    A transient system overlay can hold focus for a moment — a notification
    banner made Nova answer "you're in UserNotificationCenter", which is
    literally true and completely useless. When focus is on system chrome we
    fall back to the frontmost REAL window instead.
    """
    name = None
    try:
        from AppKit import NSWorkspace
        app = NSWorkspace.sharedWorkspace().frontmostApplication()
        name = str(app.localizedName()) if app else None
    except Exception as exc:
        log.warning(f"frontmost app lookup failed: {exc}")

    if name and name not in _CHROME_OWNERS:
        return name
    for owner, _ in visible_windows():
        return owner
    return name


# Owners that are always on screen but are never what the user means.
_CHROME_OWNERS = {
    "Window Server", "Dock", "SystemUIServer", "Control Center",
    "Notification Center", "NotificationCenter", "UserNotificationCenter",
    "Spotlight", "TextInputMenuAgent", "universalaccessd", "loginwindow",
    "Screenshot", "CoreServicesUIAgent", "AirPlayUIAgent",
}


def visible_windows() -> list[tuple[str, str]]:
    """(owner, title) for real application windows, front to back.

    Layer 0 only: that filters out the menu bar, Dock, and status items, which
    are on screen but are not what "what's on my screen" means.
    """
    try:
        from Quartz import (CGWindowListCopyWindowInfo,
                            kCGWindowListOptionOnScreenOnly, kCGNullWindowID)
        raw = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly,
                                         kCGNullWindowID) or []
    except Exception as exc:
        log.warning(f"window list failed: {exc}")
        return []

    out: list[tuple[str, str]] = []
    for w in raw:
        try:
            if w.get("kCGWindowLayer", 99) != 0:
                continue
            owner = str(w.get("kCGWindowOwnerName") or "").strip()
            if not owner or owner in _CHROME_OWNERS:
                continue
            bounds = w.get("kCGWindowBounds") or {}
            # Skip slivers — tooltips, shadows, 1px helper windows.
            if int(bounds.get("Width", 0)) < 120 or int(bounds.get("Height", 0)) < 90:
                continue
            title = str(w.get("kCGWindowName") or "").strip()
            out.append((owner, title))
        except Exception:
            continue
    return out


# ═══════════════════════════════════════════════════════════════════════════
# OCR — Apple Vision, fully on-device
# ═══════════════════════════════════════════════════════════════════════════
def ocr(path: str, min_confidence: float = 0.35,
        max_lines: int = 80) -> list[str]:
    """Recognized text lines in natural reading order (top to bottom).

    Vision reports normalized coordinates with the origin at the BOTTOM left,
    so reading order is descending y. Low-confidence fragments are dropped —
    they are usually icon glyphs, and passing them on invites the summarizer
    to treat noise as content.
    """
    try:
        import Vision
        from Foundation import NSURL
    except Exception as exc:
        log.warning(f"Vision framework unavailable: {exc}")
        return []

    try:
        url = NSURL.fileURLWithPath_(path)
        handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(url, None)
        request = Vision.VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(0)          # 0 = accurate
        request.setUsesLanguageCorrection_(True)
        handler.performRequests_error_([request], None)
        results = request.results() or []
    except Exception as exc:
        log.warning(f"OCR failed: {exc}")
        return []

    rows: list[tuple[float, float, str]] = []
    for obs in results:
        try:
            cands = obs.topCandidates_(1)
            if not cands or not cands.count():
                continue
            best = cands.objectAtIndex_(0)
            if float(best.confidence()) < min_confidence:
                continue
            text = str(best.string()).strip()
            if not text:
                continue
            box = obs.boundingBox()
            rows.append((float(box.origin.y), float(box.origin.x), text))
        except Exception:
            continue

    rows.sort(key=lambda r: (-r[0], r[1]))

    seen: set[str] = set()
    lines: list[str] = []
    for _, _, text in rows:
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        lines.append(text)
        if len(lines) >= max_lines:
            break
    return lines


# ═══════════════════════════════════════════════════════════════════════════
# Natural-language dispatch
# ═══════════════════════════════════════════════════════════════════════════
# Strict by design. These must not fire on ordinary conversation, and they run
# BEFORE the file stage — otherwise "read my screen" is a file search for a
# document named "screen".
_DESCRIBE_RE = re.compile(
    r"\b(?:"
    r"what(?:'?s| is)\s+(?:on|in)\s+(?:my|the)\s+screen"
    r"|what\s+am\s+i\s+(?:looking\s+at|seeing)"
    r"|(?:describe|read|check|look\s+at)\s+(?:my|the)\s+screen"
    r"|what\s+do\s+you\s+see"
    r"|what(?:'?s| is)\s+(?:this|that)\s+on\s+(?:my|the)\s+screen"
    r"|(?:can\s+you\s+)?see\s+(?:my|the)\s+screen"
    r")\b",
    re.I,
)

_CONTEXT_RE = re.compile(
    r"\b(?:"
    r"what\s+app\s+am\s+i\s+(?:in|using|on)"
    r"|what(?:'?s| is)\s+(?:the\s+)?(?:frontmost|active|current)\s+(?:app|application|window)"
    r"|what\s+window\s+(?:is\s+)?(?:this|am\s+i\s+in)"
    r"|what\s+am\s+i\s+(?:working\s+on|doing)\s+right\s+now"
    r"|what(?:'?s| is)\s+open\s+(?:on\s+)?(?:my|the)\s+(?:screen|mac|desktop)"
    r")\b",
    re.I,
)


def _clean_spoken(text: str) -> str:
    """Strip anything the LLM added that Nova would read aloud as punctuation
    (CLAUDE.md invariant 10: no markdown, no lists, no em dashes)."""
    text = re.sub(r"[*_`#]+", "", text)
    text = re.sub(r"^\s*[-•]\s*", "", text, flags=re.M)
    text = re.sub(r"^\s*\d+[.)]\s*", "", text, flags=re.M)
    text = text.replace("—", ", ").replace("–", ", ")
    text = re.sub(r"\s+-\s+", ", ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _spoken_list(items: list[str]) -> str:
    items = [i for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


class NovaScreen:
    """Dispatch for screen questions.

    Runs on the ``nova-llm`` worker thread (like the calendar and file layers),
    so the ``self.llm.generate`` call here is thread-safe.
    """

    _SYSTEM = (
        "You are Nova, telling Nicholas what is on his screen. You are given "
        "the real window list and the real text read off the screen. Describe "
        "ONLY what is in that data. Never invent windows, applications, "
        "objects, or people. Never mention a keyboard, mouse, or webcam. "
        "Speak DIRECTLY to him as 'you' — never say 'the user' and never "
        "describe him in the third person. Each entry in the window list is a "
        "SEPARATE application; never describe one app's window as being inside "
        "another. Be brief: one short conversational sentence is ideal, two "
        "at most. No lists, no markdown."
    )

    def __init__(self, config: dict, llm) -> None:
        self.config = config
        self.llm = llm
        scr = (config or {}).get("screen", {}) or {}
        self.enabled: bool = bool(scr.get("enabled", True))
        self.min_confidence: float = float(scr.get("min_confidence", 0.35))
        self.max_ocr_lines: int = int(scr.get("max_ocr_lines", 80))
        self.max_tokens: int = int(scr.get("max_tokens", 180))
        # Set when Nova offers a follow-up; nova.py reuses its one-shot slot.
        self.pending_offer: Optional[Callable[[], str]] = None

    # ── Detection ────────────────────────────────────────────────────────
    def detect_intent(self, text: str) -> Optional[str]:
        if not self.enabled:
            return None
        t = (text or "").strip()
        if not t:
            return None
        if _CONTEXT_RE.search(t):
            return "context"
        if _DESCRIBE_RE.search(t):
            return "describe"
        return None

    # ── Handling ─────────────────────────────────────────────────────────
    def handle(self, intent: str, text: str) -> str:
        self.pending_offer = None
        try:
            if intent == "context":
                return self._context()
            return self._describe()
        except Exception as exc:
            log.exception(f"screen handling failed: {exc}")
            return "Something went wrong trying to look at the screen."

    def _context(self) -> str:
        """Which app / window — deterministic, instant, no capture, no LLM.

        Deliberately needs NO screen-recording permission: window titles come
        from the window server, not from a screenshot.
        """
        front = frontmost_app()
        windows = visible_windows()
        if not front and not windows:
            return "I can't tell what's open right now."

        title = ""
        for owner, wtitle in windows:
            if owner == front and wtitle:
                title = wtitle
                break

        others = []
        for owner, _ in windows:
            if owner != front and owner not in others:
                others.append(owner)

        if front and title:
            lead = f"You're in {front}, on {title}."
        elif front:
            lead = f"You're in {front}."
        else:
            lead = "I can't tell which app is in front."

        if others:
            return f"{lead} You've also got {_spoken_list(others[:4])} open."
        return lead

    def _describe(self) -> str:
        """Capture, OCR, summarize. The screenshot never outlives this call."""
        if not has_permission():
            # Prompt once — harmless if already answered, and it is the only
            # way the system dialog ever appears for this app identity.
            request_permission()
            return PERMISSION_HELP

        path = capture_screen()
        if not path:
            return "I wasn't able to capture the screen just now."

        try:
            lines = ocr(path, min_confidence=self.min_confidence,
                        max_lines=self.max_ocr_lines)
        finally:
            # Always — including on an OCR exception. The image never persists.
            discard(path)

        windows = visible_windows()
        front = frontmost_app()

        if not lines and not windows:
            return "I captured the screen but couldn't make anything out on it."

        # Log counts only. Never the contents.
        log.info(f"[screen] {len(windows)} windows, {len(lines)} text lines")

        # WHICH app and WHICH window are facts we already hold exactly, so they
        # are stated deterministically and never left to the model. Measured:
        # the 3B reliably attributed one app's window title to another app
        # ("Claude ... has a window titled Coding", where Coding was a Notes
        # window), and no amount of prompt wording fixed it. The LLM's only job
        # is to characterize the TEXT — it never sees the window list.
        lead = self._lead_sentence(front, windows)

        text_block = "\n".join(lines)
        if not text_block:
            return lead

        prompt = (
            f"TEXT READ OFF THE SCREEN:\n{text_block}\n\n"
            "In ONE short sentence, say what this content is about. Start "
            "directly with the subject, not with 'this text' or 'the screen "
            "shows'. Describe only what the text shows. Do not name windows or "
            "applications unless the name appears in the text itself. Do not "
            "mention the menu bar, the clock, the battery, or the weather."
        )

        try:
            summary = self.llm.generate(self._SYSTEM, [], prompt,
                                        temperature=0.0,
                                        max_tokens=self.max_tokens).strip()
        except Exception as exc:
            log.warning(f"screen summary failed: {exc}")
            summary = ""

        if not summary:
            return self._fallback(front, windows, lines)
        return f"{lead} {_clean_spoken(summary)}"

    @staticmethod
    def _lead_sentence(front: Optional[str],
                       windows: list[tuple[str, str]]) -> str:
        """Deterministic 'where you are' sentence, built purely from the
        window server. Cannot be wrong and cannot be invented."""
        title = ""
        for owner, wtitle in windows:
            if owner == front and wtitle and wtitle != owner:
                title = wtitle
                break
        if front and title:
            return f"You're in {front}, on {title}."
        if front:
            return f"You're in {front}."
        return "Here's what's on your screen."

    @staticmethod
    def _fallback(front: Optional[str], windows: list[tuple[str, str]],
                  lines: list[str]) -> str:
        """Deterministic answer when generation fails. Says less, invents
        nothing — the same contract as the calendar and file fallbacks."""
        title = ""
        for owner, wtitle in windows:
            if owner == front and wtitle:
                title = wtitle
                break
        if front and title:
            lead = f"You're looking at {front}, on {title}."
        elif front:
            lead = f"You're looking at {front}."
        else:
            lead = "I can see your screen."
        if lines:
            return f"{lead} I can read about {len(lines)} lines of text on it, but I wasn't able to summarize them."
        return f"{lead} I wasn't able to summarize it."
