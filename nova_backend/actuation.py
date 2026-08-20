"""
Actuation — Nova types and clicks, on text she can actually see.

This is the part RileyJarvis fakes. Its `computer_click` takes raw x,y from the
model, and its screenshot tool returns a FILE PATH that is stripped before the
model ever sees it, so the model is guessing coordinates from a three-line text
summary. It clicks blind.

Nova does not have to. `screen_awareness.ocr()` already runs Apple Vision, and
Vision returns a boundingBox with every observation — the existing code uses it
for sort order and then throws the geometry away. So Nova can be told "click
Send" and click the pixels where the word "Send" actually is.

TWO WAYS TO FIND A TARGET, fast one first:

  1. Accessibility (AXUIElement). Real controls with real positions, as
     structured data. ~10-50ms, no image, and it needs Accessibility
     permission but NOT Screen Recording — so the fast path is also the less
     invasive one.
  2. Window-only screenshot + Vision OCR. ~300-800ms. The fallback for
     Electron apps and canvases that expose nothing useful over AX.

SAFETY, in the order it matters:

  * Every one of these is refused outside WORK MODE. A hard gate in code, not
    a line in a prompt — the mistake RileyJarvis makes is letting the MODEL
    decide whether an action needs confirming (`requiresConfirmation` returns
    False whenever the model omits a `risk` field).
  * Typing, scrolling and reading are free. Clicking something that sends,
    submits, deletes, buys or changes settings is CONFIRMED, by matching the
    label Nova is about to click.
  * ENTER IS AMBIGUOUS. In Messages and Mail, Return IS send. So Return is
    confirmed when the frontmost app is a messaging or mail app, and free
    everywhere else. This was Nicholas's own example and it is the sharpest
    edge in the module.
  * Nothing here raises. A failure returns a spoken sentence.

Nova never clicks a coordinate she was not able to verify by name.
"""

from __future__ import annotations

import ctypes
import logging
import re
import subprocess
import time
from typing import Optional

log = logging.getLogger("nova.actuation")

# Apps where Return sends something to another person.
_SEND_ON_RETURN = frozenset({
    "Messages", "Mail", "Slack", "Discord", "WhatsApp", "Telegram",
    "Microsoft Teams", "Signal",
})

# A control whose label means the action leaves, deletes, or costs.
_DESTRUCTIVE_LABEL_RE = re.compile(
    r"\b(?:send|submit|post|publish|share|reply|forward|"
    r"delete|remove|trash|discard|erase|clear|"
    r"buy|purchase|order|pay|checkout|confirm|"
    r"sign\s*out|log\s*out|unsubscribe|deactivate)\b",
    re.I,
)


# ── Permission ────────────────────────────────────────────────────────────────
def has_accessibility() -> bool:
    """Whether THIS process may drive the UI.

    Preflighted rather than attempted, for the same reason screen_awareness
    preflights Screen Recording: without the grant, AppleScript UI scripting
    fails with an error that reads like a bug rather than a permission problem.
    """
    try:
        from ApplicationServices import AXIsProcessTrusted
        return bool(AXIsProcessTrusted())
    except Exception:
        try:
            lib = ctypes.cdll.LoadLibrary(
                "/System/Library/Frameworks/ApplicationServices.framework/"
                "ApplicationServices")
            lib.AXIsProcessTrusted.restype = ctypes.c_bool
            return bool(lib.AXIsProcessTrusted())
        except Exception:
            return False


def _osa(script: str, timeout: float = 10.0) -> tuple[bool, str]:
    try:
        r = subprocess.run(["osascript", "-e", script],
                           capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0, (r.stdout or r.stderr).strip()
    except Exception as exc:
        return False, str(exc)[:120]


def frontmost_app() -> str:
    ok, out = _osa('tell application "System Events" to return name of '
                   'first application process whose frontmost is true')
    return out if ok else ""


# ── Finding a target ──────────────────────────────────────────────────────────
class Target:
    """Something on screen Nova can click, and how she found it."""

    __slots__ = ("label", "x", "y", "source")

    def __init__(self, label: str, x: float, y: float, source: str) -> None:
        self.label = label
        self.x = x
        self.y = y
        self.source = source        # "accessibility" or "ocr"

    def __repr__(self) -> str:
        return f"<Target {self.label!r} at ({self.x:.0f},{self.y:.0f}) via {self.source}>"


# Words that describe WHAT KIND of thing it is, not which one. He says "click
# the File menu"; the control is named "File". Left in, they turn every exact
# match into a miss and every miss into a substring gamble.
_LABEL_NOISE = re.compile(
    r"^\s*(?:the|a|an|that|this)\s+|"
    r"\s+(?:menu|button|tab|field|box|icon|link|item|option|control)\s*$",
    re.IGNORECASE,
)


def _normalise_label(label: str) -> str:
    """"the File menu" -> "file". Applied until it stops shrinking, so
    "the send button" loses both ends."""
    t = (label or "").strip()
    while True:
        stripped = _LABEL_NOISE.sub("", t).strip()
        if stripped == t:
            break
        t = stripped
    return t.lower()


def find_by_accessibility(label: str) -> Optional[Target]:
    """A real control with that name in the frontmost app.

    Matched on AXTitle/AXDescription, which is what a person reads off the
    button — and unlike OCR it cannot be fooled by the same word appearing in
    body text.
    """
    esc = label.replace('"', '\\"')
    script = f'''
    tell application "System Events"
      set frontApp to first application process whose frontmost is true
      try
        set hits to (every UI element of front window of frontApp whose ¬
          (name is "{esc}" or description is "{esc}"))
      on error
        return "NONE"
      end try
      if (count of hits) is 0 then return "NONE"
      set el to item 1 of hits
      set p to position of el
      set s to size of el
      return ((item 1 of p) + (item 1 of s) / 2) & "," & ¬
             ((item 2 of p) + (item 2 of s) / 2)
    end tell'''
    ok, out = _osa(script, timeout=8)
    if not ok or out == "NONE":
        return None
    nums = re.findall(r"-?\d+(?:\.\d+)?", out)
    if len(nums) < 2:
        return None
    return Target(label, float(nums[0]), float(nums[1]), "accessibility")


def find_by_ocr(label: str) -> Optional[Target]:
    """The fallback: read the frontmost WINDOW and locate the words.

    Deliberately the focused window rather than the whole screen — a Messages
    window is a fraction of a 5K display, and OCR cost scales with pixels.
    """
    import screen_awareness as sa

    if not sa.has_permission():
        return None

    bounds = _front_window_bounds()
    if bounds is None:
        return None
    wx, wy, ww, wh = bounds

    path = _capture_front_window()
    if not path:
        return None
    try:
        boxes = ocr_with_boxes(path)
    finally:
        # Same discipline as screen_awareness: the image never outlives the
        # call, on any path.
        _delete(path)

    wanted = _normalise_label(label)
    if not wanted:
        return None

    # Exact first, then WHOLE-WORD. Never a bare substring: "the File menu"
    # matched a folder called Coding_Files, because "file" is inside it. That
    # is the routing bug all over again — a match on a fragment rather than on
    # the thing named — except here the consequence is a click he did not ask
    # for, on something he did not name.
    exact, worded = [], []
    pattern = re.compile(rf"(?<![\w]){re.escape(wanted)}(?![\w])")
    for box in boxes:
        t = box[0].strip().lower()
        if t == wanted:
            exact.append(box)
        elif pattern.search(t):
            worded.append(box)

    hits = exact or worded
    if not hits:
        return None
    if len(hits) > 1 and not exact:
        # Several things on screen could be it. Guessing which one is exactly
        # the behaviour the "found by name or not at all" rule exists to
        # prevent, so this is a miss and the caller asks him.
        log.info(f"actuation: {label!r} is ambiguous ({len(hits)} matches)")
        return None
    best = hits[0]

    _, bx, by, bw, bh = best
    # Vision is normalized with the origin at the BOTTOM left; the screen's
    # origin is top left. Getting this flip wrong clicks the mirror image of
    # the thing you meant, which is the worst possible failure here.
    cx = wx + (bx + bw / 2) * ww
    cy = wy + (1.0 - (by + bh / 2)) * wh
    return Target(best[0], cx, cy, "ocr")


def ocr_with_boxes(path: str, min_confidence: float = 0.35
                   ) -> list[tuple[str, float, float, float, float]]:
    """(text, x, y, width, height) per observation, normalized 0-1.

    screen_awareness.ocr() computes exactly this and then discards the
    geometry, because it only ever needed reading order. Clicking needs the
    boxes, so this keeps them.
    """
    try:
        import Vision
        from Foundation import NSURL
    except Exception as exc:
        log.warning(f"Vision unavailable: {exc}")
        return []
    try:
        url = NSURL.fileURLWithPath_(path)
        handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(url, None)
        request = Vision.VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(0)
        request.setUsesLanguageCorrection_(True)
        handler.performRequests_error_([request], None)
        results = request.results() or []
    except Exception as exc:
        log.warning(f"OCR failed: {exc}")
        return []

    out = []
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
            b = obs.boundingBox()
            out.append((text, float(b.origin.x), float(b.origin.y),
                        float(b.size.width), float(b.size.height)))
        except Exception:
            continue
    return out


def _front_window_bounds() -> Optional[tuple[float, float, float, float]]:
    try:
        from Quartz import (CGWindowListCopyWindowInfo, kCGNullWindowID,
                            kCGWindowListOptionOnScreenOnly)
    except Exception:
        return None
    app = frontmost_app()
    if not app:
        return None
    try:
        for w in CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly,
                                            kCGNullWindowID) or []:
            if w.get("kCGWindowOwnerName") != app:
                continue
            b = w.get("kCGWindowBounds") or {}
            if b.get("Width", 0) > 100 and b.get("Height", 0) > 100:
                return (float(b["X"]), float(b["Y"]),
                        float(b["Width"]), float(b["Height"]))
    except Exception as exc:
        log.warning(f"window bounds failed: {exc}")
    return None


def _capture_front_window() -> Optional[str]:
    try:
        from Quartz import (CGWindowListCopyWindowInfo, kCGNullWindowID,
                            kCGWindowListOptionOnScreenOnly)
    except Exception:
        return None
    app = frontmost_app()
    wid = None
    try:
        for w in CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly,
                                            kCGNullWindowID) or []:
            b = w.get("kCGWindowBounds") or {}
            if (w.get("kCGWindowOwnerName") == app
                    and b.get("Width", 0) > 100 and b.get("Height", 0) > 100):
                wid = w.get("kCGWindowNumber")
                break
    except Exception:
        return None
    if wid is None:
        return None

    path = f"/tmp/nova_act_{int(time.time()*1000)}.png"
    try:
        subprocess.run(["screencapture", "-x", "-o", f"-l{wid}", path],
                       check=False, timeout=15)
        return path
    except Exception as exc:
        log.warning(f"window capture failed: {exc}")
        return None


def _delete(path: str) -> None:
    try:
        import os
        os.unlink(path)
    except Exception:
        pass


def find(label: str) -> Optional[Target]:
    """Accessibility first, OCR second. Never a guess."""
    return find_by_accessibility(label) or find_by_ocr(label)


# ── Doing things ──────────────────────────────────────────────────────────────
def needs_confirmation(label: str) -> bool:
    """Whether clicking this control should be confirmed first."""
    return bool(_DESTRUCTIVE_LABEL_RE.search(label or ""))


def return_sends() -> bool:
    """True when Return in the frontmost app sends something to a person.

    His example: in Messages, Enter IS send. So "press enter" is a harmless
    keystroke in a text editor and an irreversible one in a chat window, and
    Nova cannot treat them the same.
    """
    return frontmost_app() in _SEND_ON_RETURN


def click(target: Target) -> bool:
    ok, _ = _osa(f'tell application "System Events" to click at '
                 f'{{{int(target.x)}, {int(target.y)}}}', timeout=10)
    return ok


def type_text(text: str) -> bool:
    # Typing is free by design — he asked for that explicitly. What is typed is
    # visible before it goes anywhere, and SENDING is the gated step.
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    ok, _ = _osa('tell application "System Events" to keystroke '
                 f'"{escaped}"', timeout=15)
    return ok


_KEY_CODES = {"return": 36, "enter": 36, "tab": 48, "escape": 53, "esc": 53,
              "delete": 51, "space": 49, "up": 126, "down": 125,
              "left": 123, "right": 124}


def press_key(key: str, repeat: int = 1) -> bool:
    code = _KEY_CODES.get((key or "").lower())
    if code is None:
        return False
    repeat = max(1, min(20, int(repeat)))
    ok, _ = _osa(f'tell application "System Events" to repeat {repeat} times\n'
                 f'key code {code}\nend repeat', timeout=15)
    return ok


def scroll(direction: str, amount: int = 4) -> bool:
    code = _KEY_CODES.get({"up": "up", "down": "down",
                           "left": "left", "right": "right"}
                          .get((direction or "down").lower(), "down"))
    if code is None:
        return False
    amount = max(1, min(20, int(amount)))
    ok, _ = _osa(f'tell application "System Events" to repeat {amount} times\n'
                 f'key code {code}\nend repeat', timeout=15)
    return ok
