#!/usr/bin/env python3
"""
Actuation — Nova types and clicks, and refuses to do either carelessly.

The riskiest code in Nova, so this suite is mostly NEGATIVE checks. The happy
path is one line; the gates are the product.

What it proves, with the real modules and only the OS calls stubbed:
  * nothing happens outside WORK MODE — and, crucially, actuation does not
    SHADOW handlers that already work when she is not in it
  * a target that cannot be found by name is refused, never clicked at a
    guessed coordinate
  * send / delete / buy are confirmed before the click
  * RETURN is confirmed in Messages and free in TextEdit — his own example,
    and the sharpest edge in the module
  * anything that is not a clear yes cancels
  * ordinary speech containing type/press/scroll never moves his mouse

Nothing here clicks anything: `actuation._osa` is replaced, so every regex,
gate and decision runs for real while the machine is untouched.

Run:  python tests/test_actuation.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
BACKEND = TESTS_DIR.parent
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(TESTS_DIR))
os.chdir(BACKEND)

from listener import check_spoken  # noqa: E402

PASS = FAIL = 0
FAILURES: list[str] = []


def check(cond, label, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        FAILURES.append(f"{label}  ({detail})" if detail else label)
    return bool(cond)


class Fake:
    """A VoiceAssistant stand-in with only what actuation reads."""

    def __init__(self, work_mode: bool = True):
        self.work_mode = work_mode


def build(work_mode=True, front="TextEdit", findable=("Send", "OK")):
    """Real NovaActuation; only the calls that touch the Mac are replaced."""
    import actuation as A
    import nova as nova_mod
    from actuation_intents import NovaActuation

    A._did = []                       # what WOULD have happened
    A.has_accessibility = lambda: True
    A.frontmost_app = lambda: front
    A.click = lambda t: (A._did.append(("click", t.label)), True)[1]
    A.type_text = lambda s: (A._did.append(("type", s)), True)[1]
    A.press_key = lambda k, repeat=1: (A._did.append(("key", k)), True)[1]
    A.scroll = lambda d, amount=4: (A._did.append(("scroll", d)), True)[1]
    A.find = lambda label: (A.Target(label, 100, 200, "accessibility")
                            if label in findable else None)
    return NovaActuation(nova_mod.load_config(), assistant=Fake(work_mode)), A


# ── 1. Work mode is the gate ──────────────────────────────────────────────────
def test_work_mode_gate() -> None:
    print("\n1. WORK MODE IS THE GATE")
    act, A = build(work_mode=False)

    for phrase in ("type hello there", "click the Send button", "press enter",
                   "scroll down", "send it"):
        check(act.detect_intent(phrase) is None,
              f"outside work mode, '{phrase}' is not claimed",
              str(act.detect_intent(phrase)))
    check(not A._did, "and nothing was done", str(A._did))

    # The reason the gate is at DETECTION and not only in handle(): claiming
    # these outside work mode would have replaced the browser scroll that
    # already works with "say work with me first".
    import nova as nova_mod
    import tools as T

    class R:
        returncode, stdout, stderr = 0, "", ""
    saved_run, saved_sleep = T.subprocess.run, T.time.sleep
    T.subprocess.run = lambda *a, **k: R()
    T.time.sleep = lambda *a, **k: None
    try:
        tools = T.NovaTools(nova_mod.load_config())
        check(tools.match("scroll down") is not None,
              "'scroll down' still reaches the browser handler outside work mode")
    finally:
        T.subprocess.run, T.time.sleep = saved_run, saved_sleep

    act_on, _ = build(work_mode=True)
    check(act_on.detect_intent("type hello there") == "type",
          "and inside work mode it IS claimed")


# ── 2. Never clicks what it cannot find ───────────────────────────────────────
def test_never_guesses() -> None:
    print("\n2. NEVER CLICKS A GUESS")
    act, A = build(findable=("Send",))

    out = act.handle("click", "click the Reply button")
    check(not A._did, "an unfindable target is NOT clicked", str(A._did))
    check("can't find" in out.lower(), "and Nova says so", out)
    check(not check_spoken(out), "fit to speak", out)

    # Findable but destructive: confirmed, still not clicked yet.
    A._did.clear()
    out = act.handle("click", "click the Send button")
    check(not A._did, "a destructive click waits for a yes", str(A._did))
    check("want me to" in out.lower(), "and asks", out)
    check(act.pending is not None, "the click is held pending")

    out = act.resolve_pending("yes")
    check(("click", "Send") in A._did, "…and happens on a yes", str(A._did))
    check(act.pending is None, "the pending slot is cleared")


# ── 3. Return is the ambiguous key ────────────────────────────────────────────
def test_return_ambiguity() -> None:
    print("\n3. RETURN: A KEYSTROKE OR AN IRREVERSIBLE ACTION")

    # In Messages, Return IS send. His example.
    act, A = build(front="Messages")
    out = act.handle("key", "press enter")
    check(not A._did, "in Messages, Return is NOT pressed straight away",
          str(A._did))
    check("send" in out.lower() and "want me to" in out.lower(),
          "…Nova says it would send, and asks", out)
    act.resolve_pending("yes")
    check(("key", "return") in A._did, "…and presses it on a yes", str(A._did))

    # In a text editor it is just a newline.
    act2, A2 = build(front="TextEdit")
    out2 = act2.handle("key", "press enter")
    check(("key", "return") in A2._did, "in TextEdit, Return is just pressed",
          str(A2._did))
    check(act2.pending is None, "…with nothing to confirm")

    import actuation as A3
    A3.frontmost_app = lambda: "Mail"
    check(A3.return_sends(), "Mail counts as send-on-return")
    A3.frontmost_app = lambda: "Xcode"
    check(not A3.return_sends(), "Xcode does not")


# ── 4. Typing is free; sending is not ─────────────────────────────────────────
def test_typing_free() -> None:
    print("\n4. TYPING IS FREE, SENDING IS NOT")
    act, A = build()

    out = act.handle("type", 'say "okay thank you see you soon"')
    check(("type", "okay thank you see you soon") in A._did,
          "his exact example types the quoted words", str(A._did))
    check("send" in out.lower(), "…and offers to send, rather than sending", out)
    check(not check_spoken(out), "fit to speak", out)

    A._did.clear()
    act.handle("type", "type hello there")
    check(("type", "hello there") in A._did, "unquoted typing works too",
          str(A._did))


# ── 5. Anything that is not a yes cancels ─────────────────────────────────────
def test_cancel() -> None:
    print("\n5. NOT-A-YES CANCELS")
    for reply in ("no", "wait", "actually never mind", "hold on",
                  "what time is it"):
        act, A = build()
        act.handle("click", "click the Send button")
        out = act.resolve_pending(reply)
        check(not A._did, f"'{reply}' does not click", str(A._did))
        check(act.pending is None, f"'{reply}' clears the pending action")
        check(bool(out), f"'{reply}' gets an answer", str(out))


# ── 6. Ordinary speech never moves the mouse ──────────────────────────────────
def test_conversation_safe() -> None:
    print("\n6. ORDINARY SPEECH IS NOT A COMMAND")
    act, A = build()
    for phrase in ("what type of music do you like",
                   "that's my type of movie",
                   "I press my shirts on sundays",
                   "the press has been brutal lately",
                   "scroll through my photos sometime",
                   "type in my message window, yes I do, then send it",
                   "he sent it yesterday",
                   "click here was the old web"):
        got = act.detect_intent(phrase)
        check(got is None, f"'{phrase[:44]}' is not an action", str(got))
    check(not A._did, "nothing was done for any of them", str(A._did))


# ── 7. Permission is preflighted ──────────────────────────────────────────────
def test_permission() -> None:
    print("\n7. PERMISSION")
    import actuation as A
    from actuation_intents import NovaActuation
    import nova as nova_mod

    A.has_accessibility = lambda: False
    act = NovaActuation(nova_mod.load_config(), assistant=Fake(True))
    out = act.handle("type", "type hello")
    check("accessibility" in out.lower(),
          "a missing grant is named, not reported as a failure", out)
    check(not check_spoken(out), "fit to speak", out)
    A.has_accessibility = lambda: True


# ── 8. Destructive labels ─────────────────────────────────────────────────────
def test_labels() -> None:
    print("\n8. WHICH LABELS ARE DESTRUCTIVE")
    import actuation as A
    for label in ("Send", "Send Message", "Delete", "Buy now", "Submit",
                  "Post", "Confirm", "Pay", "Sign out"):
        check(A.needs_confirmation(label), f"'{label}' is confirmed")
    for label in ("Cancel", "Close", "Back", "Next", "Refresh", "Open",
                  "Bold", "Zoom in"):
        check(not A.needs_confirmation(label), f"'{label}' is not")


# ── 9. The coordinate flip, against a real window ─────────────────────────────
def test_coordinates() -> None:
    """The one piece of maths that cannot be reasoned about safely.

    Vision reports normalized boxes with the origin at the BOTTOM left; the
    screen's origin is TOP left. Getting the flip wrong does not fail loudly —
    it clicks the mirror image of the thing you meant.

    Verified against a window whose contents are known: NovaOS renders its
    state word centred under the orb, so a correct mapping puts it near x=0.5
    and low in the window. Skipped when Nova is not on screen.
    """
    print("\n9. COORDINATE FLIP (needs a real window)")
    import actuation as A

    # Restore the real implementations — build() replaced them.
    import importlib
    importlib.reload(A)

    if A.frontmost_app() != "NovaOS":
        print("     skipped: NovaOS is not frontmost")
        return
    bounds = A._front_window_bounds()
    target = A.find_by_ocr("IDLE") or A.find_by_ocr("LISTENING")
    if bounds is None or target is None:
        print("     skipped: could not read the window")
        return

    wx, wy, ww, wh = bounds
    rel_x = (target.x - wx) / ww
    rel_y = (target.y - wy) / wh
    check(wx <= target.x <= wx + ww and wy <= target.y <= wy + wh,
          "the point is inside the window", f"({target.x:.0f},{target.y:.0f})")
    check(0.35 < rel_x < 0.65, "horizontally centred, as rendered",
          f"x={rel_x:.2f}")
    check(rel_y > 0.55, "in the LOWER half — the y-flip is right",
          f"y={rel_y:.2f}; if this is <0.5 the flip is inverted")
    print(f"     '{target.label}' at x={rel_x:.2f} y={rel_y:.2f} of the window")


def main() -> int:
    print("=" * 72)
    print("ACTUATION — types and clicks, and refuses to do either carelessly")
    print("=" * 72)
    test_work_mode_gate()
    test_never_guesses()
    test_return_ambiguity()
    test_typing_free()
    test_cancel()
    test_conversation_safe()
    test_permission()
    test_labels()
    test_coordinates()

    print(f"\n  {PASS}/{PASS + FAIL} checks passed")
    if FAILURES:
        print("\n  FAILURES:")
        for f in FAILURES:
            print(f"    ✗ {f}")
    print("\n  NOT PROVEN HERE: a real click in a real app. The coordinate maths")
    print("  is checked against a live window above, but nothing in this suite")
    print("  presses a mouse button, and Accessibility is granted to the app")
    print("  bundle rather than to this interpreter.")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
