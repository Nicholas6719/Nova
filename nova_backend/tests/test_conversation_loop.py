#!/usr/bin/env python3
"""
Conversation-loop tests — the part no sweep touched.

_main_loop needs a microphone, so it was never exercised: both bugs Nicholas
reported live in it. Here the STT engine is replaced with a scripted fake, so
the REAL loop runs against controlled input and we can assert on when it stays
in conversation, when it drops to wake mode, and what order the LLM queue runs.
"""
from __future__ import annotations

import os
import sys
import threading
import time

from pathlib import Path as _Path
TESTS_DIR = _Path(__file__).resolve().parent
BACKEND = str(TESTS_DIR.parent)
sys.path.insert(0, BACKEND)
sys.path.insert(0, str(TESTS_DIR))
os.chdir(BACKEND)

PASS = FAIL = 0
FAILURES: list[str] = []


def check(cond, label, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {label}")
    else:
        FAIL += 1
        FAILURES.append(label)
        print(f"  FAIL  {label}")
    if detail:
        print(f"        {detail}")


def section(t):
    print("\n" + "=" * 70)
    print(t)
    print("=" * 70)


import nova as nova_mod
from nova import VoiceAssistant, _MAX_EMPTY_TURNS, _PRIO_TURN, _PRIO_BACKGROUND


class FakeSTT:
    """Scripted microphone. Each script entry is the text `transcribe` will
    return for one turn; None means 'user stayed silent' (record_command
    returns None, i.e. the conversation-timeout path)."""

    def __init__(self, script):
        self.script = list(script)
        self.wake_calls = 0
        self.commands = 0
        self.last_capture_reason = "ok"
        # Mirrors the real STTEngine. Without it the fake drifted from the
        # thing it stands in for and the confidence gate went untested here —
        # a clean transcript, so every scripted turn is acted on.
        self.last_confidence = -0.10

    def record_wake(self, wake_keywords=None, timeout_s=0):
        self.wake_calls += 1
        if not self.script:
            time.sleep(0.05)
            return False
        return True

    def record_command(self, max_duration_s=0, start_timeout_s=None):
        if not self.script:
            raise SystemExit
        self.commands += 1
        nxt = self.script[0]
        if nxt is None:                  # real silence
            self.script.pop(0)
            self.last_capture_reason = "silence"
            return None
        if nxt == "~faint":              # captured, but below the noise floor
            self.script.pop(0)
            self.last_capture_reason = "too_quiet"
            return None
        self.last_capture_reason = "ok"
        return b"audio"

    def transcribe(self, audio):
        return self.script.pop(0) or ""


def build(script):
    va = VoiceAssistant.__new__(VoiceAssistant)
    va.config = nova_mod.load_config()
    # Call the REAL state initializer instead of copying its fields by hand.
    # Hand-copying is how this harness drifted from __init__ and hid a crash.
    va._init_state()
    va.stt = FakeSTT(script)

    class _Mem:
        def add_turn(self, *a, **k): pass
    va.memory = _Mem()

    class _Tools:
        """The loop ducks the music on wake and restores it on the way back to
        wake mode. This suite is about the state machine, not the player, so
        the calls are recorded rather than performed — but they must EXIST,
        because the loop really makes them."""
        def __init__(self): self.ducked = 0; self.restored = 0
        def duck_music(self): self.ducked += 1
        def restore_music(self): self.restored += 1
    va.tools = _Tools()

    class _WS:
        def send_message(self, *a, **k): pass
        def broadcast_state(self, *a, **k): pass
        def stream_token(self, *a, **k): pass
    va.ws = _WS()
    va.set_state = lambda s: None

    va.ended = 0
    va.turns = []

    def fake_end():
        va.ended += 1
        va._session_turns = []
    va._end_conversation = fake_end

    def fake_submit(text, wait):
        va.turns.append(text)
    va._submit_turn = fake_submit
    return va


def run_loop(va, seconds=3.0):
    t = threading.Thread(target=_safe_loop, args=(va,), daemon=True)
    t.start()
    t.join(timeout=seconds)


def _safe_loop(va):
    try:
        va._main_loop()
    except SystemExit:
        pass
    except Exception as exc:
        print(f"        (loop raised {type(exc).__name__}: {exc})")


# ══════════════════════════════════════════════════════════════════════════
section("EMPTY TRANSCRIPTIONS MUST NOT END THE CONVERSATION")
# ══════════════════════════════════════════════════════════════════════════
print(f"  _MAX_EMPTY_TURNS = {_MAX_EMPTY_TURNS}")

# One real turn, then ONE empty (Nova's own audio tail) — the bug he hit.
va = build(["what's on my calendar", "", "what are my reminders"])
run_loop(va)
check(va.ended == 0,
      "a single empty transcription does NOT end the conversation",
      f"ended={va.ended} turns={va.turns}")
check(va.turns == ["what's on my calendar", "what are my reminders"],
      "the turn AFTER the empty one is still handled in conversation mode",
      f"{va.turns}")
check(va.stt.wake_calls == 1,
      "the wake word was only needed ONCE for all three utterances",
      f"wake_calls={va.stt.wake_calls}")

# Enough consecutive empties must still give up (no listening forever).
va = build(["hello"] + [""] * _MAX_EMPTY_TURNS + ["still here?"])
run_loop(va)
check(va.ended >= 1,
      f"{_MAX_EMPTY_TURNS} consecutive empties DO return to wake mode",
      f"ended={va.ended}")

# A real utterance resets the tolerance.
va = build(["hello", "", "", "real turn", "", "", "another"])
run_loop(va)
check(va.ended == 0,
      "a real utterance resets the empty counter (never reaches the limit)",
      f"ended={va.ended} turns={va.turns}")
check("another" in va.turns,
      "conversation survives scattered noise between real turns", f"{va.turns}")

# THE BUG FROM HIS LOG: Nova's own audio tail is captured but is too faint to
# use. record_command returns None for that AND for real silence, and the loop
# treated both as a timeout — ending the conversation the instant Nova finished
# speaking (his log: "Conversation timed out" 0 seconds after going to listening).
va = build(["what do you know about Spider-Man", "~faint", "Spider-Man is my favorite"])
run_loop(va)
check(va.ended == 0,
      "faint noise after Nova speaks does NOT end the conversation",
      f"ended={va.ended} turns={va.turns}")
check(va.turns == ["what do you know about Spider-Man", "Spider-Man is my favorite"],
      "the follow-up turn is still heard in conversation mode", f"{va.turns}")
check(va.stt.wake_calls == 1, "no second wake word was needed",
      f"wake_calls={va.stt.wake_calls}")

va = build(["hello"] + ["~faint"] * _MAX_EMPTY_TURNS + ["still there"])
run_loop(va)
check(va.ended >= 1, f"{_MAX_EMPTY_TURNS} faint captures DO give up eventually",
      f"ended={va.ended}")

# True silence (record_command returns None) still ends it immediately.
va = build(["hello", None, "after"])
run_loop(va)
check(va.ended >= 1,
      "real silence still returns to wake mode straight away", f"ended={va.ended}")


# ══════════════════════════════════════════════════════════════════════════
section("LLM QUEUE PRIORITY  (user turns beat background work)")
# ══════════════════════════════════════════════════════════════════════════
import queue as _q

va2 = VoiceAssistant.__new__(VoiceAssistant)
va2._llm_queue = _q.PriorityQueue()
import itertools
va2._llm_seq = itertools.count()

order: list[str] = []
started = threading.Event()
release = threading.Event()


def slow_background():
    order.append("background-start")
    started.set()
    release.wait(5)
    order.append("background-end")


def user_turn():
    order.append("user-turn")


# Queue a long background job, let it start, then pile on a background job and
# a user turn. The user turn must come out FIRST.
va2._llm_queue.put((_PRIO_BACKGROUND, next(va2._llm_seq), slow_background, None))
worker_done = threading.Event()


def worker():
    for _ in range(3):
        _p, _s, job, _d = va2._llm_queue.get()
        job()
    worker_done.set()


threading.Thread(target=worker, daemon=True).start()
started.wait(3)
va2._llm_queue.put((_PRIO_BACKGROUND, next(va2._llm_seq),
                    lambda: order.append("background-2"), None))
va2._llm_queue.put((_PRIO_TURN, next(va2._llm_seq), user_turn, None))
release.set()
worker_done.wait(6)

check(order.index("user-turn") < order.index("background-2"),
      "a user turn jumps AHEAD of queued background work", f"{order}")
check(_PRIO_TURN < _PRIO_BACKGROUND, "turn priority outranks background")


# ══════════════════════════════════════════════════════════════════════════
section("MUSIC IS DUCKED FOR THE CONVERSATION, AND PUT BACK AFTER")
# ══════════════════════════════════════════════════════════════════════════
# Music out of the speakers goes into the microphone: measured, Whisper's word
# error went from 0% in silence to 709% over loud music, where it starts
# transcribing the music itself. So the player is turned down while Nova
# listens. The wiring lives in _main_loop, so it is checked here, against the
# REAL loop — and the thing that would actually hurt him is a conversation
# that ends without the volume coming back.
va = build(["hello", None, "still there"])
run_loop(va, seconds=2.0)
check(va.tools.ducked >= 1, "the music is ducked when the wake word fires",
      f"ducked {va.tools.ducked}x")
check(va.tools.restored >= 1,
      "and restored once the conversation ends", f"restored {va.tools.restored}x")
check(va.tools.ducked <= va.tools.restored + 1,
      "never ducked more times than it was restored (music left quiet)",
      f"ducked {va.tools.ducked}, restored {va.tools.restored}")


# ══════════════════════════════════════════════════════════════════════════
section("RESULT")
# ══════════════════════════════════════════════════════════════════════════
print(f"\n  {PASS}/{PASS + FAIL}")
for f in FAILURES:
    print(f"    ✗ {f}")
sys.exit(1 if FAIL else 0)
