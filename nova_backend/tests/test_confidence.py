#!/usr/bin/env python3
"""
Confidence gating — Nova acts when she is sure, and says so when she is not.

The transcript is gone from the UI, so a mishear is now a SILENT failure. This
is the replacement: how sure Whisper was decides whether Nova acts, and the bar
rises with what the action costs.

Proves, with the real `confidence` module and the real Whisper engine:
  * the tiers are right — "send that to Sarah" is not treated like "what time
    is it"
  * a poor transcript is refused rather than guessed at
  * anything outbound is read back in HIS words, whatever the confidence
  * a missing confidence signal makes Nova behave exactly as she did before,
    rather than deaf
  * the measured thresholds are the ones actually in the code

The thresholds themselves came from 270 clips (18 commands x 3 voices x 5 noise
levels). That sweep is documented in confidence.py; this file guards the
behaviour built on top of it.

Run:  python tests/test_confidence.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

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


# ── 1. Consequence tiers ──────────────────────────────────────────────────────
def test_tiers() -> None:
    print("\n1. WHAT DOES IT COST TO BE WRONG?")
    import confidence as C

    high = ["send that message", "text mom I'll be late", "reply to that email",
            "delete the old screenshots", "buy the laptop stand",
            "call Sarah", "shut down the mac", "forward it to my brother"]
    # Reminders whose CONTENT names an outbound verb. Every one of these used
    # to be classified HIGH, which would have made Nova read back every
    # reminder he ever set.
    medium = ["move the budget file to Documents", "remind me to call mom at six",
              "remind me to send the invoice tomorrow",
              "remind me to delete those files later",
              "make a note to email the landlord",
              "remember to buy milk",
              "rename that to Q3 numbers", "open Xcode", "play some music",
              "set a timer for five minutes"]
    low = ["what time is it", "what's the weather tomorrow",
           "what's on my screen", "tell me something interesting",
           "how are you doing", "what's my battery at"]

    for t in high:
        check(C.consequence(t) == C.HIGH, f"HIGH: '{t}'", C.consequence(t))
    for t in medium:
        check(C.consequence(t) == C.MEDIUM, f"MEDIUM: '{t}'", C.consequence(t))
    for t in low:
        check(C.consequence(t) == C.LOW, f"LOW: '{t}'", C.consequence(t))

    check(C.consequence("") == C.HIGH,
          "nothing heard is never worth acting on")


# ── 2. The decision ───────────────────────────────────────────────────────────
def test_decisions() -> None:
    print("\n2. THE DECISION")
    import confidence as C

    # Chat: being wrong is cheap, so she acts on almost anything.
    d = C.decide("tell me something interesting", -0.70)
    check(d.should_act, "poor confidence still acts on chat", repr(d))
    check(d.is_unsure, "…but shows that she was not sure", repr(d))

    d = C.decide("tell me something interesting", -0.10)
    check(d.action == C.ACT, "clean confidence acts silently", repr(d))
    check(not d.is_unsure, "…with no unsure flag", repr(d))

    # A file move is recoverable but real: the knee applies.
    d = C.decide("move the budget file to Documents", -0.70)
    check(d.action == C.REJECT, "a poor transcript does not move his files",
          repr(d))
    d = C.decide("move the budget file to Documents", -0.20)
    check(d.should_act, "a clean transcript does", repr(d))

    # Outbound: read back EVERY time, however confident.
    for score in (-0.02, -0.10, -0.25):
        d = C.decide("send that message to Sarah", score)
        check(d.action == C.CONFIRM,
              f"outbound is read back even at {score:+.2f}", repr(d))
    d = C.decide("send that message to Sarah", -0.90)
    check(d.action == C.REJECT, "…and refused outright when badly heard", repr(d))

    # A missing signal must not make her deaf.
    d = C.decide("what time is it", None)
    check(d.action == C.ACT, "no confidence signal behaves as before", repr(d))


# ── 3. What she says ──────────────────────────────────────────────────────────
def test_spoken() -> None:
    print("\n3. WHAT SHE SAYS")
    import confidence as C

    d = C.decide("send that message to Sarah", -0.10)
    check("send that message to Sarah" in d.readback,
          "the read-back quotes HIS words, not a paraphrase", d.readback)
    check(not check_spoken(d.readback), "read-back is fit to speak", d.readback)

    for tier in (C.LOW, C.MEDIUM, C.HIGH):
        line = C.ask_again(tier)
        check(not check_spoken(line), f"'{tier}' re-ask is fit to speak", line)
        # She must not guess at what he meant — that is the whole point.
        check("did you mean" not in line.lower(),
              f"'{tier}' re-ask does not guess at his words", line)


# ── 4. The thresholds are the measured ones ───────────────────────────────────
def test_thresholds() -> None:
    print("\n4. MEASURED THRESHOLDS")
    import confidence as C

    check(C.FLOOR_MEDIUM == -0.50,
          "the medium floor is the measured knee (-0.50)", str(C.FLOOR_MEDIUM))
    check(C.FLOOR_HIGH > C.FLOOR_MEDIUM > C.FLOOR_LOW,
          "the bar rises with consequence",
          f"{C.FLOOR_LOW} < {C.FLOOR_MEDIUM} < {C.FLOOR_HIGH}")
    check(C.ACT_COMFORTABLE == C.FLOOR_HIGH,
          "the unsure band starts where the strictest floor sits")


# ── 5. Real audio, real engine ────────────────────────────────────────────────
def test_real_audio() -> None:
    """Whisper's confidence has to separate real speech from noise on the
    actual engine, not just in theory."""
    print("\n5. REAL AUDIO")
    import confidence as C
    import nova as nova_mod
    from stt_engine import STTEngine

    stt = STTEngine(nova_mod.load_config()["stt"])

    wav = "/tmp/_conf_test.wav"
    subprocess.run(["say", "-v", "Daniel", "-o", wav,
                    "--data-format=LEI16@16000", "--file-format=WAVE",
                    "what time is it"], check=True, capture_output=True)
    with wave.open(wav) as w:
        clean = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)

    text = stt.transcribe(clean)
    score = stt.last_confidence
    check(bool(text.strip()), "clean speech transcribes", repr(text))
    check(score is not None and score > C.FLOOR_MEDIUM,
          "clean speech clears the medium floor", f"{score}")
    print(f"     clean speech      -> {score:+.3f}  ({text!r})")

    noise = (np.random.default_rng(0).standard_normal(16000) * 3000).astype(np.int16)
    ntext = stt.transcribe(noise)
    nscore = stt.last_confidence
    d = C.decide(ntext, nscore)
    check(not d.should_act or not ntext.strip(),
          "pure noise never becomes an action",
          f"text={ntext!r} score={nscore} -> {d.action}")
    print(f"     pure noise        -> {nscore}  ({ntext!r}) -> {d.action}")


def main() -> int:
    print("=" * 72)
    print("CONFIDENCE GATING — acting only when sure enough for the stakes")
    print("=" * 72)
    test_tiers()
    test_decisions()
    test_spoken()
    test_thresholds()
    test_real_audio()

    print(f"\n  {PASS}/{PASS + FAIL} checks passed")
    if FAILURES:
        print("\n  FAILURES:")
        for f in FAILURES:
            print(f"    ✗ {f}")
    print("\n  NOT PROVEN HERE: the thresholds against NICHOLAS's voice. They")
    print("  were swept on synthesised speech across 5 noise levels; his own")
    print("  voice is the corpus that would settle them.")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
