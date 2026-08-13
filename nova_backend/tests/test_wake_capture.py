#!/usr/bin/env python3
"""
Wake-word capture — the two failures from Nicholas's live session, 2026-08-13.

He said "Nova", it did nothing, he said it again, and Nova answered a hundred
and ten wake words:

    10:43:28  [state] -> idle          (capture held only "Nova"; nothing said)
    10:43:33  [user] Nova. Nova. Nova. Nova. ... (x110)  -> sent to the LLM

Two independent bugs:

  1. The wake word is replayed from the pre-wake buffer, so the recorder counts
     itself as already speaking. The normal 700ms silence cutoff then applied
     immediately, giving him 700ms to begin his command. Saying "Nova",
     pausing to think, and starting late looked exactly like the wake word not
     working.

  2. Whisper looped on a capture that was only the wake word.
     `_strip_wake_prefix` removed ONE, leaving 109 for the LLM, and
     `_is_noise_hallucination` could not help because it deliberately
     whitelists the wake word so a real "Nova" always wins the WAKE decision.

Fidelity: the REAL record_command and the REAL guards. The VAD is scripted, the
way the loop suite scripts the microphone — it is the INPUT being controlled,
not the decision under test.
"""
from __future__ import annotations

import os
import sys
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


import json
import numpy as np

import nova as nova_mod
from nova import VoiceAssistant
import stt_engine as se
from stt_engine import _is_transcription_loop, _is_noise_hallucination

config = json.loads((_Path(BACKEND) / "config.json").read_text())


# ══════════════════════════════════════════════════════════════════════════
section("THE LOOP GUARD  (what actually reached the LLM)")
# ══════════════════════════════════════════════════════════════════════════
# The exact shape from the log: 110 wake words out of 1.8s of audio.
looped = " ".join(["Nova."] * 110)
check(_is_transcription_loop(looped, 1.84),
      "the live failure is caught (110 'Nova.' from 1.8s)")
check(not _is_noise_hallucination(looped),
      "…and the OLD guard would NOT have caught it (it whitelists the wake word)",
      "which is why a new guard was needed rather than reusing that one")

check(_is_transcription_loop(" ".join(["yes"] * 40), 2.0),
      "a loop on some other word is caught too")
check(_is_transcription_loop("what is the date " * 12, 3.0),
      "the beam-search loop shape is caught")

# Must never discard something he actually said.
REAL = [
    ("what time is it", 1.4),
    ("remind me to call mom at five", 2.6),
    ("what's on my calendar today", 2.2),
    ("open spotify and play some music", 2.8),
    ("tell me something interesting about the ocean", 3.2),
    ("no", 0.6),
    ("yes please", 1.0),
    ("no no that's wrong", 1.8),
    ("set a timer for five minutes and remind me to check the oven", 4.5),
    ("what does my finance degree roadmap say", 3.4),
]
for phrase, secs in REAL:
    check(not _is_transcription_loop(phrase, secs),
          f"kept: {phrase[:46]!r}")

# A genuinely fast but real utterance must survive.
check(not _is_transcription_loop("yes yes okay fine sure", 1.2),
      "a short emphatic real phrase is kept")


# ══════════════════════════════════════════════════════════════════════════
section("STRIPPING A REPEATED WAKE WORD")
# ══════════════════════════════════════════════════════════════════════════
va = VoiceAssistant.__new__(VoiceAssistant)
va.config = nova_mod.load_config()

check(va._strip_wake_prefix("Nova. Nova. Nova. Nova.") == "",
      "a capture that is only repeated wake words strips to nothing")
check(va._strip_wake_prefix("Nova") == "", "a bare wake word strips to nothing")
check(va._strip_wake_prefix("Nova, what time is it") == "what time is it",
      "a real command still survives stripping")
check(va._strip_wake_prefix("Nova. Nova. what time is it") == "what time is it",
      "a stutter before the command is stripped, command kept")
check(va._strip_wake_prefix("hey nova what's on my calendar")
      == "what's on my calendar", "'hey nova' is stripped")
check(va._strip_wake_prefix("what time is it") == "what time is it",
      "a command with no wake word is untouched")
# Do not eat a real word that merely starts with the wake word. Asserting
# "not empty" was too weak and passed while returning "k won the match" —
# check the exact string.
for phrase in ("Novak won the match", "Novartis is a company", "November is cold"):
    got = va._strip_wake_prefix(phrase)
    check(got == phrase, f"untouched: {phrase!r}", f"got {got!r}")


# ══════════════════════════════════════════════════════════════════════════
section("TIME TO START SPEAKING AFTER THE WAKE WORD  (real record_command)")
# ══════════════════════════════════════════════════════════════════════════
import queue as _queue
import threading

# The REAL constructor, not a hand-assembled stand-in. Building it by hand is
# how this harness would drift from __init__ — and it just did: a new field
# (_pre_roll_frames) was added and the hand-built object crashed on it, which
# is the same class of bug as the `screen` engine that was missing from
# __init__ for a whole session while 130 checks stayed green. Only the mic
# stream is stubbed, because there is no microphone here.
print("  building the real STTEngine (loads Whisper)…", flush=True)
_gate = threading.Event(); _gate.set()
stt = se.STTEngine(config["stt"], mic_gate=_gate,
                   wake_config=config.get("wake_word", {}))
stt._stream = object()                      # so _ensure_stream is a no-op
stt._ensure_stream = lambda: None
stt._audio_q = _queue.Queue()
stt._pending_pre_wake = b""

FRAME = se.FRAME_BYTES
LOUD = (np.full(se.FRAME_SAMPLES, 6000, dtype=np.int16)).tobytes()
QUIET = (np.zeros(se.FRAME_SAMPLES, dtype=np.int16)).tobytes()


class ScriptedVAD:
    """Scripted speech/silence per frame — the controlled INPUT, like the
    scripted microphone in the conversation-loop suite."""

    def __init__(self, script):
        self.script = list(script)

    def is_speech(self, frame, rate):
        return self.script.pop(0) if self.script else False


def frames(n, loud):
    return [LOUD if loud else QUIET] * n


def run_capture(pre_wake_frames, live_script):
    """live_script: list of booleans, one per live frame."""
    stt._pending_pre_wake = b"".join(frames(pre_wake_frames, True))
    stt._audio_q = _queue.Queue()
    for is_speech in live_script:
        stt._audio_q.put(LOUD if is_speech else QUIET)
    # VAD sees the priming frames first, then the live ones.
    stt.vad = ScriptedVAD([True] * pre_wake_frames + list(live_script))
    audio = stt.record_command(max_duration_s=15.0, start_timeout_s=None)
    return audio


F = lambda ms: ms // se.FRAME_MS      # frames for a duration

# THE BUG: wake word, a 1.5s pause to think, then the command.
audio = run_capture(F(500), [False] * F(1500) + [True] * F(1000) + [False] * F(1200))
captured_s = (len(audio) / se.SAMPLE_RATE) if audio is not None else 0.0
check(audio is not None and captured_s > 2.0,
      "a 1.5s pause before speaking no longer ends the capture",
      f"captured {captured_s:.2f}s (needs to include the command after the pause)")

# Normal case: he speaks immediately. The grace must NOT delay the cutoff.
audio = run_capture(F(500), [True] * F(1000) + [False] * F(1500))
captured_s = (len(audio) / se.SAMPLE_RATE) if audio is not None else 0.0
check(audio is not None and captured_s < 2.6,
      "speaking immediately still cuts off promptly (grace does not apply)",
      f"captured {captured_s:.2f}s")

# Silence after the wake word: gives up, but only after the grace window.
audio = run_capture(F(500), [False] * F(4000))
captured_s = (len(audio) / se.SAMPLE_RATE) if audio is not None else 0.0
check(captured_s < 3.0, "pure silence after the wake word still gives up",
      f"captured {captured_s:.2f}s")


# ══════════════════════════════════════════════════════════════════════════
section("KEEPING THE START OF WHAT HE SAID")
# ══════════════════════════════════════════════════════════════════════════
# webrtcvad fires late on soft onsets, and losing the first syllable is the
# most damaging thing that can happen to a transcript. Measured on 42 noisy
# clips: 200ms of onset lost drops exact matches from 81% to 21%, while
# carrying 500ms of extra lead-in costs nothing. The buffer is generous for
# that reason, and it matters more since the post-wake grace window — once he
# pauses after the wake word, this is the only thing protecting his first word.
check(stt._pre_roll_frames * se.FRAME_MS >= 300,
      "at least 300ms of audio is kept from before speech is detected",
      f"{stt._pre_roll_frames} frames = {stt._pre_roll_frames * se.FRAME_MS}ms")

# Speech starting immediately on live audio must still carry its lead-in.
lead = F(600)
audio = run_capture(0, [False] * lead + [True] * F(1000) + [False] * F(1200))
captured_s = (len(audio) / se.SAMPLE_RATE) if audio is not None else 0.0
expected_min = 1.0 + (stt._pre_roll_frames * se.FRAME_MS) / 1000 * 0.8
check(audio is not None and captured_s >= expected_min,
      "the capture includes audio from before speech was detected",
      f"captured {captured_s:.2f}s of a 1.0s utterance "
      f"(pre-roll {stt._pre_roll_frames * se.FRAME_MS}ms)")

check(config["stt"]["beam_size"] == 5,
      "beam search is on (greedy measured worse on 126 clips)",
      f"beam_size={config['stt']['beam_size']}")


# ══════════════════════════════════════════════════════════════════════════
section("RESULT")
# ══════════════════════════════════════════════════════════════════════════
print(f"\n  {PASS}/{PASS + FAIL}")
for f in FAILURES:
    print(f"    ✗ {f}")
sys.exit(1 if FAIL else 0)
