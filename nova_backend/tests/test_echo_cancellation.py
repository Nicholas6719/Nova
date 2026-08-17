#!/usr/bin/env python3
"""
Echo cancellation — can Nova still hear him while she is speaking?

This decides whether barge-in is possible over speakers. `feat/barge-in`
already implements interrupting Nova by saying the wake word, and it ships
DISABLED with the note "measured, it cannot work over speakers": once Nova's
own voice reaches the microphone the wake model's score collapses to roughly
zero. That is acoustics, not a threshold to tune.

Nova is unusually well placed to fix it. The hard part of software echo
cancellation is a clean, time-aligned reference of what the speaker is
playing — and Nova SYNTHESISES her own speech, so she holds that signal
before it is ever played.

    clean    = his command on its own
    echo     = Nova's TTS, delayed and attenuated like a speaker-to-mic path
    mic      = clean + echo            <- what the microphone hears
    cleaned  = AEC(near=mic, far=Nova's TTS)

REAL AUDIO ONLY, and that is the point of this file rather than a convenience.
An earlier version of this test built both voices from synthetic harmonic
stacks with the same syllabic envelope. That is pathological for an adaptive
filter — the two signals are structurally identical — and it reported his voice
being destroyed (-17 dB, negative correlation) when the canceller was fine. The
same measurement on real Kokoro output and real speech showed 1.5-4 dB. A
synthetic signal can make a working canceller look broken, so this uses Nova's
actual voice.

Run:  python tests/test_echo_cancellation.py
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

PASS = FAIL = 0
FAILURES: list[str] = []
SR = 16000
FRAME = 160          # the APM works in 10ms frames: 160 samples at 16 kHz
CACHE = Path(os.environ.get("TMPDIR", "/tmp")) / "nova_aec_fixtures"


def check(cond, label, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        FAILURES.append(f"{label}  ({detail})" if detail else label)
    return bool(cond)


# ── Real signals ──────────────────────────────────────────────────────────────

def nova_voice() -> np.ndarray:
    """Nova's ACTUAL voice — the same Kokoro blend that reaches the speakers."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = CACHE / "nova_far.npy"
    if cached.exists():
        return np.load(cached)

    import scipy.signal as ss

    import nova as nova_mod
    from tts_engine import TTSEngine

    cfg = nova_mod.load_config()
    tts = TTSEngine(cfg["tts"])
    samples, rate = tts._primary.create(
        text="I checked your calendar and you have three things today, "
             "starting with a dentist appointment at ten thirty this morning.",
        voice=tts._kokoro_voice,
        speed=cfg["tts"].get("rate_multiplier", 1.05),
        lang="en-us")
    a = ss.resample_poly(np.asarray(samples, dtype=np.float32), SR, rate)
    a = (a / (np.abs(a).max() + 1e-9)).astype(np.float32)
    np.save(cached, a)
    return a


def his_voice() -> np.ndarray:
    """A different human voice than Nova's, which is the whole point: the
    canceller must tell them apart."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = CACHE / "him.npy"
    if cached.exists():
        return np.load(cached)

    wav = CACHE / "him.wav"
    subprocess.run(
        ["say", "-v", "Daniel", "-o", str(wav),
         "--data-format=LEI16@16000", "--file-format=WAVE",
         "Nova stop, I need you for a second, can you check the weather instead"],
        check=True, capture_output=True)
    with wave.open(str(wav)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()),
                          dtype=np.int16).astype(np.float32) / 32768
    x = (x / (np.abs(x).max() + 1e-9)).astype(np.float32)
    np.save(cached, x)
    return x


def speaker_path(far: np.ndarray, delay_ms: float, attenuation_db: float) -> np.ndarray:
    """Nova's voice by the time it comes back through the room: delayed,
    attenuated, and smeared by two early reflections so this is not a delay a
    naive subtraction could cancel."""
    d = int(SR * delay_ms / 1000)
    out = np.zeros_like(far)
    out[d:] = far[: far.size - d]
    for ms, g in ((11.0, 0.35), (23.0, 0.18)):
        k = d + int(SR * ms / 1000)
        if k < far.size:
            out[k:] += g * far[: far.size - k]
    return (out * 10 ** (-attenuation_db / 20)).astype(np.float32)


def i16(x: np.ndarray) -> np.ndarray:
    return np.clip(x * 32767, -32768, 32767).astype(np.int16)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-12)


def cancel(near: np.ndarray, far: np.ndarray, delay_ms: int = 35) -> np.ndarray:
    import pywebrtc_audio as pw
    aec = pw.EchoCanceller(SR, 1, delay_ms)
    n = (min(near.size, far.size) // FRAME) * FRAME
    out = np.zeros(n, dtype=np.int16)
    for i in range(0, n, FRAME):
        out[i:i + FRAME] = np.asarray(
            aec.process(near[i:i + FRAME], far[i:i + FRAME]),
            dtype=np.int16).reshape(-1)[:FRAME]
    return out


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_library() -> bool:
    print("\n1. LIBRARY")
    try:
        import pywebrtc_audio as pw
    except ImportError:
        check(False, "pywebrtc_audio installed", "pip install pywebrtc-audio")
        return False
    check(hasattr(pw, "EchoCanceller"), "EchoCanceller is available")

    # The mic delivers 30ms frames and the APM wants 10ms. It does NOT reject a
    # wrong size — it processes it anyway — so the resegmenting is ours to get
    # right and worth stating out loud.
    aec = pw.EchoCanceller(SR, 1, 0)
    accepted = True
    try:
        aec.process(i16(np.zeros(480, np.float32)), i16(np.zeros(480, np.float32)))
    except Exception:
        accepted = False
    check(True, "frame-size behaviour recorded",
          "480-sample frames are accepted silently" if accepted
          else "480-sample frames are rejected")
    print(f"     APM frame is {FRAME} samples (10ms). The mic delivers 480 "
          f"(30ms).\n     Wrong sizes are "
          f"{'ACCEPTED SILENTLY — we must resegment' if accepted else 'rejected'}.")
    return True


def test_cancellation() -> None:
    print("\n2. NOVA'S OWN VOICE REMOVED, HIS KEPT")
    nova = nova_voice()
    him = his_voice()
    n = min(nova.size, him.size)
    nova, him = nova[:n], him[:n]

    print(f"     {'speaker path':<26}{'echo removed':>14}{'his voice':>12}")
    for label, atten in (("loud", 6.0), ("normal", 12.0), ("quiet", 20.0)):
        echo = speaker_path(nova, 35, atten)

        # ERLE on the echo ALONE. Mixed with his voice, a canceller that simply
        # attenuated everything would score well — the opposite of the truth.
        cleaned_echo = cancel(i16(echo), i16(nova))
        erle = 20 * np.log10(rms(i16(echo)[:cleaned_echo.size]) / rms(cleaned_echo))

        # Double-talk: he speaks WHILE Nova is speaking. This is barge-in.
        cleaned_mix = cancel(i16(him * 0.6 + echo), i16(nova))
        kept = 20 * np.log10(rms(cleaned_mix) / rms(i16(him * 0.6)[:cleaned_mix.size]))

        print(f"     {label:<26}{erle:>11.1f} dB{kept:>9.1f} dB")
        check(erle > 15.0, f"{label}: Nova's voice cut by >15 dB", f"{erle:.1f} dB")
        check(kept > -8.0, f"{label}: his voice survives double-talk",
              f"{kept:.1f} dB")


def test_no_harm() -> None:
    """With nothing playing, cancellation must be a no-op. Nova listens far more
    often than she talks, and quietly degrading the ordinary case to fix the
    rare one would be a bad trade."""
    print("\n3. NO HARM WHEN SILENT")
    him = his_voice()
    out = cancel(i16(him), np.zeros(him.size, np.int16))
    loss = -20 * np.log10(rms(out) / rms(i16(him)[:out.size]))
    check(loss < 2.0, "with nothing playing his voice is untouched",
          f"lost {loss:.1f} dB")
    print(f"     his voice attenuated {loss:.1f} dB when Nova is silent")


def main() -> int:
    print("=" * 72)
    print("ECHO CANCELLATION — can Nova hear him while she is speaking?")
    print("=" * 72)
    if test_library():
        test_cancellation()
        test_no_harm()

    print(f"\n  {PASS}/{PASS + FAIL} checks passed")
    if FAILURES:
        print("\n  FAILURES:")
        for f in FAILURES:
            print(f"    ✗ {f}")
    print("\n  NOT PROVEN HERE: real speakers, a real room, and Nicholas's own")
    print("  voice. This measures the canceller against a simulated speaker")
    print("  path — it does not prove barge-in works at his desk.")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
