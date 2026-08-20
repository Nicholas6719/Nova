#!/usr/bin/env python3
"""
Does the echo canceller converge, and if so how fast — measured per frame.

The suite tells us the shipping canceller removes ~0 dB against an echo with
realistic room delay. That single number cannot tell us WHY, and the three
possible answers need three different fixes:

  never converges     the reference is not reaching the filter in a form it can
                      use at all — a frame-contract problem, not a tuning one
  converges slowly    it works, just far too late to matter for a wake word
  converges then dies  something is resetting or corrupting the filter state

So this prints ERLE per 0.5s of audio, for several room delays at once.

  python nova_backend/tests/aec_converge.py
"""
from __future__ import annotations
import math, os, sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND)); sys.path.insert(0, str(BACKEND / "tests"))
os.chdir(BACKEND)
import logging; logging.disable(logging.WARNING)

import numpy as np
from echo_canceller import EchoCanceller

SR, FRAME = 16000, 480
BLOCK = int(0.5 * SR / FRAME)          # frames per reported row


def speaker_path(far, delay_ms, atten_db=6.0):
    """Delayed, attenuated, plus two early reflections — a room, not a wire."""
    d = int(SR * delay_ms / 1000)
    out = np.zeros_like(far)
    out[d:] = far[: far.size - d]
    for ms, g in ((11.0, 0.35), (23.0, 0.18)):
        k = d + int(SR * ms / 1000)
        if k < far.size:
            out[k:] += g * far[: far.size - k]
    return out * (10 ** (-atten_db / 20))


def run(delay_ms, seconds=8.0, voice=None):
    far = voice if voice is not None else (
        np.random.default_rng(0).standard_normal(int(SR * seconds)) * 0.25)
    far = far.astype(np.float32)
    echo = speaker_path(far, delay_ms).astype(np.float32)
    f_i = np.clip(far * 32767, -32768, 32767).astype(np.int16)
    n_i = np.clip(echo * 32767, -32768, 32767).astype(np.int16)

    ec = EchoCanceller(stream_delay_ms=35)
    if not ec.enabled:
        print("canceller unavailable"); return

    rows, raw_a, cln_a, k = [], 0.0, 0.0, 0
    total = (min(len(f_i), len(n_i)) // FRAME) * FRAME
    for i in range(0, total, FRAME):
        ec.feed_far(f_i[i:i + FRAME].astype(np.float32) / 32768.0, SR)
        out = ec.process(n_i[i:i + FRAME].tobytes(), residual=False)
        a = n_i[i:i + FRAME].astype(np.float64)
        b = np.frombuffer(out, dtype=np.int16).astype(np.float64)
        raw_a += math.sqrt((a * a).mean()); cln_a += math.sqrt((b * b).mean()); k += 1
        if k == BLOCK:
            erle = 20 * math.log10(raw_a / cln_a) if cln_a > 1e-9 else 99.0
            rows.append(erle); raw_a = cln_a = 0.0; k = 0
    lag = ec.delay_ms
    print(f"  delay {delay_ms:3d}ms  measured_lag={str(lag):>6}  " +
          " ".join(f"{r:5.1f}" for r in rows))


def main() -> int:
    print("ERLE per half-second of audio (dB removed). Higher is better.\n")
    print("  " + " " * 32 + "  ".join(f"{t/2:4.1f}s" for t in range(1, 17)))
    for d in (0, 10, 20, 35, 60):
        run(d)
    print("\n  0ms is the case the OLD suite effectively tested.")
    print("  READ IT: all-zero rows = never converges (frame contract).")
    print("           rising rows   = converges, just too slowly.")
    print("           rise then fall = state being reset or corrupted.")

    # Real speech, because noise is the easy case for an adaptive filter.
    try:
        from kokoro_onnx import Kokoro
        k = Kokoro(str(BACKEND / "kokoro-v1.0.onnx"), str(BACKEND / "voices-v1.0.bin"))
        a, sr = k.create("Nova speaking a sentence so the canceller has real "
                         "speech to adapt against, not white noise.",
                         voice="af_heart", lang="en-us")
        a = np.asarray(a, dtype=np.float32)
        a = np.interp(np.linspace(0, len(a), int(len(a) * SR / sr)),
                      np.arange(len(a)), a).astype(np.float32)
        print("\n  with REAL Kokoro speech as the far end:")
        run(35, voice=a)
    except Exception as exc:
        print(f"\n  (Kokoro unavailable: {exc})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
