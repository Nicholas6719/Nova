#!/usr/bin/env python3
"""
Say "Nova" over the top of a voice that is NOT Nova's, and see if she stops.

WHY A DIFFERENT VOICE. Barge-in is hard to judge when the thing you are
interrupting sounds like the thing listening — you cannot tell by ear whether
she stopped because she heard you or because she happened to finish. So this
speaks in a male Kokoro voice that Nova never uses. It is a TEST FIXTURE: it
reads `tts.kokoro_voice` nowhere, writes config nowhere, and cannot change
Nova's voice by running.

Run it, wait for the long sentence to start, then say "Nova". If barge-in is
working the voice cuts off mid-word. If it is not, it reads to the end.

  python tests/barge_in_probe.py            # one long sentence
  python tests/barge_in_probe.py --voice bm_george
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
BACKEND = TESTS_DIR.parent
sys.path.insert(0, str(BACKEND))
os.chdir(BACKEND)

LINE = ("This is the test voice, not Nova. I am going to keep talking for a "
        "while so you have plenty of time to interrupt me. Say the wake word "
        "whenever you like, and if barge-in is working I should stop in the "
        "middle of a word rather than finishing this sentence, which is quite "
        "a long one on purpose so that there is no doubt about it.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="am_michael",
                    help="a Kokoro voice Nova does not use (am_michael, "
                         "am_adam, bm_george, bm_lewis)")
    ap.add_argument("--text", default=LINE)
    args = ap.parse_args()

    import numpy as np
    import sounddevice as sd
    try:
        from kokoro_onnx import Kokoro
    except Exception as exc:
        print(f"Kokoro unavailable ({exc}) — cannot run the probe.")
        return 1

    model = BACKEND / "kokoro-v1.0.onnx"
    voices = BACKEND / "voices-v1.0.bin"
    if not model.exists() or not voices.exists():
        print("Kokoro model files are missing; see CLAUDE.md First Run.")
        return 1

    print(f"voice: {args.voice}   (Nova's own voice is untouched)")
    k = Kokoro(str(model), str(voices))
    audio, rate = k.create(args.text, voice=args.voice, speed=1.0, lang="en-us")

    print(f"speaking {len(audio) / rate:.1f}s — say \"Nova\" to interrupt")
    t0 = time.time()
    sd.play(np.asarray(audio, dtype=np.float32), rate, blocking=True)
    print(f"finished after {time.time() - t0:.1f}s "
          f"(of {len(audio) / rate:.1f}s — a shorter number means it was cut)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
