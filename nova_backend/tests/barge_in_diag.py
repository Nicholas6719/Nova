#!/usr/bin/env python3
"""
Why the wake word does not interrupt her — measured, not guessed.

Plays a long line through the speakers in a voice that is NOT Nova's, records
the microphone throughout, and scores EVERY frame with OpenWakeWord three ways:

    raw        the microphone exactly as it arrives
    aec        after the linear echo canceller
    aec+res    after the residual suppressor as well  (what Nova actually uses)

Say "Nova" four or five times while it talks. Then read the table. Whichever
column stays near zero is the one destroying the wake word, and that is the
thing to fix. If ALL THREE stay near zero the problem is not cancellation at
all — it is that his voice never reaches the microphone loudly enough over the
speakers, which is a different fix entirely.

  python nova_backend/tests/barge_in_diag.py
"""
from __future__ import annotations

import os, sys, threading, time
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND)); os.chdir(BACKEND)
import logging; logging.disable(logging.WARNING)

import numpy as np, sounddevice as sd
import nova as N
from echo_canceller import EchoCanceller

FRAME = 480          # 30ms at 16k
LINE = ("This is the test voice, not Nova, and I am going to keep talking for a "
        "good long while so that you have every chance to interrupt me. Please "
        "say the wake word several times while I am speaking, clearly, at the "
        "volume you would normally use, and I will keep going regardless so "
        "that the microphone hears both of us at once for the whole recording.")


def make_detector(cfg, trigger=1):
    """Build a detector and expose its real score.

    NOT via `det.last_score` — that attribute does not exist and reading it
    returns 0.000 for everything, which is exactly how an earlier attempt at
    this measured nothing and concluded the wrong thing (CLAUDE.md records it).
    The score is taken by wrapping the model's own predict, so it is the number
    the detector actually decided on rather than a guess about it.
    """
    from wake_openwakeword import OpenWakeWordDetector
    mp = cfg.get("oww_model_path", "")
    if mp and not Path(mp).is_absolute():
        mp = str(BACKEND / mp)
    det = OpenWakeWordDetector(model_path=mp,
                               threshold=float(cfg.get("oww_threshold", 0.5)),
                               trigger_level=trigger,
                               vad_threshold=float(cfg.get("oww_vad_threshold", 0.5)))
    det.peak = 0.0
    real_predict = det._model.predict
    keys = det._keys

    def wrapped(chunk):
        scores = real_predict(chunk)
        try:
            det.peak = max(det.peak, max((scores.get(k, 0.0) for k in keys),
                                         default=0.0))
        except Exception:
            pass
        return scores

    det._model.predict = wrapped
    return det


def main() -> int:
    cfg = N.load_config()
    wake_cfg = cfg.get("wake_word", {})

    try:
        from kokoro_onnx import Kokoro
        k = Kokoro(str(BACKEND / "kokoro-v1.0.onnx"), str(BACKEND / "voices-v1.0.bin"))
        speech, srate = k.create(LINE, voice="am_michael", speed=1.0, lang="en-us")
        speech = np.asarray(speech, dtype=np.float32)
    except Exception as exc:
        print(f"Kokoro unavailable ({exc})"); return 1

    ec_a = EchoCanceller(35)          # linear only
    ec_b = EchoCanceller(35)          # linear + residual (what Nova runs)
    if not ec_a.enabled:
        print("echo canceller unavailable"); return 1
    # Strip the residual stage from the first one so the two are comparable.
    class _Passthrough:
        def process(self, near, far): return None
    ec_a._residual = _Passthrough()

    det = {n: make_detector(wake_cfg) for n in ("raw", "aec", "aec+res")}
    hits = {n: 0 for n in det}
    peak = {n: 0.0 for n in det}
    frames = [0]

    def score(name, pcm_bytes):
        d = det[name]
        try:
            fired = d.process(pcm_bytes)
            peak[name] = max(peak[name], float(getattr(d, "peak", 0.0)))
            if fired:
                hits[name] += 1
        except Exception:
            pass

    def cb(indata, n, t, status):
        raw = bytes(indata); frames[0] += 1
        score("raw", raw)
        score("aec", ec_a.process(raw))
        score("aec+res", ec_b.process(raw))

    print(f"speaking {len(speech)/srate:.1f}s — say \"Nova\" several times, clearly\n")
    stream = sd.InputStream(samplerate=16000, channels=1, dtype="int16",
                            blocksize=FRAME, callback=cb)
    stream.start()

    def feed_far():
        # Both cancellers get the reference, exactly as Kokoro feeds Nova's.
        step = 2400
        played = 0
        t0 = time.time()
        while played < len(speech):
            chunk = speech[played:played+step]
            ec_a.feed_far(chunk, srate); ec_b.feed_far(chunk, srate)
            played += step
            time.sleep(max(0.0, (played/srate) - (time.time()-t0)))
    threading.Thread(target=feed_far, daemon=True).start()
    sd.play(speech, srate, blocking=True)
    time.sleep(0.4); stream.stop(); stream.close()

    print(f"{'path':10s} {'wake fired':>11s} {'peak score':>11s}")
    print("-" * 34)
    for n in ("raw", "aec", "aec+res"):
        print(f"{n:10s} {hits[n]:>11d} {peak[n]:>11.3f}")
    print(f"\n{frames[0]} frames ({frames[0]*0.03:.1f}s) of microphone scored.")
    print("\nREAD IT LIKE THIS:")
    print("  raw fires, aec does not      -> the linear canceller eats his voice")
    print("  aec fires, aec+res does not  -> the residual suppressor does")
    print("  none of them fire            -> his voice never reaches the mic over")
    print("                                  the speakers; cancellation is innocent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
