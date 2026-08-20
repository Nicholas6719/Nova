#!/usr/bin/env python3
"""
The four cues — that they exist, that they are short, and that they are safe.

Sound is the least important thing Nova does and the easiest place to break
something important. This suite is mostly about what the cues must NOT do:
clip, run long, block the pipeline, or take Nova down when there is no audio
device.

What it CANNOT check is the only thing that really matters — whether they
sound right. That is his ears, and the suite says so at the end rather than
implying a pass means good.

Run:  python tests/test_sounds.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
BACKEND = TESTS_DIR.parent
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(TESTS_DIR))
os.chdir(BACKEND)

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


def main() -> int:
    print("=" * 72)
    print("SOUND CUES — short, clean, and never load-bearing")
    print("=" * 72)

    import numpy as np
    import sounds
    import nova as nova_mod

    print("\n1. THE FOUR CUES EXIST")
    for name in ("boot", "ready", "wake", "rest"):
        check(name in sounds.CUES, f"{name} was synthesised")

    print("\n2. LENGTHS")
    for name, limit in (("wake", 0.25), ("rest", 0.35),
                        ("ready", 1.0), ("boot", 4.0)):
        d = sounds.duration(name)
        check(0.0 < d <= limit, f"{name} is {d:.2f}s, within {limit}s", f"{d:.2f}s")
    # The wake cue is spent out of the time he has to start talking, so its
    # length is a real constraint rather than taste.
    check(sounds.duration("wake") <= 0.20,
          "the wake cue costs under 200ms of his head start",
          f"{sounds.duration('wake'):.3f}s")

    print("\n3. NOTHING CLIPS, NOTHING CLICKS")
    for name, wave in sounds.CUES.items():
        peak = float(np.max(np.abs(wave)))
        check(peak <= 0.95, f"{name} peaks at {peak:.2f}, no clipping", f"{peak:.2f}")
        check(abs(float(wave[0])) < 0.02,
              f"{name} starts from silence (no click)", f"{float(wave[0]):.3f}")
        check(abs(float(wave[-1])) < 0.05,
              f"{name} ends in silence (no click)", f"{float(wave[-1]):.3f}")

    print("\n4. ARRIVING AND LEAVING ARE OPPOSITES")
    # wake rises, rest falls. Compared as the spectral centroid of the first
    # half against the second — the actual perceptual difference he relies on.
    # Compared by the DOMINANT frequency of each half rather than the spectral
    # centroid: the centroid is dragged around by the attack transient, and it
    # called a plainly falling cue "rising".
    def slope(name):
        w = sounds.CUES[name]
        # The FIRST THIRD against the LAST THIRD. Halves overlap the two notes
        # in a short cue, so both sides peaked on the same one and a plainly
        # falling gesture measured as flat.
        third = max(1, len(w) // 3)
        def peak_hz(seg):
            spec = np.abs(np.fft.rfft(seg))
            freqs = np.fft.rfftfreq(len(seg), 1 / sounds.SAMPLE_RATE)
            return float(freqs[int(np.argmax(spec))])
        return peak_hz(w[-third:]) - peak_hz(w[:third])
    up, down = slope("wake"), slope("rest")
    check(up > 0, "the wake cue rises", f"{up:+.0f} Hz")
    check(down < 0, "the rest cue falls", f"{down:+.0f} Hz")
    print(f"     wake {up:+7.0f} Hz    rest {down:+7.0f} Hz")

    print("\n5. SILENCE IS ALWAYS AN OPTION")
    quiet = sounds.NovaSounds({"sounds": {"enabled": False}})
    check(not quiet.enabled, "enabled false disables them")
    t0 = time.time()
    quiet.play("wake"); quiet.play_and_wait("wake")
    check(time.time() - t0 < 0.05, "and costs nothing when off")
    # An unknown name must be a no-op, not a KeyError on the voice path.
    quiet.play("nope"); sounds.NovaSounds({}).play("nope")
    check(True, "an unknown cue is ignored, not raised")

    print("\n6. NEVER LOAD-BEARING")
    # A dead audio device must cost the cue and nothing else.
    player = sounds.NovaSounds({"sounds": {"enabled": True}})
    real, sounds.sd = sounds.sd, None          # every call will now raise
    try:
        player.play_and_wait("wake")
        check(True, "a broken audio device does not raise")
    except Exception as exc:
        check(False, "a broken audio device does not raise", repr(exc))
    finally:
        sounds.sd = real

    print("\n7. WIRED WHERE HE ASKED")
    src = Path(nova_mod.__file__).read_text()
    check('self.sounds.play("boot")' in src, "boot plays while the engines load")
    check('self.sounds.play("ready")' in src, "ready plays when she is online")
    wake_block = src[src.index("wake_detected = self.stt.record_wake"):]
    wake_block = wake_block[:wake_block.index("# ── Mic health")]
    check('play_and_wait("wake")' in wake_block,
          "the wake cue fires on the wake word — the one path into a "
          "conversation, so puck mode gets it too")
    check('play_and_wait' in wake_block,
          "...and BLOCKS, so it is over before the microphone opens")
    check('self.sounds.play("rest")' in src,
          "rest plays on the way back to the wake word")
    check("first_wake" in src, "...but not at startup, where ready just played")

    print(f"\n  {PASS}/{PASS + FAIL} checks passed")
    if FAILURES:
        print("\n  FAILURES:")
        for f in FAILURES:
            print(f"    ✗ {f}")
    print("\n  NOT PROVEN HERE: whether they sound any good, or whether the wake\n"
          "  cue is audible over his speakers at his desk. Numbers cannot hear.")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
