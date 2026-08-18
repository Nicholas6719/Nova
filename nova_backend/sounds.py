"""
Nova's voice before she has words — four short cues, synthesised, never files.

Nothing here is a recording. Every cue is generated from oscillators at import,
which is deliberate on three counts: no binary assets in a repo that is read as
often as it is run, no licence questions about someone else's sound design, and
the character lives in code where it can be tuned by ear and by argument.

The palette is one idea at two dispositions. A rising perfect fifth means
"coming toward you" — Nova waking, Nova arriving. The same interval falling
means "stepping back" — the conversation closing, Nova returning to the wake
word. He should be able to tell those two apart from across the room without
ever being told which is which.

RULES THIS FILE KEEPS
  * Nothing raises. A Mac with no output device, a busy audio stack, a missing
    numpy — every one of them costs the cue and nothing else. Sound is the last
    thing that should ever take Nova down.
  * Nothing blocks the pipeline unless the caller asks. `play` returns
    immediately; `play_and_wait` exists solely for the wake cue, which must
    finish BEFORE the microphone opens or Nova records her own chime and hands
    it to Whisper.
  * Everything is short. The longest is the boot sequence; the cues he hears
    all day are under a fifth of a second.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

log = logging.getLogger("nova.sounds")

# 48 kHz, the rate the output device actually runs at. At 24 kHz CoreAudio
# resamples every cue on the fly, and during startup — where the boot sequence
# overlaps Kokoro opening its OWN output stream to warm up — that came out
# audibly staticky. Matching the hardware removes the resampler from the path.
SAMPLE_RATE = 48_000

try:
    import numpy as np
    import sounddevice as sd
    _AUDIO = True
except Exception as exc:                       # pragma: no cover - env dependent
    np = None                                  # type: ignore
    sd = None                                  # type: ignore
    _AUDIO = False
    log.info(f"sound cues unavailable ({exc})")


# ── Synthesis ─────────────────────────────────────────────────────────────────

def _tone(freq: float, seconds: float, *, amp: float = 0.5,
          attack: float = 0.006, decay: float = 0.5,
          detune: float = 0.0, shimmer: float = 0.0):
    """One voice: a sine, its octave, and an optional detuned twin.

    The octave at a sixth of the level is what stops these sounding like a
    test-tone generator — it gives the cue a body without making it a chord.
    `detune` beats two oscillators against each other for the small metallic
    motion the boot sequence wants.
    """
    n = max(1, int(SAMPLE_RATE * seconds))
    t = np.linspace(0.0, seconds, n, endpoint=False)
    wave = np.sin(2 * np.pi * freq * t)
    wave += 0.17 * np.sin(2 * np.pi * freq * 2 * t)
    if detune:
        wave += 0.5 * np.sin(2 * np.pi * (freq * (1 + detune)) * t)
    if shimmer:
        wave *= 1.0 + shimmer * np.sin(2 * np.pi * 7.0 * t)
    # Exponential decay, not linear: a linear tail sounds like a fade, an
    # exponential one sounds like something that was struck.
    env = np.exp(-t / max(1e-4, seconds * decay))
    a = max(1, int(SAMPLE_RATE * attack))
    env[:a] *= np.linspace(0.0, 1.0, a)         # no click on the leading edge
    return (wave * env * amp).astype("float32")


def _silence(seconds: float):
    return np.zeros(max(1, int(SAMPLE_RATE * seconds)), dtype="float32")


def _layer(*parts):
    """Sum voices of unequal length, longest wins."""
    if not parts:
        return _silence(0.01)
    out = np.zeros(max(len(p) for p in parts), dtype="float32")
    for p in parts:
        out[:len(p)] += p
    return out


def _at(offset: float, part):
    """Place a voice at a point in time."""
    return np.concatenate([_silence(offset), part])


def _normalise(wave, peak: float = 0.72):
    hi = float(np.max(np.abs(wave))) if len(wave) else 0.0
    if hi <= 0:
        return wave
    out = (wave * (peak / hi)).astype("float32")
    # Fade both edges. A cue that stops mid-cycle is a step change, and a step
    # change through a speaker is a click.
    edge = min(len(out) // 8, int(SAMPLE_RATE * 0.008))
    if edge > 1:
        out[:edge] *= np.linspace(0.0, 1.0, edge)
        out[-edge:] *= np.linspace(1.0, 0.0, edge)
    # A little silence after it, so playback never ends exactly on the last
    # sample of a decaying tone.
    return np.concatenate([out, _silence(0.02)]).astype("float32")


# ── The four cues ─────────────────────────────────────────────────────────────
# Frequencies are a D-major-ish set. The exact notes matter less than the
# INTERVALS: a fifth up to arrive, a fifth down to withdraw.
_D4, _A4, _D5, _F5, _A5, _D6 = 293.66, 440.00, 587.33, 739.99, 880.00, 1174.66


def _make_boot():
    """Systems coming online: five rising ticks, then the room opens up.

    The ticks are deliberately mechanical — short, evenly spaced, climbing — so
    it reads as a sequence completing rather than a piece of music. The long
    detuned pad underneath is what makes it feel like something powering up
    instead of a countdown.
    """
    ticks = [
        _at(0.00, _tone(_D4, 0.16, amp=0.30, decay=0.28)),
        _at(0.34, _tone(_A4, 0.16, amp=0.32, decay=0.28)),
        _at(0.68, _tone(_D5, 0.16, amp=0.34, decay=0.28)),
        _at(1.02, _tone(_F5, 0.16, amp=0.34, decay=0.28)),
        _at(1.36, _tone(_A5, 0.22, amp=0.36, decay=0.32)),
    ]
    # No detune, no shimmer: clean tones only. Depth comes from the octave
    # partial inside _tone and from the pad being long and quiet, not from
    # oscillators fighting each other.
    pad = _at(0.00, _tone(_D4 / 2, 2.40, amp=0.13, attack=0.45, decay=0.8))
    swell = _at(1.15, _tone(_D5, 1.20, amp=0.10, attack=0.60, decay=0.7))
    return _normalise(_layer(pad, swell, *ticks), peak=0.50)


def _make_ready():
    """Online. The fifth lands and holds for a moment."""
    return _normalise(_layer(
        _at(0.00, _tone(_D5, 0.30, amp=0.40, decay=0.35)),
        _at(0.10, _tone(_A5, 0.55, amp=0.34, decay=0.45)),
        _at(0.10, _tone(_D6, 0.45, amp=0.12, decay=0.35)),
    ), peak=0.60)


def _make_wake():
    """She is listening. Two notes, up, and gone — 150ms start to finish.

    Length is a hard constraint, not a taste: this plays in the gap between the
    wake word firing and the microphone opening, and every millisecond of it is
    taken from the time he has to start talking.
    """
    return _normalise(_layer(
        _at(0.000, _tone(_A4, 0.075, amp=0.42, decay=0.30)),
        _at(0.055, _tone(_D5, 0.110, amp=0.46, decay=0.34)),
    ), peak=0.55)


def _make_rest():
    """Back to the wake word. The same two notes, the other way up."""
    # The notes barely overlap, unlike the wake cue. Arriving should feel like
    # one gesture; leaving should feel like two steps down and away.
    return _normalise(_layer(
        _at(0.000, _tone(_D5, 0.085, amp=0.38, decay=0.22)),
        _at(0.085, _tone(_A4, 0.170, amp=0.36, decay=0.34)),
    ), peak=0.45)


CUES: dict = {}
if _AUDIO:
    try:
        CUES = {"boot": _make_boot(), "ready": _make_ready(),
                "wake": _make_wake(), "rest": _make_rest()}
    except Exception as exc:                   # pragma: no cover
        log.warning(f"could not synthesise sound cues ({exc})")
        CUES = {}


def duration(name: str) -> float:
    wave = CUES.get(name)
    return (len(wave) / SAMPLE_RATE) if wave is not None else 0.0


# ── Playback ──────────────────────────────────────────────────────────────────

class NovaSounds:
    """Plays the cues. Independent of the TTS player on purpose.

    Routing these through `tts_engine`'s queue would put a chime behind
    whatever Nova is in the middle of saying, and the wake cue in particular is
    only useful at the instant it is earned.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = (config or {}).get("sounds", {})
        self.enabled = bool(cfg.get("enabled", True)) and _AUDIO and bool(CUES)
        self.volume = float(cfg.get("volume", 0.5))
        self._lock = threading.Lock()

    def play(self, name: str) -> None:
        """Start a cue and return immediately."""
        if not self.enabled or name not in CUES:
            return
        threading.Thread(target=self._play_now, args=(name,),
                         name=f"nova-sound-{name}", daemon=True).start()

    def play_and_wait(self, name: str, timeout: float = 1.5) -> None:
        """Start a cue and wait for it to finish.

        Only the wake cue needs this, and only because the microphone opens
        immediately afterwards: overlap would put Nova's own chime at the head
        of the recording, where the VAD reads it as speech onset and Whisper
        gets handed a note instead of a word.
        """
        if not self.enabled or name not in CUES:
            return
        t = threading.Thread(target=self._play_now, args=(name,),
                             name=f"nova-sound-{name}", daemon=True)
        t.start()
        t.join(timeout)

    def _play_now(self, name: str) -> None:
        try:
            wave = CUES[name] * max(0.0, min(1.0, self.volume))
            # Serialised: two cues at once is never intentional, and overlapping
            # streams on the same device is how you get a click.
            with self._lock:
                sd.play(wave.astype("float32"), SAMPLE_RATE,
                        blocking=True, latency="high")
        except Exception as exc:
            # A cue is the least important thing Nova does.
            log.debug(f"sound cue {name} failed: {exc}")
