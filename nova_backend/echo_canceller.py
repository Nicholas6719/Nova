"""
Echo cancellation — hearing him while Nova is talking.

Nova currently DROPS every microphone frame while she speaks (`mic_gate` in
stt_engine). That is why barge-in cannot work over speakers: the audio he
interrupts with is thrown away before anything sees it, and even when it was
kept, the wake model's score collapsed once Nova's own voice reached the mic.

The fix is not a threshold. It is subtracting Nova's voice from what the mic
hears — and Nova is unusually well placed to do that, because she SYNTHESISES
her own speech. The hard part of software AEC is a clean reference of what the
speaker is playing; Kokoro hands us that signal before it is ever played.

Measured on real Kokoro output against real speech (tests/test_echo_cancellation):

    speaker path        Nova's voice removed      his voice kept
    loud   (6 dB)             36.5 dB                -3.9 dB
    normal (12 dB)           32.6 dB                -2.1 dB
    quiet  (20 dB)           27.7 dB                -1.5 dB
    silent                      —                    -0.2 dB

Two traps this module exists to handle:

  * The APM works in 10ms frames (160 samples at 16 kHz). The mic delivers
    30ms (480). The library does NOT reject a wrong size — it processes it
    anyway and quietly degrades — so resegmenting is ours to get right.
  * The reference has to be time-aligned with what the mic actually hears.
    Playback runs ahead of capture, so the far signal is buffered and consumed
    in step with incoming mic frames rather than read "now".

DEFAULT OFF (`stt.echo_cancellation`). Nova listens far more often than she
talks, and the ordinary case must not be risked for the rare one. Nothing here
raises: a failure disables cancellation and leaves the raw microphone, because
degraded hearing beats no hearing.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from typing import Optional

import numpy as np

log = logging.getLogger("nova.aec")

SAMPLE_RATE = 16000
APM_FRAME = 160          # 10ms — what the APM requires
MIC_FRAME = 480          # 30ms — what the mic stream delivers

# How much played audio to keep waiting for the mic to catch up. Beyond this the
# reference is too old to describe what the microphone is hearing, and stale
# reference is worse than none: it makes the filter adapt to the wrong thing.
_MAX_FAR_SECONDS = 1.5


class EchoCanceller:
    """Subtracts Nova's own voice from the microphone.

    Thread-safe by design: `feed_far` is called from the TTS playback thread
    and `process` from the audio callback, which are different threads.
    """

    def __init__(self, stream_delay_ms: int = 35) -> None:
        self.enabled = False
        self._aec = None
        self._lock = threading.Lock()
        self._far = deque()              # queued reference samples (int16)
        self._far_len = 0
        self._max_far = int(_MAX_FAR_SECONDS * SAMPLE_RATE)
        self._stream_delay_ms = stream_delay_ms
        # Rolling attenuation, so "is this actually working" is answerable at
        # runtime instead of only in a test. Cheap: two RMS values per frame.
        self._erle_raw = 0.0
        self._erle_out = 0.0
        self._erle_n = 0

        try:
            import pywebrtc_audio as pw
            self._aec = pw.EchoCanceller(SAMPLE_RATE, 1, stream_delay_ms)
            self.enabled = True
            log.info(f"Echo cancellation ready (stream delay {stream_delay_ms}ms)")
        except Exception as exc:
            # Not fatal, and deliberately quiet about it: without the library
            # Nova behaves exactly as she did before.
            log.info(f"Echo cancellation unavailable ({exc}); using the raw mic")

    # ── Reference signal ──────────────────────────────────────────────────────
    def feed_far(self, audio: np.ndarray, rate: int) -> None:
        """Hand over audio Nova is about to play.

        Called from the TTS worker with Kokoro's float32 output at its own
        sample rate; resampled to the mic's rate so the two can be compared.
        """
        if not self.enabled or audio is None or len(audio) == 0:
            return
        try:
            mono = np.asarray(audio, dtype=np.float32).reshape(-1)
            if rate != SAMPLE_RATE:
                mono = _resample(mono, rate, SAMPLE_RATE)
            pcm = np.clip(mono * 32767, -32768, 32767).astype(np.int16)
            with self._lock:
                self._far.append(pcm)
                self._far_len += pcm.size
                # Drop the OLDEST reference when overfull. Keeping the newest
                # matters: the mic is always behind, and a reference from two
                # seconds ago describes nothing it is hearing now.
                while self._far_len > self._max_far and self._far:
                    self._far_len -= self._far.popleft().size
        except Exception as exc:
            log.warning(f"could not queue reference audio: {exc}")

    def reset(self) -> None:
        """Nova stopped speaking: drop the reference and let the filter start
        clean, so the next utterance does not adapt against stale audio."""
        if not self.enabled:
            return
        with self._lock:
            self._far.clear()
            self._far_len = 0
        try:
            self._aec.reset()
        except Exception:
            pass

    @property
    def has_reference(self) -> bool:
        """True when there is played audio to cancel against. When Nova is
        silent, `process` is a pass-through and costs nothing."""
        with self._lock:
            return self._far_len > 0

    # ── The live path ─────────────────────────────────────────────────────────
    def process(self, mic_frame: bytes) -> bytes:
        """Clean one 30ms microphone frame.

        Returns the frame unchanged when cancellation is off, when there is
        nothing playing, or on any failure. The caller never has to care.
        """
        if not self.enabled:
            return mic_frame
        try:
            near = np.frombuffer(mic_frame, dtype=np.int16)
            if near.size != MIC_FRAME:
                return mic_frame

            far = self._take_far(near.size)
            if far is None:
                return mic_frame        # nothing playing: leave the mic alone

            out = np.empty_like(near)
            for i in range(0, near.size, APM_FRAME):
                res = self._aec.process(near[i:i + APM_FRAME],
                                        far[i:i + APM_FRAME])
                out[i:i + APM_FRAME] = np.asarray(
                    res, dtype=np.int16).reshape(-1)[:APM_FRAME]
            cleaned = out.tobytes()
            self._note_attenuation(mic_frame, cleaned)
            return cleaned
        except Exception as exc:
            log.warning(f"echo cancellation failed on a frame: {exc}")
            return mic_frame

    def _note_attenuation(self, raw: bytes, clean: bytes) -> None:
        try:
            a = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
            b = np.frombuffer(clean, dtype=np.int16).astype(np.float32)
            if a.size and b.size:
                self._erle_raw += float(np.sqrt((a * a).mean()))
                self._erle_out += float(np.sqrt((b * b).mean()))
                self._erle_n += 1
        except Exception:
            pass

    @property
    def attenuation_db(self) -> Optional[float]:
        """How much of the speaker mix is being removed, averaged since the
        last read. None when nothing has been cancelled yet."""
        if self._erle_n == 0:
            return None
        raw = self._erle_raw / self._erle_n
        out = self._erle_out / self._erle_n
        self._erle_raw = self._erle_out = 0.0
        self._erle_n = 0
        if out <= 1e-6 or raw <= 1e-6:
            return None
        import math
        return round(20 * math.log10(raw / out), 1)

    def _take_far(self, n: int) -> Optional[np.ndarray]:
        """Pull exactly n reference samples, consuming the queue in step with
        the mic. Returns None when Nova is not speaking."""
        with self._lock:
            if self._far_len == 0:
                return None
            out = np.zeros(n, dtype=np.int16)
            filled = 0
            while filled < n and self._far:
                chunk = self._far[0]
                take = min(n - filled, chunk.size)
                out[filled:filled + take] = chunk[:take]
                filled += take
                if take == chunk.size:
                    self._far.popleft()
                else:
                    self._far[0] = chunk[take:]
                self._far_len -= take
            return out


def _resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    """Rate conversion, preferring scipy and falling back to linear
    interpolation so a missing scipy costs quality rather than the feature."""
    if src == dst:
        return x
    try:
        import scipy.signal as ss
        from math import gcd
        g = gcd(src, dst)
        return ss.resample_poly(x, dst // g, src // g).astype(np.float32)
    except Exception:
        n = int(round(x.size * dst / src))
        return np.interp(np.linspace(0, x.size - 1, n),
                         np.arange(x.size), x).astype(np.float32)
