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
_RING_SECONDS = 3.0        # reference history to search within
_ENV_SECONDS = 1.5         # envelope history used for the correlation
_MAX_LAG_MS = 400          # how far behind the mic can plausibly be
_ESTIMATE_EVERY = 8        # mic frames between lag estimates (~240ms)
# 0.35, not 0.50. The floor was raised to 0.50 to stop the estimate jumping,
# but the MEDIAN is what actually fixed that — and at 0.50 music never cleared
# the bar at all, so the lag stayed unmeasured. Permissive detection plus a
# robust median beats a strict threshold with nothing behind it.
_MIN_CONFIDENCE = 0.35
_LAG_VOTES = 9             # estimates kept; the MEDIAN of them is the lag


class _ResidualSuppressor:
    """Removes what the linear filter cannot: the nonlinear residue.

    WHY THIS EXISTS. AEC3 models the echo path as a linear filter, and against
    a simulated speaker that is exactly what the path is — which is why the
    own-voice suite measures 27-36 dB. A real laptop speaker driven at volume
    clips and resonates, so a large part of what reaches the microphone was
    never a linear function of the reference and cannot be subtracted from it.
    Measured on this Mac: 1.6-12.7 dB with alignment already correct.

    So this works in the frequency domain instead of subtracting. For each bin
    it keeps a running estimate of how strongly the reference COUPLES into the
    microphone — learned only while the far end dominates, so his voice never
    teaches it — and then attenuates bins where the predicted echo accounts for
    most of what is there. Nonlinearity does not matter to a magnitude estimate
    the way it does to a subtraction.

    Two properties it must have, and does:
      * When the speakers are silent the reference is silent, the predicted
        echo is zero, every gain is 1, and his voice passes through untouched.
      * Gains are smoothed across frequency AND time. An unsmoothed spectral
        gain is what makes suppressors sound like a broken radio, and it would
        wreck Whisper long before it helped it.

    STFT with 50% overlap-add, so the output is continuous. Costs 16ms of
    latency, which is invisible next to the wake word.
    """

    N = 512                      # 32ms window
    H = 256                      # 50% hop
    _FLOOR = 0.05                # never more than 26dB in one bin
    _OVER = 1.6                  # over-subtract a little; residue is bursty

    def __init__(self) -> None:
        self.win = np.hanning(self.N + 1)[:self.N].astype(np.float32)
        self._near = np.zeros(0, dtype=np.float32)
        self._far = np.zeros(0, dtype=np.float32)
        self._ola = np.zeros(0, dtype=np.float32)
        self._out = np.zeros(0, dtype=np.float32)
        bins = self.N // 2 + 1
        self._coupling = np.full(bins, 0.2, dtype=np.float32)
        self._gain = np.ones(bins, dtype=np.float32)

    def process(self, near: np.ndarray, far: np.ndarray) -> Optional[np.ndarray]:
        """Suppress residue in `near` using `far`. Returns the same number of
        samples once primed, or None while filling the first window."""
        self._near = np.concatenate([self._near, near.astype(np.float32)])
        self._far = np.concatenate([self._far, far.astype(np.float32)])
        want = near.size

        while self._near.size >= self.N and self._far.size >= self.N:
            n_blk = self._near[:self.N] * self.win
            f_blk = self._far[:self.N] * self.win
            Y = np.fft.rfft(n_blk)
            X = np.fft.rfft(f_blk)
            ymag = np.abs(Y) + 1e-6
            xmag = np.abs(X) + 1e-6

            # Learn the coupling only where the reference clearly dominates, so
            # his voice can never be mistaken for echo and taught as one.
            ratio = ymag / xmag
            learn = xmag > (4.0 * np.median(xmag))
            if np.any(learn):
                self._coupling[learn] = (0.9 * self._coupling[learn]
                                         + 0.1 * np.minimum(ratio[learn], 4.0))

            predicted = self._OVER * self._coupling * xmag
            g = np.clip(1.0 - predicted / ymag, self._FLOOR, 1.0)
            # Smooth across frequency (3-bin) then across time. Unsmoothed
            # gains are what make a suppressor sound like a broken radio.
            g = np.convolve(g, np.array([0.25, 0.5, 0.25], dtype=np.float32),
                            mode="same")
            self._gain = 0.6 * self._gain + 0.4 * g
            block = np.fft.irfft(Y * self._gain, n=self.N).astype(np.float32) * self.win

            if self._ola.size < self.N:
                self._ola = np.pad(self._ola, (0, self.N - self._ola.size))
            self._ola[:self.N] += block
            self._out = np.concatenate([self._out, self._ola[:self.H]])
            self._ola = np.concatenate([self._ola[self.H:],
                                        np.zeros(self.H, dtype=np.float32)])
            self._near = self._near[self.H:]
            self._far = self._far[self.H:]

        if self._out.size < want:
            return None
        out, self._out = self._out[:want], self._out[want:]
        return out


class EchoCanceller:
    """Subtracts Nova's own voice from the microphone.

    Thread-safe by design: `feed_far` is called from the TTS playback thread
    and `process` from the audio callback, which are different threads.
    """

    def __init__(self, stream_delay_ms: int = 35) -> None:
        self.enabled = False
        self._aec = None
        self._lock = threading.Lock()
        # A RING, not a queue. A FIFO forced the alignment to be whatever the
        # queue depth happened to be, and that drifts: the reference arrives at
        # render time while the microphone hears it later, so the two run at the
        # same rate but a shifting offset. Cancellation collapsed from 27-36 dB
        # (Nova's own voice, handed over microseconds before playback) to 2-8 dB
        # (the speaker mix, arriving through ScreenCaptureKit) for exactly this
        # reason. With a ring we can read at a MEASURED offset instead.
        self._ring_n = int(_RING_SECONDS * SAMPLE_RATE)
        self._far_ring = np.zeros(self._ring_n, dtype=np.int16)
        self._far_w = 0                  # monotonic samples ever written
        self._far_len = 0                # samples available (for has_reference)
        # Decimated |amplitude| histories, one bin per millisecond. The lag is
        # found by correlating ENVELOPES rather than waveforms: it is ~16x less
        # arithmetic, and it locks onto the shape of speech and music instead of
        # their phase, which is what survives a speaker and a room.
        self._env_bins = int(_ENV_SECONDS * 1000)
        self._far_env = np.zeros(self._env_bins, dtype=np.float32)
        self._mic_env = np.zeros(self._env_bins, dtype=np.float32)
        self._lag_ms: Optional[float] = None
        self._lag_votes = deque(maxlen=_LAG_VOTES)
        self._residual = _ResidualSuppressor()
        self._lag_conf = 0.0
        self._since_estimate = 0
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
                self._write_ring(pcm)
                self._push_env(self._far_env, pcm)
        except Exception as exc:
            log.warning(f"could not queue reference audio: {exc}")

    def reset(self) -> None:
        """Nova stopped speaking: drop the reference and let the filter start
        clean, so the next utterance does not adapt against stale audio."""
        if not self.enabled:
            return
        with self._lock:
            self._far_ring[:] = 0
            self._far_w = 0
            self._far_len = 0
            self._far_env[:] = 0
            self._mic_env[:] = 0
            # The lag is a property of the PATH, not of an utterance, so it
            # survives a reset. Re-measuring from scratch every time Nova stops
            # speaking would throw away the alignment on every turn.
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
    def process(self, mic_frame: bytes, residual: bool = True) -> bytes:
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

            with self._lock:
                self._push_env(self._mic_env, near)
                self._since_estimate += 1
                if self._since_estimate >= _ESTIMATE_EVERY:
                    self._since_estimate = 0
                    self._estimate_lag()
            far = self._take_far(near.size)
            if far is None:
                return mic_frame        # nothing playing: leave the mic alone

            out = np.empty_like(near)
            for i in range(0, near.size, APM_FRAME):
                res = self._aec.process(near[i:i + APM_FRAME],
                                        far[i:i + APM_FRAME])
                out[i:i + APM_FRAME] = np.asarray(
                    res, dtype=np.int16).reshape(-1)[:APM_FRAME]
            # Second stage: whatever the linear filter could not subtract.
            # Feeding it the AEC OUTPUT rather than the raw mic means it only
            # ever has to deal with the residue.
            # `residual=False` for the WAKE path, and this is the whole of
            # barge-in working or not.
            #
            # The suppressor was built for MUSIC, where the far end plays
            # continuously and his voice is the exception. It attenuates
            # whichever frequency bins the reference explains — and while Nova
            # is talking that is most of the bins a wake word lives in, so it
            # was removing HIM along with her. Confirmed by the two facts that
            # narrowed it: the wake word fires reliably in a quiet room, and he
            # can hear himself over her, so neither the detector nor his volume
            # was ever the problem.
            #
            # The linear canceller alone is measured safe for his voice: 27-36
            # dB off Nova against 1.5-3.9 dB off him. That is ample for a wake
            # word, and it is all the wake path gets.
            # The two stages want DIFFERENT references, and giving them the
            # same one is why cancellation kept trading places.
            #
            # AEC3 gets the reference unshifted, because it aligns internally
            # and moving it underneath makes it re-converge forever (measured:
            # ~30 dB decaying to single digits). The residual suppressor is not
            # an adaptive filter — it compares magnitudes per bin — so it wants
            # the reference actually LINED UP with what the mic heard, and
            # without that it fell from ~19 dB to 4.
            far_aligned = far
            if self._lag_ms is not None:
                with self._lock:
                    shifted = self._read_ring(
                        int(self._lag_ms * SAMPLE_RATE / 1000), near.size)
                if shifted is not None:
                    far_aligned = shifted
            res = (self._residual.process(out.astype(np.float32),
                                          far_aligned.astype(np.float32))
                   if residual else None)
            if res is not None:
                out = np.clip(res, -32768, 32767).astype(np.int16)
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

    # ── Ring + envelope plumbing ──────────────────────────────────────────────
    def _write_ring(self, pcm: np.ndarray) -> None:
        """Append to the reference ring, wrapping. Caller holds the lock."""
        n = pcm.size
        if n >= self._ring_n:                       # absurdly large chunk
            self._far_ring[:] = pcm[-self._ring_n:]
            self._far_w += n
            self._far_len = self._ring_n
            return
        start = self._far_w % self._ring_n
        end = start + n
        if end <= self._ring_n:
            self._far_ring[start:end] = pcm
        else:
            split = self._ring_n - start
            self._far_ring[start:] = pcm[:split]
            self._far_ring[:end - self._ring_n] = pcm[split:]
        self._far_w += n
        self._far_len = min(self._ring_n, self._far_len + n)

    def _read_ring(self, end_offset: int, n: int) -> Optional[np.ndarray]:
        """n samples ending `end_offset` samples before the newest one."""
        newest = self._far_w - end_offset
        first = newest - n
        if n <= 0 or first < 0 or (self._far_w - first) > self._far_len:
            return None
        idx = np.arange(first, newest) % self._ring_n
        return self._far_ring[idx]

    @staticmethod
    def _push_env(env: np.ndarray, pcm: np.ndarray) -> None:
        """Fold |samples| into 1ms bins and shift them into the history.

        Envelopes rather than waveforms because the lag search only needs the
        SHAPE of what was played. A room and a speaker mangle phase; they leave
        the shape of a syllable or a drum hit intact.
        """
        per_bin = SAMPLE_RATE // 1000                # 16 samples at 16 kHz
        usable = (pcm.size // per_bin) * per_bin
        if usable <= 0:
            return
        bins = np.abs(pcm[:usable].astype(np.float32)).reshape(-1, per_bin).mean(axis=1)
        k = min(bins.size, env.size)
        env[:-k] = env[k:]
        env[-k:] = bins[-k:]

    def _estimate_lag(self) -> None:
        """Find how far the microphone is behind the speakers, and remember it.

        Normalised cross-correlation of the two envelopes over 0..400ms. Only
        adopted when it is CONFIDENT: a correlation peak on near-silence is
        noise, and acting on it would be worse than keeping a stale but
        plausible alignment. The accepted value is smoothed, because the true
        delay does not jump around and a jumping estimate would make the filter
        re-adapt from scratch every quarter second.
        """
        try:
            mic = self._mic_env
            far = self._far_env
            win = int(0.6 * 1000)                    # 600ms of mic to match
            if mic.size < win + _MAX_LAG_MS:
                return
            m = mic[-win:]
            m = m - m.mean()
            m_norm = float(np.sqrt((m * m).sum()))
            if m_norm < 1e-3:
                return                               # microphone is silent
            best_lag, best_score = None, 0.0
            for lag in range(0, _MAX_LAG_MS + 1, 2):
                seg = far[far.size - win - lag: far.size - lag] if lag else far[-win:]
                if seg.size != win:
                    continue
                f = seg - seg.mean()
                f_norm = float(np.sqrt((f * f).sum()))
                if f_norm < 1e-3:
                    continue
                score = float((m * f).sum() / (m_norm * f_norm))
                if score > best_score:
                    best_score, best_lag = score, lag
            if best_lag is None or best_score < _MIN_CONFIDENCE:
                return
            self._lag_conf = best_score
            # MEDIAN of recent confident estimates, not an exponential slew.
            # Measured: a slew still chased outliers — the estimate walked
            # 2.5 -> 94.6 -> 16 -> 47 ms and cancellation collapsed on every
            # jump, because the filter re-adapts whenever the alignment moves.
            # The true delay is a property of the speaker-mic path and barely
            # changes, so a median throws single bad reads away entirely
            # instead of averaging them in.
            self._lag_votes.append(float(best_lag))
            if len(self._lag_votes) < 3:
                return
            lag = float(np.median(np.fromiter(self._lag_votes, dtype=np.float32)))
            first = self._lag_ms is None
            self._lag_ms = lag
            if first:
                log.info(f"speaker-to-mic delay measured at {lag:.0f}ms "
                         f"(confidence {best_score:.2f})")
        except Exception as exc:
            log.debug(f"lag estimate failed: {exc}")

    def _take_far(self, n: int) -> Optional[np.ndarray]:
        """The n reference samples that match the mic frame just received.

        Read at the MEASURED offset rather than consumed FIFO. Until a lag has
        been measured this falls back to the newest reference, which is what the
        queue effectively did and is right for Nova's own voice, where the
        handoff happens microseconds before playback.
        """
        with self._lock:
            if self._far_len == 0:
                return None
            # The measured lag is NOT applied here, and this is the whole of
            # why cancellation kept collapsing.
            #
            # AEC3 has its own delay estimator and adapts to whatever offset it
            # observes. Shifting the reference underneath it means that offset
            # keeps moving, so the filter re-converges forever and never
            # settles. Measured per half-second, the shape is unmistakable:
            # every configuration where this estimator LOCKED decayed from
            # ~40 dB to single digits, while the one where it never locked held
            # ~30 dB for the whole run. With real Kokoro speech it fell from
            # 35.5 dB to 1.6 in three seconds.
            #
            # So the reference is handed over in order and AEC3 is left to
            # align it, which is what the original FIFO did and why it measured
            # 27-36 dB. The estimator stays — `delay_ms` is genuinely useful to
            # the system tap's magnitude-domain suppressor, which is not an
            # adaptive filter and does not mind being nudged.
            lag_samples = 0
            far = self._read_ring(lag_samples, n)
            if far is None:
                far = self._read_ring(0, n)          # not enough history yet
            return far

    @property
    def delay_ms(self) -> Optional[float]:
        """The measured speaker-to-mic delay, for /api/status."""
        return None if self._lag_ms is None else round(self._lag_ms, 1)


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
