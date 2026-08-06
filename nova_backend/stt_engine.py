"""
STT Engine — faster-whisper transcription + webrtcvad VAD + microphone recording.

Design (ported from the proven Jarvis capture pipeline):

  - ONE persistent sd.RawInputStream runs continuously and pushes 30ms frames
    into a shared queue. Both wake detection and command recording read from
    that same queue, so no audio is lost at the boundary between "hearing the
    wake word" and "recording the command".
  - Wake detection scans a ROLLING transcript window (not disjoint fixed clips),
    and keeps a ~2s pre-wake ring buffer so a wake word spoken right before the
    command ("Nova, what time is it?") is captured in one breath.
  - Command recording is VAD-gated with ADAPTIVE silence: a short cutoff for
    quick commands, a longer one once you've been speaking a while, so long
    sentences are never cut off mid-thought.
  - Transcription is hardened against Whisper hallucinations (the phantom "you"
    on silence): temperature=0, no conditioning on previous text, an initial
    prompt biasing toward "Nova", and an RMS noise-floor gate.
"""

from __future__ import annotations

import collections
import logging
import queue
import re
import threading
import time
from typing import Optional

import numpy as np
import sounddevice as sd
import webrtcvad
from faster_whisper import WhisperModel

log = logging.getLogger("nova.stt")

# ── Audio constants ────────────────────────────────────────────────────────────────
SAMPLE_RATE   = 16000                                 # Hz — Whisper + webrtcvad requirement
FRAME_MS      = 30                                    # webrtcvad frame: 10 / 20 / 30 ms
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)   # 480 samples per frame
FRAME_BYTES   = FRAME_SAMPLES * 2                    # int16 = 2 bytes/sample

# Wake detection: keep a rolling ~2s buffer and transcribe it periodically.
PRE_WAKE_SECONDS  = 2.0
PRE_WAKE_FRAMES   = int(PRE_WAKE_SECONDS * 1000 / FRAME_MS)   # ~66 frames — command priming
# Wake DETECTION scans a shorter window than the 2s command-priming buffer: a
# spoken "Nova" should dominate its window rather than being diluted by ~2s of
# surrounding background hum (which makes Whisper transcribe the whole window as
# noise). ~1.2s is long enough to hold "hey nova", short enough to stay clean.
WAKE_WINDOW_SECONDS = 1.2
WAKE_WINDOW_FRAMES  = int(WAKE_WINDOW_SECONDS * 1000 / FRAME_MS)  # ~40 frames
WAKE_SCAN_EVERY_S   = 0.5   # transcribe the wake window this often

# Adaptive silence (ms). Short cutoff for quick commands; longer once the user
# has been speaking a while, so long sentences aren't cut off on natural pauses.
SILENCE_CUTOFF_SHORT_MS  = 700
SILENCE_CUTOFF_LONG_MS   = 900
LONG_SPEECH_THRESHOLD_MS = 2500   # switch to the long cutoff after 2.5s of speech
MAX_RECORDING_MS         = 15000  # hard safety cap on one utterance
NOISE_FLOOR_RMS          = 150    # below this RMS, treat the buffer as silence


class STTEngine:
    def __init__(self, config: dict, mic_gate: Optional[threading.Event] = None,
                 wake_config: Optional[dict] = None) -> None:
        self.config = config
        # Wake-word settings (engine choice + OpenWakeWord tuning). Kept separate
        # from the stt block so the wake engine can be swapped without touching
        # STT. Defaults keep the legacy transcript engine if unspecified.
        self._wake_config = wake_config or {}
        self._oww = None            # lazily-loaded OpenWakeWordDetector
        self._oww_failed = False    # don't retry a broken OWW load every wait
        # Shared gate with the TTS engine: cleared while Nova is speaking so we
        # ignore mic audio during playback (avoids self-capture and the CoreAudio
        # input/output device conflict). If unset, defaults to always-open.
        self._mic_gate = mic_gate if mic_gate is not None else threading.Event()
        if mic_gate is None:
            self._mic_gate.set()

        log.info(f"Loading Whisper model: {config['model']}")
        self.model = WhisperModel(
            config["model"],
            device="auto",
            compute_type="int8",
        )
        self.vad = webrtcvad.Vad(config.get("vad_aggressiveness", 2))

        # Persistent input stream + shared frame queue (started lazily).
        self._audio_q: "queue.Queue[bytes]" = queue.Queue()
        self._stream: Optional[sd.RawInputStream] = None
        # Rolling pre-wake buffer; on a wake hit its contents prime the command
        # recording so a single-breath "Nova, <command>" is fully captured.
        self._pre_wake_buf: "collections.deque[bytes]" = collections.deque(maxlen=PRE_WAKE_FRAMES)
        self._pending_pre_wake: bytes = b""

        log.info("STT ready.")

    # ── Persistent audio stream ────────────────────────────────────────────────────
    def _ensure_stream(self) -> None:
        """Start the always-on mic stream once. Idempotent."""
        if self._stream is not None:
            return

        def _cb(indata, frames, time_info, status):  # noqa: ANN001
            # Runs on PortAudio's thread. Drop frames while Nova is speaking so
            # its own TTS never enters the queue.
            if self._mic_gate.is_set():
                self._audio_q.put(bytes(indata))

        self._stream = sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=FRAME_SAMPLES,
            dtype="int16",
            channels=1,
            callback=_cb,
        )
        self._stream.start()
        log.info("Persistent mic stream started.")


    # ── Wake-word detection (engine dispatch) ───────────────────────────────────────
    def record_wake(self, wake_keywords: list[str], timeout_s: float = 300.0) -> bool:
        """Wait for the wake word. Dispatches to the configured engine:
        'openwakeword' (neural, noise-robust) or 'transcript' (legacy Whisper
        scan). Falls back to transcript if OpenWakeWord can't load, so Nova is
        never left unable to wake."""
        engine = str(self._wake_config.get("engine", "transcript")).lower()
        if engine == "openwakeword" and not self._oww_failed:
            det = self._get_oww_detector()
            if det is not None:
                return self._record_wake_oww(det, timeout_s)
            # OWW unavailable (model missing / load error) → transcript fallback.
        return self._record_wake_transcript(wake_keywords, timeout_s)

    def _get_oww_detector(self):
        """Lazily construct the OpenWakeWord detector. Returns None (and latches
        _oww_failed) if the model is missing or the load fails."""
        if self._oww is not None:
            return self._oww
        try:
            from pathlib import Path as _Path
            from wake_openwakeword import OpenWakeWordDetector
            model_path = self._wake_config.get("oww_model_path", "")
            if model_path and not _Path(model_path).is_absolute():
                # Resolve relative to this backend directory.
                model_path = str(_Path(__file__).parent / model_path)
            self._oww = OpenWakeWordDetector(
                model_path=model_path,
                threshold=float(self._wake_config.get("oww_threshold", 0.5)),
                trigger_level=int(self._wake_config.get("oww_trigger_level", 2)),
                vad_threshold=float(self._wake_config.get("oww_vad_threshold", 0.5)),
            )
            return self._oww
        except Exception as e:
            log.warning(f"OpenWakeWord unavailable ({e}); using transcript wake.")
            self._oww_failed = True
            return None

    def _record_wake_oww(self, detector, timeout_s: float) -> bool:
        """Neural wake wait: score every frame from the persistent stream, keep
        the pre-wake ring buffer primed so a single-breath 'Nova, <command>' is
        still captured, and return True on trigger."""
        self._ensure_stream()
        detector.reset()
        start = time.time()
        while time.time() - start < timeout_s:
            try:
                frame = self._audio_q.get(timeout=0.1)
            except queue.Empty:
                continue
            if len(frame) != FRAME_BYTES:
                continue
            self._pre_wake_buf.append(frame)
            if detector.process(frame):
                self._pending_pre_wake = b"".join(self._pre_wake_buf)
                self._pre_wake_buf.clear()
                return True
        return False

    # ── Wake-word detection (legacy transcript engine) ──────────────────────────────
    def _record_wake_transcript(self, wake_keywords: list[str], timeout_s: float = 300.0) -> bool:
        """
        Listen for a wake keyword on a rolling transcript window.

        Returns True when a keyword is detected (and stashes the pre-wake audio
        so record_command can replay it), or False on timeout.
        """
        self._ensure_stream()
        keywords_lower = [kw.lower() for kw in wake_keywords]
        start          = time.time()
        last_scan      = 0.0
        # Short rolling window for wake detection (separate from the longer
        # pre-wake buffer, which primes command capture).
        window: "collections.deque[bytes]" = collections.deque(maxlen=WAKE_WINDOW_FRAMES)

        while time.time() - start < timeout_s:
            try:
                frame = self._audio_q.get(timeout=0.1)
            except queue.Empty:
                continue

            window.append(frame)
            self._pre_wake_buf.append(frame)

            # Only transcribe every WAKE_SCAN_EVERY_S so we're not running Whisper
            # continuously, and only once the window holds enough audio.
            now = time.time()
            if now - last_scan < WAKE_SCAN_EVERY_S or len(window) < WAKE_WINDOW_FRAMES // 2:
                continue
            last_scan = now

            audio = np.frombuffer(b"".join(window), dtype=np.int16)
            # Cheap energy gate first: skip transcription of near-silence, which
            # is where Whisper hallucinates a phantom "you".
            if _rms(audio) < NOISE_FLOOR_RMS:
                continue

            text = self._transcribe_raw(audio, vad_filter=False).lower().strip()
            if not text:
                continue

            # Steady background noise (a fan, AC) makes Whisper hallucinate a
            # single word repeated many times ("no no no", "okay okay okay").
            # These pollute detection and can drown out a real "Nova". Discard
            # them so noise never blocks the wake word.
            if _is_noise_hallucination(text):
                continue

            matched = _matches_wake(text, keywords_lower)
            log.info(f"[wake-heard] '{text}'" + ("  <-- MATCH" if matched else ""))
            if matched:
                log.info(f"Wake word detected in transcript: '{text}'")
                # Hand the rolling pre-wake audio to record_command so a
                # single-breath "Nova, <command>" is captured in full.
                self._pending_pre_wake = b"".join(self._pre_wake_buf)
                self._pre_wake_buf.clear()
                return True

        return False

    # ── Command recording (VAD-gated, adaptive silence) ────────────────────────────
    def record_command(
        self,
        max_duration_s: float = 15.0,
        start_timeout_s: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        """
        Record a full user utterance from the persistent stream.

        Replays the pre-wake buffer first (so the command spoken right after the
        wake word isn't lost), then reads live audio until an adaptive silence
        cutoff — short for quick commands, longer once the user has been speaking
        a while, so long sentences aren't cut off. Returns int16 audio or None.

        start_timeout_s: if set and no speech begins within that many seconds,
        return None. Used for conversation mode — silence returns to wake mode.
        """
        self._ensure_stream()

        buf         = b""
        silence_ms  = 0
        speech_ms   = 0
        total_ms    = 0
        speaking    = False
        pre_roll: "collections.deque[bytes]" = collections.deque(maxlen=5)
        hard_cap_ms = int(max_duration_s * 1000) if max_duration_s else MAX_RECORDING_MS

        # Prime with the pre-wake frames captured during wake detection.
        priming = []
        if self._pending_pre_wake:
            raw = self._pending_pre_wake
            self._pending_pre_wake = b""
            for i in range(0, len(raw) - FRAME_BYTES + 1, FRAME_BYTES):
                priming.append(raw[i:i + FRAME_BYTES])
        processing_priming = bool(priming)

        # Deadline for the user to START speaking (conversation-mode timeout).
        start_deadline = (time.time() + start_timeout_s) if start_timeout_s else None
        deadline = time.time() + max(max_duration_s, 1.0) + 5.0  # absolute safety
        while time.time() < deadline:
            # Give up if the user never started speaking within start_timeout_s.
            if (not speaking and not priming and start_deadline is not None
                    and time.time() > start_deadline):
                return None
            if priming:
                frame = priming.pop(0)
            else:
                if processing_priming:
                    processing_priming = False
                try:
                    frame = self._audio_q.get(timeout=0.2)
                except queue.Empty:
                    # No frames right this moment. If we're still waiting for the
                    # user to START speaking, keep waiting for the FULL silence
                    # window rather than bailing on a brief mic-queue gap — the
                    # conversation must only end after start_timeout_s of true
                    # silence (and the top-of-loop start_deadline check handles
                    # that). Bail early only when there's no start window at all
                    # (post-wake) so we don't hang.
                    if not speaking:
                        if start_deadline is not None and time.time() <= start_deadline:
                            continue
                        break
                    continue

            if len(frame) != FRAME_BYTES:
                continue

            if self.vad.is_speech(frame, SAMPLE_RATE):
                if not speaking:
                    buf = b"".join(pre_roll)   # preserve the first syllable
                buf       += frame
                silence_ms = 0
                speaking   = True
                speech_ms += FRAME_MS
                total_ms  += FRAME_MS
            elif speaking:
                buf        += frame
                silence_ms += FRAME_MS
                total_ms   += FRAME_MS
                # Only apply the silence cutoff once we're on live audio: a pause
                # inside the priming buffer is just the gap between "Nova" and the
                # command, not the end of the utterance.
                if not processing_priming:
                    cutoff = (
                        SILENCE_CUTOFF_LONG_MS
                        if speech_ms >= LONG_SPEECH_THRESHOLD_MS
                        else SILENCE_CUTOFF_SHORT_MS
                    )
                    if silence_ms > cutoff:
                        break
            else:
                pre_roll.append(frame)

            if speaking and total_ms >= hard_cap_ms:
                break

        audio = np.frombuffer(buf, dtype=np.int16)
        if len(audio) == 0 or _rms(audio) < NOISE_FLOOR_RMS:
            return None
        return audio

    # ── Transcription ─────────────────────────────────────────────────────────────
    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe a numpy int16 audio array to text (full command path)."""
        return self._transcribe_raw(audio, vad_filter=True)

    def _transcribe_raw(self, audio: np.ndarray, vad_filter: bool = True) -> str:
        if audio is None or len(audio) == 0:
            return ""

        # faster-whisper expects float32 in [-1.0, 1.0]
        audio_f32 = audio.astype(np.float32) / 32768.0

        segments, _ = self.model.transcribe(
            audio_f32,
            language=self.config.get("language", "en"),
            beam_size=5,
            temperature=0,                     # deterministic — no random sampling
            condition_on_previous_text=False,  # don't hallucinate from prior context
            vad_filter=vad_filter,
            vad_parameters={"threshold": 0.45, "min_silence_duration_ms": 500},
            initial_prompt=(
                "Nova, hey Nova. Yes, no, remember, remind me, what time is it, "
                "what is the date, open, play, search, tell me."
            ),
        )

        text = " ".join(seg.text for seg in segments).strip()
        # Strip leading filler words.
        text = re.sub(r"^\s*(um+|uh+|hmm+|ah+)\s*", "", text, flags=re.IGNORECASE)
        return text.strip()


def _rms(audio: np.ndarray) -> float:
    if audio is None or len(audio) == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))


def _is_noise_hallucination(text: str) -> bool:
    """True if a transcript looks like Whisper hallucinating from steady noise:
    a small set of words repeated ('no no no', 'okay okay okay'). Real commands
    have varied vocabulary, so they pass through. But NEVER discard a window that
    contains a wake variant — a real 'Nova' must always win."""
    words = re.findall(r"[a-z']+", text.lower())
    if len(words) < 4:
        return False
    if any(v.replace(" ", "") in "".join(words) for v in ("nova", "novah")):
        return False  # protect a real wake word from being filtered as noise
    unique = set(words)
    if len(unique) <= 2:
        return True
    # One token dominating the whole utterance is the repetition signature.
    from collections import Counter
    most = Counter(words).most_common(1)[0][1]
    return most / len(words) >= 0.6


# Close variants Whisper emits for "nova", especially with background noise.
_NOVA_VARIANTS = ("nova", "novah", "no va", "nolva", "knova", "gnova")


def _matches_wake(text: str, keywords_lower: list[str]) -> bool:
    """Match the wake word, tolerant of the near-misses Whisper produces in
    noise. Word-boundary aware so 'innovate' doesn't trigger 'nova'."""
    # Normalize to space-separated tokens; match on whole words only.
    padded = f" {re.sub(r'[^a-z ]', ' ', text.lower())} "
    padded = re.sub(r"\s+", " ", padded)
    for kw in keywords_lower:
        if f" {kw} " in padded:
            return True
    if any("nova" in kw for kw in keywords_lower):
        return any(f" {v} " in padded for v in _NOVA_VARIANTS)
    return False
