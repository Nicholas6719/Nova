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
PRE_WAKE_FRAMES   = int(PRE_WAKE_SECONDS * 1000 / FRAME_MS)   # ~66 frames
WAKE_SCAN_EVERY_S = 0.7   # transcribe the rolling window this often

# Adaptive silence (ms). Short cutoff for quick commands; longer once the user
# has been speaking a while, so long sentences aren't cut off on natural pauses.
SILENCE_CUTOFF_SHORT_MS  = 700
SILENCE_CUTOFF_LONG_MS   = 900
LONG_SPEECH_THRESHOLD_MS = 2500   # switch to the long cutoff after 2.5s of speech
MAX_RECORDING_MS         = 15000  # hard safety cap on one utterance
NOISE_FLOOR_RMS          = 150    # below this RMS, treat the buffer as silence


class STTEngine:
    def __init__(self, config: dict, mic_gate: Optional[threading.Event] = None) -> None:
        self.config = config
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

    def _drain_q(self) -> None:
        """Discard any queued frames (e.g. residual TTS echo)."""
        try:
            while True:
                self._audio_q.get_nowait()
        except queue.Empty:
            pass

    # ── Wake-word detection ───────────────────────────────────────────────────────
    def record_wake(self, wake_keywords: list[str], timeout_s: float = 300.0) -> bool:
        """
        Listen for a wake keyword on a rolling transcript window.

        Returns True when a keyword is detected (and stashes the pre-wake audio
        so record_command can replay it), or False on timeout.
        """
        self._ensure_stream()
        keywords_lower = [kw.lower() for kw in wake_keywords]
        start          = time.time()
        last_scan      = 0.0
        # Local rolling window of raw frames to transcribe.
        window: "collections.deque[bytes]" = collections.deque(maxlen=PRE_WAKE_FRAMES)

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
            if now - last_scan < WAKE_SCAN_EVERY_S or len(window) < PRE_WAKE_FRAMES // 2:
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

            matched = any(kw in text for kw in keywords_lower)
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
    def record_command(self, max_duration_s: float = 15.0) -> Optional[np.ndarray]:
        """
        Record a full user utterance from the persistent stream.

        Replays the pre-wake buffer first (so the command spoken right after the
        wake word isn't lost), then reads live audio until an adaptive silence
        cutoff — short for quick commands, longer once the user has been speaking
        a while, so long sentences aren't cut off. Returns int16 audio or None.
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

        deadline = time.time() + max(max_duration_s, 1.0) + 5.0  # absolute safety
        while time.time() < deadline:
            if priming:
                frame = priming.pop(0)
            else:
                if processing_priming:
                    processing_priming = False
                try:
                    frame = self._audio_q.get(timeout=1.0)
                except queue.Empty:
                    # No audio at all and nothing captured yet → give up.
                    if not speaking:
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
