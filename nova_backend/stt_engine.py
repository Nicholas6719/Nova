"""
STT Engine — faster-whisper transcription + webrtcvad VAD + microphone recording.

Wake-word detection:
  Primary path:   openwakeword (if installed)
  Fallback path:  2-second transcript chunks scanned for keyword

Recording:
  VAD-gated via webrtcvad — stops after silence_duration_s of silence
  following at least min_speech_duration_s of speech.
"""

from __future__ import annotations

import logging
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
OWW_FRAME_SAMPLES = 1280                              # 80 ms at 16 kHz (openwakeword)


class STTEngine:
    def __init__(self, config: dict, mic_gate: Optional[threading.Event] = None) -> None:
        self.config = config
        # Shared gate with the TTS engine: cleared while Nova is speaking so we
        # do not record during playback (avoids self-capture and the CoreAudio
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
        log.info("STT ready.")

    def _await_mic(self) -> None:
        """Block until the mic gate is open (Nova is not speaking)."""
        self._mic_gate.wait()

    # ── Wake-word detection ───────────────────────────────────────────────────────
    def record_wake(
        self,
        wake_keywords: list[str],
        timeout_s: float = 300.0,
    ) -> bool:
        """
        Listen for a wake keyword.
        Returns True when wake word detected, False on timeout.

        Tries openwakeword first; falls back to transcript-based scanning.
        """
        try:
            return self._oww_wake(timeout_s)
        except (ImportError, Exception) as exc:
            if isinstance(exc, ImportError):
                log.debug("openwakeword not installed; using transcript-based wake detection.")
            else:
                log.debug(f"openwakeword error ({exc}); falling back.")
            return self._transcript_wake(wake_keywords, timeout_s)

    def _oww_wake(self, timeout_s: float) -> bool:
        """openwakeword neural wake-word detection."""
        from openwakeword.model import Model
        oww = Model(inference_framework="onnx")
        start = time.time()

        # Don't open the mic while Nova is speaking (device conflict + self-hear).
        self._await_mic()

        with sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=OWW_FRAME_SAMPLES
        ) as stream:
            while time.time() - start < timeout_s:
                frame, _ = stream.read(OWW_FRAME_SAMPLES)
                chunk = np.squeeze(frame)
                oww.predict(chunk)
                for scores in oww.prediction_buffer.values():
                    if scores and scores[-1] > 0.5:
                        log.info("Wake word detected (openwakeword).")
                        return True

        return False

    def _transcript_wake(self, wake_keywords: list[str], timeout_s: float) -> bool:
        """
        Fallback: record 2-second windows, transcribe each, check for keyword.
        Accepts 'nova' or 'hey nova' by default.
        """
        keywords_lower = [kw.lower() for kw in wake_keywords]
        start          = time.time()

        while time.time() - start < timeout_s:
            audio = self._record_fixed(duration_s=2.0)
            # No VAD filter on the short wake window: faster-whisper's VAD was
            # stripping entire 2s clips of real-room speech ("removed 00:02.000
            # of audio"), leaving nothing to transcribe so the wake word never
            # matched. Let Whisper transcribe the raw clip instead.
            text  = self._transcribe_raw(audio, vad_filter=False).lower().strip()
            # Log what we heard (even non-matches) so wake-word misses are
            # visible — otherwise a failure to trigger is completely silent.
            if text:
                matched = any(kw in text for kw in keywords_lower)
                log.info(f"[wake-heard] '{text}'" + ("  <-- MATCH" if matched else ""))
                if matched:
                    log.info(f"Wake word detected in transcript: '{text}'")
                    return True

        return False

    # ── Command recording (VAD-gated) ─────────────────────────────────────────────
    def record_command(self, max_duration_s: float = 15.0) -> Optional[np.ndarray]:
        """
        Record audio until VAD detects end of speech.
        Returns int16 numpy array, or None if no speech detected.
        """
        silence_threshold_s = self.config.get("silence_duration_s", 1.2)
        min_speech_s        = self.config.get("min_speech_duration_s", 0.5)

        audio_buffer   = []
        speech_started = False
        silence_start  = None
        speech_start   = None
        start          = time.time()

        # Don't open the mic while Nova is speaking (device conflict + self-hear).
        self._await_mic()

        with sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=FRAME_SAMPLES
        ) as stream:
            while time.time() - start < max_duration_s:
                frame, _    = stream.read(FRAME_SAMPLES)
                frame_bytes = frame.tobytes()
                audio_buffer.extend(frame.flatten().tolist())

                is_speech = self.vad.is_speech(frame_bytes, SAMPLE_RATE)

                if is_speech:
                    if not speech_started:
                        speech_started = True
                        speech_start   = time.time()
                    silence_start = None
                elif speech_started:
                    if silence_start is None:
                        silence_start = time.time()
                    elapsed_speech  = time.time() - speech_start
                    elapsed_silence = time.time() - silence_start
                    if elapsed_speech >= min_speech_s and elapsed_silence >= silence_threshold_s:
                        break

        if not speech_started:
            return None

        return np.array(audio_buffer, dtype=np.int16)

    # ── Fixed-duration recording ──────────────────────────────────────────────────
    def _record_fixed(self, duration_s: float = 2.0) -> np.ndarray:
        # Don't open the mic while Nova is speaking (device conflict + self-hear).
        self._await_mic()
        samples = int(SAMPLE_RATE * duration_s)
        audio   = sd.rec(samples, samplerate=SAMPLE_RATE, channels=1, dtype="int16")
        sd.wait()
        return audio.flatten()

    # ── Transcription ─────────────────────────────────────────────────────────────
    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe a numpy int16 audio array to text."""
        return self._transcribe_raw(audio)

    def _transcribe_raw(self, audio: np.ndarray, vad_filter: bool = True) -> str:
        if audio is None or len(audio) == 0:
            return ""

        # faster-whisper expects float32 in [-1.0, 1.0]
        audio_f32 = audio.astype(np.float32) / 32768.0

        segments, _ = self.model.transcribe(
            audio_f32,
            language=self.config.get("language", "en"),
            beam_size=5,
            vad_filter=vad_filter,
        )

        text = " ".join(seg.text for seg in segments).strip()

        # Strip leading filler words
        text = re.sub(r"^\s*(um+|uh+|hmm+|ah+)\s*", "", text, flags=re.IGNORECASE)
        return text.strip()
