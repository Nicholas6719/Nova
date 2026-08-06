"""
Neural wake-word detection via OpenWakeWord.

Why this exists: transcript-based wake detection (running faster-whisper on a
rolling window and string-matching "nova") fundamentally can't handle steady
background noise — Whisper locks onto the noise and transcribes IT, so the
spoken "Nova" never appears. OpenWakeWord runs a purpose-built classifier on
each frame instead: cheap, and trained to be robust to noise. This is the same
approach Jarvis uses ("Hey Jarvis" is a shipped OWW model); Nova ships a custom
"Nova" model trained via OWW's pipeline.

Fully local at runtime — inference uses onnxruntime on-device, no network. The
only network touch is a ONE-TIME download of OWW's shared feature extractors
(melspectrogram + embedding + VAD), the same one-time-model-fetch pattern Nova
already uses for the MLX and Whisper models.

Contract: feed 16 kHz mono int16 frames via ``process(frame)``; it returns True
once the wake word is detected (score over threshold for ``trigger_level``
consecutive scoring windows). Format matches the STT capture stream exactly, so
the same persistent mic stream feeds both.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

log = logging.getLogger("nova.wake")

# OpenWakeWord scores on contiguous 80 ms (1280-sample) chunks. The capture
# stream delivers 30 ms (480-sample) frames, so we accumulate frames until we
# have at least this many samples, then score on exactly one chunk.
_OWW_CHUNK_SAMPLES = 1280


def _models_dir() -> str:
    import openwakeword
    return os.path.join(os.path.dirname(openwakeword.__file__), "resources", "models")


def ensure_feature_models() -> None:
    """Download ONLY OpenWakeWord's shared feature extractors (melspectrogram,
    embedding) and the Silero VAD — NOT the pretrained wake presets. Idempotent;
    a no-op once the files are present. These are required by every OWW model,
    including our custom 'Nova' one."""
    import openwakeword
    from openwakeword.utils import download_file

    target = _models_dir()
    os.makedirs(target, exist_ok=True)

    for fm in openwakeword.FEATURE_MODELS.values():
        onnx_url = fm["download_url"].replace(".tflite", ".onnx")
        dest = os.path.join(target, onnx_url.split("/")[-1])
        if not os.path.exists(dest):
            log.info(f"Downloading OWW feature model: {onnx_url.split('/')[-1]}")
            download_file(onnx_url, target)

    for vm in openwakeword.VAD_MODELS.values():
        dest = os.path.join(target, vm["download_url"].split("/")[-1])
        if not os.path.exists(dest):
            log.info(f"Downloading OWW VAD model: {vm['download_url'].split('/')[-1]}")
            download_file(vm["download_url"], target)


class OpenWakeWordDetector:
    """Wraps an OpenWakeWord model for streaming wake detection.

    threshold      — per-window score (0..1) that counts as a hit.
    trigger_level  — consecutive hit windows required to fire (debounce against
                     a single spurious spike).
    vad_threshold  — OWW's built-in Silero VAD gate; scoring is suppressed
                     unless speech is present, so steady noise scores nothing.
                     0 disables it.
    """

    def __init__(
        self,
        model_path: str,
        threshold: float = 0.5,
        trigger_level: int = 2,
        vad_threshold: float = 0.5,
    ) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"wake model not found: {model_path}")
        ensure_feature_models()

        from openwakeword.model import Model
        self._model = Model(
            wakeword_models=[model_path],
            inference_framework="onnx",
            vad_threshold=vad_threshold,
        )
        self.threshold = float(threshold)
        self.trigger_level = int(trigger_level)
        self._count = 0
        self._pending = np.empty(0, dtype=np.int16)
        # Model key: the sole loaded model's name (predict() returns {name: score}).
        self._keys = list(self._model.models.keys())
        log.info(
            f"OpenWakeWord ready: model={os.path.basename(model_path)} "
            f"keys={self._keys} threshold={self.threshold} trigger={self.trigger_level}"
        )

    def reset(self) -> None:
        """Clear debounce + OWW's internal audio buffer between sessions so a
        stale score can't leak into the next wake wait."""
        self._count = 0
        self._pending = np.empty(0, dtype=np.int16)
        try:
            self._model.reset()
        except Exception:
            pass

    def process(self, frame: bytes) -> bool:
        """Feed one 16 kHz mono int16 frame (bytes). Returns True on wake."""
        samples = np.frombuffer(frame, dtype=np.int16)
        if samples.size == 0:
            return False
        self._pending = np.concatenate((self._pending, samples))

        fired = False
        # Score each full 80 ms chunk we've accumulated (contiguous, no overlap).
        while self._pending.size >= _OWW_CHUNK_SAMPLES:
            chunk = self._pending[:_OWW_CHUNK_SAMPLES]
            self._pending = self._pending[_OWW_CHUNK_SAMPLES:]
            if self._score_chunk(chunk):
                fired = True
                break
        return fired

    def _score_chunk(self, chunk: np.ndarray) -> bool:
        try:
            scores = self._model.predict(chunk)
        except Exception as e:
            log.warning(f"OWW predict failed: {e}")
            return False
        score = max((scores.get(k, 0.0) for k in self._keys), default=0.0)
        if score >= self.threshold:
            self._count += 1
            if self._count >= self.trigger_level:
                self._count = 0
                log.info(f"Wake word detected (OpenWakeWord, score={score:.2f}).")
                return True
        else:
            self._count = 0
        return False
