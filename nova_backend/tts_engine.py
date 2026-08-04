"""
TTS Engine — Kokoro ONNX (primary) + macOS `say` (fallback).

100% local — no cloud TTS unless explicitly added later.

Playback model:
  - Sentences are queued and played sequentially via a background worker thread.
  - Nova's streaming pipeline calls speak() per sentence as the LLM streams,
    so TTS overlaps with continued LLM generation (gapless feel).
  - wait_until_done() blocks until the queue drains (used before set_state("idle")).
"""

from __future__ import annotations

import logging
import queue
import subprocess
import threading
import time
from typing import Optional

import numpy as np
import sounddevice as sd

log = logging.getLogger("nova.tts")


class TTSEngine:
    def __init__(self, config: dict) -> None:
        self.config   = config
        self._queue   = queue.Queue()
        self._stop    = threading.Event()
        self._primary = None

        # Start playback worker before loading model so the thread is ready
        self._worker = threading.Thread(target=self._playback_worker, daemon=True)
        self._worker.start()

        self._load_primary()

    # ── Engine loading ────────────────────────────────────────────────────────────
    def _load_primary(self) -> None:
        engine = self.config.get("primary", "kokoro")
        if engine == "kokoro":
            try:
                self._load_kokoro()
                log.info("TTS primary: Kokoro ONNX")
            except Exception as exc:
                log.warning(f"Kokoro unavailable ({exc}) — falling back to macOS say.")
                self._primary = None
        else:
            self._primary = None

    def _load_kokoro(self) -> None:
        from kokoro_onnx import Kokoro
        self._kokoro_voice = self.config.get("kokoro_voice", "af_nova")
        self._primary = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
        # Warm-up: speak a near-silent utterance to pre-load the ONNX graph
        log.info("Warming up Kokoro…")
        self._primary.create(
            text=".",
            voice=self._kokoro_voice,
            speed=1.0,
            lang="en-us",
        )

    # ── Public API ────────────────────────────────────────────────────────────────
    def speak(self, text: str) -> None:
        """Queue text for immediate playback. Returns instantly."""
        text = text.strip()
        if not text:
            return
        self._queue.put(text)

    def wait_until_done(self, timeout: Optional[float] = None) -> None:
        """Block until the sentence queue is fully played out.

        A timeout is important: this is called from the single LLM worker thread,
        so if a playback call ever hangs (e.g. a wedged CoreAudio stream), an
        unbounded wait would freeze the entire turn pipeline — status stuck on
        "speaking", no further turns processed. With a timeout we give up waiting
        and let the pipeline continue instead of deadlocking.
        """
        if timeout is None:
            self._queue.join()
            return
        deadline = time.time() + timeout
        while self._queue.unfinished_tasks > 0:
            if time.time() >= deadline:
                log.warning("TTS wait timed out; continuing without blocking pipeline.")
                return
            time.sleep(0.05)

    def stop_and_flush(self) -> None:
        """Barge-in: discard queued sentences and stop current playback."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break
        sd.stop()

    # ── Playback worker ───────────────────────────────────────────────────────────
    def _playback_worker(self) -> None:
        while not self._stop.is_set():
            try:
                text = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                if self._primary:
                    self._play_kokoro(text)
                else:
                    self._play_macos_say(text)
            except Exception as exc:
                log.error(f"TTS playback error: {exc}")
            finally:
                self._queue.task_done()

    # ── Kokoro playback ───────────────────────────────────────────────────────────
    def _play_kokoro(self, text: str) -> None:
        samples, sample_rate = self._primary.create(
            text=text,
            voice=self._kokoro_voice,
            speed=self.config.get("rate_multiplier", 1.05),
            lang="en-us",
        )
        audio = np.array(samples, dtype=np.float32)

        # Guard playback so a wedged output device (seen as PaMacCore -50 when the
        # mic is also open) can neither hang the worker nor kill the turn. On any
        # audio error, fall back to macOS `say`, which uses a separate process and
        # sidesteps the PortAudio input/output contention.
        try:
            sd.play(audio, samplerate=sample_rate)
            sd.wait()
        except Exception as exc:
            log.warning(f"Kokoro playback failed ({exc}); using macOS say for this line.")
            try:
                sd.stop()
            except Exception:
                pass
            self._play_macos_say(text)

        delay = self.config.get("post_utterance_delay_s", 0.1)
        if delay > 0:
            time.sleep(delay)

    # ── macOS say fallback ────────────────────────────────────────────────────────
    def _play_macos_say(self, text: str) -> None:
        voice = self.config.get("macos_say_voice", "Ava (Premium)")
        subprocess.run(["say", "-v", voice, text], check=False)

    # ── Cleanup ───────────────────────────────────────────────────────────────────
    def stop(self) -> None:
        self._stop.set()
