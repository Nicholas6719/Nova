"""
TTS Engine — Kokoro ONNX (primary) + macOS `say` (fallback).

100% local — no cloud TTS unless explicitly added later.

Playback model (ported from Jarvis's SeamlessPlayer):
  - The streaming pipeline calls speak() per sentence as the LLM streams.
  - A background worker synthesizes each sentence with Kokoro and FEEDS the
    float32 samples into ONE continuous sd.OutputStream (SeamlessPlayer), so
    sentences are joined at sample level — no gap, click, or re-open latency
    between them. (The old per-sentence sd.play/sd.wait produced the ~2s stutter
    and crackle between sentences.)
  - The mic gate is held closed for the WHOLE spoken response (not toggled per
    sentence), so recording never contends with playback mid-utterance.
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

KOKORO_RATE      = 24000   # Kokoro output sample rate
PLAYER_BLOCKSIZE = 1024    # ~42ms at 24kHz — small enough to avoid head-of-speech gap


class SeamlessPlayer:
    """Plays a continuous stream of float32 mono audio fed from sentences.

    Uses one sd.OutputStream with a callback so chunks are joined at sample
    level — no gap or click between sentences.
    """

    def __init__(self, sample_rate: int = KOKORO_RATE) -> None:
        self._sr      = sample_rate
        self._buf     = np.empty(0, dtype=np.float32)
        self._lock    = threading.Lock()
        self._done    = threading.Event()
        self._feeding = True
        self._stream: Optional[sd.OutputStream] = None

    def start(self) -> None:
        self._done.clear()
        self._feeding = True
        self._buf = np.empty(0, dtype=np.float32)
        self._stream = sd.OutputStream(
            samplerate=self._sr,
            channels=1,
            dtype="float32",
            blocksize=PLAYER_BLOCKSIZE,
            callback=self._callback,
        )
        self._stream.start()

    def feed(self, audio: np.ndarray) -> None:
        with self._lock:
            self._buf = np.concatenate((self._buf, audio.ravel().astype(np.float32)))

    def mark_done(self) -> None:
        """No more audio will be fed; drain then signal done."""
        self._feeding = False

    def wait(self) -> None:
        self._done.wait()
        self._close()

    def stop(self) -> None:
        self._feeding = False
        self._done.set()
        self._close()

    def _close(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def _callback(self, outdata: np.ndarray, frames: int, _time, _status) -> None:
        with self._lock:
            have = len(self._buf)
            if have >= frames:
                outdata[:, 0] = self._buf[:frames]
                self._buf = self._buf[frames:]
            elif have > 0:
                outdata[:have, 0] = self._buf
                outdata[have:, 0] = 0.0
                self._buf = np.empty(0, dtype=np.float32)
                if not self._feeding:
                    threading.Timer(0.05, self._done.set).start()
            else:
                outdata[:, 0] = 0.0
                if not self._feeding:
                    self._done.set()


class TTSEngine:
    def __init__(self, config: dict, mic_gate: Optional[threading.Event] = None) -> None:
        self.config   = config
        self._queue   = queue.Queue()
        self._stop    = threading.Event()
        self._primary = None

        # Shared gate with the STT engine. Cleared while audio plays so the mic
        # loops pause — recording + playback on one device causes a CoreAudio
        # conflict (PaMacCore -50) that garbles output. No-op Event if unset.
        self._mic_gate = mic_gate if mic_gate is not None else threading.Event()
        self._mic_gate.set()

        # SeamlessPlayer is created lazily per spoken response.
        self._player: Optional[SeamlessPlayer] = None
        self._speaking = False

        # Start playback worker before loading model so the thread is ready.
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
        # Warm-up: synthesize a near-silent utterance to pre-load the ONNX graph.
        log.info("Warming up Kokoro…")
        self._primary.create(
            text=".",
            voice=self._kokoro_voice,
            speed=1.0,
            lang="en-us",
        )

    # ── Public API ────────────────────────────────────────────────────────────────
    def speak(self, text: str) -> None:
        """Queue a sentence for playback. Returns instantly."""
        text = text.strip()
        if not text:
            return
        self._queue.put(text)

    def wait_until_done(self, timeout: Optional[float] = None) -> None:
        """Block until the queued sentences are fully played out.

        A timeout matters: this runs on the single LLM worker thread, so if
        playback ever wedges, an unbounded wait would freeze the pipeline
        (status stuck on 'speaking'). With a timeout we give up and continue.
        """
        if timeout is None:
            self._queue.join()
        else:
            deadline = time.time() + timeout
            while self._queue.unfinished_tasks > 0:
                if time.time() >= deadline:
                    log.warning("TTS wait timed out; continuing without blocking pipeline.")
                    break
                time.sleep(0.05)
        # Finalize the continuous player and drain any remaining audio.
        self._finish_player()

    def stop_and_flush(self) -> None:
        """Barge-in: discard queued sentences and stop current playback."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break
        if self._player is not None:
            self._player.stop()
            self._player = None
        self._speaking = False
        self._mic_gate.set()

    # ── Playback worker ───────────────────────────────────────────────────────────
    def _playback_worker(self) -> None:
        while not self._stop.is_set():
            try:
                text = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # First sentence of a response: pause the mic and open the player.
            if not self._speaking:
                self._mic_gate.clear()
                self._speaking = True
                if self._primary:
                    self._player = SeamlessPlayer(KOKORO_RATE)
                    self._player.start()

            try:
                if self._primary and self._player is not None:
                    self._synth_kokoro(text)
                else:
                    self._play_macos_say(text)
            except Exception as exc:
                log.error(f"TTS playback error: {exc}")
            finally:
                self._queue.task_done()

    def _finish_player(self) -> None:
        """Called when a response's sentences are all queued: drain the player,
        then re-open the mic."""
        if self._player is not None:
            self._player.mark_done()
            self._player.wait()   # blocks until the buffer empties
            self._player = None
        self._speaking = False
        self._mic_gate.set()

    # ── Kokoro synthesis → SeamlessPlayer ──────────────────────────────────────────
    def _synth_kokoro(self, text: str) -> None:
        samples, sample_rate = self._primary.create(
            text=text,
            voice=self._kokoro_voice,
            speed=self.config.get("rate_multiplier", 1.05),
            lang="en-us",
        )
        audio = np.array(samples, dtype=np.float32)
        if self._player is not None:
            self._player.feed(audio)

    # ── macOS say fallback ────────────────────────────────────────────────────────
    def _play_macos_say(self, text: str) -> None:
        voice = self.config.get("macos_say_voice", "Ava (Premium)")
        subprocess.run(["say", "-v", voice, text], check=False)

    # ── Cleanup ───────────────────────────────────────────────────────────────────
    def stop(self) -> None:
        self._stop.set()
        if self._player is not None:
            self._player.stop()
            self._player = None
