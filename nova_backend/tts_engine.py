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
import re
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


# Audio queued before playback begins. ~0.35s at 24 kHz: enough to ride out the
# pause while the LLM produces the next sentence, short enough that Nova still
# starts speaking promptly.
_PREROLL_SAMPLES = 8192


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
        self._primed  = False
        self._stream: Optional[sd.OutputStream] = None

    def start(self) -> None:
        self._done.clear()
        self._feeding = True
        self._primed = False
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
        """Fill one audio block.

        THE STUTTER LIVED HERE. The old version, whenever the buffer held less
        than a full block, played the fragment and padded the rest of the block
        with SILENCE, then discarded the fragment. While the LLM is still
        generating the next sentence that happens constantly, so speech was
        chopped mid-word with little silences — audible as stuttering.

        A partial buffer while more audio is COMING is an underrun, not the end
        of speech. The fix is to leave the audio in the buffer and emit a whole
        block of silence, so a word is never cut in half; combined with the
        pre-roll in feed(), underruns became rare instead of routine.
        """
        with self._lock:
            have = len(self._buf)

            # Pre-roll: hold silent until enough audio is queued to ride out the
            # gaps between sentences. Without it, playback starts on the first
            # fragment and underruns immediately, which is what made speech
            # break up while the model was still generating.
            if not self._primed:
                if have >= _PREROLL_SAMPLES or not self._feeding:
                    self._primed = True
                else:
                    outdata[:, 0] = 0.0
                    return

            if have >= frames:
                outdata[:, 0] = self._buf[:frames]
                self._buf = self._buf[frames:]
                return

            if self._feeding:
                # More is coming. Wait for a whole block rather than slicing a
                # word in half, and re-arm the pre-roll so we don't restart on
                # a nearly-empty buffer and immediately underrun again.
                self._primed = False
                outdata[:, 0] = 0.0
                return

            # Feeding is finished: drain whatever is left, then stop.
            if have > 0:
                outdata[:have, 0] = self._buf
                outdata[have:, 0] = 0.0
                self._buf = np.empty(0, dtype=np.float32)
                threading.Timer(0.05, self._done.set).start()
            else:
                outdata[:, 0] = 0.0
                self._done.set()


# Kokoro hands text to espeak for phonemization, and espeak is fussy: newlines
# make it treat one utterance as several lines, and a spaced hyphen, symbol, or
# stray markdown character can drop or duplicate phonemes — heard as a stutter.
# Everything spoken is normalized through here first.
_SPEECH_SUBS = (
    ("\u2014", ", "), ("\u2013", ", "),        # em / en dash
    ("&", " and "), ("%", " percent "), ("+", " plus "),
    ("\u221a", " square root of "), ("\u00d7", " times "), ("\u00f7", " divided by "),
    ("\u2192", " to "), ("/", " slash "), ("@", " at "),
    ("\u201c", ""), ("\u201d", ""), ("\u2018", "'"), ("\u2019", "'"),
)


def _normalize_for_speech(text: str) -> str:
    """Make text safe for the phonemizer. Never changes meaning."""
    if not text:
        return ""
    t = str(text)
    # Collapse ALL whitespace, newlines included — this is the 2/1 mismatch.
    t = re.sub(r"\s+", " ", t)
    for a, b in _SPEECH_SUBS:
        t = t.replace(a, b)
    # A hyphen BETWEEN words is a pause; inside a word it belongs to the word.
    t = re.sub(r"\s+-\s+", ", ", t)
    # Markdown and control characters are never spoken.
    t = re.sub(r"[*_`#>|\\]+", "", t)
    t = "".join(ch for ch in t if ch.isprintable())
    # Collapse punctuation runs that make espeak stumble ("?!?!", "....").
    t = re.sub(r"([,.!?;:])\1+", r"\1", t)
    t = re.sub(r"\s+([,.!?;:])", r"\1", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


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
        import numpy as np
        from kokoro_onnx import Kokoro
        self._primary = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")

        # Voice selection. A "kokoro_voice_blend" (list of [name, weight]) defines
        # a CUSTOM voice by weighted-averaging the style vectors of several stock
        # voices — this is Nova's designed voice. Falls back to the single
        # "kokoro_voice" name if no blend is configured. create() accepts either
        # a name string or a style ndarray, so downstream code is unchanged.
        blend = self.config.get("kokoro_voice_blend")
        if blend:
            style = sum(
                self._primary.get_voice_style(name) * float(w) for name, w in blend
            ).astype(np.float32)
            self._kokoro_voice = style
            log.info(f"Kokoro voice: custom blend of {[b[0] for b in blend]}")
        else:
            self._kokoro_voice = self.config.get("kokoro_voice", "af_nova")
            log.info(f"Kokoro voice: {self._kokoro_voice}")

        # Warm-up: synthesize a near-silent utterance to pre-load the ONNX graph.
        log.info("Warming up Kokoro…")
        self._primary.create(
            text=".",
            voice=self._kokoro_voice,
            speed=self.config.get("rate_multiplier", 1.0),
            lang="en-us",
        )

    # ── Public API ────────────────────────────────────────────────────────────────
    def speak(self, text: str) -> None:
        """Queue a sentence for playback. Returns instantly.

        Normalizes here, at the ONE choke point every response passes through.
        Only the LLM path was being cleaned, so deterministic replies reached
        Kokoro raw — including newlines, which made espeak report
        "words count mismatch on 200.0% of the lines (2/1)" and produced the
        stutter Nicholas heard on the screen-awareness and square-root replies.
        """
        text = _normalize_for_speech(text)
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

    def is_speaking(self) -> bool:
        """True while audio is playing or sentences are still queued. Used by
        unprompted announcements (timers) to wait for a safe moment rather than
        talking over an in-flight response."""
        return self._speaking or not self._queue.empty()

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
        settle briefly, then re-open the mic."""
        if self._player is not None:
            self._player.mark_done()
            self._player.wait()   # blocks until the buffer empties
            self._player = None
        # Settle delay before re-opening the mic: the speaker's acoustic tail
        # lingers a beat after the buffer drains, and re-capturing it makes Nova
        # transcribe the end of its own voice. Frames aren't queued while the
        # gate is closed, so this delay is what actually clears the tail.
        delay = float(self.config.get("post_utterance_delay_s", 0.1))
        if delay > 0:
            time.sleep(delay)
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
