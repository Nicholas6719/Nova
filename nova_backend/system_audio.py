"""
The speaker mix, arriving from the app, on its way into the echo canceller.

`echo_canceller.py` could always remove a known sound from the microphone. Its
limit was never the filter, it was the REFERENCE: Nova synthesises her own
voice, so she has a clean copy of that and of nothing else. His music, a video,
a call — all unknowable, so the only defence was pausing the player.

`SystemAudioTap.swift` captures the system mix with ScreenCaptureKit and sends
it here. This module is the seam: a socket, a liveness clock, and a handoff to
`feed_far`. It deliberately holds no audio of its own — buffering belongs to
the canceller, which already drops the oldest reference when it overfills,
because a reference from two seconds ago describes nothing the microphone is
hearing now.

TWO PROPERTIES WORTH STATING PLAINLY
  * It never raises. A missing tap, a closed socket, a malformed packet — each
    costs cancellation and nothing else, and Nova falls back to exactly the
    behaviour she had before: half duplex, music paused while she listens.
  * It knows when it has gone quiet. `is_live` is a clock, not a flag, so a
    tap that dies mid-session degrades honestly instead of leaving Nova
    convinced she is cancelling audio that is really reaching her microphone.
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from typing import Optional

import numpy as np

log = logging.getLogger("nova.systemaudio")

HOST = "127.0.0.1"
PORT = 8767                 # Nova's third port; 5001 and 8766 are invariant 1
RATE = 16_000               # the microphone's rate, so nothing is resampled
_LIVE_WINDOW_S = 0.75       # silence longer than this means the tap is gone


class SystemAudioReceiver:
    """Receives the speaker mix and feeds it to the canceller as the far end."""

    def __init__(self, echo, port: int = PORT) -> None:
        self._echo = echo
        self._port = port
        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._last_packet = 0.0
        self._packets = 0
        self._samples = 0
        self.started = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    def start(self) -> bool:
        if self.started:
            return True
        if self._echo is None or not getattr(self._echo, "enabled", False):
            log.info("no echo canceller, so no use for the speaker mix")
            return False
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # A big receive buffer, because the cost of a full one is a DROPPED
            # reference frame and a moment of uncancelled speaker audio.
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
            except OSError:
                pass
            sock.bind((HOST, self._port))
            sock.settimeout(0.5)
        except Exception as exc:
            log.warning(f"could not open the speaker-mix socket: {exc}")
            return False

        self._sock = sock
        self._thread = threading.Thread(target=self._run, name="nova-system-audio",
                                        daemon=True)
        self._thread.start()
        self.started = True
        log.info(f"listening for the speaker mix on {HOST}:{self._port}")
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
        self.started = False

    # ── State ─────────────────────────────────────────────────────────────────
    @property
    def is_live(self) -> bool:
        """True only while packets are actually arriving.

        A clock rather than a flag on purpose: everything downstream changes
        behaviour based on this — whether Nova keeps listening while she speaks,
        whether the music has to be paused — and all of those should revert the
        moment the tap stops, not the moment somebody remembers to unset a flag.
        """
        return (time.time() - self._last_packet) < _LIVE_WINDOW_S

    @property
    def stats(self) -> dict:
        return {"packets": self._packets, "samples": self._samples,
                "live": self.is_live,
                "age_s": round(time.time() - self._last_packet, 2)
                if self._last_packet else None}

    # ── The receive loop ──────────────────────────────────────────────────────
    def _run(self) -> None:
        buf = bytearray(65536)
        while not self._stop.is_set():
            try:
                n = self._sock.recv_into(buf)
            except socket.timeout:
                continue
            except OSError:
                break                      # socket closed under us; we are done
            except Exception as exc:
                log.debug(f"speaker-mix receive failed: {exc}")
                continue
            if n < 2:
                continue
            try:
                # int16 mono at the mic's own rate. An odd byte count means a
                # truncated packet: keep the whole samples and drop the tail
                # rather than throwing the frame away.
                pcm = np.frombuffer(bytes(buf[: n - (n % 2)]), dtype="<i2")
                if pcm.size == 0:
                    continue
                self._echo.feed_far(pcm.astype(np.float32) / 32768.0, RATE)
                self._last_packet = time.time()
                self._packets += 1
                self._samples += pcm.size
            except Exception as exc:
                log.debug(f"could not queue the speaker mix: {exc}")
