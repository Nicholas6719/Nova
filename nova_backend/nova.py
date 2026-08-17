#!/usr/bin/env python3
"""
Nova — Neural Omniscient Voice Assistant
macOS backend · Entry point

Pipeline:
  Wake word → STT (faster-whisper + webrtcvad)
  → Fast-path routing → Calendar/Reminders → Files → Memory → Tools
  → LLM (MLX Llama, streaming) → Sentence-chunked TTS (Kokoro ONNX)

Communication with SwiftUI:
  HTTP  :5001  — /api/status, /api/messages, /api/message
  WS    :8766  — real-time state + token stream

Run: python nova.py
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import queue
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

# ── Logging ──────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("nova")

# Priorities on the single MLX worker queue. Lower runs first.
_PRIO_TURN       = 0    # the user is waiting on this
_PRIO_BACKGROUND = 1    # reconciliation and friends

# How many unintelligible turns in a row before conversation mode gives up.
# 1 was the old behaviour and it ended the conversation on Nova's own audio
# tail; too high and a noisy room keeps the mic open indefinitely.
_MAX_EMPTY_TURNS = 3

_SLEEP_PHRASES = (
    "go to sleep", "back to sleep", "sleep mode", "stop listening",
    "take a break", "return to wake mode", "wake mode", "that's all",
    "that is all", "never mind", "nevermind", "we're done", "we are done",
    "that'll be all", "that will be all", "goodbye", "good night",
)


# Filler that may trail a sign-off without changing it ("that's all for now").
_SLEEP_TRAILER_RE = re.compile(r"^[\s,.!]*(?:now|please|then|ok|okay|thanks?|"
                               r"thank you|for now|nova)?[\s,.!]*$", re.I)


def find_signoff(text: str):
    """Locate a trailing sign-off. Returns (index, phrase) or (None, None).

    A sleep phrase only ends the conversation when it is the LAST thing said.
    "I was so tired I had to go to sleep early" is a story, not an instruction,
    and used to put Nova to sleep mid-anecdote.
    """
    low = (text or "").lower()
    best = None
    for phrase in _SLEEP_PHRASES:
        start = 0
        while True:
            idx = low.find(phrase, start)
            if idx < 0:
                break
            after = text[idx + len(phrase):]
            if _SLEEP_TRAILER_RE.match(after):        # nothing meaningful after
                if best is None or idx < best[0]:
                    best = (idx, phrase)
            start = idx + 1
    return best if best else (None, None)


def _content_before_sleep(text: str, idx: int) -> str:
    """What the user said BEFORE the sign-off, with any other trailing
    sign-off clauses removed too.

    Stripping only the matched phrase left "...my favorite movie. That's all",
    so the model answered the sign-off ("Goodnight, Nicholas.") instead of the
    sentence he cared about.
    """
    lead = text[:idx].strip()
    changed = True
    while changed:
        changed = False
        stripped = lead.strip().rstrip(",;.! ").strip()
        # Politeness sits BETWEEN the content and the sign-off ("That's all.
        # Please go to sleep"), and left in place it blocked the sign-off strip
        # — Nova was handed "...my favorite movie. That's all. Please".
        stripped = re.sub(r"[,.]?\s*(?:please|now|thanks|thank you)$", "",
                          stripped, flags=re.I).strip().rstrip(",;.! ").strip()
        for phrase in _SLEEP_PHRASES:
            if stripped.lower().endswith(phrase):
                lead = stripped[: -len(phrase)]
                changed = True
                break
        else:
            lead = stripped
    lead = re.sub(r"[,.]?\s*(?:and|then|so|ok|okay|alright)$", "", lead,
                  flags=re.I).strip()
    # Needs to be a real sentence, not a stray word.
    return lead if len(lead.split()) >= 3 else ""

# An imperative that survived every handler. The verbs are ones Nova either
# performs deterministically or cannot perform at all — so if one reaches the
# LLM stage, the honest answer is "I can't", never the model's guess.
_ACTION_REQUEST_RE = re.compile(
    r"^\s*(?:hey\s+)?(?:nova[,\s]+)?(?:please\s+)?"
    r"(?:(?:can|could|would|will)\s+you\s+)?(?:please\s+)?"
    r"(?:go\s+ahead\s+and\s+)?"
    r"(?:open|close|shut|quit|launch|play|pause|resume|skip|type|send|write|"
    r"click|press|move|copy|rename|delete|erase|scroll|drag|install|"
    r"uninstall|download|upload|print|email|text|message|call|order|buy)\b",
    re.I,
)

# ── Paths ─────────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent
DATA_DIR = Path(
    os.environ.get(
        "NOVA_DATA_DIR",
        Path.home() / "Library" / "Application Support" / "Nova",
    )
)
DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_config() -> dict:
    with open(ROOT / "config.json") as f:
        return json.load(f)


# ── Greeting helper ───────────────────────────────────────────────────────────────
def _time_of_day_greeting() -> str:
    hour = datetime.now().hour
    if hour < 12:
        return "Good morning"
    if hour < 17:
        return "Good afternoon"
    return "Good evening"


def _ordinal(n: int) -> str:
    """1 -> '1st', 2 -> '2nd', 4 -> '4th', 21 -> '21st'."""
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _spoken_time(now: datetime) -> str:
    """Read a time the way a person would: '9:02 PM', not '09:02 PM'."""
    hour12 = now.hour % 12 or 12   # 0 -> 12
    return f"{hour12}:{now.minute:02d} {now.strftime('%p')}"


def _spoken_date(now: datetime) -> str:
    """Read a date naturally: 'Tuesday, August 4th, 2026' (no leading zero)."""
    return f"{now.strftime('%A, %B')} {_ordinal(now.day)}, {now.year}"


def _clean_for_tts(text: str) -> str:
    """Strip anything Nova shouldn't speak aloud before synthesis: markdown
    emphasis/heading/quote/code marks, and em/en dashes (turned into a spoken
    pause). Nova speaks its output, so these would be read literally or garble
    phrasing (CLAUDE.md invariant #10)."""
    if not text:
        return ""
    t = text.replace("—", ", ").replace("–", ", ")
    t = re.sub(r"[*_`#>]+", "", t)
    # Numbered list markers, before whitespace is collapsed and the line starts
    # are lost. The bullet characters are already gone with the markdown strip
    # above, but "1." survived it and was read aloud as "one." — measured, the
    # 3B still answers document questions with a numbered list despite the
    # prompt forbidding it (CLAUDE.md invariant #10).
    t = re.sub(r"(?m)^[ \t]*\d+[.)][ \t]+", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ═══════════════════════════════════════════════════════════════════════════════════
# VoiceAssistant
# ═══════════════════════════════════════════════════════════════════════════════════
class VoiceAssistant:
    """
    Central orchestrator for the Nova voice pipeline.

    Routing order (first match wins):
      1. System commands  (sleep / wake / mute)
      2. Calendar follow-up offer ("want to hear what's coming up?" -> yes/no)
      2b. Pending file question ("which one?" / "move it to Documents?")
      2c. UI navigation  (go home / show me the menu / go to finance)
      3. Calendar intents (read/create/complete/delete/update events + reminders)
      4. Screen awareness (what's on my screen / what app am I in)
      4b. Weather (current / tomorrow / next few days, here or a named place)
      5. File intents     (find / read / open / move / copy / rename)
      6. Memory intents   (remember / recall / update / forget)
      7. Fast-path intents (greeting / date / time / repeat)
      8. Tool intents     (open app / volume / battery / search / screenshot)
      9. RAG context enrichment
      10. LLM fallback    (MLX streaming + sentence-chunked TTS)
    """

    # ── Init ──────────────────────────────────────────────────────────────────────
    def __init__(self) -> None:
        self.config = load_config()
        self._init_state()
        self._init_engines()

    def _init_state(self) -> None:
        """All plain per-run state, in ONE place.

        Extracted so test harnesses can call it instead of hand-assigning the
        fields. They used to reproduce this list by hand, which is how a test
        object drifted from the real one — the `screen` engine was missing from
        __init__ for a whole session because the harness set it up itself, and
        adding `_sleep_after_turn` here broke the loop suite the same way.
        A harness that calls this cannot drift again.
        """
        self.is_muted         = False
        self.is_awake         = True
        self._last_response   = ""
        # A SOFT one-shot follow-up armed by a calendar read ("want to hear
        # what's coming up?"). It only fires on an affirmative reply and never
        # eats an unrelated next command.
        self._calendar_offer: Optional[Callable[[], str]] = None
        # A STRICT one-shot confirmation armed by a destructive tool action
        # (shutdown / restart / sleep). Unlike the soft offer, anything that
        # isn't a clear yes cancels it — we never guess about powering off.
        self._tool_confirm: Optional[Callable[[], str]] = None
        self._rag_ready       = False
        self._rag             = None
        # Set by a "return to wake mode" voice command to drop out of
        # conversation mode immediately (without full sleep).
        self._return_to_wake  = False
        # Transcript of the CURRENT conversation (user+assistant), used by the
        # wake-mode memory reconciliation pass. Reset when a conversation ends.
        self._session_turns: list[dict] = []
        self._turn_was_command = False
        # Set when a sleep phrase rode along with real content: answer the
        # content first, then return to wake mode.
        self._sleep_after_turn = False

        # Shared microphone gate. Set = capture allowed; cleared = mic paused.
        # A voice assistant must not record while it speaks — both to avoid
        # hearing its own TTS and to avoid a CoreAudio input/output device
        # conflict (PaMacCore -50) that garbles playback. TTS clears this around
        # playback; the STT record loops skip capture while it is clear.
        self.mic_gate = threading.Event()
        self.mic_gate.set()

    def _init_engines(self) -> None:
        """Load every engine, in order. Separate from _init_state so a test
        harness can take the state without booting the mic and the servers."""
        log.info("Nova initializing…")
        self._init_stt()
        self._init_llm()
        self._init_tts()
        self._init_memory()
        self._init_rag()
        self._init_tools()
        self._init_calendar()
        self._init_files()
        self._init_screen()
        self._init_weather()
        self._init_ws()
        self._init_views()          # after _init_ws: it needs the WS server
        self._verify_engines()
        self._warm_prompt_cache()
        log.info("Nova ready.")

    def _warm_prompt_cache(self) -> None:
        """Pre-process the unchanging head of the system prompt so the FIRST
        thing Nicholas says is answered as fast as the second.

        Submitted as a BACKGROUND job because it has to run on the nova-llm
        thread (MLX is thread-local) — and at background priority a real turn
        arriving mid-warm jumps ahead of it rather than waiting."""
        if not self.config["llm"].get("prompt_cache", True):
            return

        def _warm() -> None:
            try:
                self.llm.warm(self._build_system_prompt(
                    self.memory.get_context_for_llm(), ""))
            except Exception as exc:
                # Never fatal: an unwarmed cache is just today's speed.
                log.warning(f"Prompt cache warm-up skipped: {exc}")

        self._submit_job(_warm)

    # Every engine the router dereferences in _handle_turn_impl. A missing one
    # is fatal: it raises on EVERY turn, so Nova starts, connects, and then
    # fails silently on everything the user says. That happened — `screen` was
    # added to the router while its `_init_screen()` call never made it into
    # __init__, and the behaviour suite missed it because the test harness
    # initialized engines by hand. Fail loudly at startup instead.
    _REQUIRED_ENGINES = ("stt", "llm", "tts", "memory", "tools",
                         "calendar", "files", "screen", "weather", "ws", "views")

    def _verify_engines(self) -> None:
        missing = [name for name in self._REQUIRED_ENGINES
                   if getattr(self, name, None) is None]
        if missing:
            raise RuntimeError(
                "Nova cannot start: routing depends on engines that were never "
                f"initialized: {', '.join(missing)}. Add the matching "
                "_init_<name>() call to VoiceAssistant.__init__."
            )

    def _init_stt(self) -> None:
        log.info("Loading STT (faster-whisper)…")
        from stt_engine import STTEngine
        self.stt = STTEngine(
            self.config["stt"],
            mic_gate=self.mic_gate,
            wake_config=self.config.get("wake_word", {}),
        )

    def _init_llm(self) -> None:
        # MLX arrays and Metal GPU streams are thread-local: the model must be
        # loaded on, and every generation run on, ONE consistent thread. We use a
        # single long-lived worker thread that owns MLX end-to-end. Both the voice
        # loop (main thread) and text input (HTTP/WS threads) submit turns to it
        # via a queue, so MLX is never touched off-thread.
        self.llm: Optional["object"] = None
        # PRIORITY queue, not FIFO. A user turn and the wake-mode fact
        # reconciliation share this thread (MLX is thread-local, so everything
        # must run here). Measured: a reconcile pass blocks for ~2.4s, and with
        # a plain FIFO the user's next turn queued BEHIND it — so speaking again
        # right after a conversation ended felt sluggish. User turns now jump
        # ahead of background work. The tie-breaker counter is required because
        # the payloads are callables and are not comparable.
        self._llm_queue: "queue.PriorityQueue[tuple]" = queue.PriorityQueue()
        self._llm_seq = itertools.count()
        self._llm_ready = threading.Event()
        self._llm_thread = threading.Thread(
            target=self._llm_worker, daemon=True, name="nova-llm"
        )
        self._llm_thread.start()
        # Block startup until the model is loaded on the worker thread so the
        # "Nova ready." ordering and first-turn latency stay predictable.
        self._llm_ready.wait()

    def _llm_worker(self) -> None:
        """Owns MLX for the process lifetime: loads the model, then serially
        executes every job submitted via ``self._llm_queue``. A job is a
        (callable, done_event) pair — usually a conversation turn, but also the
        wake-mode fact-reconciliation pass. Serializing everything through this
        one thread is required because MLX is thread-local."""
        log.info("Loading LLM (MLX)…")
        from llm_engine import LLMEngine
        self.llm = LLMEngine(self.config["llm"])
        self._llm_ready.set()

        while True:
            _prio, _seq, job, done = self._llm_queue.get()
            try:
                job()
            except Exception:
                log.exception("LLM job failed")
                self.set_state("idle")
            finally:
                if done is not None:
                    done.set()
                self._llm_queue.task_done()

    def _submit_turn(self, text: str, wait: bool) -> None:
        """Enqueue a conversation turn for the MLX worker thread. Voice waits for
        completion (so the wake loop doesn't resume mid-response); text input
        returns immediately and receives results over the WebSocket."""
        done = threading.Event() if wait else None
        self._llm_queue.put(
            (_PRIO_TURN, next(self._llm_seq),
             lambda: self._handle_turn_impl(text), done)
        )
        if done is not None:
            done.wait()

    def _submit_job(self, fn, wait: bool = False,
                    priority: int = None) -> None:
        """Enqueue an arbitrary LLM job (e.g. wake-mode reconciliation) onto the
        MLX worker thread. Off the live path; not user-facing, so it runs at
        BACKGROUND priority and never delays a turn the user is waiting on."""
        done = threading.Event() if wait else None
        prio = _PRIO_BACKGROUND if priority is None else priority
        self._llm_queue.put((prio, next(self._llm_seq), fn, done))
        if done is not None:
            done.wait()

    def _init_tts(self) -> None:
        log.info("Loading TTS (Kokoro ONNX)…")
        from tts_engine import TTSEngine
        self.tts = TTSEngine(self.config["tts"], mic_gate=self.mic_gate)

    def _init_memory(self) -> None:
        from memory import NovaMemory
        self.memory = NovaMemory(DATA_DIR / "nova_memory.db")

    def _init_rag(self) -> None:
        """RAG index loads in background — never blocks startup."""
        if self.config["memory"].get("rag_enabled", True):
            t = threading.Thread(target=self._load_rag_background, daemon=True)
            t.start()

    def _load_rag_background(self) -> None:
        try:
            from rag import NovaRAG, DEFAULT_MAX_DISTANCE
            docs_dir = Path(self.config["memory"]["rag_docs_dir"]).expanduser()
            self._rag = NovaRAG(
                DATA_DIR / "rag_index",
                docs_dir=docs_dir,
                max_distance=self.config["memory"].get(
                    "rag_max_distance", DEFAULT_MAX_DISTANCE),
            )
            self._rag_ready = True
            log.info("RAG index ready.")
        except Exception as exc:
            log.warning(f"RAG unavailable: {exc}")

    def _init_tools(self) -> None:
        from tools import NovaTools
        # on_announce lets a timer speak when it fires — nothing is asking at
        # that moment, so it needs its own safe path to the floor.
        self.tools = NovaTools(self.config, on_announce=self._announce)

    def _init_calendar(self) -> None:
        # NovaCalendar's LLM extraction/summarize calls run on the nova-llm
        # worker thread (calendar handling is dispatched from _handle_turn_impl,
        # which is already on that thread), so passing self.llm is thread-safe.
        from calendar_intents import NovaCalendar
        self.calendar = NovaCalendar(self.config, self.llm, self.memory)

    def _init_files(self) -> None:
        # Same thread story as the calendar: file handling is dispatched from
        # _handle_turn_impl on the nova-llm worker, so its summarization call
        # into self.llm is thread-safe.
        from file_intents import NovaFiles
        self.files = NovaFiles(self.config, self.llm)

    def _init_screen(self) -> None:
        # Same thread story as the calendar and files: dispatched from
        # _handle_turn_impl on the nova-llm worker, so the summarization call
        # into self.llm is thread-safe. Nothing loads at import — the Vision
        # framework is imported lazily inside ocr().
        from screen_awareness import NovaScreen
        self.screen = NovaScreen(self.config, self.llm)

    @property
    def _duck_enabled(self) -> bool:
        return bool(self.config.get("music", {}).get("duck_while_listening", True))

    def _init_weather(self) -> None:
        # Deterministic end to end: no model is involved in phrasing a
        # temperature. See weather_engine.py for the invariant-3 decision.
        from weather_intents import NovaWeather
        self.weather = NovaWeather(self.config)

    def _init_ws(self) -> None:
        from ws_server import NovaWSServer
        self.ws = NovaWSServer(
            http_port=self.config["server"]["http_port"],
            ws_port=self.config["server"]["ws_port"],
            on_text_message=self._handle_text_input,
        )
        self.ws.start()
        # Let the maps engine ask the app for a location when it needs one.
        # Only the app bundle can obtain a fix, and it used to volunteer one
        # exactly once at launch — a POST that raced this very startup.
        try:
            import maps_engine
            maps_engine.set_location_requester(self.ws.request_location)
        except Exception as exc:
            log.warning(f"could not wire the location requester: {exc}")

    def _init_views(self) -> None:
        # UI navigation ("go home", "show me the menu"). Needs the WS server, so
        # this must stay AFTER _init_ws. Deterministic regex only — the 3B is
        # never asked which screen he meant.
        # getattr, not self.ws: the routing harness calls the REAL _init_views()
        # without booting a server, and a harness that had to build NovaViews
        # itself is exactly the drift rule 1 exists to prevent.
        from views import NovaViews
        self.views = NovaViews(self.config, ws=getattr(self, "ws", None))

    # ── State broadcasting ────────────────────────────────────────────────────────
    def set_state(self, state: str) -> None:
        log.info(f"[state] → {state}")
        self.ws.broadcast_state(state)

    # ── Main run loop ─────────────────────────────────────────────────────────────
    def run(self) -> None:
        log.info("Starting voice pipeline.")
        self._start_parent_watchdog()
        self.set_state("idle")
        try:
            self._main_loop()
        except KeyboardInterrupt:
            log.info("Nova shutting down.")
            self.ws.stop()

    def _start_parent_watchdog(self) -> None:
        """Exit if the launching app dies.

        The Swift app terminates the backend on a clean quit, but if it is
        SIGKILLed (e.g. stopped from Xcode) it never runs cleanup and the backend
        would linger headless — still holding the mic and responding to speech.
        BackendManager passes its PID as NOVA_PARENT_PID; poll it and exit when it
        disappears. No-op when launched standalone (no parent PID set)."""
        parent_pid_env = os.environ.get("NOVA_PARENT_PID")
        if not parent_pid_env:
            return
        try:
            parent_pid = int(parent_pid_env)
        except ValueError:
            return

        def watch() -> None:
            while True:
                time.sleep(2.0)
                try:
                    os.kill(parent_pid, 0)   # signal 0 = liveness check only
                except OSError:
                    log.info(f"Parent app (pid {parent_pid}) gone — shutting down.")
                    os._exit(0)              # hard exit: end all threads + audio

        threading.Thread(target=watch, daemon=True, name="nova-parent-watchdog").start()
        log.info(f"Parent watchdog armed (pid {parent_pid}).")

    def _main_loop(self) -> None:
        in_conversation = False   # once awake, keep listening without the wake word
        empty_turns = 0           # consecutive unintelligible turns
        while True:
            if self.is_muted or not self.is_awake:
                in_conversation = False
                time.sleep(0.1)
                continue

            # ── Phase 1: Get the user's turn ─────────────────────────────────
            if not in_conversation:
                # Wake mode: wait for "Nova" before listening.
                self.set_state("idle")
                # Back to waiting means the conversation is over, whatever
                # ended it — so the music goes back up here. Restoring on this
                # ONE path covers every exit (timeout, sign-off, empty turns,
                # a handler raising) and is a no-op when nothing was ducked.
                if self._duck_enabled:
                    self.tools.restore_music()
                log.info("Waiting for wake word…")
                wake_detected = self.stt.record_wake(
                    wake_keywords=self.config["wake_word"]["keywords"],
                    timeout_s=300.0,
                )
                if not wake_detected:
                    continue
                just_woke = True
                empty_turns = 0
                # Duck for the WHOLE conversation, not per turn: restoring
                # between turns would pump the volume up and down every time
                # Nova answered. It stays down until we are back up top.
                if self._duck_enabled:
                    self.tools.duck_music()
            else:
                just_woke = False

            # ── Phase 2: Record command (VAD-gated) ──────────────────────────
            self.set_state("listening")
            # In conversation mode, cap how long we wait for the user to start
            # speaking; if they stay silent past the timeout, drop back to wake.
            conv_timeout = float(self.config["wake_word"].get("conversation_timeout_s", 15.0))
            command_audio = self.stt.record_command(
                max_duration_s=self.config["stt"]["command_max_duration_s"],
                start_timeout_s=None if just_woke else conv_timeout,
            )

            if command_audio is None:
                if in_conversation:
                    # record_command returns None for TWO different things, and
                    # conflating them ended the conversation the instant Nova
                    # stopped speaking: its own audio tail was captured, fell
                    # below the noise floor, and looked identical to silence.
                    if self.stt.last_capture_reason == "too_quiet":
                        empty_turns += 1
                        if empty_turns < _MAX_EMPTY_TURNS:
                            log.info(f"Caught only faint noise ({empty_turns}/"
                                     f"{_MAX_EMPTY_TURNS}) — still listening.")
                            continue
                        log.info(f"{empty_turns} unusable turns — "
                                 "returning to wake mode.")
                    else:
                        log.info("Conversation timed out — returning to wake mode.")
                    self._end_conversation()
                    in_conversation = False
                    empty_turns = 0
                    continue
                # Woke but heard nothing usable — just re-listen for the wake word.
                continue

            # ── Phase 3: Transcribe ──────────────────────────────────────────
            self.set_state("processing")
            text = self.stt.transcribe(command_audio)
            if just_woke:
                text = self._strip_wake_prefix(text)
            if not text or not text.strip():
                # An empty transcription is NOT the user leaving. After Nova
                # speaks, the mic routinely catches its own tail or a room noise:
                # VAD fires, a fragment is recorded, and Whisper returns nothing.
                # Ending the conversation on the FIRST one dropped him back to
                # wake mode the instant Nova finished answering — and, because
                # every end fires a reconciliation pass, it also queued LLM work
                # that slowed his next turn. Tolerate a few, then give up so we
                # can never get stuck listening to a noisy room forever.
                if in_conversation:
                    empty_turns += 1
                    if empty_turns >= _MAX_EMPTY_TURNS:
                        log.info(f"{empty_turns} unintelligible turns — "
                                 "returning to wake mode.")
                        self._end_conversation()
                        in_conversation = False
                    else:
                        log.info(f"Heard nothing usable ({empty_turns}/"
                                 f"{_MAX_EMPTY_TURNS}) — still listening.")
                continue

            empty_turns = 0        # a real utterance resets the tolerance
            log.info(f"[user] {text}")
            self.memory.add_turn("user", text)
            self._session_turns.append({"role": "user", "content": text})
            self._turn_was_command = False
            self.ws.send_message("user", text)
            # Run on the MLX worker thread and wait: capture must not resume
            # until the response (and its TTS) is done, or it self-captures.
            self._submit_turn(text, wait=True)
            if self._sleep_after_turn:
                # The utterance carried both content and a sign-off; the content
                # has now been answered, so honor the sign-off.
                self._sleep_after_turn = False
                self._return_to_wake = True
            # A "return to wake mode" command during the turn drops us out of
            # conversation immediately, without waiting for the silence timeout.
            if self._return_to_wake:
                self._return_to_wake = False
                self._end_conversation()
                in_conversation = False
            else:
                # Otherwise stay in conversation mode — the user and Nova are
                # talking; only silence (Phase 2 timeout) returns us to wake mode.
                in_conversation = True

    def _mark_turn_was_command(self) -> None:
        """Drop the just-recorded user turn from the learning transcript.

        Called from _respond, i.e. whenever a deterministic stage answered.
        "Open Brave" and "close downloads" say nothing durable about Nicholas,
        and letting the reconciler chew on them produced pure garbage facts.
        """
        self._turn_was_command = True
        if self._session_turns and self._session_turns[-1].get("role") == "user":
            self._session_turns.pop()

    def _end_conversation(self) -> None:
        """Called when the conversation ends and Nova returns to wake mode.
        Kicks off the memory reconciliation pass over what was just said — on the
        nova-llm worker thread, off the live path, silent."""
        turns = self._session_turns
        self._session_turns = []
        if not turns:
            return

        # Only the USER's turns are a source of facts about Nicholas. Feeding
        # Nova's own replies to the reconciler made it extract facts about itself
        # ("assistant/name = Nova") and waffle on existing facts. Skip them.
        user_lines = [t["content"] for t in turns if t["role"] == "user"]
        if not user_lines:
            return

        def job() -> None:
            from fact_reconciler import reconcile
            convo = "\n".join(f"user: {line}" for line in user_lines)
            try:
                reconcile(self.memory, self.llm, convo)
            except Exception:
                log.exception("Wake-mode memory reconciliation failed")

        # Non-blocking: queued behind any in-flight turn, runs while idle.
        self._submit_job(job, wait=False)

    def _strip_wake_prefix(self, text: str) -> str:
        """Remove a leading wake phrase from a single-breath command.

        With the always-on stream, "Nova, what time is it?" transcribes whole, so
        strip the leading "nova"/"hey nova" (and trailing punctuation) to get the
        actual command. Returns "" for a bare wake phrase with no command."""
        if not text:
            return ""
        stripped = text.strip()
        # Longest keywords first so "hey nova" wins over "nova".
        keywords = sorted(
            (kw.lower() for kw in self.config["wake_word"]["keywords"]),
            key=len, reverse=True,
        )
        # REPEATEDLY, not once. Whisper can return the wake word over and over
        # from a capture that was only the wake word ("Nova. Nova. Nova." a
        # hundred times, seen live). Stripping a single one left the other
        # hundred to be sent to the LLM as though Nicholas had said them.
        while True:
            low = stripped.lower()
            for kw in keywords:
                # startswith alone is not enough: it turned "Novak won the
                # match" into "k won the match". The wake word only counts when
                # a word actually ends there.
                if low.startswith(kw) and not stripped[len(kw):len(kw) + 1].isalnum():
                    remainder = stripped[len(kw):]
                    # Drop the punctuation/space that follows the wake word.
                    stripped = remainder.lstrip(" ,.!?—-").strip()
                    break
            else:
                return stripped
            if not stripped:
                return ""

    # ── Text input from SwiftUI (typed / programmatic) ────────────────────────────
    def _handle_text_input(self, text: str) -> None:
        if not text.strip():
            return
        log.info(f"[text-input] {text}")
        self.memory.add_turn("user", text)
        self.ws.send_message("user", text)
        # Submit to the MLX worker and return immediately; the response streams
        # back over the WebSocket. (Never touch MLX from this HTTP/WS thread.)
        self._submit_turn(text, wait=False)

    # ═════════════════════════════════════════════════════════════════════════════
    # Core turn handler
    # ═════════════════════════════════════════════════════════════════════════════
    def _handle_turn_impl(self, text: str) -> None:
        """Full pipeline for one user utterance. First match wins.

        Always runs on the ``nova-llm`` worker thread (via ``_submit_turn``) so
        every MLX access stays on the thread that owns the model."""
        text = text.strip()

        # ── 1. System commands ───────────────────────────────────────────────
        resp = self._handle_system_commands(text)
        if resp is not None:
            self._respond(resp)
            return

        # ── 1b. Destructive-action confirmation (STRICT yes/no) ──────────────
        # Armed by a power command. Only an explicit yes proceeds; ANYTHING
        # else cancels — we never power off the Mac on an ambiguous reply.
        if self._tool_confirm is not None:
            confirm = self._tool_confirm
            self._tool_confirm = None
            if re.match(r"^\s*(yes|yeah|yep|yup|do it|confirm|go ahead|please do)\b",
                        text.lower().strip()):
                self._respond(confirm())
            else:
                self._respond("Cancelled.")
            return

        # ── 2. Soft calendar follow-up offer ─────────────────────────────────
        # A read may have offered "want to hear what's coming up?". Honor a
        # yes/no reply here, but if the user says something else, drop the offer
        # and let their command flow normally (never eat an unrelated command).
        if self._calendar_offer is not None:
            offer = self._calendar_offer
            self._calendar_offer = None
            low = text.lower().strip()
            if re.match(r"^(ye(s|ah|p)|sure|please|ok|okay|go ahead|do it|yes\s+please)\b", low):
                self._respond(offer())
                return
            if re.match(r"^(no|nope|nah|no\s+thanks|that'?s\s+all|i'?m\s+good)\b", low):
                self._respond(f"Alright, {self.config['user']['address_as']}.")
                return
            # Otherwise: not a reply to the offer — fall through to normal routing.

        # ── 2b. File question Nova is waiting on ─────────────────────────────
        # "Which one?" or "Move it to Documents?" — answered here so the reply
        # can't be re-read as a fresh command. resolve_pending returns None for
        # anything that isn't an answer, which drops the question and lets the
        # utterance route normally. Nothing on disk changes without a clear yes.
        if self.files.has_pending():
            resp = self.files.resolve_pending(text)
            if resp is not None:
                if self.files.pending_offer is not None:
                    self._calendar_offer = self.files.pending_offer
                    self.files.pending_offer = None
                self._respond(resp)
                return

        # ── 2c. UI navigation ────────────────────────────────────────────────
        # "go home", "show me the menu", "go to finance". BEFORE calendar,
        # weather, files and tools, all of which would otherwise claim these:
        # "go to the weather" has a weather word, "open the memory panel" looks
        # like an app launch, "go back home" looked like a browser Back.
        # Detection is anchored at BOTH ends and the destination must be a known
        # view, so "I go home every friday" and "I want to go to italy someday"
        # fall straight through to conversation.
        view_intent = self.views.detect_intent(text)
        if view_intent is not None:
            self._respond(self.views.handle(view_intent, text))
            return

        # ── 3. Calendar / reminders intents ──────────────────────────────────
        # BEFORE memory + fast-path + tools: detection is strict (needs an
        # unambiguous calendar word), and running it early stops the greedy
        # memory-recall regex ("what's my X") from swallowing "what's my
        # calendar today" and the tool regexes ("find X"/"open X") from
        # grabbing calendar phrasing.
        cal_intent = self.calendar.detect_intent(text)
        if cal_intent is not None:
            resp = self.calendar.handle(cal_intent, text)
            # A read handler may arm a soft follow-up (e.g. read_rest_of_week).
            followup = self.calendar.pending_intent
            self.calendar.pending_intent = None
            if followup:
                self._calendar_offer = lambda fi=followup: self.calendar.handle(fi, "")
            self._respond(resp)
            return

        # ── 4. Screen awareness ──────────────────────────────────────────────
        # BEFORE files: "screen" is not a file stopword, so "read my screen"
        # would otherwise become a file search for a document named "screen".
        # Detection is strict — it needs an explicit screen phrase.
        screen_intent = self.screen.detect_intent(text)
        if screen_intent is not None:
            self._respond(self.screen.handle(screen_intent, text))
            return

        # ── 4b. Weather ───────────────────────────────────────────────────────
        # BEFORE files and tools: "what's the weather" contains no file words,
        # but "is it going to rain tomorrow" would otherwise fall through to
        # the LLM, which would happily invent a forecast. Detection is strict:
        # a weather word must appear, and an action verb rules it out.
        weather_intent = self.weather.detect_intent(text)
        if weather_intent is not None:
            self._respond(self.weather.handle(text, weather_intent))
            return

        # ── 5. File management intents ───────────────────────────────────────
        # BEFORE memory and tools. Tools' web-search rule ("find X") and app
        # launch ("open X") would both swallow file phrasing, and memory recall
        # ("what's my X") would answer "I don't have your resume stored yet".
        # Detection is strict — it needs a file word AND a searchable name —
        # so ordinary conversation still falls straight through.
        file_intent = self.files.detect_intent(text)
        if file_intent is not None:
            resp = self.files.handle(file_intent, text)
            if self.files.pending_offer is not None:
                self._calendar_offer = self.files.pending_offer
                self.files.pending_offer = None
            self._respond(resp)
            return

        # ── 6. Memory intents ────────────────────────────────────────────────
        resp = self._handle_memory_intent(text)
        if resp is not None:
            self._respond(resp)
            return

        # ── 7. Fast-path intents ─────────────────────────────────────────────
        resp = self._fast_path(text)
        if resp is not None:
            self._respond(resp)
            return

        # ── 8. Tool intents ──────────────────────────────────────────────────
        resp = self.tools.match(text)
        if resp is not None:
            # A destructive tool (power) asks first and hands back the action
            # to run only if the user confirms on the next turn.
            self._tool_confirm = self.tools.pending_confirm
            self.tools.pending_confirm = None
            # A SOFT offer ("want me to pull up directions?") reuses the same
            # one-shot slot as the calendar follow-up: yes runs it, anything
            # else just drops it and routes normally.
            if self.tools.pending_offer is not None:
                self._calendar_offer = self.tools.pending_offer
                self.tools.pending_offer = None
            self._respond(resp)
            return

        # ── 8b. Unsupported ACTION request ───────────────────────────────────
        # Reaching this point means no handler claimed the utterance. If it is
        # phrased as an ACTION ("open the coding projects folder", "type this
        # and send it", "close downloads"), then by definition Nova cannot do
        # it — and the LLM must never be the one to answer, because it says it
        # already did. Measured: told to open a folder it cannot open, the 3B
        # replied "I've opened the coding projects folder for you." A system
        # prompt rule cut that from 4 cases to 1; only refusing deterministically
        # makes it zero.
        if _ACTION_REQUEST_RE.match(text):
            log.info(f"[unsupported-action] {text}")
            self._respond(
                "I can't do that one yet. I'll tell you when I can, rather "
                "than pretend I did it."
            )
            return

        # ── 9. RAG context enrichment ────────────────────────────────────────
        rag_ctx = ""
        if self._rag_ready and self._rag:
            try:
                rag_ctx = self._rag.query(text, n_results=3)
            except Exception:
                pass

        # ── 10. LLM fallback (streaming) ──────────────────────────────────────
        self.set_state("thinking")
        self._stream_response(text, rag_context=rag_ctx)

    # ── Unprompted announcements (timers) ─────────────────────────────────────────
    def _announce(self, text: str) -> None:
        """Speak something the user did NOT just ask for (a timer firing).

        Called from a timer thread, so it must find a safe moment rather than
        cutting in: wait out any in-flight TTS, and hold while muted instead of
        dropping the message. Nova is half-duplex, so speaking on top of a
        response would also make it hear itself."""
        deadline = time.time() + 120
        while time.time() < deadline:
            if self.is_muted:
                time.sleep(0.5)
                continue
            if self.tts.is_speaking():
                time.sleep(0.2)
                continue
            break
        log.info(f"[announce] {text}")
        self.memory.add_turn("assistant", text)
        self.ws.send_message("assistant", text)
        self.set_state("speaking")
        self.tts.speak(_clean_for_tts(text))     # same boundary as _respond
        self.tts.wait_until_done(timeout=60)
        self.set_state("idle")

    # ── Respond helper ────────────────────────────────────────────────────────────
    def _respond(self, text: str) -> None:
        """Canonical response path: log → store → broadcast → speak."""
        text = text.strip()
        if not text:
            return
        self._last_response = text
        self.memory.add_turn("assistant", text)
        # Deliberately NOT added to _session_turns. This is the deterministic
        # command path; only real conversation is worth learning from. Feeding
        # commands to the reconciler is how the memory DB ended up holding
        # ('adjective', 'adjective', 'open,brave,brave,brave,brave').
        self._mark_turn_was_command()
        self.ws.send_message("assistant", text)
        log.info(f"[nova] {text}")
        self.set_state("speaking")
        # Clean at the SPEAKING boundary, not just on the LLM path. Everything
        # deterministic came through here uncleaned — and the offending text is
        # not always Nova's own words: real window titles contain em dashes, so
        # screen awareness said "You're in Xcode, on Nova — NovaApp.swift" and
        # spoke the dash. Invariant 10 is about what reaches the speaker,
        # whatever produced it. The UI and the log keep the original.
        self.tts.speak(_clean_for_tts(text))
        self.tts.wait_until_done(timeout=60)
        self.set_state("idle")

    # ═════════════════════════════════════════════════════════════════════════════
    # Fast-path intents (date / time / greeting / repeat)
    # ═════════════════════════════════════════════════════════════════════════════
    _GREETING_WORDS = frozenset(
        {"hi", "hello", "hey", "morning", "afternoon", "evening", "howdy", "sup", "yo"}
    )
    _DATE_PHRASES = frozenset(
        {"what day is it", "what's today", "what is today", "today's date", "what date is it",
         "what is the date", "what's the date"}
    )
    _TIME_PHRASES = frozenset(
        {"what time is it", "what's the time", "current time", "time please"}
    )

    def _fast_path(self, text: str) -> Optional[str]:
        low  = text.lower().strip(" .!?")
        name = self.config["user"]["address_as"]

        # Pure greeting (only greeting words, possibly with name)
        words = set(low.split())
        allowed = self._GREETING_WORDS | {name.lower(), "nova"}
        if words and words.issubset(allowed):
            return f"{_time_of_day_greeting()}, {name}."

        # Date and/or time. WORD-BOUNDARY matching, and compound requests are
        # answered in full. Substring tests used to route "what's 10 TIMES 10
        # minus 4" to the clock, and "what's the date and what time is it?"
        # answered only the date.
        wants_date = bool(re.search(r"\bdate\b|\bwhat day is it\b|\btoday'?s date\b", low))
        wants_time = bool(re.search(r"\btime\b", low)) and not re.search(
            r"\btimes\b|\btimer\b|\btime\s+left\b|\bhow\s+many\s+times\b", low)
        # Only a genuine ASK, not any sentence mentioning the words.
        asking = bool(re.match(r"\s*(?:hey\s+|nova[, ]\s*)?(?:what|when|tell me)\b", low)) \
            or low in self._DATE_PHRASES or low in self._TIME_PHRASES
        if asking and wants_date and wants_time:
            now = datetime.now()
            return (f"Today is {_spoken_date(now)}, and it's {_spoken_time(now)}.")
        if asking and wants_date:
            return f"Today is {_spoken_date(datetime.now())}."

        if asking and wants_time:
            return f"It's {_spoken_time(datetime.now())}."

        # Day of week
        if "what day" in low and "is it" in low:
            return f"Today is {datetime.now().strftime('%A')}."

        # Repeat last response
        # "what did you JUST say" is the natural phrasing and used to miss,
        # because the literal list required "what did you say" with nothing
        # in between.
        if any(p in low for p in ("say that again", "repeat that", "come again")) \
           or re.search(r"what did you (?:just )?say", low) \
           or re.search(r"repeat (?:that|yourself|it)", low):
            return self._last_response or "I haven't said anything yet."

        return None

    # ═════════════════════════════════════════════════════════════════════════════
    # System commands
    # ═════════════════════════════════════════════════════════════════════════════
    def _handle_system_commands(self, text: str) -> Optional[str]:
        low  = text.lower().strip()
        name = self.config["user"]["address_as"]

        # End the conversation and return to wake mode. One level only: "go to
        # sleep", "that's all", 15s of silence — all just drop back to waiting for
        # "Nova". Nova never goes fully dormant, so saying "Nova" always wakes it
        # (no way to get stuck asleep).
        sleep_idx, sleep_hit = find_signoff(text)
        if sleep_hit:
            # A sleep phrase TACKED ONTO real content must not throw the content
            # away. "Spider-Man No Way Home is my favorite movie. That's all, go
            # to sleep." answered only the sleep part — the sentence he actually
            # cared about was never replied to and never learned from.
            lead = _content_before_sleep(text, sleep_idx)
            if lead:
                # Answer the content on this turn, and sleep once it is spoken.
                self._sleep_after_turn = True
                log.info(f"[system] sleep deferred; answering {lead!r} first")
                return None
            self._return_to_wake = True
            self.set_state("idle")
            return f"Alright, {name}. Just say Nova when you need me."

        if any(p in low for p in ("mute yourself", "stop talking", "be quiet", "shut up")):
            self.is_muted = True
            return "Muted."

        # NOTE: "(un)mute the audio/sound/speakers/volume" is a SYSTEM-AUDIO
        # command and belongs to Tools, not to Nova's own mute. Without this
        # guard the bare "unmute" test below swallowed "unmute the audio" —
        # Nova said "Unmuted" while the speakers stayed muted.
        if not re.search(r"\b(audio|sound|speakers?|volume)\b", low):
            if "unmute" in low or "start listening" in low:
                self.is_muted = False
                return "Unmuted."

        return None

    # ═════════════════════════════════════════════════════════════════════════════
    # Memory intents
    # ═════════════════════════════════════════════════════════════════════════════
    _SAVE_RE    = re.compile(r"remember (?:that )?my ([\w][\w ]*?) is (.+)", re.I)
    # "who's my favorite superhero" missed entirely and fell to the LLM, which
    # invented an answer ("Deadpool") for a fact Nova was supposed to know.
    # People ask about people with WHO, and about things with WHAT.
    _RECALL_RE  = re.compile(
        r"(?:what|who)(?:'?s| is|'?re| are)\s+my\s+([\w][\w ]*?)[\s.?!]*$", re.I)
    _UPDATE_RE  = re.compile(r"actually (?:my ([\w][\w ]*?) is |it'?s )(.+)", re.I)
    _FORGET_RE  = re.compile(r"forget (?:that )?my ([\w][\w ]*?)$", re.I)

    # Explicit user-directed memory commands store under the 'explicit' category,
    # source='explicit', so they supersede by canonical key and sit alongside the
    # facts Nova learns passively (via wake-mode reconciliation).
    _EXPLICIT_CAT = "explicit"

    # Nouns that mean "ask the system", not "recall a stored fact". "what's my
    # battery at?" matches the recall regex just as well as "what's my favorite
    # colour?", and Memory routes BEFORE Tools — so without this, tool questions
    # were answered with "I don't have your battery at stored yet."
    _SYSTEM_QUERY_WORDS = frozenset({
        "battery", "charge", "power", "ip", "wifi", "wi-fi", "network",
        "internet", "connection", "volume", "sound", "brightness", "screen",
        "storage", "disk", "cpu", "ram", "mac", "computer", "machine",
        "system", "hostname", "uptime",
    })

    def _is_system_query(self, key: str) -> bool:
        return bool(set(re.findall(r"[\w-]+", key.lower())) & self._SYSTEM_QUERY_WORDS)

    def _remember_fact(self, key: str, value: str) -> None:
        """Store a user-stated fact under the category that already holds it, so
        an explicit correction UPDATES the passively-learned fact instead of
        creating a rival copy under 'explicit' (which would make recall
        ambiguous)."""
        found = self.memory.find_fact(key)
        category = found[0] if found else self._EXPLICIT_CAT
        self.memory.upsert_fact(category, key, value, source="explicit")

    def _handle_memory_intent(self, text: str) -> Optional[str]:
        name = self.config["user"]["address_as"]

        # Save — "remember that my X is Y"
        m = self._SAVE_RE.search(text)
        if m:
            key   = m.group(1).strip()
            value = m.group(2).strip().rstrip(".")
            self._remember_fact(key, value)
            return f"Got it. I'll remember your {key} is {value}."

        # Recall — "what's my X" (searches EVERY category, not just 'explicit')
        m = self._RECALL_RE.search(text)
        if m:
            key   = m.group(1).strip()
            found = self.memory.find_fact(key)
            if found:
                return f"Your {key} is {found[1]}."
            # Nothing stored. If this reads as a system question, fall through so
            # Tools can answer it; otherwise say honestly that it isn't stored.
            if self._is_system_query(key):
                return None
            return f"I don't have your {key} stored yet."

        # Update / correction — "actually my X is Y" / "actually it's Y"
        m = self._UPDATE_RE.search(text)
        if m:
            key_part = m.group(1)
            value    = m.group(2).strip().rstrip(".")
            if key_part:
                key = key_part.strip()
            else:
                last = self.memory.last_identity()
                key = last[1] if last else None
            if key:
                self._remember_fact(key, value)   # updates in place, wherever it lives
                return f"Updated. Your {key} is now {value}."
            return "I'm not sure what you'd like me to update."

        # Forget — "forget that my X" (removes it from whichever category holds it)
        m = self._FORGET_RE.search(text)
        if m:
            key = m.group(1).strip()
            removed = self.memory.delete_fact_anywhere(key)
            if removed:
                return f"Done. I've forgotten your {key}."
            return f"I don't have your {key} stored, {name}."

        # Meta-recall — "what do you know about me"
        low = text.lower()
        if any(p in low for p in ("what do you know about me", "what do you remember", "what have i told you")):
            facts = self.memory.facts_for_readback()
            if not facts:
                return f"I don't have anything stored about you yet, {name}."
            # Each rendered fact is already a full sentence, so join with spaces
            # (a "; " join produced "coffee.; Your favorite…" and a doubled ".").
            return "Here's what I know. " + " ".join(facts)

        return None

    # ═════════════════════════════════════════════════════════════════════════════
    # LLM streaming with sentence-chunked TTS overlap
    # ═════════════════════════════════════════════════════════════════════════════
    _SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
    # Sentence-boundary detector for INCREMENTAL emission during token streaming:
    # terminal .?! (optionally followed by a closing quote/bracket) at a word end.
    _TTS_SENT_END = re.compile(r"[.!?][\"')\]]?(?=\s|$)")
    # The "1." of a numbered list also matches _TTS_SENT_END, which made the
    # chunk end there: Nova said "your roadmap outlines the following steps,
    # one." out loud. A period is not a sentence end when the only thing before
    # it on the line is a number.
    _LIST_MARKER_TAIL = re.compile(r"(?:^|\n)[ \t]*\d+$")

    @classmethod
    def _next_sentence_end(cls, text: str):
        """First real sentence boundary in `text`, skipping list markers."""
        for m in cls._TTS_SENT_END.finditer(text):
            if cls._LIST_MARKER_TAIL.search(text[:m.start()]):
                continue
            return m
        return None
    # Clause boundary, used for the FIRST chunk only. Nothing can be heard until
    # the first chunk is both generated AND synthesised, so a long opening
    # sentence is a long silence: measured, "The shortest war in history was
    # between Britain and Zanzibar, and lasted…" took 2.83s to reach audio,
    # against 1.89s when the opening clause was spoken on its own. Only real
    # punctuation counts as a split point — a comma is a place the voice
    # naturally pauses, so the seam is inaudible, whereas cutting at an
    # arbitrary word count would audibly chop the sentence. Later chunks stay
    # whole sentences: once audio is playing, synthesis runs 3-4x faster than
    # speech and never catches up with the player.
    _TTS_CLAUSE_END = re.compile(r"[,;:](?=\s)")
    _TTS_FIRST_CHUNK_MIN_WORDS = 4

    def _stream_response(self, user_text: str, rag_context: str = "") -> None:
        """
        Stream LLM tokens to the UI and speak sentences AS they complete, so
        Nova starts talking on sentence 1 while it's still generating the rest
        (the proven Jarvis 3-thread overlap: LLM → sentence queue → TTS worker →
        SeamlessPlayer, which joins sentences at sample level so there's no gap).

        Overlap is gated by tts.stream_while_generating (default on). If it's
        ever set false, we fall back to the blocking "generate fully, then speak"
        path — an instant, code-free rollback if playback ever starves.
        """
        memory_ctx    = self.memory.get_context_for_llm()
        system_prompt = self._build_system_prompt(memory_ctx, rag_context)
        history       = self.memory.get_recent_turns(n=10)
        stream_tts    = self.config["tts"].get("stream_while_generating", True)
        split_first   = self.config["tts"].get("split_first_clause", True)

        full_response = ""
        pending       = ""            # tokens not yet emitted as a full sentence
        state         = {"spoke": False}

        def _emit(text: str) -> None:
            text = _clean_for_tts(text)
            if not text:
                return
            if not state["spoke"]:
                self.set_state("speaking")   # first audio → flip orb to speaking
                state["spoke"] = True
            self.tts.speak(text)

        def _flush(force: bool) -> None:
            nonlocal pending
            while True:
                m = self._next_sentence_end(pending)
                if not m:
                    break
                cut = m.end()
                sentence = pending[:cut].strip()
                pending = pending[cut:]
                if sentence:
                    _emit(sentence)

            # Nothing has been heard yet: take the opening clause rather than
            # waiting for the whole sentence. Only ever done for the first
            # chunk — see _TTS_CLAUSE_END.
            if not force and not state["spoke"] and split_first:
                m = self._TTS_CLAUSE_END.search(pending)
                if m and len(pending[:m.end()].split()) >= self._TTS_FIRST_CHUNK_MIN_WORDS:
                    clause = pending[:m.end()].strip()
                    pending = pending[m.end():]
                    if clause:
                        _emit(clause)

            if force and pending.strip():
                _emit(pending.strip())
                pending = ""

        def on_token(token: str) -> None:
            nonlocal full_response, pending
            full_response += token
            self.ws.stream_token(token)   # UI shows text live
            if stream_tts:
                pending += token
                _flush(force=False)

        self.set_state("thinking")
        self.llm.stream(
            system_prompt=system_prompt,
            history=history,
            user_message=user_text,
            on_token=on_token,
        )

        if stream_tts:
            _flush(force=True)            # speak the trailing partial sentence
        else:
            # Blocking fallback: whole reply generated, now speak it back-to-back.
            self.set_state("speaking")
            for sentence in self._SENT_SPLIT.split(full_response.strip()):
                s = sentence.strip()
                if s:
                    self.tts.speak(_clean_for_tts(s))

        self.tts.wait_until_done(timeout=60)

        self._last_response = full_response
        self.memory.add_turn("assistant", full_response)
        self._session_turns.append({"role": "assistant", "content": full_response})
        self.ws.send_message("assistant", full_response)
        # Fact learning does NOT happen per-turn. The wake-mode reconciliation
        # pass reviews the whole conversation when it ends (see _main_loop), so
        # extraction stays off the live path and costs one LLM call per session.

        self.set_state("idle")

    def _build_system_prompt(self, memory_ctx: str, rag_ctx: str) -> str:
        from system_prompt import build_system_prompt
        return build_system_prompt(self.config, memory_context=memory_ctx, rag_context=rag_ctx)


# ── Entry point ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    VoiceAssistant().run()
