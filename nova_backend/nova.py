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
import collections
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

# Musing, not commanding. "Scroll through my photos sometime" opens with an
# action verb and survives every handler, so the unsupported guard refused it
# with "I can't do that" — when he was thinking out loud, not asking. Same
# family as the sign-off guard: "I was so tired I had to go to sleep early" is
# a story, not an instruction.
_HYPOTHETICAL_RE = re.compile(
    r"\b(?:sometime|someday|some\s+day|one\s+day|at\s+some\s+point|"
    r"eventually|one\s+of\s+these\s+days|when\s+i\s+get\s+a\s+chance)\b"
    r"[\s.!?]*$",
    re.I,
)


def _is_unsupported_action(text: str) -> bool:
    """Phrased as an action, and no handler claimed it: Nova must refuse it
    honestly rather than let the LLM say it already did it (stage 8b)."""
    return bool(_ACTION_REQUEST_RE.match(text) and not _HYPOTHETICAL_RE.search(text))

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
        # The boot sequence starts before anything else does, because it is
        # describing exactly this: the engines coming up. It runs on its own
        # thread, so it costs startup nothing.
        from sounds import NovaSounds
        self.config = load_config()
        self.sounds = NovaSounds(self.config)
        self.sounds.play("boot")
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
        # True once he has actually started talking to Nova this session — the
        # wake word, or typing. Home's greeting and status row are the RESTING
        # state and go the moment this flips. Kept separate from
        # `_last_response` because that only becomes true after she ANSWERS,
        # which is a beat too late to be a greeting.
        self._conversation_started = False
        # What Nova has DONE, newest last. Small on purpose: this is a log he
        # glances at after the fact, not an audit trail, and an unbounded one
        # would be a memory leak with a screen attached.
        self.activity_log: "collections.deque" = collections.deque(maxlen=40)
        # A harness builds the assistant through _init_state alone, so the
        # cues have to exist here too — silent by default in that case, since
        # nothing has configured or started them.
        if not hasattr(self, "sounds"):
            from sounds import NovaSounds
            self.sounds = NovaSounds(self.config)
        # A SOFT one-shot follow-up armed by a calendar read ("want to hear
        # what's coming up?"). It only fires on an affirmative reply and never
        # eats an unrelated next command.
        self._calendar_offer: Optional[Callable[[], str]] = None
        # A STRICT one-shot confirmation armed by a destructive tool action
        # (shutdown / restart / sleep). Unlike the soft offer, anything that
        # isn't a clear yes cancels it — we never guess about powering off.
        self._tool_confirm: Optional[Callable[[], str]] = None
        self.brain = None                # brain.py; set by _init_brain
        self._brain_chat = False         # this turn: brain sure it's conversation
        self._brain_fell_through = False # this turn: brain declined (learnable)
        self._brain_offer = None         # pending "did you mean X?"
        self._last_reflex = None         # what "no, I meant ..." would correct
        self._last_turn = None           # turnstats: the last decision, for "why?"
        self.turnstats = None
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
        # This turn came from the keyboard and must not be spoken aloud. Only
        # ever set for the duration of one typed turn.
        self._silent_turn = False
        # Working alongside him on his Mac: Nova is parked in the puck, the
        # conversation timeout is long, and she stays there until he says go
        # home. Entered by phrase OR automatically the moment she does
        # something on his machine.
        self.work_mode = False
        # Said once per episode, not every pass round the wake loop.
        self._mic_warned = False

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
        self._init_market()
        self._init_actuation()
        self._init_ws()
        # After the WS server exists: the tap is created during _init_stt, which
        # runs earlier, so this cannot be done at the point of creation.
        # /api/status reports it because "is barge-in actually on" is otherwise
        # invisible — it needs a Screen Recording grant, macOS 13+, and a live
        # stream, and when any of those is missing Nova quietly goes back to
        # pausing his music.
        if getattr(self, "system_audio", None) is not None:
            self.ws._system_audio = self.system_audio
        # Space bar. Deterministic where acoustics never are: no threshold, no
        # cancellation, nothing to mishear. It also means the WAKE WORD no
        # longer has to survive her own voice, which was the hardest problem in
        # the build and is now simply not on the critical path.
        self.ws.on_interrupt = self._barged_in
        self._init_views()          # after _init_ws: it needs the WS server
        self._init_brain()          # last: its skills reach every engine above
        self._verify_engines()
        self._show_home()
        self._warm_prompt_cache()
        log.info("Nova ready.")
        self.sounds.play("ready")   # systems online


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
                         "calendar", "files", "screen", "weather", "market", "actuation",
                         "ws",
                         "views")

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
        self._init_echo()
        self.stt = STTEngine(
            self.config["stt"],
            mic_gate=self.mic_gate,
            wake_config=self.config.get("wake_word", {}),
            echo=self.echo,
        )
        # Barge-in: the wake detector is allowed to hear Nova, and when it does
        # she stops mid-sentence. Wired here because the STT engine must not
        # know what a TTS engine is.
        try:
            self.stt.on_barge_in = self.tts.stop_and_flush
        except AttributeError:
            pass                       # TTS is built later; re-pointed below

    def _init_echo(self) -> None:
        """Optional echo canceller, shared by TTS (which supplies the
        reference) and STT (which consumes it). Default OFF: it also means
        keeping mic frames during playback, and Nova listens far more often
        than she speaks."""
        self.echo = None
        self.system_audio = None
        tap_cfg = self.config.get("audio", {}).get("system_tap", {})
        want_tap = bool(tap_cfg.get("enabled", False))
        # The tap is a REASON to cancel, not an extra on top of it. Nova only
        # ever had a reference for her own voice; with the speaker mix arriving
        # she has one for everything, so cancellation turns itself on.
        if not (self.config["stt"].get("echo_cancellation", False) or want_tap):
            return
        try:
            from echo_canceller import EchoCanceller
            self.echo = EchoCanceller(
                stream_delay_ms=int(self.config["stt"].get(
                    "echo_stream_delay_ms", 35)))
            if not self.echo.enabled:
                self.echo = None
        except Exception as exc:
            log.warning(f"echo cancellation unavailable: {exc}")
            self.echo = None

        if want_tap and self.echo is not None:
            try:
                from system_audio import SystemAudioReceiver
                self.system_audio = SystemAudioReceiver(self.echo)
                if not self.system_audio.start():
                    self.system_audio = None

            except Exception as exc:
                log.warning(f"speaker mix unavailable: {exc}")
                self.system_audio = None

    @property
    def hears_speakers(self) -> bool:
        """True while the speaker mix is actually arriving.

        Everything that exists to work AROUND Nova hearing the speakers is
        conditioned on this: pausing his music, and dropping mic frames while
        she talks. Both of those are the old workaround, and both should stop
        the moment the real fix is running — and start again the moment it
        stops, which is why this asks the receiver rather than reading a flag
        set at startup.
        """
        rx = getattr(self, "system_audio", None)
        return bool(rx is not None and rx.is_live)

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
        self.tts = TTSEngine(self.config["tts"], mic_gate=self.mic_gate,
                             echo=self.echo)
        self.stt.on_barge_in = self._barged_in
        # Continuous, because the main loop is blocked inside
        # tts.wait_until_done() for the whole time she is speaking — which is
        # exactly the window barge-in has to cover.
        self.stt.start_barge_watch()
        # Kokoro is already in the speaker mix; see tts_engine.suppress_reference.
        try:
            self.tts.suppress_reference = lambda: self.hears_speakers
        except Exception:
            pass

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
        self.tools = NovaTools(self.config, on_announce=self._timer_fired,
                               on_progress=self._show_work,
                               on_activity=self.record_activity)

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
        # Pausing his music was always a workaround for not being able to
        # cancel it. With the speaker mix arriving it is cancelled for real, so
        # the music keeps playing — which is the whole point of barge-in.
        if self.hears_speakers:
            return False
        return bool(self.config.get("music", {}).get("duck_while_listening", True))

    def _init_weather(self) -> None:
        # Deterministic end to end: no model is involved in phrasing a
        # temperature. See weather_engine.py for the invariant-3 decision.
        from weather_intents import NovaWeather
        self.weather = NovaWeather(self.config)

    def _init_actuation(self) -> None:
        # Typing and clicking. Needs the assistant so it can check work mode,
        # which is the gate that keeps it from driving his Mac unasked.
        from actuation_intents import NovaActuation
        self.actuation = NovaActuation(self.config, assistant=self)

    def _init_market(self) -> None:
        # Deterministic end to end: no model ever phrases a price. See
        # market_engine.py for the invariant-3 decision.
        from market_intents import NovaMarket
        self.market = NovaMarket(self.config)

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
        self.views = NovaViews(self.config, ws=getattr(self, "ws", None),
                               assistant=self)

    def _sounds_like_me(self, text: str) -> bool:
        """True when a capture is mostly Nova's own last reply coming back.

        Word overlap rather than equality: the speaker-to-mic path mangles
        edges, so an exact match almost never happens, but the middle of the
        sentence survives intact.
        """
        said = (self._last_response or "").lower()
        heard = (text or "").lower()
        if len(heard.split()) < 3 or not said:
            return False
        mine = set(re.findall(r"[a-z']+", said))
        theirs = re.findall(r"[a-z']+", heard)
        if not mine or not theirs:
            return False
        overlap = sum(1 for w in theirs if w in mine) / len(theirs)
        return overlap >= 0.7

    def _check_mic_health(self) -> None:
        """Notice a microphone that is delivering nothing, and say so.

        Measured on his Mac: with AirPods Pro as the default input the stream
        returned RMS 0.0 and peak 0, while the built-in mic in the same room
        returned RMS 29. Nova scored 0.000 forever and looked perfectly fine.
        """
        if not getattr(self.stt, "mic_is_silent", False):
            self._mic_warned = False
            return
        if self._mic_warned:
            return
        self._mic_warned = True

        # Reopening picks up the CURRENT default device, which fixes the common
        # case: he switched headphones and the old stream is bound to nothing.
        recovered = False
        try:
            recovered = self.stt.reopen_stream()
        except Exception as exc:
            log.warning(f"mic reopen failed: {exc}")
        if recovered:
            log.info("Mic reopened; waiting to see whether audio returns.")
            return

        self.set_state("unsure")
        self._respond(
            "I can't hear anything. My microphone is connected but it isn't "
            "picking up any sound. Check which input device is selected.")

    def set_work_mode(self, on: bool, *, reason: str = "") -> None:
        """Enter or leave work mode, and tell the app to park or restore.

        Idempotent: entering while already in work mode must not re-park the
        window, or every app launch during a session would yank it back to the
        corner while he is dragging it somewhere else.
        """
        if self.work_mode == on:
            return
        self.work_mode = on
        log.info(f"[work-mode] {'on' if on else 'off'}"
                 + (f" ({reason})" if reason else ""))
        try:
            self.ws.send_mode(puck=on)
        except Exception as exc:
            log.warning(f"could not send mode: {exc}")

    def _confidence_gate(self, text: str):
        """Decide whether to act on what was heard. Returns the Decision, or
        None when gating is off.

        Three outcomes reach Nicholas:
          REJECT  — she says she did not catch it and waits. No guess.
          CONFIRM — she reads back HIS words and acts only on a clear yes.
          ACT_UNSURE — she acts, with the orb showing she was not certain, so a
                       wrong answer is explainable instead of mysterious.
        """
        if not self.config["stt"].get("confidence_gate", True):
            return None
        import confidence as C

        # getattr, not attribute access: a missing signal is designed to mean
        # "behave exactly as before", and that must not become a crash in the
        # voice loop just because an STT implementation predates this.
        # The brain's view of what this sentence would DO can only raise the
        # bar (a file move the regex tier read as chat), never lower it.
        min_tier = None
        if self.brain is not None and self.brain.ready:
            min_tier = self.brain.tier_for(text, self)
        decision = C.decide(text, getattr(self.stt, "last_confidence", None),
                            min_tier=min_tier)
        log.info(f"[confidence] {decision.score:.2f} tier={decision.tier} "
                 f"-> {decision.action}")

        if decision.action == C.REJECT:
            self._respond(C.ask_again(decision.tier))
            return decision

        if decision.action == C.CONFIRM:
            # Route the ORIGINAL words on a yes — not a re-transcription, and
            # not a paraphrase. What he confirms is exactly what was heard.
            self._tool_confirm = lambda t=text: self._run_confirmed(t)
            self._respond(decision.readback)
            return decision

        if decision.is_unsure:
            # Visible, not spoken. Interrupting every borderline turn with
            # "I think you said..." would be worse than the problem.
            self.set_state("unsure")
        return decision

    def _run_confirmed(self, text: str) -> str:
        """He confirmed a read-back: run the utterance for real.

        Returns "" because the handlers speak for themselves through _respond;
        returning their text would say it twice.
        """
        self._handle_turn_impl(text)
        return ""

    def _show_home(self) -> None:
        """Put Nova on the home screen at startup.

        The WS held view="home" with EMPTY data and nothing ever built the
        payload, so the app was told "home" and found no blocks to render — it
        came up as a bare orb and Nicholas could not find the home screen at
        all. Best-effort: home degrades block by block, and a failure here must
        never stop Nova starting.
        """
        try:
            self.views.handle("home")
        except Exception as exc:
            log.warning(f"could not show home at startup: {exc}")
        self._start_home_ticker()
        self._start_proactive()
        self._init_standing()

    def _start_proactive(self) -> None:
        """Let Nova speak first, occasionally.

        Deliberately started LAST, after every engine exists, because the
        monitor reads the calendar and the reminders and asks the assistant
        whether it is a good moment — all of which have to be there before it
        wakes up. Best effort: nothing about volunteering information should be
        able to stop Nova answering questions.
        """
        try:
            from proactive import ProactiveMonitor
            d = self._delivery()

            def _deliver(text: str, key: Optional[str] = None,
                         kind: str = "proactive") -> None:
                # Through the single door: claimed before it is said, counted
                # against the ceilings, shown instead of spoken when over them.
                if d is not None:
                    d.deliver(kind, text, key=key, claimed=key is not None)
                else:
                    self._show_then_announce(text)

            claim = (lambda key, kind, text: d.claim(key, kind, text)) if d else None
            self.proactive = ProactiveMonitor(self.config, self, _deliver, claim=claim)
            self.proactive.start()
        except Exception as exc:
            log.warning(f"proactive monitor unavailable: {exc}")
            self.proactive = None

    def _show_then_announce(self, text: str) -> None:
        # On screen first: it costs nothing and _announce blocks until it
        # finds a gap, which can be a while if she is mid-sentence.
        try:
            self.views.flash_notice(text)
        except Exception as exc:
            log.debug(f"could not show the notice: {exc}")
        self._announce(text)

    def _delivery(self):
        """The single door for unprompted speech (delivery.py), made once."""
        d = getattr(self, "delivery", None)
        if d is not None:
            return d
        try:
            from delivery import Delivery

            def _show(text: str) -> None:
                self.views.flash_notice(text)
            self.delivery = Delivery(DATA_DIR / "nova_memory.db", self.config,
                                     speak=self._announce, show=_show)
        except Exception as exc:
            log.warning(f"delivery unavailable, speaking directly: {exc}")
            self.delivery = None
        return self.delivery

    def _init_standing(self) -> None:
        """Standing watches, on the single door. Started with the proactive
        monitor (after every engine exists); readings are local only."""
        self.standing = None
        try:
            from standing import StandingWatches
            d = self._delivery()

            def _read(metric: str):
                if metric == "battery":
                    r = self.tools.battery_reading()
                    return r and r.get("level")
                if metric == "cpu":
                    r = self.tools.cpu_reading()
                    return r and r.get("busy")
                if metric == "memory":
                    r = self.tools.memory_reading()
                    return r and r.get("pct")
                return None

            def _deliver(kind: str, text: str, key: str):
                if d is not None:
                    d.deliver(kind, text, key=key)
                else:
                    self._show_then_announce(text)

            self.standing = StandingWatches(DATA_DIR / "nova_memory.db", _read, _deliver,
                                            self.config)
            self.standing.start()
        except Exception as exc:
            log.warning(f"standing watches unavailable: {exc}")
            self.standing = None

    def _timer_fired(self, msg: str) -> None:
        """A timer he set has gone off: through the door (requested kind, so
        never held by a ceiling), from the timer's own thread."""
        d = self._delivery()
        if d is not None:
            d.deliver("timer", msg)
        else:
            self._announce(msg)

    def _start_home_ticker(self) -> None:
        """Keep home honest about the world while he is looking at it.

        This is what makes Now Playing arrive when HE starts a song rather than
        when he asks Nova to. It is deliberately one ticker rather than a
        poller per card: every tile already knows its own refresh interval, and
        `refresh_home` sends nothing unless the payload actually changed, so a
        tick where the world stood still costs a dict comparison.

        Daemon, like every other background worker here, so it can never hold
        the process open (invariant 4's reasoning, applied to a second thread).
        """
        seconds = float(self.config.get("ui", {}).get("home_tick_seconds", 4))
        if seconds <= 0:
            return

        def _tick() -> None:
            while True:
                time.sleep(seconds)
                try:
                    self.views.refresh_home()
                except Exception as exc:
                    log.warning(f"home tick failed: {exc}")

        threading.Thread(target=_tick, name="nova-home-tick", daemon=True).start()

    def _emit_panel(self, engine) -> None:
        """Send the panel a handler just built, if it built one.

        One-shot and cleared on read, like `pending_offer`. A handler that
        produced no panel simply leaves the previous screen alone rather than
        blanking it — asking the time should not clear his weather.
        """
        panel = getattr(engine, "last_panel", None)
        if panel is None:
            return
        engine.last_panel = None
        try:
            view_name, payload = panel
            self.ws.send_view(view_name, payload)
            # The ticker refreshes home in the background. Without this it
            # would not know an ANSWER is on screen and would push home
            # straight over the top of it a few seconds later.
            self.views.current = view_name
            self._arm_panel_dismiss()
        except Exception as exc:
            # A panel failing must never cost him the spoken answer.
            log.warning(f"could not send panel: {exc}")

    def record_activity(self, summary: str, failed: bool = False,
                        kind: str = "browser") -> None:
        """One line in the history, stamped when it happened."""
        try:
            self.activity_log.append({
                "time": datetime.now().strftime("%-I:%M %p"),
                "text": summary, "failed": bool(failed), "kind": kind})
            self.views.refresh_home()
        except Exception as exc:
            log.debug(f"could not record activity: {exc}")

    def _show_work(self, payload: dict) -> None:
        """Put a live step list on screen while a handler is working.

        Called repeatedly by panels.Progress as a handler advances, so this is
        the one place the screen learns that Nova is DOING something rather
        than thinking about it. The orb goes amber on the first push and the
        panel becomes the work surface.

        Each push re-arms the dismissal, so the surface clears a while after
        the LAST step rather than mid-run — the same token trick the answer
        panels use, for the same reason.
        """
        try:
            self.set_state("working")
            self.ws.send_view("working", payload)
            # Without this the home ticker would push home straight over the
            # top of the work she is showing him.
            self.views.current = "working"
            self._arm_panel_dismiss()
        except Exception as exc:
            # The screen, never the work.
            log.warning(f"could not show work: {exc}")

    def _arm_panel_dismiss(self) -> None:
        """Send the panel away again after a while, back to home.

        Panels used to persist until something replaced them, so the weather
        sat on screen long after he had heard the answer. An answer is
        transient; the screen should agree with that. Any NEW panel cancels the
        pending dismissal, so a run of questions never yanks the screen away
        mid-read.
        """
        seconds = float(self.config.get("ui", {}).get("panel_seconds", 25))
        if seconds <= 0:
            return
        token = self._panel_token = getattr(self, "_panel_token", 0) + 1

        def _dismiss() -> None:
            # Only the LATEST panel's timer may fire.
            if getattr(self, "_panel_token", 0) != token:
                return
            try:
                self.views.handle("home")
            except Exception as exc:
                log.warning(f"panel dismiss failed: {exc}")

        t = threading.Timer(seconds, _dismiss)
        t.daemon = True
        t.start()

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
        # The first pass through is Nova arriving, not a conversation ending —
        # `ready` has just played and a `rest` on top of it would be two cues
        # for one event.
        first_wake = True
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
                # Deliberately the wake cue inverted: a fifth DOWN, so he can
                # tell arriving from leaving without being told which is which.
                # Only when a conversation actually ended — not at startup,
                # where `ready` has just played.
                if not first_wake:
                    self.sounds.play("rest")
                first_wake = False
                log.info("Waiting for wake word…")
                wake_detected = self.stt.record_wake(
                    wake_keywords=self.config["wake_word"]["keywords"],
                    timeout_s=300.0,
                )
                if not wake_detected:
                    continue
                just_woke = True
                empty_turns = 0
                # Blocking, and short. It has to be OVER before record_command
                # opens the mic, or the VAD reads Nova's own chime as speech
                # onset and Whisper is handed a note instead of a word. Fires
                # in puck mode too — this is the one path into a conversation.
                self.sounds.play_and_wait("wake")
                # He said "Nova". The conversation has started, so the welcome
                # and the status row go now — not when she finishes answering.
                # Hooking this to the reply meant "Good morning, NICHOLAS" was
                # still sitting under the orb while she was already listening
                # to him, which is the opposite of what a greeting is for.
                self._begin_conversation()
                # Duck for the WHOLE conversation, not per turn: restoring
                # between turns would pump the volume up and down every time
                # Nova answered. It stays down until we are back up top.
                if self._duck_enabled:
                    self.tools.duck_music()
            else:
                just_woke = False

            # ── Mic health ───────────────────────────────────────────────
            # A dead input is the one failure Nova cannot notice by listening
            # harder. Say it out loud ONCE, after trying to recover, so being
            # deaf is never silent.
            self._check_mic_health()

            # ── Phase 2: Record command (VAD-gated) ──────────────────────────
            self.set_state("listening")
            # In conversation mode, cap how long we wait for the user to start
            # speaking; if they stay silent past the timeout, drop back to wake.
            # Work mode gets a much longer leash. While they are working
            # together he reads an email, thinks, scrolls — and the ordinary
            # 15s would drop the session constantly, forcing a wake word every
            # time he paused. The session does NOT end at this timeout either:
            # saying "Nova" resumes in the puck with the conversation intact.
            conv_timeout = float(
                self.config["wake_word"].get(
                    "work_mode_timeout_s" if self.work_mode
                    else "conversation_timeout_s",
                    60.0 if self.work_mode else 15.0))
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

            # ── Confidence gate ──────────────────────────────────────────
            # The transcript used to be on screen, so a mishear was visible.
            # It isn't any more, so how sure Whisper was now decides whether
            # Nova acts — and the bar rises with what the action costs.
            # Nova must never answer HERSELF. With echo cancellation on, mic
            # frames are kept while she speaks, and if the room defeats the
            # canceller Whisper transcribes her own reply — she then answers it,
            # and the pair of them talk in a loop. Belt and braces: whatever the
            # canceller does, a transcript that is mostly her own last sentence
            # is discarded here.
            if self._sounds_like_me(text):
                log.info(f"[self-heard] discarding my own voice: {text[:60]!r}")
                continue

            gate = self._confidence_gate(text)
            if gate is not None and not gate.should_act:
                continue

            log.info(f"[user] {text}")
            self.memory.add_turn("user", text)
            self._session_turns.append({"role": "user", "content": text})
            self._turn_was_command = False
            # He SPOKE this one. Clear any silent flag left by a typed turn, or
            # a single use of the keyboard would mute Nova for the rest of the
            # session.
            self._silent_turn = False
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
        # Tell the UI she has gone back to waiting for her name. Nothing ever
        # broadcast this, so "waiting for the wake word" and "mid-conversation"
        # looked identical on screen — which is precisely the distinction the
        # readout under the orb exists to make.
        self.set_state("sleeping")
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
    def _handle_text_input(self, text: str, silent: bool = False) -> None:
        if not text.strip():
            return
        log.info(f"[text-input] {text}{' (silent)' if silent else ''}")
        self._silent_turn = silent
        self.memory.add_turn("user", text)
        self.ws.send_message("user", text)
        # Submit to the MLX worker and return immediately; the response streams
        # back over the WebSocket. (Never touch MLX from this HTTP/WS thread.)
        self._submit_turn(text, wait=False)

    # ═════════════════════════════════════════════════════════════════════════════
    # Core turn handler
    # ═════════════════════════════════════════════════════════════════════════════
    def _handle_turn_impl(self, text: str) -> None:
        """One turn, timed and recorded (turnstats.py): which path answered it,
        how long the decision took, how long until the first word. The record
        is the evidence future routing work starts from, and what "why did you
        do that?" is answered from. Nested calls (a corrected request routed
        again, a confirmed read-back) are part of the outer turn."""
        depth = getattr(self, "_turn_depth", 0)
        self._turn_depth = depth + 1
        if depth == 0:
            self._turn_t0 = time.time()
            self._turn_note = None
            self._turn_first = None
        try:
            self._route_turn(text)
        finally:
            self._turn_depth = depth
            if depth == 0:
                self._record_turn(text)

    def _note(self, path: str, skill: Optional[str] = None,
              confidence: Optional[float] = None, matched: Optional[str] = None) -> None:
        """Which path answered this turn. The first one to claim it wins."""
        if getattr(self, "_turn_note", None) is None:
            t0 = getattr(self, "_turn_t0", None)
            self._turn_note = {"path": path, "skill": skill, "confidence": confidence,
                               "matched": matched,
                               "decide_ms": (time.time() - t0) * 1000 if t0 else None}

    def _mark_first(self) -> None:
        """The first word of this turn's answer is going out."""
        if getattr(self, "_turn_first", None) is None and getattr(self, "_turn_t0", None):
            self._turn_first = (time.time() - self._turn_t0) * 1000

    def _record_turn(self, text: str) -> None:
        note = getattr(self, "_turn_note", None) or {"path": "unknown"}
        row = {"text": text.strip(), **note, "first_ms": getattr(self, "_turn_first", None)}
        if note.get("path") != "explain":
            self._last_turn = row
        try:
            if getattr(self, "turnstats", None) is None:
                from turnstats import TurnStats
                self.turnstats = TurnStats(DATA_DIR / "nova_memory.db")
            self.turnstats.record(row)
        except Exception as exc:
            log.warning(f"turn stats unavailable: {exc}")

    def _route_turn(self, text: str) -> None:
        """Full pipeline for one user utterance. First match wins.

        Always runs on the ``nova-llm`` worker thread (via ``_submit_turn``) so
        every MLX access stays on the thread that owns the model."""
        text = text.strip()

        # ── 1. System commands ───────────────────────────────────────────────
        resp = self._handle_system_commands(text)
        if resp is not None:
            self._note("system")
            self._respond(resp)
            return

        # ── 1b. Destructive-action confirmation (STRICT yes/no) ──────────────
        # Armed by a power command. Only an explicit yes proceeds; ANYTHING
        # else cancels — we never power off the Mac on an ambiguous reply.
        if self._tool_confirm is not None:
            self._note("pending")
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
                self._note("pending")
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
                self._note("pending")
                if self.files.pending_offer is not None:
                    self._calendar_offer = self.files.pending_offer
                    self.files.pending_offer = None
                self._respond(resp)
                return

        # ── 2b′. The brain's own questions and lessons ───────────────────────
        # An answer to "did you mean X?", a correction of the last reflex
        # ("no, I meant ..."), teaching or forgetting a phrase ("when I say
        # lights out, mute the sound and lock my Mac"), and a taught phrase
        # itself. All before routing, so none can be re-read as a new command.
        if self._brain_lessons(text):
            self._note("lesson")
            return

        # ── 2b″. "Why did you do that?" / "how often do you answer by yourself"
        # Answered from the turn record (turnstats.py), never invented.
        import turnstats
        if turnstats.wants_explanation(text):
            self._note("explain")
            self._respond(turnstats.explain(getattr(self, "_last_turn", None), self.brain))
            return
        if turnstats.wants_report(text):
            self._note("explain")
            try:
                if getattr(self, "turnstats", None) is None:
                    self.turnstats = turnstats.TurnStats(DATA_DIR / "nova_memory.db")
                stats = self.turnstats.summary()
                learned = self.brain.learned() if self.brain is not None and self.brain.ready else []
                self._respond(turnstats.report(stats, learned))
            except Exception as exc:
                log.warning(f"brain report failed: {exc}")
                self._respond("I couldn't put that together just now.")
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
            self._note("view")
            self._run_view(view_intent, text)
            return

        # ── 2d. Actuation confirmation ────────────────────────────────────────
        # "That would send it. Want me to?" — answered before routing, so the
        # yes cannot be re-read as a fresh command.
        if self.actuation.pending is not None:
            resp = self.actuation.resolve_pending(text)
            if resp is not None:
                self._note("pending")
                self._respond(resp)
                return

        # ── 2e-0. Arithmetic, percentages, tips, unit conversion ─────────────
        # In CODE (mathskill.py, units.py — ported from JARVIS), never by the
        # model: a wrong number sounds exactly like a right one. The parser
        # accepts a sentence only when every word of it is part of the sum,
        # so "set the volume to 50 percent" and "I have 2 kids" never land here.
        if self._run_math(text):
            return

        # ── 2e-0b. Standing watches ───────────────────────────────────────────
        # "Tell me when my battery is charged" is a WATCH, not a question:
        # stored, checked by a loop, said once through the single door
        # (standing.py). Before the brain and the regex stages, both of which
        # would read it as a battery question and answer with today's level.
        st = getattr(self, "standing", None)
        if st is not None:
            try:
                resp = st.handle(text)
            except Exception as exc:
                log.warning(f"[watch] failed, routing on: {exc}")
                resp = None
            if resp is not None:
                self._note("watch")
                self._respond(resp)
                return

        # ── 2e. THE BRAIN ─────────────────────────────────────────────────────
        # A learned intent router (brain.py, skills.py). A confident match
        # whose skill can extract its arguments runs directly — the same stage
        # method the regex below would call, no LLM. Anything else falls to the
        # regex command stages exactly as they were, which stay as the fallback
        # until the brain has matched each of them on the corpus.
        if self._brain_turn(text):
            return

        # ── 2f-8. Regex command stages (the fallback) ────────────────────────
        # Skipped only when the brain is CONFIDENT the sentence is
        # conversation — as sure as it would need to be to act. The
        # unsupported-action refusal below still runs either way.
        if not self._brain_chat:
            claim = self._regex_intent(text) if self._brain_fell_through else None
            if self._regex_commands(text):
                self._note(f"regex:{(claim or self._regex_intent(text))[0]}")
                # The brain fell through and a regex stage knew what this
                # was: that is a lesson (brain.learn applies the bar).
                if claim is not None:
                    self._learn_from_regex(text, claim)
                return

        # ── 8a. "Did you mean ...?" ──────────────────────────────────────────
        # Nothing claimed it, but the brain NEARLY knew: a skill ranked top
        # with a raw similarity over brain.ask_threshold, and that skill's own
        # slots can run the sentence. Asking costs one short question; the
        # alternative is the LLM guessing, or "I can't do that" for something
        # Nova can do. A yes runs it and teaches the brain (source=user); a no
        # is remembered so the same wrong guess is never offered again.
        if self._offer_brain_guess(text):
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
        if _is_unsupported_action(text):
            self._note("unsupported")
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
        self._note("conversation" if self._brain_chat else "llm")
        self.set_state("thinking")
        self._stream_response(text, rag_context=rag_ctx)

    # ═════════════════════════════════════════════════════════════════════════════
    # Routing: the brain, and the regex stages behind it
    # ═════════════════════════════════════════════════════════════════════════════
    def _init_brain(self) -> None:
        """The learned router. Loads in the background (the first run embeds
        ~700 seeds, ~4s; later runs read them from nova_memory.db). Until it
        is ready, and whenever it is disabled or fails, every turn takes the
        regex chain exactly as before."""
        self.brain = None
        if not self.config.get("brain", {}).get("enabled", True):
            log.info("[brain] disabled in config; regex routing only")
            return
        try:
            from brain import Brain
            from skills import SKILLS
            self.brain = Brain(DATA_DIR / "nova_memory.db", SKILLS, self.config, va=self)
            self.brain.load_background()
        except Exception as exc:
            log.warning(f"[brain] unavailable, regex routing only: {exc}")
            self.brain = None

    def _brain_turn(self, text: str) -> bool:
        """Route through the brain. True if it handled the turn."""
        # Per turn, never carried over: a stale verdict from the last turn must
        # not skip the regex stages for this one (nor while loading).
        self._brain_chat = False
        self._brain_fell_through = False
        b = self.brain
        if b is None or not b.ready:
            return False
        t0 = time.time()
        try:
            d = b.decide(text, self)
        except Exception as exc:
            log.warning(f"[brain] decide failed, falling back: {exc}")
            return False
        ms = (time.time() - t0) * 1000
        self._brain_chat = d is None and b.conversation
        self._brain_fell_through = d is None and not b.conversation
        if d is not None:
            # What a correction ("no, I meant ...") would be correcting.
            self._last_reflex = {"text": text, "skill": d.skill.name,
                                 "match": dict(b.last_match or {}), "ts": time.time()}
        m = b.last_match or {}
        if self.config.get("brain", {}).get("log_disagreements", True):
            regex = self._regex_route(text)
            mine = d.skill.stage if d is not None else ("llm" if b.conversation else None)
            if mine is not None and regex != mine:
                log.info(f"[brain-disagree] {text!r}: brain {mine} "
                         f"({m.get('skill')}@{m.get('confidence')}), regex {regex}")
        if d is None:
            log.info(f"[brain] {'conversation' if b.conversation else 'fallthrough'} "
                     f"(top {m.get('skill')}@{m.get('confidence')}, "
                     f"rival {m.get('rival')}, {ms:.0f}ms)")
            return False
        log.info(f"[brain] reflex {d.skill.name}@{d.confidence:.2f} {d.args} ({ms:.0f}ms)")
        self._note("reflex", d.skill.name, d.confidence, m.get("text"))
        try:
            d.skill.run(self, text, d.args)
        except Exception as exc:
            # The skill's handler raised. Never cost him the turn: say so.
            log.exception(f"[brain] {d.skill.name} failed: {exc}")
            self._respond("Something went wrong doing that.")
        return True

    # ── Stage 2: learning, "did you mean", taught phrases, corrections ───────
    _YES_RE = re.compile(r"^\s*(?:yes|yeah|yep|yup|sure|please|do it|go ahead|"
                         r"that'?s right|correct|right|exactly|uh huh)\b", re.I)
    _NO_RE = re.compile(r"^\s*(?:no|nope|nah|not that|wrong|neither)\b", re.I)
    _MEANT_RE = re.compile(
        r"^\s*(?:no|nope|nah|wrong|not that)\b[,.!]?\s*(?:,\s*)?"
        r"(?:i\s+(?:meant|mean|said|wanted|asked(?:\s+for)?)|"
        r"that'?s\s+not\s+what\s+i\s+(?:meant|said|asked(?:\s+for)?))\b[,:]?\s*(?P<rest>.*)$",
        re.I)
    _WRONG_RE = re.compile(
        r"^\s*(?:no[,.!]?\s+)?(?:that'?s\s+(?:wrong|not\s+right|not\s+what\s+i\s+"
        r"(?:meant|said|asked(?:\s+for)?))|wrong\s+one|that\s+was\s+wrong)[.!]*\s*$", re.I)
    _TEACH_RE = re.compile(
        r"^\s*(?:from\s+now\s+on[,\s]+)?when(?:ever)?\s+i\s+say\s+[\"']?(?P<rest>.+)$", re.I)
    _FORGET_CMD_RE = re.compile(
        r"^\s*(?:forget|delete|remove|stop\s+using)\s+(?:the\s+)?(?:phrase|command|"
        r"shortcut)\s+[\"']?(?P<p>.+?)[\"']?[.!?]*\s*$"
        r"|^\s*forget\s+what\s+(?:to\s+do|you\s+do|i\s+said)\s+when\s+i\s+say\s+"
        r"[\"']?(?P<q>.+?)[\"']?[.!?]*\s*$", re.I)
    _LIST_CMD_RE = re.compile(
        r"^\s*what\s+(?:phrases|commands|shortcuts)\s+(?:have\s+i\s+taught\s+you|"
        r"do\s+(?:you\s+know|i\s+have))\b", re.I)
    CORRECTION_WINDOW_S = 90.0

    def _brain_lessons(self, text: str) -> bool:
        b = self.brain
        if b is None or not b.ready:
            self._brain_offer = None
            return False
        # An answer to "did you mean X?" — one shot: anything else drops it.
        offer, self._brain_offer = self._brain_offer, None
        if offer is not None:
            m = self._MEANT_RE.match(text)
            if m:
                b.reject(offer["text"], offer["skill"])
                rest = m.group("rest").strip()
                if rest:
                    self._handle_turn_impl(rest)
                else:
                    self._respond("Okay. What did you mean?")
                return True
            if self._YES_RE.match(text):
                sk = b.by_name.get(offer["skill"])
                if sk is not None:
                    b.learn(offer["text"], sk.name, source="user", va=self)
                    log.info(f"[brain] confirmed: {offer['text']!r} -> {sk.name}")
                    self._last_reflex = None      # he chose it; nothing to correct
                    sk.run(self, offer["text"], offer["args"])
                    return True
            if self._NO_RE.match(text):
                b.reject(offer["text"], offer["skill"])
                self._respond("Okay.")
                return True
            # not an answer: route normally

        # A correction of the last reflex. Not while an actuation question is
        # open: "no, the other button" answers THAT question (stage 2d).
        last = self._last_reflex
        if last is not None and self.actuation.pending is None \
                and time.time() - last["ts"] <= self.CORRECTION_WINDOW_S:
            m = self._MEANT_RE.match(text)
            if m or self._WRONG_RE.match(text):
                self._last_reflex = None
                b.reject(last["text"], last["skill"])
                removed = b.unlearn(last["match"])
                log.info(f"[brain] corrected: {last['text']!r} was not {last['skill']}"
                         + (f" (unlearned the {removed} example)" if removed else ""))
                rest = m.group("rest").strip() if m else ""
                if rest:
                    self._handle_turn_impl(rest)
                else:
                    self._respond("Sorry about that. What did you mean?")
                return True

        # Teaching, forgetting and listing taught phrases.
        if self._LIST_CMD_RE.match(text):
            cmds = b.commands()
            if not cmds:
                self._respond("You haven't taught me any phrases yet.")
            else:
                parts = [f"{c['phrase']}, which {self._steps_spoken(c['steps'])}"
                         for c in cmds]
                self._respond("You've taught me " + "; ".join(parts) + ".")
            return True
        m = self._FORGET_CMD_RE.match(text)
        if m:
            phrase = (m.group("p") or m.group("q") or "").strip()
            if b.forget_command(phrase):
                self._respond(f"Done. {phrase} doesn't do anything special any more.")
            else:
                self._respond(f"I don't have a phrase called {phrase}.")
            return True
        m = self._TEACH_RE.match(text)
        if m:
            self._respond(self._teach_phrase(m.group("rest")))
            return True

        # A taught phrase.
        hit = b.match_command(text)
        if hit is not None:
            phrase, steps, sim = hit
            log.info(f"[brain] taught phrase {phrase!r}@{sim:.2f}: {len(steps)} step(s)")
            self._note("phrase", phrase, sim)
            for step in steps:
                self._run_step(step["text"])
            return True
        return False

    def _teach_phrase(self, rest: str) -> str:
        """"lights out, mute the sound and lock my Mac" -> a stored phrase.

        Every step must be something Nova can already do (the brain or a regex
        stage claims it), and nothing high-consequence: a shortcut that could
        shut the Mac down on a near-miss of a casual phrase is not worth having.
        """
        b = self.brain
        m = re.match(r"(?P<phrase>.+?)[\"']?\s*(?:,|\s+then\s+|\s+do\s+|\s+you\s+should\s+|"
                     r"\s+i\s+want\s+you\s+to\s+)\s*(?:please\s+)?(?P<steps>.+)$", rest.strip(), re.I)
        if not m:
            return ("Say it like this: when I say lights out, mute the sound and "
                    "lock my Mac.")
        phrase = m.group("phrase").strip().strip("\"'").strip()
        raw_steps = [x.strip(" .!") for x in re.split(
            r",\s*(?:and\s+)?(?:then\s+)?|\s+and\s+then\s+|\s+then\s+|\s+and\s+",
            m.group("steps").strip(" .!")) if x.strip(" .!")]
        if not phrase or not raw_steps:
            return "I didn't catch what you want that phrase to do."
        if len(raw_steps) > 6:
            return "That's a lot of steps for one phrase. Try six or fewer."
        steps = []
        for st in raw_steps:
            d = b.decide(st, self)
            if d is not None:
                if d.skill.tier == "high":
                    return (f"I won't put {st} behind a shortcut. Things that can't "
                            "be undone should always be asked for directly.")
                steps.append({"text": st, "skill": d.skill.name})
                continue
            stage, intent = self._regex_intent(st)
            if stage in ("llm", "unsupported") or stage.startswith("error"):
                return f"I don't know how to {st} yet, so I didn't save that phrase."
            import confidence as C
            if C.consequence(st) == C.HIGH:
                return (f"I won't put {st} behind a shortcut. Things that can't "
                        "be undone should always be asked for directly.")
            steps.append({"text": st, "skill": f"regex:{stage}"})
        b.teach_command(phrase, steps)
        return f"Got it. When you say {phrase}, I'll {self._steps_spoken(steps, verb=False)}."

    def _steps_spoken(self, steps: list, verb: bool = True) -> str:
        """Each step as Nova would SAY it: the skill's label ("check your
        battery"), not his words — "I'll what's my battery at" was the first
        version of this sentence."""
        b = self.brain
        texts = []
        for st in steps:
            sk = b.by_name.get(st.get("skill")) if b is not None else None
            texts.append(sk.label if sk is not None and sk.label else st["text"])
        joined = texts[0] if len(texts) == 1 else ", ".join(texts[:-1]) + " and " + texts[-1]
        return ("will " + joined) if verb else joined

    def _run_step(self, text: str) -> None:
        """One step of a taught phrase: the brain if it is sure, else the regex
        stages. Never the LLM — a step that nothing claims says so."""
        b = self.brain
        d = b.decide(text, self) if b is not None and b.ready else None
        if d is not None:
            d.skill.run(self, text, d.args)
            return
        if not self._regex_commands(text):
            self._respond(f"I couldn't {text}.")

    def _learn_from_regex(self, text: str, claim: tuple) -> None:
        b = self.brain
        if b is None or not b.ready:
            return
        try:
            from skills import skill_for_regex
            skill = skill_for_regex(self, text, claim[0], claim[1])
            if skill and b.learn(text, skill, va=self):
                log.info(f"[brain] learned from the {claim[0]} stage: {text!r} -> {skill}")
        except Exception as exc:
            log.warning(f"[brain] learning failed (turn unaffected): {exc}")

    def _offer_brain_guess(self, text: str) -> bool:
        b = self.brain
        if b is None or not b.ready or not b.unsure or not self._brain_fell_through:
            return False
        u = b.unsure
        sk = b.by_name.get(u["skill"])
        if sk is None or b.was_rejected(text, sk.name):
            return False
        label = sk.label or sk.name.split(".", 1)[-1].replace("_", " ")
        self._brain_offer = {"text": text, "skill": sk.name, "args": u["args"]}
        self._note("ask", sk.name, u.get("confidence"))
        log.info(f"[brain] asking: did you mean {sk.name}? (top {u['top']}, "
                 f"confidence {u['confidence']})")
        self._respond(f"Did you mean {label}?")
        return True

    def _run_math(self, text: str) -> bool:
        try:
            import mathskill
            r = mathskill.slots(text)
        except Exception as exc:
            log.warning(f"[math] parse failed, routing on: {exc}")
            return False
        if not r or not r.get("said"):
            return False
        log.info(f"[math] {r.get('expr')!r} = {r.get('value')}")
        self._note("math")
        self._respond(r["said"])
        return True

    def _regex_route(self, text: str) -> str:
        """Which regex stage WOULD claim this, with no side effects."""
        return self._regex_intent(text)[0]

    def _regex_intent(self, text: str) -> tuple[str, Optional[str]]:
        """(stage, intent) the regex chain WOULD give this, with no side
        effects. Used to log disagreements — how the brain earns the removal
        of a regex stage, or shows it has not — and as the TEACHER for
        learning (skills.skill_for_regex maps it to a skill)."""
        try:
            import mathskill
            import standing
            if mathskill.slots(text):
                return "math", None
            if standing.claims(text):
                return "watch", None
            for stage, det in (("view", self.views.detect_intent),
                               ("actuation", self.actuation.detect_intent),
                               ("calendar", self.calendar.detect_intent),
                               ("screen", self.screen.detect_intent),
                               ("weather", self.weather.detect_intent),
                               ("market", self.market.detect_intent),
                               ("files", self.files.detect_intent)):
                got = det(text)
                if got is not None:
                    return stage, got
            if self._memory_would_claim(text):
                return "memory", None
            if self._fast_path(text) is not None:
                return "fastpath", None
            got = self.tools.detect(text)
            if got is not None:
                return "tools", got
            if _is_unsupported_action(text):
                return "unsupported", None
        except Exception as exc:
            return f"error:{exc}", None
        return "llm", None

    def _memory_would_claim(self, text: str) -> bool:
        """_handle_memory_intent's verdict WITHOUT its writes (it saves and
        forgets as it matches, like tools.match did)."""
        if self._SAVE_RE.search(text) or self._NOTE_SAVE_RE.match(text) \
                or self._UPDATE_RE.search(text) or self._FORGET_RE.search(text) \
                or self._NOTE_FORGET_RE.match(text):
            return True
        m = self._RECALL_RE.search(text)
        if m:
            key = m.group(1).strip()
            if self.memory.find_fact(key):
                return True
            return not (self._is_system_query(key) or self.tools.detect(text) is not None)
        return any(p in text.lower() for p in self._META_RECALL)

    def _regex_commands(self, text: str) -> bool:
        """Stages 2f-8: the regex chain as it was, behind the brain. True if
        a stage claimed the turn."""
        # ── 2f. Actuation ─────────────────────────────────────────────────────
        # Typing and clicking, WORK MODE ONLY. Before tools, whose "open X" and
        # "search X" rules would claim some of this phrasing.
        act_intent = self.actuation.detect_intent(text)
        if act_intent is not None:
            self._run_actuation(act_intent, text)
            return True

        # ── 3. Calendar / reminders intents ──────────────────────────────────
        # BEFORE memory + fast-path + tools: detection is strict (needs an
        # unambiguous calendar word), and running it early stops the greedy
        # memory-recall regex ("what's my X") from swallowing "what's my
        # calendar today" and the tool regexes ("find X"/"open X") from
        # grabbing calendar phrasing.
        cal_intent = self.calendar.detect_intent(text)
        if cal_intent is not None:
            self._run_calendar(cal_intent, text)
            return True

        # ── 4. Screen awareness ──────────────────────────────────────────────
        # BEFORE files: "screen" is not a file stopword, so "read my screen"
        # would otherwise become a file search for a document named "screen".
        # Detection is strict — it needs an explicit screen phrase.
        screen_intent = self.screen.detect_intent(text)
        if screen_intent is not None:
            self._run_screen(screen_intent, text)
            return True

        # ── 4b. Weather ───────────────────────────────────────────────────────
        # BEFORE files and tools: "what's the weather" contains no file words,
        # but "is it going to rain tomorrow" would otherwise fall through to
        # the LLM, which would happily invent a forecast. Detection is strict:
        # a weather word must appear, and an action verb rules it out.
        weather_intent = self.weather.detect_intent(text)
        if weather_intent is not None:
            self._run_weather(weather_intent, text)
            return True

        # ── 4c. Market data ───────────────────────────────────────────────────
        # AFTER weather (both answer "how's the X") and BEFORE files and tools,
        # whose "find X" and "open X" rules would swallow "what's Apple stock
        # at". Detection needs a finance word AND a resolvable subject, so
        # "how are you doing" and "I had apple pie" fall straight through.
        market_intent = self.market.detect_intent(text)
        if market_intent is not None:
            self._run_market(market_intent, text)
            return True

        # ── 5. File management intents ───────────────────────────────────────
        # BEFORE memory and tools. Tools' web-search rule ("find X") and app
        # launch ("open X") would both swallow file phrasing, and memory recall
        # ("what's my X") would answer "I don't have your resume stored yet".
        # Detection is strict — it needs a file word AND a searchable name —
        # so ordinary conversation still falls straight through.
        file_intent = self.files.detect_intent(text)
        if file_intent is not None:
            self._run_files(file_intent, text)
            return True

        # ── 6. Memory intents ────────────────────────────────────────────────
        resp = self._handle_memory_intent(text)
        if resp is not None:
            self._respond(resp)
            return True

        # ── 7. Fast-path intents ─────────────────────────────────────────────
        resp = self._fast_path(text)
        if resp is not None:
            self._respond(resp)
            return True

        # ── 8. Tool intents ──────────────────────────────────────────────────
        resp = self.tools.match(text)
        if resp is not None:
            self._finish_tool(resp)
            return True

        return False

    # ═════════════════════════════════════════════════════════════════════════════
    # Stage runners — ONE code path per handler
    # ═════════════════════════════════════════════════════════════════════════════
    # Each is what a routing stage does once it has decided the utterance is
    # its own. The regex chain calls them after detect_intent; the brain's
    # skills (skills.py) call the SAME methods after classifying. Two copies of
    # "what happens after a calendar read" would drift — the follow-up offer,
    # the panel, work mode — and the drift would only show up in use.
    def _run_view(self, intent: str, text: str) -> None:
        self._respond(self.views.handle(intent, text))

    def _run_actuation(self, intent: str, text: str) -> None:
        self._respond(self.actuation.handle(intent, text))

    def _run_calendar(self, intent: str, text: str) -> None:
        resp = self.calendar.handle(intent, text)
        # A read handler may arm a soft follow-up (e.g. read_rest_of_week).
        followup = self.calendar.pending_intent
        self.calendar.pending_intent = None
        if followup:
            self._calendar_offer = lambda fi=followup: self.calendar.handle(fi, "")
        self._emit_panel(self.calendar)
        self._respond(resp)

    def _run_screen(self, intent: str, text: str) -> None:
        self._respond(self.screen.handle(intent, text))

    def _run_weather(self, intent: str, text: str) -> None:
        resp = self.weather.handle(text, intent)
        self._emit_panel(self.weather)
        self._respond(resp)

    def _run_market(self, intent: str, text: str) -> None:
        resp = self.market.handle(intent, text)
        self._emit_panel(self.market)
        self._respond(resp)

    def _run_files(self, intent: str, text: str) -> None:
        resp = self.files.handle(intent, text)
        if self.files.pending_offer is not None:
            self._calendar_offer = self.files.pending_offer
            self.files.pending_offer = None
        self._emit_panel(self.files)
        self._respond(resp)

    def _finish_tool(self, resp: str) -> None:
        """Everything that follows a tool answering, whoever routed to it."""
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
        # She just did something on his Mac, so she steps aside into the
        # corner — the second way into work mode, alongside asking for it.
        if getattr(self.tools, "touched_mac", False):
            self.tools.touched_mac = False
            self.set_work_mode(True, reason="acted on the Mac")
        # He asked what his CPU was doing: the number she just said should
        # also be a number he can see, so the status row comes back for a
        # while and then settles away on its own.
        if getattr(self.tools, "showed_system", False):
            self.tools.showed_system = False
            try:
                self.views.recall_system()
            except Exception as exc:
                log.warning(f"could not recall status row: {exc}")
        self._respond(resp)

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
    def _barged_in(self) -> None:
        """He talked over her: stop, and SAY so on screen immediately.

        The orb read "Speaking" for a moment after she had gone quiet, because
        nothing broadcast a state until the main loop unblocked from
        `tts.wait_until_done`. From where he is sitting that is Nova ignoring
        him — the one impression barge-in exists to prevent.
        """
        self.tts.stop_and_flush()
        self.set_state("listening")

    def _begin_conversation(self) -> None:
        """The welcome is over: he is talking to her now.

        The greeting and the status row are a RESTING state — what home looks
        like when nobody has said anything yet. The wake word is exactly the
        moment that stops being true, so this fires there rather than on her
        reply: waiting for the answer left "Good morning, NICHOLAS" under the
        orb while she was already listening to him.

        Typing counts too (see _respond), because Cmd-T is a conversation that
        never passes the wake word.
        """
        self._conversation_started = True
        try:
            self.views.refresh_home()
        except Exception as exc:
            log.warning(f"could not retire the greeting: {exc}")

    def _respond(self, text: str) -> None:
        """Canonical response path: log → store → broadcast → speak."""
        text = text.strip()
        if not text:
            return
        self._mark_first()
        self._last_response = text
        self._begin_conversation()
        self.memory.add_turn("assistant", text)
        # Deliberately NOT added to _session_turns. This is the deterministic
        # command path; only real conversation is worth learning from. Feeding
        # commands to the reconciler is how the memory DB ended up holding
        # ('adjective', 'adjective', 'open,brave,brave,brave,brave').
        self._mark_turn_was_command()
        self.ws.send_message("assistant", text)
        log.info(f"[nova] {text}")

        # Typed in, typed back. When Nicholas types instead of talking he is
        # usually somewhere he cannot talk, so speaking the answer aloud is the
        # last thing he wants. The reply still reaches the UI over the WS.
        if self._silent_turn:
            self.set_state("idle")
            self._settle_panels()
            return

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
        # She has stopped talking, so anything held on screen for him to READ
        # can start its clock now. Armed here rather than in the handler
        # because a timer started before she spoke is already half spent by
        # the time he has heard the answer.
        self._settle_panels()

    def _settle_panels(self) -> None:
        try:
            self.views.settle_system(
                float(self.config.get("ui", {}).get("status_recall_seconds", 10)))
        except Exception as exc:
            log.warning(f"could not settle the status row: {exc}")

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
    # [\w' ] not [\w ]: "my sister's name" has an apostrophe, and without it
    # the save regex failed, the sentence fell to the LLM, and the LLM said
    # "got it" for a fact nothing stored (found 2026-11-19).
    _SAVE_RE    = re.compile(r"remember (?:that )?my ([\w][\w' ]*?) is (.+)", re.I)
    # "who's my favorite superhero" missed entirely and fell to the LLM, which
    # invented an answer ("Deadpool") for a fact Nova was supposed to know.
    # People ask about people with WHO, and about things with WHAT.
    _RECALL_RE  = re.compile(
        r"(?:what|who)(?:'?s| is|'?re| are)\s+my\s+([\w][\w' ]*?)[\s.?!]*$", re.I)
    _UPDATE_RE  = re.compile(r"actually (?:my ([\w][\w' ]*?) is |it'?s )(.+)", re.I)
    _FORGET_RE  = re.compile(r"forget (?:that )?my ([\w][\w' ]*?)[\s.?!]*$", re.I)
    _META_RECALL = ("what do you know about me", "what do you remember",
                    "what have i told you")
    # Free-form notes: "remember that I like tea". Anchored at the start, and
    # the things "remember ..." also introduces are excluded: a reminder
    # ("remember to call mom" is the calendar's), a question ("remember when
    # we...", "remember what I said"), and a bare pointer ("remember that").
    _NOTE_SAVE_RE = re.compile(
        r"^\s*(?:please\s+)?(?:remember|keep in mind)\s+(?:that\s+)?"
        r"(?!(?:to|when|what|who|where|how|why|if|whether|me|this|it)\b)"
        r"(?!that\s*[.?!]*$)(.{3,}?)[\s.?!]*$", re.I)
    _NOTE_FORGET_RE = re.compile(
        r"^\s*(?:please\s+)?forget\s+(?:that\s+|about\s+)?"
        r"(?!(?:it|about it|everything|this|me)\b)(?!that\s*[.?!]*$)(.{3,}?)[\s.?!]*$",
        re.I)

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

        # Save a free-form note — "remember that I like tea". After the
        # structured form, so "remember that my X is Y" keeps its key.
        m = self._NOTE_SAVE_RE.match(text)
        if m:
            from memory import to_second_person
            stored = self.memory.add_note(m.group(1))
            if stored:
                return f"Got it. I'll remember that {to_second_person(stored)}."

        # Recall — "what's my X" (searches EVERY category, not just 'explicit')
        m = self._RECALL_RE.search(text)
        if m:
            key   = m.group(1).strip()
            found = self.memory.find_fact(key)
            if found:
                return f"Your {key} is {found[1]}."
            # Nothing stored. If this reads as a system question, fall through so
            # Tools can answer it; otherwise say honestly that it isn't stored.
            # "what's my memory usage" answered "I don't have your memory
            # usage stored yet": the word list below never knew "memory" or
            # "usage", and the routing harness hid it by asking tools FIRST
            # (the pipeline asks memory first). Asking tools itself — a dry
            # run, nothing happens — cannot drift out of step with what tools
            # can actually answer.
            if self._is_system_query(key) or self.tools.detect(text) is not None:
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

        # Forget a free-form note — "forget that I put my resume in Downloads"
        m = self._NOTE_FORGET_RE.match(text)
        if m:
            if self.memory.forget_note(m.group(1)):
                return "Done. I've forgotten that."
            return f"I don't have that stored, {name}."

        # Meta-recall — "what do you know about me"
        low = text.lower()
        if any(p in low for p in self._META_RECALL):
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
        # Nova's LLM runs no tools, so a sentence promising to watch, remind or
        # follow up — or claiming it already set/opened/sent something — is
        # never true here (honesty.py). Checked per sentence BEFORE it is
        # spoken, so the streaming overlap is untouched.
        import honesty
        honest = honesty.Filter() if self.config["llm"].get("honesty_filter", True) \
            else (lambda x: x)
        # Typed in, typed back — on this path too. _respond always honoured
        # _silent_turn; this one spoke every LLM answer aloud, so a question
        # he TYPED because he couldn't talk was read out through his speakers.
        silent = bool(getattr(self, "_silent_turn", False))

        def _emit(text: str) -> None:
            text = honest(text)
            text = _clean_for_tts(text)
            if not text or silent:
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
            if not full_response:
                self._mark_first()
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
            for sentence in self._SENT_SPLIT.split(full_response.strip()):
                s = sentence.strip()
                if s:
                    _emit(s)

        self.tts.wait_until_done(timeout=60)

        if getattr(honest, "replaced", None):
            for said, truth in honest.replaced:
                log.info(f"[honesty] replaced {said!r}")
            # What is stored and shown is what she SAID, not what the model
            # wrote: a promise kept in history would be repeated next turn.
            full_response = honest.text

        self._last_response = full_response
        self._begin_conversation()
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
