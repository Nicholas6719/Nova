#!/usr/bin/env python3
"""
Nova — Neural Omniscient Voice Assistant
macOS backend · Entry point

Pipeline:
  Wake word → STT (faster-whisper + webrtcvad)
  → Fast-path routing → Memory → Calendar/Reminders → Tools
  → LLM (MLX Llama, streaming) → Sentence-chunked TTS (Kokoro ONNX)

Communication with SwiftUI:
  HTTP  :5001  — /api/status, /api/messages, /api/message
  WS    :8766  — real-time state + token stream

Run: python nova.py
"""

from __future__ import annotations

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
      2. Pending confirmation flow
      3. Calendar intents (read/create/complete/delete/update events + reminders)
      4. Memory intents   (remember / recall / update / forget)
      5. Fast-path intents (greeting / date / time / repeat)
      6. Tool intents     (open app / volume / battery / search / screenshot)
      7. RAG context enrichment
      8. LLM fallback     (MLX streaming + sentence-chunked TTS)
    """

    # ── Init ──────────────────────────────────────────────────────────────────────
    def __init__(self) -> None:
        self.config           = load_config()
        self.is_muted         = False
        self.is_awake         = True
        self._last_response   = ""
        self._pending_action: Optional[Callable[[], str]] = None
        # A SOFT one-shot follow-up armed by a calendar read ("want to hear
        # what's coming up?"). Unlike _pending_action it only fires on an
        # affirmative reply and never eats an unrelated next command.
        self._calendar_offer: Optional[Callable[[], str]] = None
        self._rag_ready       = False
        self._rag             = None
        # Set by a "return to wake mode" voice command to drop out of
        # conversation mode immediately (without full sleep).
        self._return_to_wake  = False
        # Transcript of the CURRENT conversation (user+assistant), used by the
        # wake-mode memory reconciliation pass. Reset when a conversation ends.
        self._session_turns: list[dict] = []

        # Shared microphone gate. Set = capture allowed; cleared = mic paused.
        # A voice assistant must not record while it speaks — both to avoid
        # hearing its own TTS and to avoid a CoreAudio input/output device
        # conflict (PaMacCore -50) that garbles playback. TTS clears this around
        # playback; the STT record loops skip capture while it is clear.
        self.mic_gate = threading.Event()
        self.mic_gate.set()

        log.info("Nova initializing…")
        self._init_stt()
        self._init_llm()
        self._init_tts()
        self._init_memory()
        self._init_rag()
        self._init_tools()
        self._init_calendar()
        self._init_ws()
        log.info("Nova ready.")

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
        self._llm_queue: "queue.Queue[tuple]" = queue.Queue()
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
            job, done = self._llm_queue.get()
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
        self._llm_queue.put((lambda: self._handle_turn_impl(text), done))
        if done is not None:
            done.wait()

    def _submit_job(self, fn, wait: bool = False) -> None:
        """Enqueue an arbitrary LLM job (e.g. wake-mode reconciliation) onto the
        MLX worker thread. Off the live path; not user-facing."""
        done = threading.Event() if wait else None
        self._llm_queue.put((fn, done))
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
            from rag import NovaRAG
            docs_dir = Path(self.config["memory"]["rag_docs_dir"]).expanduser()
            self._rag = NovaRAG(DATA_DIR / "rag_index", docs_dir=docs_dir)
            self._rag_ready = True
            log.info("RAG index ready.")
        except Exception as exc:
            log.warning(f"RAG unavailable: {exc}")

    def _init_tools(self) -> None:
        from tools import NovaTools
        self.tools = NovaTools(self.config)

    def _init_calendar(self) -> None:
        # NovaCalendar's LLM extraction/summarize calls run on the nova-llm
        # worker thread (calendar handling is dispatched from _handle_turn_impl,
        # which is already on that thread), so passing self.llm is thread-safe.
        from calendar_intents import NovaCalendar
        self.calendar = NovaCalendar(self.config, self.llm, self.memory)

    def _init_ws(self) -> None:
        from ws_server import NovaWSServer
        self.ws = NovaWSServer(
            http_port=self.config["server"]["http_port"],
            ws_port=self.config["server"]["ws_port"],
            on_text_message=self._handle_text_input,
        )
        self.ws.start()

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
        while True:
            if self.is_muted or not self.is_awake:
                in_conversation = False
                time.sleep(0.1)
                continue

            # ── Phase 1: Get the user's turn ─────────────────────────────────
            if not in_conversation:
                # Wake mode: wait for "Nova" before listening.
                self.set_state("idle")
                log.info("Waiting for wake word…")
                wake_detected = self.stt.record_wake(
                    wake_keywords=self.config["wake_word"]["keywords"],
                    timeout_s=300.0,
                )
                if not wake_detected:
                    continue
                just_woke = True
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
                    # Silence during conversation → return to wake mode quietly,
                    # and let Nova review the conversation for anything to learn.
                    log.info("Conversation timed out — returning to wake mode.")
                    self._end_conversation()
                    in_conversation = False
                    continue
                # Woke but heard nothing usable — just re-listen for the wake word.
                continue

            # ── Phase 3: Transcribe ──────────────────────────────────────────
            self.set_state("processing")
            text = self.stt.transcribe(command_audio)
            if just_woke:
                text = self._strip_wake_prefix(text)
            if not text or not text.strip():
                # Bare wake phrase / no command. Stay in conversation so the user
                # can just speak, but don't get stuck: fall back to wake on repeat.
                if in_conversation:
                    self._end_conversation()
                    in_conversation = False
                continue

            log.info(f"[user] {text}")
            self.memory.add_turn("user", text)
            self._session_turns.append({"role": "user", "content": text})
            self.ws.send_message("user", text)
            # Run on the MLX worker thread and wait: capture must not resume
            # until the response (and its TTS) is done, or it self-captures.
            self._submit_turn(text, wait=True)
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
        low = stripped.lower()
        for kw in keywords:
            if low.startswith(kw):
                remainder = stripped[len(kw):]
                # Drop the punctuation/space that follows the wake word.
                return remainder.lstrip(" ,.!?—-").strip()
        return stripped

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

        # ── 2. Pending confirmation flow ─────────────────────────────────────
        if self._pending_action is not None:
            resp = self._resume_pending(text)
            if resp is not None:
                self._respond(resp)
                return

        # ── 2b. Soft calendar follow-up offer ────────────────────────────────
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

        # ── 4. Memory intents ────────────────────────────────────────────────
        resp = self._handle_memory_intent(text)
        if resp is not None:
            self._respond(resp)
            return

        # ── 5. Fast-path intents ─────────────────────────────────────────────
        resp = self._fast_path(text)
        if resp is not None:
            self._respond(resp)
            return

        # ── 6. Tool intents ──────────────────────────────────────────────────
        resp = self.tools.match(text)
        if resp is not None:
            self._respond(resp)
            return

        # ── 7. RAG context enrichment ────────────────────────────────────────
        rag_ctx = ""
        if self._rag_ready and self._rag:
            try:
                rag_ctx = self._rag.query(text, n_results=3)
            except Exception:
                pass

        # ── 8. LLM fallback (streaming) ──────────────────────────────────────
        self.set_state("thinking")
        self._stream_response(text, rag_context=rag_ctx)

    # ── Respond helper ────────────────────────────────────────────────────────────
    def _respond(self, text: str) -> None:
        """Canonical response path: log → store → broadcast → speak."""
        text = text.strip()
        if not text:
            return
        self._last_response = text
        self.memory.add_turn("assistant", text)
        self._session_turns.append({"role": "assistant", "content": text})
        self.ws.send_message("assistant", text)
        log.info(f"[nova] {text}")
        self.set_state("speaking")
        self.tts.speak(text)
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

        # Date
        if low in self._DATE_PHRASES or ("what" in low and "date" in low):
            return f"Today is {_spoken_date(datetime.now())}."

        # Time
        if low in self._TIME_PHRASES or ("what" in low and "time" in low and "date" not in low):
            return f"It's {_spoken_time(datetime.now())}."

        # Day of week
        if "what day" in low and "is it" in low:
            return f"Today is {datetime.now().strftime('%A')}."

        # Repeat last response
        if any(p in low for p in ("say that again", "repeat that", "what did you say")):
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
        if any(p in low for p in (
            "go to sleep", "back to sleep", "sleep mode", "stop listening",
            "take a break", "return to wake mode", "wake mode", "that's all",
            "that is all", "never mind", "nevermind", "we're done", "we are done",
            "that'll be all", "that will be all", "goodbye", "good night",
        )):
            self._return_to_wake = True
            self.set_state("idle")
            return f"Alright, {name}. Just say Nova when you need me."

        if any(p in low for p in ("mute yourself", "stop talking", "be quiet", "shut up")):
            self.is_muted = True
            return "Muted."

        if "unmute" in low or "start listening" in low:
            self.is_muted = False
            return "Unmuted."

        return None

    # ═════════════════════════════════════════════════════════════════════════════
    # Memory intents
    # ═════════════════════════════════════════════════════════════════════════════
    _SAVE_RE    = re.compile(r"remember (?:that )?my ([\w][\w ]*?) is (.+)", re.I)
    _RECALL_RE  = re.compile(r"what(?:'?s| is) my ([\w][\w ]*?)[\s.?]*$", re.I)
    _UPDATE_RE  = re.compile(r"actually (?:my ([\w][\w ]*?) is |it'?s )(.+)", re.I)
    _FORGET_RE  = re.compile(r"forget (?:that )?my ([\w][\w ]*?)$", re.I)

    # Explicit user-directed memory commands store under the 'explicit' category,
    # source='explicit', so they supersede by canonical key and sit alongside the
    # facts Nova learns passively (via regex fast-path + wake-mode reconciliation).
    _EXPLICIT_CAT = "explicit"

    def _handle_memory_intent(self, text: str) -> Optional[str]:
        name = self.config["user"]["address_as"]

        # Save — "remember that my X is Y"
        m = self._SAVE_RE.search(text)
        if m:
            key   = m.group(1).strip()
            value = m.group(2).strip().rstrip(".")
            self.memory.upsert_fact(self._EXPLICIT_CAT, key, value, source="explicit")
            return f"Got it. I'll remember your {key} is {value}."

        # Recall — "what's my X"
        m = self._RECALL_RE.search(text)
        if m:
            key   = m.group(1).strip()
            value = self.memory.get_fact(self._EXPLICIT_CAT, key)
            return (f"Your {key} is {value}." if value
                    else f"I don't have your {key} stored yet.")

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
                self.memory.upsert_fact(self._EXPLICIT_CAT, key, value, source="explicit")
                return f"Updated. Your {key} is now {value}."
            return "I'm not sure what you'd like me to update."

        # Forget — "forget that my X"
        m = self._FORGET_RE.search(text)
        if m:
            key = m.group(1).strip()
            self.memory.delete_fact(self._EXPLICIT_CAT, key)
            return f"Done. I've forgotten your {key}."

        # Meta-recall — "what do you know about me"
        low = text.lower()
        if any(p in low for p in ("what do you know about me", "what do you remember", "what have i told you")):
            facts = self.memory.facts_for_readback()
            if not facts:
                return f"I don't have anything stored about you yet, {name}."
            return "Here's what I know: " + "; ".join(facts) + "."

        return None

    # ═════════════════════════════════════════════════════════════════════════════
    # LLM streaming with sentence-chunked TTS overlap
    # ═════════════════════════════════════════════════════════════════════════════
    _SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
    # Sentence-boundary detector for INCREMENTAL emission during token streaming:
    # terminal .?! (optionally followed by a closing quote/bracket) at a word end.
    _TTS_SENT_END = re.compile(r"[.!?][\"')\]]?(?=\s|$)")

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
                m = self._TTS_SENT_END.search(pending)
                if not m:
                    break
                cut = m.end()
                sentence = pending[:cut].strip()
                pending = pending[cut:]
                if sentence:
                    _emit(sentence)
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

    # ═════════════════════════════════════════════════════════════════════════════
    # Confirmation flow
    # ═════════════════════════════════════════════════════════════════════════════
    def _resume_pending(self, text: str) -> Optional[str]:
        low = text.lower()
        if any(w in low for w in ("yes", "yeah", "yep", "do it", "confirm", "go ahead", "sure", "ok")):
            result = self._pending_action()
            self._pending_action = None
            return result
        else:
            self._pending_action = None
            return "Cancelled."

    def set_pending(self, action: Callable[[], str], prompt: str) -> None:
        """Register a pending confirmation action and speak the prompt."""
        self._pending_action = action
        self._respond(prompt)


# ── Entry point ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    VoiceAssistant().run()
