# Nova — Claude Code Briefing

Read this entire file before touching anything.

---

## What Nova Is

Nova is a macOS voice assistant app built by Nicholas Coppola.
It is his personal AI — built from the ground up to know him, serve him, and grow smarter about him over time. The reference point is Jarvis from Iron Man: capable, personal, loyal, always improving.

Nova has two parts:
1. A **Python backend** (`nova_backend/`) — the brain. Handles all voice, AI, memory, and tools.
2. A **SwiftUI frontend** (`Nova/`) — the face. Handles the chat UI and communicates with the Python backend.

---

## Project Location

```
/Users/nicholascoppola/Documents/Coding_Projects/Nova/
  Nova/                  ← SwiftUI source (existing)
  NovaTests/
  NovaUITests/
  Nova.xcodeproj
  nova_backend/          ← Python backend (the brain)
    nova.py                  entry point + routing pipeline
    config.json              all tuneables
    system_prompt.py         Nova's personality
    stt_engine.py            mic capture, wake dispatch, transcription
    wake_openwakeword.py     neural wake-word detector
    nova.onnx                trained "Nova" wake model
    llm_engine.py            MLX inference (streaming + blocking)
    tts_engine.py            Kokoro TTS + SeamlessPlayer
    memory.py                SQLite facts + rendering
    fact_reconciler.py       wake-mode passive learning
    calendar_reminders.py    EventKit engine (pure functions)
    calendar_intents.py      calendar NL dispatch
    tools.py                 macOS system tools
    rag.py                   local document retrieval
    ws_server.py             HTTP :5001 + WS :8766 bridge
    training/                wake-model training kit (Colab)
    requirements.txt
    ARCHITECTURE.md
  CLAUDE.md              ← this file
```

---

## Python Backend — What Each File Does

**`nova.py`** — Main entry point. `VoiceAssistant` class owns the entire pipeline. Initializes all engines, runs the wake word loop, routes every user utterance through an 8-stage handler chain (first match wins).

**`config.json`** — All tuneable settings. Model name, ports, voice, user name, silence timing, RAG toggle. Edit this without touching Python code.

**`system_prompt.py`** — Nova's personality. Rebuilt fresh on every LLM call. Injects live memory context and RAG context. Defines Nova's identity, tone, and communication rules.

**`stt_engine.py`** — Microphone input. One persistent 16 kHz mono stream feeds both wake detection and command capture. Wake detection dispatches on `wake_word.engine`: `openwakeword` (neural, noise-robust — see `wake_openwakeword.py`) or `transcript` (legacy Whisper rolling-window scan), auto-falling back to transcript if the OWW model can't load. VAD-gated command recording via webrtcvad with adaptive silence cutoffs. Transcription via faster-whisper (local, offline).

**`wake_openwakeword.py`** — Neural wake-word detection via OpenWakeWord. `OpenWakeWordDetector` accumulates the stream's 30 ms frames into 80 ms chunks, scores them with onnxruntime (fully local), and fires on a score threshold held for N consecutive windows; OWW's built-in Silero VAD gate suppresses scoring on non-speech so steady noise never triggers. The custom "Nova" model (`nova.onnx`) is trained via `training/` (Colab). Replaces transcript-based wake, which couldn't survive steady background noise.

**`llm_engine.py`** — Local AI inference via MLX (Apple Silicon optimized). Streaming generation — tokens arrive one by one via callback. Default model: Llama 3.2 3B. Upgrade path: Llama 3.1 8B (one line in config.json).

**`tts_engine.py`** — Text to speech. Primary: Kokoro ONNX (local, no cloud). Fallback: macOS `say`. Background queue worker for gapless playback. TTS overlaps with LLM streaming — Nova speaks sentence 1 while still generating sentence 2.

**`memory.py`** — SQLite persistent memory at `~/Library/Application Support/Nova/nova_memory.db`. Facts are STRUCTURED: `(category, key)` is a UNIQUE canonical identity, so a correction supersedes instead of piling up contradictions. `find_fact(key)` searches across ALL categories (recall must see passively-learned facts, not just explicit ones); `_render_fact` renders facts as English — third person for prompt injection, second person for spoken readback. Degenerate model-extracted facts (value merely restates the key) are rejected on write.

**`fact_reconciler.py`** — The LLM slow path of passive learning. Runs once per conversation in wake mode, on the `nova-llm` thread, off the live path. Decides insert/update/delete as strict JSON, then validates: contradictory decisions collapse per key, and every value must be GROUNDED in what the user actually said (`_is_grounded`) so the small model can't fabricate detail it never heard.

**`rag.py`** — Local document retrieval via ChromaDB. Watches `~/Documents`, indexes supported file types (.txt, .md, .pdf, .py, .swift, .js, .json), stores embeddings locally. Enriches LLM context with relevant personal documents. Zero network calls — all on-device.

**`tools.py`** — macOS system tools. App launch, volume control, battery status, web search, screenshot, system info. All deterministic — never touches the LLM. Easy to extend.

**`calendar_reminders.py`** — Apple Calendar & Reminders engine. Pure functions over EventKit (PyObjC), with an AppleScript fallback. Reads (today / this week / open reminders), writes (create event, create reminder), and edits (complete / delete / update by fuzzy title match). Zero side effects at import. Every function may raise RuntimeError — callers must catch. Ported from Jarvis, where it ran in production.

**`calendar_intents.py`** — Natural-language dispatch on top of `calendar_reminders.py`. `NovaCalendar.detect_intent` is strict regex only (no calendar word → None → normal chat); `handle` runs LLM JSON extraction (temp=0, heavily post-processed) and returns a single spoken string. Runs on the `nova-llm` worker thread, so its `self.llm.generate` calls are thread-safe. Reads are LLM-summarized with deterministic template fallbacks so a bad generation never invents a day/time.

**`ws_server.py`** — Bridge between Python and Swift. HTTP server on :5001, WebSocket server on :8766. Swift connects here to receive state changes, message content, and streaming tokens in real time.

**`requirements.txt`** — All Python dependencies. Install with `pip install -r requirements.txt`.

---

## Pipeline Routing Order (nova.py — first match wins)

1. System commands — sleep / wake / mute (always intercepted first)
2. Calendar follow-up offer — answers a "want to hear what's coming up?" with yes/no
3. Calendar / reminders intents — read / create / complete / delete / update (EventKit)
4. Memory intents — remember / recall / update / forget
5. Fast-path intents — greetings / date / time / repeat last response
6. Tool intents — open app / volume / battery / search / screenshot
7. RAG context enrichment — query personal documents for context
8. LLM fallback — MLX streaming with sentence-chunked TTS overlap

---

## Communication Protocol (Python ↔ Swift)

### HTTP (:5001)
| Method | Path           | Purpose                        |
|--------|----------------|--------------------------------|
| GET    | /api/status    | Health check + current state   |
| GET    | /api/messages  | Message history (last 50)      |
| POST   | /api/message   | Send text from UI to pipeline  |
| POST   | /api/mute      | Toggle mute from UI            |

### WebSocket (:8766)
```json
{"type": "state",   "state": "idle|listening|thinking|speaking"}
{"type": "message", "role": "user|assistant", "content": "..."}
{"type": "token",   "token": "..."}
```

---

## Swift Side — Current State

The migration to the Python backend is DONE. Every remaining Swift file is live
and in the build (verified by removing the dead ones and rebuilding):

- **`Nova/NovaApp.swift`** — app entry; starts `BackendManager`.
- **`Nova/SwiftBackend/BackendManager.swift`** — locates Python + `nova_backend/nova.py`,
  launches it as a child process with `NOVA_DATA_DIR` set, polls `/api/status`
  until ready, restarts on crash, passes its own PID as `NOVA_PARENT_PID` so the
  backend exits if the app is SIGKILLed. **Runs the backend from the REPO path
  (nova_backend is NOT bundled) — so Python-only changes need only an app
  relaunch, never an Xcode rebuild.**
- **`Nova/SwiftBackend/NovaAPIClient.swift`** — HTTP (:5001) + WebSocket (:8766)
  client; publishes state/messages/tokens via `@Published`.
- **`Nova/ContentView.swift`**, **`Features/Chat/ChatViewModel.swift`**,
  **`Features/Chat/Message.swift`** — the chat UI, wired to `NovaAPIClient`.
- **`Nova/Core/DebugLog.swift`**, **`Nova/Core/NovaLogger.swift`** — logging (active).
- **`Nova/Voice/SpeechManager.swift`**, **`SpeechRecognizer.swift`** — legacy
  on-device voice. **Python owns all voice I/O now**, so these are effectively
  inert, but `ChatViewModel` is still wired to them through Combine. Untangling
  that is a real refactor, not a cleanup — do it deliberately, with a build to
  verify, not as a drive-by.

Already deleted (2026-08-06 cleanup, build verified): `Nova/Core/`
NovaEngine, NovaEngineCore, LLMClient, IntentDetector, MathRouter,
NovaPersonality, APIKeyProvider; all of `Nova/Memory/` and `Nova/Tools/`;
`Nova/Voice/AudioSessionQueue.swift`.

---

## Invariants — Never Break These

1. **Ports are 5001 (HTTP) and 8766 (WS).** These differ from Jarvis (3000/8765) intentionally so both apps can run simultaneously. Do not change them.
2. **LLM n_ctx stays at 4096.** Do not raise the context window without explicit instruction.
3. **No cloud dependencies.** Nova is fully local and private. Do not add any API keys, external HTTP calls, or cloud services.
4. **Memory summarization runs on a daemon thread.** It must never block the main pipeline or the voice loop.
5. **RAG loads in the background.** Queries must fail gracefully (return empty string) if the index is not yet ready.
6. **TTS and LLM overlap.** The sentence-chunked streaming pipeline in `nova.py._stream_response` must not be changed to a blocking pattern.
7. **Kokoro model files** (`kokoro-v1.0.onnx`, `voices-v1.0.bin`) live in `nova_backend/` — same files as Jarvis. TTS falls back to macOS `say` if they are missing.
8. **NOVA_DATA_DIR** must be set by the Swift BackendManager so data survives app updates.
9. **Wake word keywords** are `["nova", "hey nova"]` — do not change without updating config.json.
10. **Do not add markdown, bullet points, numbered lists, or em dashes** to anything the LLM outputs. Nova speaks its responses aloud.

---

## Privacy Rules

- No user data leaves the machine. Ever.
- No OpenAI API key. No external LLM calls.
- SQLite memory database: `~/Library/Application Support/Nova/nova_memory.db`
- ChromaDB RAG index: `~/Library/Application Support/Nova/rag_index/`
- Both paths are local-only. Never transmit their contents.

---

## First Run (Python backend)

```bash
cd /Users/nicholascoppola/Documents/Coding_Projects/Nova/nova_backend
pip install -r requirements.txt
# Copy Kokoro model files from Jarvis:
cp /Users/nicholascoppola/Documents/Coding_Projects/Jarvis/kokoro-v1.0.onnx .
cp /Users/nicholascoppola/Documents/Coding_Projects/Jarvis/voices-v1.0.bin .
python nova.py
```

The MLX model (~2GB) downloads automatically on first run.

---

## Not Built Yet

1. Proactive notifications — background monitor for upcoming events / due reminders
2. Barge-in over speakers — needs acoustic echo cancellation (see branch `feat/barge-in`;
   measured: the wake model's score collapses to ~0 once Nova's own voice reaches the mic,
   so this is NOT a tuning problem). Works on headphones today.
3. UI overhaul — deliberately late, after capabilities are concrete
4. iOS app — far back burner, separate on-device architecture

