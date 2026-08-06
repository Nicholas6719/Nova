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
  nova_backend/          ← Python backend (new — already placed here)
    nova.py
    config.json
    system_prompt.py
    stt_engine.py
    llm_engine.py
    tts_engine.py
    memory.py
    ws_server.py
    tools.py
    rag.py
    requirements.txt
    ARCHITECTURE.md
  CLAUDE.md              ← this file
```

---

## Python Backend — What Each File Does

**`nova.py`** — Main entry point. `VoiceAssistant` class owns the entire pipeline. Initializes all engines, runs the wake word loop, routes every user utterance through a 7-stage handler chain (first match wins).

**`config.json`** — All tuneable settings. Model name, ports, voice, user name, silence timing, RAG toggle. Edit this without touching Python code.

**`system_prompt.py`** — Nova's personality. Rebuilt fresh on every LLM call. Injects live memory context and RAG context. Defines Nova's identity, tone, and communication rules.

**`stt_engine.py`** — Microphone input. One persistent 16 kHz mono stream feeds both wake detection and command capture. Wake detection dispatches on `wake_word.engine`: `openwakeword` (neural, noise-robust — see `wake_openwakeword.py`) or `transcript` (legacy Whisper rolling-window scan), auto-falling back to transcript if the OWW model can't load. VAD-gated command recording via webrtcvad with adaptive silence cutoffs. Transcription via faster-whisper (local, offline).

**`wake_openwakeword.py`** — Neural wake-word detection via OpenWakeWord. `OpenWakeWordDetector` accumulates the stream's 30 ms frames into 80 ms chunks, scores them with onnxruntime (fully local), and fires on a score threshold held for N consecutive windows; OWW's built-in Silero VAD gate suppresses scoring on non-speech so steady noise never triggers. The custom "Nova" model (`nova.onnx`) is trained via `training/` (Colab). Replaces transcript-based wake, which couldn't survive steady background noise.

**`llm_engine.py`** — Local AI inference via MLX (Apple Silicon optimized). Streaming generation — tokens arrive one by one via callback. Default model: Llama 3.2 3B. Upgrade path: Llama 3.1 8B (one line in config.json).

**`tts_engine.py`** — Text to speech. Primary: Kokoro ONNX (local, no cloud). Fallback: macOS `say`. Background queue worker for gapless playback. TTS overlaps with LLM streaming — Nova speaks sentence 1 while still generating sentence 2.

**`memory.py`** — SQLite persistent memory at `~/Library/Application Support/Nova/nova_memory.db`. Four tables: facts (explicit), episodes (semantic), conversations (turn history), user_profile (inferred). Background summarization thread — never blocks the pipeline.

**`rag.py`** — Local document retrieval via ChromaDB. Watches `~/Documents`, indexes supported file types (.txt, .md, .pdf, .py, .swift, .js, .json), stores embeddings locally. Enriches LLM context with relevant personal documents. Zero network calls — all on-device.

**`tools.py`** — macOS system tools. App launch, volume control, battery status, web search, screenshot, system info. All deterministic — never touches the LLM. Easy to extend.

**`calendar_reminders.py`** — Apple Calendar & Reminders engine. Pure functions over EventKit (PyObjC), with an AppleScript fallback. Reads (today / this week / open reminders), writes (create event, create reminder), and edits (complete / delete / update by fuzzy title match). Zero side effects at import. Every function may raise RuntimeError — callers must catch. Ported from Jarvis, where it ran in production.

**`calendar_intents.py`** — Natural-language dispatch on top of `calendar_reminders.py`. `NovaCalendar.detect_intent` is strict regex only (no calendar word → None → normal chat); `handle` runs LLM JSON extraction (temp=0, heavily post-processed) and returns a single spoken string. Runs on the `nova-llm` worker thread, so its `self.llm.generate` calls are thread-safe. Reads are LLM-summarized with deterministic template fallbacks so a bad generation never invents a day/time.

**`ws_server.py`** — Bridge between Python and Swift. HTTP server on :5001, WebSocket server on :8766. Swift connects here to receive state changes, message content, and streaming tokens in real time.

**`requirements.txt`** — All Python dependencies. Install with `pip install -r requirements.txt`.

---

## Pipeline Routing Order (nova.py — first match wins)

1. System commands — sleep / wake / mute (always intercepted first)
2. Pending confirmation — yes/no flow for destructive actions
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

## Swift Side — What Needs to Change

### Files to ADD to Nova.xcodeproj

**`Nova/SwiftBackend/BackendManager.swift`**
- Locates Python and `nova_backend/nova.py`
- Launches `python nova.py` as a subprocess
- Sets `NOVA_DATA_DIR` environment variable to `~/Library/Application Support/Nova`
- Polls `/api/status` until backend confirms ready
- Restarts on crash
- Exposes `start()`, `stop()`, `isRunning: Bool`

**`Nova/SwiftBackend/NovaAPIClient.swift`**
- HTTP client: wraps GET /api/status, GET /api/messages, POST /api/message
- WebSocket client connecting to ws://localhost:8766
- On WS message: parses JSON, publishes state changes and messages via @Published / Combine
- Exposes `sendMessage(_ text: String)`, `currentState: String`, `messages: [Message]`

### Files to REMOVE (or gut) from Nova.xcodeproj

These are replaced entirely by the Python backend:
- `Nova/Core/NovaEngineCore.swift`
- `Nova/Core/LLMClient.swift`
- `Nova/Core/IntentDetector.swift`
- `Nova/Core/MathRouter.swift`
- `Nova/Core/NovaPersonality.swift`
- `Nova/Core/APIKeyProvider.swift`
- `Nova/Memory/` (entire directory)
- `Nova/Tools/` (entire directory)

### Files to KEEP and ADAPT

- `Nova/ContentView.swift` — keep the UI, wire it to `NovaAPIClient` instead of `ChatViewModel`'s old engine
- `Nova/Features/Chat/ChatViewModel.swift` — adapt to use `NovaAPIClient` for sending/receiving messages and state
- `Nova/Features/Chat/Message.swift` — keep as-is
- `Nova/NovaApp.swift` — add `BackendManager` initialization here
- `Nova/Voice/SpeechManager.swift` — TTS is now handled by Python; simplify or remove
- `Nova/Voice/SpeechRecognizer.swift` — STT is now handled by Python; simplify or remove

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

## What Does Not Exist Yet (build in order)

1. `Nova/SwiftBackend/BackendManager.swift` — launch and supervise Python process
2. `Nova/SwiftBackend/NovaAPIClient.swift` — HTTP + WebSocket client
3. `ChatViewModel.swift` adaptations — wire to NovaAPIClient
4. Calendar integration — EventKit (same approach as Jarvis's `calendar_reminders.py`)
5. Proactive notifications — background monitor for upcoming events
6. iOS app — independent, on-device (future phase, separate architecture)
