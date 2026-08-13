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
    maps_engine.py           MapKit location / nearby / ETA (subprocess)
    browser_control.py       Brave/Chrome/Safari: sites, tabs, history, scroll
    file_manager.py          filesystem engine: search, read, move/copy/rename
    file_intents.py          file NL dispatch: detect, disambiguate, confirm
    screen_awareness.py      window list + Vision OCR: "what's on my screen"
    rag.py                   local document retrieval
    ws_server.py             HTTP :5001 + WS :8766 bridge
    training/                wake-model training kit (Colab)
    tests/                   the test harness — see "Testing" below
    requirements.txt
    ARCHITECTURE.md
  CLAUDE.md              ← this file
```

---

## Python Backend — What Each File Does

**`nova.py`** — Main entry point. `VoiceAssistant` class owns the entire pipeline. Initializes all engines, runs the wake word loop, routes every user utterance through an 8-stage handler chain (first match wins).

**`config.json`** — All tuneable settings. Model name, ports, voice, user name, silence timing, RAG toggle. Edit this without touching Python code.

**`system_prompt.py`** — Nova's personality. Rebuilt fresh on every LLM call. Injects live memory context and RAG context. Defines Nova's identity, tone, and communication rules.

**`stt_engine.py`** — Microphone input. One persistent 16 kHz mono stream feeds both wake detection and command capture. Wake detection dispatches on `wake_word.engine`: `openwakeword` (neural, noise-robust — see `wake_openwakeword.py`) or `transcript` (legacy Whisper rolling-window scan), auto-falling back to transcript if the OWW model can't load. VAD-gated command recording via webrtcvad with adaptive silence cutoffs. Transcription via faster-whisper (local, offline) with **`beam_size=1` (greedy)** — faster *and* more accurate here, which is not the usual trade: measured under fan-level noise, beam search latched onto the `initial_prompt` and looped ("what time is it" → "What time is it, what is the date, what is the date…" forty-odd times, 0.555s vs 0.296s), scoring 6/8 against greedy's 8/8. Same failure family as the fan-noise hallucinations that drove the neural wake word; `_is_noise_hallucination` does not catch it (too many distinct words). Tune via `stt.beam_size`.

**`wake_openwakeword.py`** — Neural wake-word detection via OpenWakeWord. `OpenWakeWordDetector` accumulates the stream's 30 ms frames into 80 ms chunks, scores them with onnxruntime (fully local), and fires on a score threshold held for N consecutive windows; OWW's built-in Silero VAD gate suppresses scoring on non-speech so steady noise never triggers. The custom "Nova" model (`nova.onnx`) is trained via `training/` (Colab). Replaces transcript-based wake, which couldn't survive steady background noise.

**`llm_engine.py`** — Local AI inference via MLX (Apple Silicon optimized). Streaming generation — tokens arrive one by one via callback. Default model: Llama 3.2 3B. Upgrade path: Llama 3.1 8B (one line in config.json). **Carries a PROMPT CACHE between turns:** most of every prompt is identical to the last one (the identity block alone is ~930 tokens and never changes), and reprocessing it was 1.67s of the 1.77s wait before Nova spoke. The KV cache for the longest shared prefix is reused, so only new tokens are processed — time to first token 1.67s → 0.21s. The reuse length is always recomputed against the real token sequence, never assumed, and any exception drops the cache: a stale one would not raise, it would answer from a context the model never saw. Two measured traps are documented in the code — `generate_step` leaves a phantom sampled token in the cache (use a direct `model()` call), and mlx-lm skips special tokens when a prompt already starts with BOS (encode the same way or the model silently gets a second BOS). Rollback: `llm.prompt_cache` false.

**`tts_engine.py`** — Text to speech. Primary: Kokoro ONNX (local, no cloud). Fallback: macOS `say`. Background queue worker for gapless playback. TTS overlaps with LLM streaming — Nova speaks sentence 1 while still generating sentence 2. The FIRST chunk is additionally split at a clause break when the opening sentence is long (`tts.split_first_clause`), because nothing is audible until that chunk is both generated and synthesised — measured worst case 2.83s → 1.89s to first audio. It splits only at a comma/semicolon/colon, where the voice already pauses. **Kokoro stays on the default ONNX provider:** CoreML was measured slower (0.586s → 0.669s; only 129 of 2256 nodes partitioned) AND it changed the audio (correlation 0.998, different sample counts) — that is Nicholas's custom designed voice, so a provider that quietly alters it is a regression, not an optimisation.

**`memory.py`** — SQLite persistent memory at `~/Library/Application Support/Nova/nova_memory.db`. Facts are STRUCTURED: `(category, key)` is a UNIQUE canonical identity, so a correction supersedes instead of piling up contradictions. `find_fact(key)` searches across ALL categories (recall must see passively-learned facts, not just explicit ones); `_render_fact` renders facts as English — third person for prompt injection, second person for spoken readback. Degenerate model-extracted facts (value merely restates the key) are rejected on write.

**`fact_reconciler.py`** — The LLM slow path of passive learning. Runs once per conversation in wake mode, on the `nova-llm` thread, off the live path. Decides insert/update/delete as strict JSON, then validates: contradictory decisions collapse per key, and every value must be GROUNDED in what the user actually said (`_is_grounded`) so the small model can't fabricate detail it never heard.

**`rag.py`** — Local document retrieval via ChromaDB. Watches `~/Documents`, indexes supported file types (.txt, .md, .pdf, .py, .swift, .js, .json), stores embeddings locally. Zero network calls — all on-device. **Dependency and build directories are never indexed** (`EXCLUDED_DIRS`: node_modules, site-packages, .venv, DerivedData, .git, `__pycache__`…): measured, 14,699 of 15,059 indexable files under `~/Documents` were third-party source, so "what's on your mind" retrieved base64 from `draco_encoder.js` and glyph tables from a font and told the 3B they were Nicholas's personal documents. **Retrieval is relevance-gated** (`memory.rag_max_distance`, default 0.45) — Chroma returns the nearest k chunks no matter how far away they are, so without a threshold every single turn injected its three nearest neighbours. The threshold is a swept trade, not a clean split (the chatty and document-question distance distributions genuinely overlap): 0 of 40 conversational phrases retrieve anything, 6 of 8 real document questions still do. Narrowing the rules cannot clean what is already stored, so `INGEST_RULES_VERSION` rebuilds the collection when the rules change.

**`weather_engine.py`** — Weather via Open-Meteo (no API key, no account). Pure functions over HTTP with a hard timeout; NOTHING raises, so a failed lookup returns an error dict and Nova says it couldn't get the forecast instead of inventing one. **Every spoken number is templated, never phrased by the LLM** — a wrong temperature sounds exactly like a right one to someone listening, and the 3B has invented figures before. WMO codes map to spoken words ("mostly clear", "raining lightly"). See invariant 3 for the privacy decision.

**`weather_intents.py`** — NL dispatch for weather. Strict regex: a weather word must appear AND an action verb ("open", "play", "remind") rules it out, so "play some rain sounds" stays a music command. Handles now / tomorrow / the next few days, here or a named place ("weather in Boston" works with no location permission at all). Routing stage 4b, ahead of files and tools.

**`maps_engine.py`** — Location, nearby search, and travel time via Apple MapKit. **Runs every MapKit call in a short-lived SUBPROCESS**: MapKit/CoreLocation deliver results on the MAIN queue, which the `nova-llm` worker thread never services — verified, an MKLocalSearch started from a worker thread hangs forever. The subprocess owns its own main thread, prints one JSON line, and gives us a hard timeout. Location needs a real app identity (headless auth stays `notDetermined`), so it only works under Nova.app with `NSLocationWhenInUseUsageDescription` + the location entitlement. See invariant 3 for the privacy tradeoff. **Location is requested ON DEMAND** (`set_location_requester` → WS `need_location` → Swift `LocationProvider` → `POST /api/location`): the app used to volunteer a fix only at launch, in a POST that raced the backend's own startup and was dropped, so Nova had no coordinate for entire sessions and told him it couldn't find him. The Swift POST now retries until the backend is listening.

**`browser_control.py`** — Brave / Chrome / Safari control. Picks whichever supported browser is RUNNING (Brave preferred) rather than hardcoding one, and never probes with `tell application "X"` — that would launch it. Chromium and Safari differ in vocabulary (`active tab` vs `current tab`, `title` vs `name`, native `go back` vs JavaScript-only), so everything goes through one adapter. Back/forward/reload work natively on Chromium. Scroll needs "Allow JavaScript from Apple Events" — for Chromium that is the PROFILE pref `browser.allow_javascript_apple_events` in `~/Library/Application Support/BraveSoftware/Brave-Browser/Default/Preferences` (not a `defaults write` key, and Brave must be CLOSED when editing it or it is overwritten on quit). Enabled 2026-08-08 and verified. If it is ever off, scroll explains how to enable it rather than pretending.

**`file_manager.py`** — The filesystem engine. Search is three-pass and TCC-tolerant: Spotlight literal, Spotlight AND-tokens, then a direct walk of the user's common folders (Spotlight results are filtered per-process by macOS privacy, so `mdfind` can return nothing for a file plainly sitting on the Desktop). Permission errors are reported ONLY for top-level user folders — reporting every one made `~/Pictures/Photos Library.photoslibrary`, unreadable on every Mac, trigger a false "grant Nova folder access" on every miss. Nova's own project and data directories are PROTECTED: never surfaced as a candidate, and refused again at the filesystem call. Move and copy never overwrite. **There is deliberately no delete.**

**`screen_awareness.py`** — "What's on my screen?" answered from GROUND TRUTH, never a guess: the frontmost app (NSWorkspace), real window titles (`CGWindowListCopyWindowInfo`), and the text actually on screen (Apple Vision OCR, on-device). The LLM only characterizes the OCR text and never sees the window list — measured, the 3B kept attaching one app's window title to another, so the "you're in X, on Y" lead is DETERMINISTIC. Deliberately not a vision model: Moondream 0.5B produced filler and the 2B invented a keyboard and mouse that weren't on screen, and it pins huggingface-hub/tokenizers to versions `mlx_lm` can't import. **The permission trap:** `screencapture` does NOT fail without Screen Recording permission, it writes a valid PNG of just the wallpaper — so permission is always preflighted with `CGPreflightScreenCaptureAccess` and declined honestly. "What app am I in" needs no permission and works regardless. The screenshot is deleted in a `finally` on every path; contents are never logged, stored, or transmitted.

**`file_intents.py`** — Natural-language dispatch over `file_manager.py`. `detect_intent` is strict regex AND requires a distinctive search token to survive tokenizing, which is what keeps "open my photos" pointed at the Photos app instead of a file search. Extraction is fully deterministic (the 3B mangles "from Downloads to Documents"); the LLM is used only to summarize contents, at temperature 0 with an explicit no-arithmetic instruction after it invented a leftover-budget figure. Nothing on disk changes without an explicit spoken yes, and ambiguity is resolved by asking "which one?" BEFORE the confirmation rather than walking candidates one at a time. Runs on the `nova-llm` worker thread, so its `self.llm.generate` call is thread-safe.

**`tools.py`** — macOS system tools. App launch, volume control, battery status, web search, screenshot, system info. All deterministic — never touches the LLM. Easy to extend.

**`calendar_reminders.py`** — Apple Calendar & Reminders engine. Pure functions over EventKit (PyObjC), with an AppleScript fallback. Reads (today / this week / open reminders), writes (create event, create reminder), and edits (complete / delete / update by fuzzy title match). Zero side effects at import. Every function may raise RuntimeError — callers must catch. Ported from Jarvis, where it ran in production.

**`calendar_intents.py`** — Natural-language dispatch on top of `calendar_reminders.py`. `NovaCalendar.detect_intent` is strict regex only (no calendar word → None → normal chat); `handle` runs LLM JSON extraction (temp=0, heavily post-processed) and returns a single spoken string. Runs on the `nova-llm` worker thread, so its `self.llm.generate` calls are thread-safe. Reads are LLM-summarized with deterministic template fallbacks so a bad generation never invents a day/time.

**`ws_server.py`** — Bridge between Python and Swift. HTTP server on :5001, WebSocket server on :8766. Swift connects here to receive state changes, message content, and streaming tokens in real time.

**`requirements.txt`** — All Python dependencies. Install with `pip install -r requirements.txt`.

---

## Pipeline Routing Order (nova.py — first match wins)

1. System commands — sleep / wake / mute (always intercepted first)
2. Calendar follow-up offer — answers a "want to hear what's coming up?" with yes/no
2b. Pending file question — "which one?" / "move it to Documents?" answered before routing
3. Calendar / reminders intents — read / create / complete / delete / update (EventKit)
4. Screen awareness — what's on my screen / what app am I in
4b. Weather — now / tomorrow / next few days, here or a named place
5. File intents — find / read / open / move / copy / rename
6. Memory intents — remember / recall / update / forget
7. Fast-path intents — greetings / date / time / repeat last response
8. Tool intents — open app / volume / battery / search / screenshot
9. RAG context enrichment — query personal documents for context
10. LLM fallback — MLX streaming with sentence-chunked TTS overlap

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

## Testing

```bash
python nova_backend/tests/run_tests.py --quick   # env + routing + loop (fast, silent)
python nova_backend/tests/run_tests.py --all     # everything (plays audio, ~1 min)
```

| Suite | Proves | Fidelity |
|---|---|---|
| `verify_environment.py` | every engine imports, MLX generates | real engines |
| `test_routing_corpus.py` | every phrase that has broken Nova still routes right | real code, side effects stubbed |
| `test_conversation_loop.py` | when Nova keeps listening vs returns to wake | real `_main_loop`, scripted mic |
| `test_prompt_cache.py` | the prompt cache changed speed and NOTHING else | real engine + real model |
| `test_rag_relevance.py` | Nova only quotes a document when it has one | real rag.py + real on-disk index |
| `test_wake_capture.py` | the wake word gives him time; loops never reach the LLM | real `record_command`, scripted VAD |
| `test_weather.py` | weather answers are real numbers, and steal nothing else | real intents + canned payloads, live calls optional |
| `test_tts_chunking.py` | Nova starts speaking sooner and says the SAME words | real `_stream_response`, scripted LLM |
| `smoke_launch.py` | the REAL process starts and answers over HTTP | real process |
| `test_full_sweep.py` | every subsystem vs real system state | real code + real system |

**Rules these encode — every one came from a bug that reached Nicholas:**

1. **The harness may not construct what the product constructs.** Tests call
   `VoiceAssistant._init_state()`, never a hand-copied field list. A harness
   that builds its own object tests its own construction — that is how
   `_init_screen()` went missing from `__init__` while 130 checks stayed green
   and every single utterance crashed.
2. **Stub side effects, never verdicts.** `tools.match()` performs as it
   matches. Stubbing whole handlers made the corpus report false regressions,
   because the real `_resolve_app("downloads")` returns None. Replace only
   `subprocess.run` / `time.sleep`.
3. **Listener rules apply to every response, not per feature** (`listener.py`):
   no spoken filesystem paths, no markdown, no third person, no invented advice
   or numbers, sane length.
4. **Never trust a single run of anything model-dependent.** A 3B is a sampling
   process; the calendar editorializing passed one run and came back in real
   use. Sample repeatedly, or make the guarantee deterministic in code. The
   deterministic version has won every time.
5. **A green run is not "Nova works."** `run_tests.py` prints what it could NOT
   verify after every run — mic, speakers, and anything TCC-gated inside
   NovaOS.app.
6. **Regression-first.** A phrase that breaks Nova goes into
   `tests/adversarial_phrases.txt` BEFORE the fix; confirm it fails, then fix.

**Reporting to Nicholas** — every test report is four sections: what I tested /
how I tested it (always naming the fidelity above) / what succeeded and what
did not / what he needs to do. Plus an explicit "could not verify" line. Never
lead with a pass count.

**When to run what:** targeted suites for a targeted fix; `--quick` before any
merge; `--all` before a merge touching shared plumbing. Do NOT full-sweep every
small change — Nicholas asked for this explicitly and it is faster and sharper.

---

## Invariants — Never Break These

1. **Ports are 5001 (HTTP) and 8766 (WS).** These differ from Jarvis (3000/8765) intentionally so both apps can run simultaneously. Do not change them.
2. **LLM n_ctx stays at 4096.** Do not raise the context window without explicit instruction.
3. **No cloud dependencies — TWO narrow, documented exceptions.** Nova is local and private: no API keys, and the LLM/STT/TTS/memory NEVER leave the machine. The exceptions:
   - `maps_engine.py` — "how far is the nearest X" and travel times are Apple MapKit network lookups; getting a location fix sends an approximate position to Apple.
   - `weather_engine.py` — **added 2026-08-13, Nicholas's explicit choice** after being shown WeatherKit (needs a paid developer account), weather.gov (grid lookup) and browser scraping. Open-Meteo needs **no API key and no account**, so nothing ties a request to him beyond the IP any HTTP call carries. What leaves the machine is an approximate coordinate (rounded to ~100m) or a place name he said aloud.

   Both are used ONLY when he asks that kind of question, touch neither the LLM nor memory, are never stored, and degrade honestly ("I can't get your location", "I couldn't reach the weather service") rather than guessing. `weather.enabled: false` disables the second entirely. **Do not widen further without asking him — this is his privacy line, and he decides where it sits.**
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

1. Describing IMAGES (photos) — screen awareness reads text and windows, it does not
   describe pictures. `file_intents` still says "I can't see inside images yet".
   Needs a real vision model; see screen_awareness.py for why Moondream was rejected.
2. Proactive notifications — background monitor for upcoming events / due reminders
3. Barge-in over speakers — needs acoustic echo cancellation (see branch `feat/barge-in`;
   measured: the wake model's score collapses to ~0 once Nova's own voice reaches the mic,
   so this is NOT a tuning problem). Works on headphones today.
4. UI overhaul — deliberately late, after capabilities are concrete
5. iOS app — far back burner, separate on-device architecture

