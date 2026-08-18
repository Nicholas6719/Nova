# Nova — Claude Code Briefing

Read this entire file before touching anything.

---

## What Nova Is

Nova is a macOS voice assistant app built by Nicholas Coppola.
It is his personal AI — built from the ground up to know him, serve him, and grow smarter about him over time. The reference point is Jarvis from Iron Man: capable, personal, loyal, always improving.

Nova has two parts:
1. A **Python backend** (`nova_backend/`) — the brain. Handles all voice, AI, memory, and tools.
2. A **SwiftUI frontend** (`Nova/`) — the face. An orb and a panel; no chat, no transcript. Communicates with the Python backend.

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
    views.py                 voice navigation between UI screens
    panels.py                structured payloads for the screen + live progress
    sounds.py                four synthesised cues: boot, ready, wake, rest
    system_audio.py          the speaker mix, received from the app, for AEC
    market_engine.py         quotes, indices, analyst ratings (Yahoo + Finnhub)
    market_intents.py        market NL dispatch
    actuation.py             find-by-name, click, type (work mode only)
    actuation_intents.py     actuation NL dispatch + confirmations
    confidence.py            how sure Nova must be before acting
    echo_canceller.py        subtracts Nova's own voice from the mic
    ws_server.py             HTTP :5001 + WS :8766 bridge
    training/                wake-model training kit (Colab)
    tests/                   the test harness — see "Testing" below
    requirements.txt
    ARCHITECTURE.md
  CLAUDE.md              ← this file
```

---

## Python Backend — What Each File Does

**`nova.py`** — Main entry point. `VoiceAssistant` class owns the entire pipeline. Initializes all engines, runs the wake word loop, routes every user utterance through a 10-stage handler chain (first match wins).

**`config.json`** — All tuneable settings. Model name, ports, voice, user name, silence timing, RAG toggle. Edit this without touching Python code.

**`system_prompt.py`** — Nova's personality. Rebuilt fresh on every LLM call. Injects live memory context and RAG context. Defines Nova's identity, tone, and communication rules.

**`stt_engine.py`** — Microphone input. One persistent 16 kHz mono stream feeds both wake detection and command capture. Wake detection dispatches on `wake_word.engine`: `openwakeword` (neural, noise-robust — see `wake_openwakeword.py`) or `transcript` (legacy Whisper rolling-window scan), auto-falling back to transcript if the OWW model can't load. VAD-gated command recording via webrtcvad with adaptive silence cutoffs. Transcription via faster-whisper (local, offline) with **`beam_size=5` (beam search)**. This reversed an earlier call and the reversal is the lesson: an 8-clip test at ONE noise level made greedy look better, it shipped, and Nicholas was then misheard in real use ("went to 25% of 10%"). A proper sweep — 126 clips, 3 voices, 3 noise levels — says the opposite on both corpora: easy 0.5% WER / 71-of-72 exact for beam against greedy's 2.9% / 65-of-72; hard 20.4% / 61.1% against 23.6% / 53.2%, for 44ms. Also measured and rejected: `small.en` (2.6-3.8% WER on the easy corpus vs base.en's 0.5%, 3x slower), `distil-small.en` (5.4%), and float32/int8_float32 compute types. Beam search CAN loop on the `initial_prompt` — `patience=2` turned "what time is it" into forty repetitions — which is what `_is_transcription_loop` exists to catch, at any decode setting. Tune via `stt.beam_size`.

**`wake_openwakeword.py`** — Neural wake-word detection via OpenWakeWord. `OpenWakeWordDetector` accumulates the stream's 30 ms frames into 80 ms chunks, scores them with onnxruntime (fully local), and fires on a score threshold held for N consecutive windows; OWW's built-in Silero VAD gate suppresses scoring on non-speech so steady noise never triggers. The custom "Nova" model (`nova.onnx`) is trained via `training/` (Colab). Replaces transcript-based wake, which couldn't survive steady background noise.

**`llm_engine.py`** — Local AI inference via MLX (Apple Silicon optimized). Streaming generation — tokens arrive one by one via callback. Default model: Llama 3.2 3B. Upgrade path: Llama 3.1 8B (one line in config.json). **Carries a PROMPT CACHE between turns:** most of every prompt is identical to the last one (the identity block alone is ~930 tokens and never changes), and reprocessing it was 1.67s of the 1.77s wait before Nova spoke. The KV cache for the longest shared prefix is reused, so only new tokens are processed — time to first token 1.67s → 0.21s. The reuse length is always recomputed against the real token sequence, never assumed, and any exception drops the cache: a stale one would not raise, it would answer from a context the model never saw. Two measured traps are documented in the code — `generate_step` leaves a phantom sampled token in the cache (use a direct `model()` call), and mlx-lm skips special tokens when a prompt already starts with BOS (encode the same way or the model silently gets a second BOS). Rollback: `llm.prompt_cache` false.

**`tts_engine.py`** — Text to speech. Primary: Kokoro ONNX (local, no cloud). Fallback: macOS `say`. Background queue worker for gapless playback. TTS overlaps with LLM streaming — Nova speaks sentence 1 while still generating sentence 2. The FIRST chunk is additionally split at a clause break when the opening sentence is long (`tts.split_first_clause`), because nothing is audible until that chunk is both generated and synthesised — measured worst case 2.83s → 1.89s to first audio. It splits only at a comma/semicolon/colon, where the voice already pauses. **Kokoro stays on the default ONNX provider:** CoreML was measured slower (0.586s → 0.669s; only 129 of 2256 nodes partitioned) AND it changed the audio (correlation 0.998, different sample counts) — that is Nicholas's custom designed voice, so a provider that quietly alters it is a regression, not an optimisation.

**`memory.py`** — SQLite persistent memory at `~/Library/Application Support/Nova/nova_memory.db`. Facts are STRUCTURED: `(category, key)` is a UNIQUE canonical identity, so a correction supersedes instead of piling up contradictions. `find_fact(key)` searches across ALL categories (recall must see passively-learned facts, not just explicit ones); `_render_fact` renders facts as English — third person for prompt injection, second person for spoken readback. Degenerate model-extracted facts (value merely restates the key) are rejected on write.

**`fact_reconciler.py`** — The LLM slow path of passive learning. Runs once per conversation in wake mode, on the `nova-llm` thread, off the live path. Decides insert/update/delete as strict JSON, then validates: contradictory decisions collapse per key, and every value must be GROUNDED in what the user actually said (`_is_grounded`) so the small model can't fabricate detail it never heard.

**`rag.py`** — Local document retrieval via ChromaDB. Watches `~/Documents`, indexes supported file types (.txt, .md, .pdf, .py, .swift, .js, .json), stores embeddings locally. Zero network calls — all on-device. **Dependency and build directories are never indexed** (`EXCLUDED_DIRS`: node_modules, site-packages, .venv, DerivedData, .git, `__pycache__`…): measured, 14,699 of 15,059 indexable files under `~/Documents` were third-party source, so "what's on your mind" retrieved base64 from `draco_encoder.js` and glyph tables from a font and told the 3B they were Nicholas's personal documents. **Retrieval is relevance-gated** (`memory.rag_max_distance`, default 0.45) — Chroma returns the nearest k chunks no matter how far away they are, so without a threshold every single turn injected its three nearest neighbours. The threshold is a swept trade, not a clean split (the chatty and document-question distance distributions genuinely overlap): 0 of 40 conversational phrases retrieve anything, 6 of 8 real document questions still do. Narrowing the rules cannot clean what is already stored, so `INGEST_RULES_VERSION` rebuilds the collection when the rules change.

**`weather_engine.py`** — Weather via Open-Meteo (no API key, no account). Pure functions over HTTP with a hard timeout; NOTHING raises, so a failed lookup returns an error dict and Nova says it couldn't get the forecast instead of inventing one. **Every spoken number is templated, never phrased by the LLM** — a wrong temperature sounds exactly like a right one to someone listening, and the 3B has invented figures before. WMO codes map to spoken words ("mostly clear", "raining lightly"). See invariant 3 for the privacy decision.

**`weather_intents.py`** — NL dispatch for weather. Strict regex: a weather word must appear AND an action verb ("open", "play", "remind") rules it out, so "play some rain sounds" stays a music command. **The action-verb list must cover every file verb**, because weather runs at stage 4b and files at stage 5: with only "move" and "rename" on it, "put my weather report in Documents" was answered with the current temperature while the file sat where it was. Handles now / tomorrow / the next few days, here or a named place ("weather in Boston" works with no location permission at all). Routing stage 4b, ahead of files and tools.

**`maps_engine.py`** — Location, nearby search, and travel time via Apple MapKit. **Runs every MapKit call in a short-lived SUBPROCESS**: MapKit/CoreLocation deliver results on the MAIN queue, which the `nova-llm` worker thread never services — verified, an MKLocalSearch started from a worker thread hangs forever. The subprocess owns its own main thread, prints one JSON line, and gives us a hard timeout. Location needs a real app identity (headless auth stays `notDetermined`), so it only works under Nova.app with `NSLocationWhenInUseUsageDescription` + the location entitlement. See invariant 3 for the privacy tradeoff. **Location is requested ON DEMAND** (`set_location_requester` → WS `need_location` → Swift `LocationProvider` → `POST /api/location`): the app used to volunteer a fix only at launch, in a POST that raced the backend's own startup and was dropped, so Nova had no coordinate for entire sessions and told him it couldn't find him. The Swift POST now retries until the backend is listening.

**`browser_control.py`** — Brave / Chrome / Safari control. Picks whichever supported browser is RUNNING (Brave preferred) rather than hardcoding one, and never probes with `tell application "X"` — that would launch it. Chromium and Safari differ in vocabulary (`active tab` vs `current tab`, `title` vs `name`, native `go back` vs JavaScript-only), so everything goes through one adapter. Back/forward/reload work natively on Chromium. Scroll needs "Allow JavaScript from Apple Events" — for Chromium that is the PROFILE pref `browser.allow_javascript_apple_events` in `~/Library/Application Support/BraveSoftware/Brave-Browser/Default/Preferences` (not a `defaults write` key, and Brave must be CLOSED when editing it or it is overwritten on quit). Enabled 2026-08-08 and verified. If it is ever off, scroll explains how to enable it rather than pretending. `run_js` / `read_results` read the search results off the page Nova just opened — titles and hostnames only, structural selectors only. **One escaping trap, already paid for:** a `[data-testid='...']` selector's inner quotes collapsed on the way through AppleScript, and it did NOT fail loudly — querySelectorAll threw, the function returned nothing, and the caller reported "still loading" until it timed out. The extractor now avoids nested quoting entirely and reports a thrown selector instead of swallowing it.

**`file_manager.py`** — The filesystem engine. Search is three-pass and TCC-tolerant: Spotlight literal, Spotlight AND-tokens, then a direct walk of the user's common folders (Spotlight results are filtered per-process by macOS privacy, so `mdfind` can return nothing for a file plainly sitting on the Desktop). Permission errors are reported ONLY for top-level user folders — reporting every one made `~/Pictures/Photos Library.photoslibrary`, unreadable on every Mac, trigger a false "grant Nova folder access" on every miss. Nova's own project and data directories are PROTECTED: never surfaced as a candidate, and refused again at the filesystem call. Move and copy never overwrite. **There is deliberately no delete.**

**`screen_awareness.py`** — "What's on my screen?" answered from GROUND TRUTH, never a guess: the frontmost app (NSWorkspace), real window titles (`CGWindowListCopyWindowInfo`), and the text actually on screen (Apple Vision OCR, on-device). The LLM only characterizes the OCR text and never sees the window list — measured, the 3B kept attaching one app's window title to another, so the "you're in X, on Y" lead is DETERMINISTIC. Deliberately not a vision model: Moondream 0.5B produced filler and the 2B invented a keyboard and mouse that weren't on screen, and it pins huggingface-hub/tokenizers to versions `mlx_lm` can't import. **The permission trap:** `screencapture` does NOT fail without Screen Recording permission, it writes a valid PNG of just the wallpaper — so permission is always preflighted with `CGPreflightScreenCaptureAccess` and declined honestly. "What app am I in" needs no permission and works regardless. The screenshot is deleted in a `finally` on every path; contents are never logged, stored, or transmitted.

**`file_intents.py`** — Natural-language dispatch over `file_manager.py`. `detect_intent` is strict regex AND requires a distinctive search token to survive tokenizing, which is what keeps "open my photos" pointed at the Photos app instead of a file search. **A destination may be introduced by "in", not just to/into/onto** — "put the invoice in Documents" is simply how he says it and it used to fall through to the model — but only when the folder RESOLVES. That asymmetry is the whole safety argument: widening the preposition alone would have claimed "send the report in an email" and answered it with "name the folder", which is worse than the model's answer, not better. The move and copy VERBS are also subtracted from the search query, using the regexes that define them rather than a second hand-written list: `file_manager`'s stopwords know "move" and "put" but not "stick" or "drop", so "stick that report in the desktop folder" searched for a file matching stick AND report.

**The rule that makes "in" safe is position.** WHEN SOMEBODY NAMES A PLACE TO PUT A THING, THEY STOP TALKING; when "in the X" describes the thing, the sentence carries on. So an "in" destination must END the utterance, give or take a politeness, and the captured phrase must BE a folder rather than merely start with one. Without both halves, "put a caption in the photos from the party", "put a warning in the docs for the new API" and "put Sarah in the picture too" were all claimed as file moves. Matched against the ORIGINAL text, never the from-stripped copy — deleting "from the party" leaves "put a caption in the photos", which looks end-anchored and is not. `detect_intent` also declines anything opening with **remember/forget** (that regression silently dropped the fact AND returned a file-search failure) and the unambiguous **web-search** forms, while leaving "search for my resume" a file find. Extraction is fully deterministic (the 3B mangles "from Downloads to Documents"); the LLM is used only to summarize contents, at temperature 0 with an explicit no-arithmetic instruction after it invented a leftover-budget figure. Nothing on disk changes without an explicit spoken yes, and ambiguity is resolved by asking "which one?" BEFORE the confirmation rather than walking candidates one at a time. Runs on the `nova-llm` worker thread, so its `self.llm.generate` call is thread-safe.

**`spotify_search.py`** — **entirely optional.** "Play music", play/pause/skip/resume and "what's playing" all work with no credentials and always have — Spotify resumes whatever context it was last in, which is usually the playlist he was listening to. This module exists only to turn a NAME into a Spotify URI, and without credentials Nova opens that search in the desktop app instead of asking him for an API key. Spotify's desktop app exposes exactly six AppleScript commands (`next track`, `previous track`, `playpause`, `pause`, `play`, `play track`) and **none of them is `search`** — `play track` needs a URI. So the app can play anything, it just can't say what a name refers to. One HTTPS lookup fills that gap; **playback stays local through the desktop app**. Client-credentials only: no user login, no OAuth, no Premium — and therefore **no access to his own library or personal/algorithmic playlists** (Discover Weekly and friends need a user token), which Nova says plainly rather than reporting "not found". Credentials live at `NOVA_DATA_DIR/spotify_credentials.json`, **never in config.json, which is committed**. Nothing raises. See invariant 3.

**`tools.py`** — macOS system tools. **Command words that are also ordinary English must be SHAPED like commands.** NINE matchers tested a bare substring and each one fired on innocent sentences — found by routing 65 ordinary sentences through the real pipeline, of which 19 were claimed. `space` answered "I need more space in my closet" with disk usage; `memory` answered "in memory of my grandfather" with RAM; `volume` answered "the volume of work is insane"; `brighter` tried to change the display for "brighter days ahead"; `shuffle`/`repeat`/`loop` changed playback for "repeat after me" and "shuffle the deck of cards"; a bare `refresh` reloaded his browser for "refresh my memory on that"; `play` sent every idiom to Spotify ("play it cool" searched for "it cool"); and an app name that is really a phrase produced "I couldn't find an app called your feelings". The four loudest: `\bresume\b` meant "put my resume in Downloads" launched Spotify and played AC/DC; `screenshot` anywhere meant "shove the old screenshots in the archive folder" took a fresh capture, wrote a PNG to the Desktop and reported success; `\b(next|skip)\b` meant "next month" and "what's next" skipped his track; and `\bsearch\b` anywhere grabbed the word inside "job search", so "I'm resuming my job search next month" ACTIVATED the browser and searched for "next month". All four now require the utterance to be about them — heading the sentence, or naming the object. His resume is the worked example in this very file, which is how long the first one sat there. **PAUSES the music while Nova listens** (`music.pause_while_listening`, default true) — ducking to 20% still left him mis-heard at his desk, because 20% of loud is not quiet, and paused there is nothing of it in the microphone at all. Music HE paused is never started by Nova finishing a conversation. Falls back to **ducking, OFF-THREAD** (`music.duck_while_listening`, level `music.duck_level`). The AppleScript round trip is 534ms and it used to sit between the wake word firing and the microphone opening — against a head start of about 700ms — so ducking ate the window and clipped the first word of his command. It runs on a thread now and the recording starts on time.: music out of the speakers goes back into the microphone, and measured with music mixed into real command audio, Whisper went from 0% word error in silence to 9.7% at full volume and **709% when loud — transcribing the music itself**. This is the same physics as barge-in, but tractable because Nova owns the volume knob and does not have to cancel anything. The PLAYER's volume is ducked, never the system's, or Nova's own voice would go down with it. Ducked once on wake and restored on the single path back to wake mode, so it never pumps between turns and never strands his music quiet. App launch, volume control, battery status, web search, screenshot, system info. All deterministic — never touches the LLM. **CPU / memory / battery have structured twins** (`cpu_reading`, `memory_reading`, `battery_reading`, `status_row`) because those numbers now go two places — Nova says them and home shows them — and a spoken sentence is a terrible thing to parse back into a percentage. `status_row` is cached and refreshed off-thread: `top -l 1` alone is 363ms and home redraws constantly, while a question still gets a fresh reading. **Web search reads the page now.** It used to build a Google URL, `open` it, and say "Searching for X" — Nova never looked. It goes through `browser_control` (the browser already RUNNING, not the default one) and reads the results back, which costs no extra network call because the browser already fetched them. Only titles and hostnames are extracted and **none of it reaches the LLM**: a web page is untrusted text and this same Nova can type, click and move files, so the spoken answer is templated in Python like every other fact she states.

**`calendar_reminders.py`** — Apple Calendar & Reminders engine. Pure functions over EventKit (PyObjC), with an AppleScript fallback. Reads (today / this week / open reminders), writes (create event, create reminder), and edits (complete / delete / update by fuzzy title match). Zero side effects at import. Every function may raise RuntimeError — callers must catch. Ported from Jarvis, where it ran in production.

**`calendar_intents.py`** — Natural-language dispatch on top of `calendar_reminders.py`. `NovaCalendar.detect_intent` is strict regex only (no calendar word → None → normal chat); **its file-word guard is load-bearing and was incomplete** — there is a bare "rename X to Y" rule carrying no calendar word of its own, and "resume" was missing from the guard, so "rename my resume to resume final draft" reached the reminder editor. Fuzzy matching drops words of two characters or fewer, so "my resume" reduced to "resume" and ANY open reminder containing that word was silently retitled, with no confirmation, while the file was never renamed; `handle` runs LLM JSON extraction (temp=0, heavily post-processed) and returns a single spoken string. Runs on the `nova-llm` worker thread, so its `self.llm.generate` calls are thread-safe. Reads are LLM-summarized with deterministic template fallbacks so a bad generation never invents a day/time.

**`views.py`** — Voice navigation between UI destinations ("go home", "show me the menu", "go to finance"), and **the home surface itself**. Nova's UI has no sidebar and nothing to click, so **speech is the navigation**. Detection is strict regex anchored at BOTH ends and the destination must be a name in the registry, because navigation words are ordinary English: "go home" navigates, "I go home every friday" is conversation. Routing stage 2c — ahead of calendar, weather, files and tools, every one of which would otherwise claim these. A view whose panel does not exist yet **says so out loud and shows nothing**. The menu is generated from the registry so it cannot drift. Never touches the LLM.

Home is a GRID of six named slots (L1-L3, R1-R3) plus a status row, and its whole character is what it hides: the greeting and the status row leave on the WAKE WORD (not on her reply — that is a beat too late, and left "Good morning, NICHOLAS" under the orb while she was already listening), the status row comes back when he asks about CPU and is HELD until Nova stops talking before its ten seconds start (a timer armed in the handler is half spent by the time he has heard the number), Now Playing exists only while something is playing, and "clear home" strips it to the orb. **Slots are a stamp on each block, not a new payload shape**, so a client that knows nothing about them still renders the list. `move the now playing to the bottom right` SWAPS with whatever is there — evicting the occupant would silently lose him a card he never mentioned — and the layout is saved to `NOVA_DATA_DIR/home_layout.json`. Both ends of the move regex must be known names, which is the only reason "move my resume to Downloads" still reaches the file handler five stages later.

**Every tile refreshes BEHIND the screen** (`_Tile`). Measured: the music check is 352ms of AppleScript and the calendar read 218ms of EventKit, and home is re-rendered at startup, after every answer when the panel dismisses, and on every tick — 742ms → 0.34ms. When a tile's value CHANGES it calls back, which is what makes home live: a song he started himself appears on its own. `refresh_home` sends nothing unless the payload actually differs, so a quiet tick costs a dict comparison, and it asks the WS SERVER whether home is on screen rather than trusting `views.current` — that is a second copy of the same fact, nova.py writes it too, and when the two drift home goes stale for the life of the process with nothing to say why. The music tile is checked every 2s because he can HEAR it change; measured, a song appears in ~4.5s whoever started it, and clears in 3.3s. **Player detection asks the KERNEL (`pgrep`), never `NSWorkspace.runningApplications()`** — that list is maintained by run-loop notifications and the backend has no run loop, so it only ever knows what was running at startup. Measured: with Spotify launched by Nova mid-session it said False while AppleScript in the same process read back "War Pigs, playing", so Now Playing never appeared for any music Nova herself started. Same family as the MapKit trap; it passed every test and failed only in use. **Every AppleScript is bounded** for the same reason a tile must never wedge: an unbounded call leaves `_busy` set and that card silently stops updating for the life of the process. **The trap this cost a debugging session:** EventKit's reminder fetch delivers its completion block on its own dispatch queue, and if the interpreter finalises mid-fetch the block takes a GIL that is being torn down — Foundation kills the process outright, EXC_BREAKPOINT, SIGKILL, no traceback. `_drain` at exit waits (bounded) for in-flight refreshes — and then SETTLES, because joining the threads is not sufficient on its own: EventKit releases the completion block on its own queue a moment after the fetch returns, and that release runs PyObjC's dispose helper, which takes the GIL. Measured 4 crashes in 20 runs of a suite that builds a NovaViews and exits within half a second; 0 in 30 after.

**`confidence.py`** — How sure Nova has to be before she acts. The UI has no transcript any more, so a mishear is a SILENT failure; Whisper's `avg_logprob` gates action instead. It is not a calibrated probability and this does not pretend otherwise — it is a score that separates right from wrong well enough to threshold, **measured over 270 clips** (18 commands x 3 voices x 5 noise levels, -2 to 40 dB SNR): usable transcripts median -0.137, unusable -0.768; a -0.50 floor catches 75% of mishears while asking again unnecessarily only 5.3% of the time. **The bar scales with consequence** — chat acts almost always, file moves and calendar writes sit at the knee, and anything outbound (send, delete, buy, call) is READ BACK in his own words regardless of confidence, because Whisper can be confidently wrong. A verb inside a reminder is content, not a command: "remind me to call mom" is a calendar write, not a call, or Nova would read back every reminder he ever set. A missing signal means "behave exactly as before", never a crash. Rollback: `stt.confidence_gate` false.

**`actuation.py`** — Nova types and clicks, **on text she can actually see**. This is the part RileyJarvis fakes: its `computer_click` takes raw x,y from the model, and its screenshot tool returns a file path that is stripped before the model ever sees it, so the model guesses coordinates from a three-line summary. Nova doesn't have to — `screen_awareness.ocr()` already runs Vision, and Vision returns a `boundingBox` with every observation that the existing code used for sort order and threw away. Two ways to find a target, fast first: **Accessibility** (`AXUIElement`, structured, ~10-50ms, needs Accessibility but NOT Screen Recording, so the fast path is also the less invasive one), then **window-only screenshot + OCR** (~300-800ms) for Electron and canvases. **The coordinate flip is the dangerous bit** — Vision is normalized bottom-left, the screen is top-left, and getting it wrong doesn't fail loudly, it clicks the mirror image. Verified against a live window: Nova's own state word came back at x=0.50, y=0.80, exactly where it renders.

**`actuation_intents.py`** — The gates, in the order they matter. **Work mode only, checked at DETECTION** rather than only in `handle` — gating later would have let actuation SHADOW handlers that already work ("scroll down" drives the browser through `tools` outside work mode, and claiming it here replaced that with "say work with me first"). **Accessibility preflighted**, so a missing grant is named rather than surfacing as a bug. **A target must be found BY NAME**; not finding it is the answer, never a guessed coordinate. Typing is free — what is typed is visible before it goes anywhere — and **sending is the gated step**. **Return is the sharpest edge**: in Messages and Mail, Return IS send, so it is confirmed there and free in a text editor. Anything that is not a clear yes cancels. Never touches the LLM.

**`panels.py`** — The structured half of Nova's answers. Every deterministic handler already computed real structure and then flattened it to one spoken line; this is how that reaches the screen. **The spoken and shown halves are different channels with different rules**: the voice stays under the listener rules (no markdown, no lists, invariant 10), the panel can be as rich as it likes because it is read with eyes. The LLM never touches a panel. The block vocabulary is deliberately tiny (`stat` / `rows` / `items` / `text` / `note` / `metrics` / `steps`) so a new panel is a backend change and never needs a new SwiftUI view. `metrics` is the bottom-left instrumentation ROW — his call, and the right one: a box would give CPU and battery the same weight as his calendar. `Progress` is a step list a handler updates AS IT WORKS, declaring the whole sequence up front so what is still to come is visible from the first frame; a failed step stays on screen, because red says more than blank.

**`sounds.py`** — Nova's voice before she has words. Four cues, SYNTHESISED at import from oscillators — no audio files in the repo, no licence questions, and the character lives in code where it can be argued with. One idea at two dispositions: a rising fifth means arriving (`wake`), the same fifth falling means withdrawing (`rest`), so he can tell them apart across a room without being told which is which. `boot` runs while the engines come up and `ready` lands when she is online. **The wake cue BLOCKS and is 170ms**, both for the same reason: it plays in the gap between the wake word firing and the microphone opening, so it must be over before recording starts (or the VAD reads Nova's own chime as speech onset) and every millisecond is taken from the ~700ms he has to start talking. Nothing raises — no audio device costs the cue and nothing else.

**`system_audio.py`** + **`Nova/SwiftBackend/SystemAudioTap.swift`** — The speaker mix, so the canceller has a reference for sounds Nova did NOT synthesise. ScreenCaptureKit captures the system mix in the app (it needs a run loop and the Screen Recording grant, neither of which the headless backend has), converts to 16 kHz mono, and sends it over **UDP :8767** — a third port, deliberately separate from 5001/8766 so audio cannot interleave with control, and unreliable on purpose because a late reference frame is worthless while a stalled socket would wedge capture. With the tap live, Kokoro stops feeding a second copy of Nova's voice (already in the mix) and music-pausing turns itself off. **The crash that cost most of the build:** the first version built an `AudioStreamBasicDescription` and passed `[asbd].withUnsafeBufferPointer { $0.baseAddress! }` to `AVAudioFormat` — a pointer into a temporary that is dangling by the time it is read. The garbage format made `AVAudioPCMBuffer.init` raise an Objective-C exception, which Swift cannot catch, so the APP ABORTED rather than degrading. `AVAudioFormat(cmAudioFormatDescription:)` takes the CMFormatDescription directly and has no pointer to outlive. **Measured, real speaker music through the real mic: 2-8 dB removed at `echo_stream_delay_ms` 35, roughly 8 dB average and 14.8 dB peak at 120.** That is working but WEAK — Nova's own voice cancels at 27-36 dB — and the gap is delay alignment, because 35ms was tuned for Kokoro's path. Barge-in is not trustworthy until that closes. `audio.system_tap.enabled`.

**`echo_canceller.py`** — Subtracts Nova's own voice from the microphone, so she can be interrupted while speaking. Nova is unusually well placed for this: the hard part of software AEC is a clean reference of what the speaker is playing, and **Nova synthesises her own speech**, so Kokoro hands over that signal before it is ever played. Measured on real Kokoro output against real speech: **27-36 dB of her voice removed, his attenuated only 1.5-3.9 dB, and 0.2 dB when she is silent.** Two traps it exists to handle: the APM works in 10 ms frames (160 samples) while the mic delivers 30 ms (480) and the library does NOT reject a wrong size — it degrades silently; and the reference must be consumed in step with incoming mic frames, because playback runs ahead of capture. **Default OFF** (`stt.echo_cancellation`): enabling it also means keeping mic frames during playback, and Nova listens far more often than she speaks. Nothing raises — a failure returns the raw microphone. `feat/barge-in` is the other half.

**`ws_server.py`** — Bridge between Python and Swift. HTTP server on :5001, WebSocket server on :8766. Swift connects here to receive state changes, message content, streaming tokens, and **which screen to show** in real time. The current view and its data are held like the current state, so a client that connects — or reconnects after an app relaunch — is told immediately instead of coming up blank.

**`requirements.txt`** — All Python dependencies. Install with `pip install -r requirements.txt`.

---

## Pipeline Routing Order (nova.py — first match wins)

1. System commands — sleep / wake / mute (always intercepted first)
2. Calendar follow-up offer — answers a "want to hear what's coming up?" with yes/no
2b. Pending file question — "which one?" / "move it to Documents?" answered before routing
2c. UI navigation — go home / the menu / a named screen / work mode, plus the
    home surface itself: clear home, restore home, move a card to a slot
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
| GET    | /api/view      | Current screen + its panel data|
| POST   | /api/message   | Send text from UI to pipeline  |
| POST   | /api/mute      | Toggle mute from UI            |

### WebSocket (:8766)
```json
{"type": "state",   "state": "idle|listening|thinking|speaking"}
{"type": "message", "role": "user|assistant", "content": "..."}
{"type": "token",   "token": "..."}
{"type": "view",    "view": "home", "data": {...}}
```

---

## Swift Side — Current State

**There is no chat UI.** No transcript, no message bubbles, no mic button. The
orb is the entire interface and the word under it is the only thing that says
what Nova is doing.

- **`Nova/NovaApp.swift`** — app entry; starts `BackendManager`, owns
  `LocationProvider`. `.windowStyle(.hiddenTitleBar)` is load-bearing: AppKit is
  the wrong layer for it (see WindowChrome).
- **`Nova/Shell/OrbView.swift`** — **Reactor** at full size, **Array** in the
  puck: the same design at two densities, so Nova sheds detail as she shrinks
  rather than becoming a different shape. The core is a TORUS — a ring of light
  white-hot at its crest, built from gradient stops peaking at the ring radius,
  which is the thing a stroked circle cannot do. Canvas + TimelineView.
- **`Nova/Shell/NovaState.swift`** — the seven states. `idle` reads **Say "Nova"**, not "Idle": that state IS the wake-word wait, Nova is running the detector continuously, and the word he needs at that moment is the one that gets her attention. Cyan throughout except
  `working` (amber, real progress sweep) and `unsure` (coral, arcs that will not
  lock), because those two must be unmistakable at a glance.
- **`Nova/Shell/Panel.swift`** — renders `panels.py`'s vocabulary generically,
  decoding defensively so a malformed payload costs the panel and never the
  answer. Slot and card live on the block WRAPPER rather than inside every
  content case: they are placement, not content. `StatusRow` is the bottom
  line of instrumentation; `StepList` is Nova showing her work.
- **`Nova/Shell/ShellView.swift`** / **`ShellViewModel.swift`** — the window.
  Replaced `ChatViewModel` (1,022 lines of the old on-device pipeline) outright
  rather than untangling it. Cmd-T types to Nova and she answers in TEXT, not
  aloud. Three layouts: **home** (orb dead centre, cards in slots either side,
  status row along the bottom), **answer** (orb steps aside), **work** (orb
  shrinks and slides left, the room it gives up becomes the live step list).
  An icon rail down the left and a status strip across the top frame all
  three. Neither is a control — speech is the navigation — so the rail says
  what exists and where you are, and the strip says she is awake and what day
  it is. Without the strip the date in every payload had nowhere to render.
  Home STACKS rather than flanking — an HStack means an empty column shoves the
  orb off centre, and home spends most of its life with an uneven number of
  cards. `matchedGeometryEffect` keyed on the card id is what makes a moved
  card FLY to its new slot instead of blinking between two places.
- **`Nova/Shell/WindowChrome.swift`** — frameless, drags from anywhere, and the
  130pt puck that floats above everything including fullscreen apps and other
  Spaces. **Measured traps:** `titlebarAppearsTransparent` +
  `fullSizeContentView` + a black background + hiding `NSTitlebarContainerView`
  all applied cleanly and STILL left a 32pt #1D1F20 strip, because the scene
  paints that area itself; macOS restores the previous frame AFTER `onAppear`,
  so Nova came back puck-sized after being quit while parked; and changing the
  style mask rebuilds the frame view, which brought the traffic lights back.
- **`Nova/SwiftBackend/`** — `BackendManager` (locates Python, supervises the
  child process, runs it from the REPO path so Python-only changes need only an
  app relaunch), `NovaAPIClient` (HTTP + WS), `LocationProvider`.

**The Xcode target takes its files from disk** (`PBXFileSystemSynchronizedRootGroup`).
Anything under `Nova/` is in the build because it exists — there is no list to
forget to update.

Deleted in the overhaul: `ContentView.swift`, `Features/Chat/ChatViewModel.swift`,
`Voice/SpeechManager.swift`, `Voice/SpeechRecognizer.swift`.

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
| `test_music.py` | play-by-name works and shadows no transport command | real router, AppleScript + network stubbed |
| `test_views.py` | voice navigation reaches the right screen and fakes nothing | real views + real WS server, transport captured |
| `test_home.py` | home shows what belongs and hides what doesn't | real views/panels/WS, music + market stubbed |
| `test_sounds.py` | the cues are short, clean, and never load-bearing | real synthesis, device failure simulated |
| `test_confidence.py` | Nova acts only when sure enough for what it costs | real confidence module + real Whisper |
| `test_market.py` | market answers are real numbers, and never advice | real engine, canned payloads (NOVA_TEST_LIVE=1 for live) |
| `test_actuation.py` | Nova types and clicks, and refuses to do either carelessly | real gates, OS calls stubbed + one live coordinate check |
| `test_echo_cancellation.py` | Nova's own voice is removed from the mic, his is not | real Kokoro + real speech, simulated speaker path |
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
7. **The sweep leaves his Mac as it found it, and never acts on his windows.**
   It records what was open first and closes only what it opened — Finder
   windows, TextEdit, its own browser window by id. And because
   `browser_control` always drives `window 1`, the live browser tests REFUSE to
   run unless the front window is provably the one the test created. Without
   that guard, "close this tab" and "go back" ran against whatever Nicholas had
   open.

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
   - `spotify_search.py` — **added 2026-08-13, his explicit choice** after being shown the alternatives (hand off to Spotify's own search with no key; automate Spotify's UI). Name→URI lookup only; the words he asked for are what leaves the machine. Playback is local.
   - `market_engine.py` — **added 2026-08-17, his explicit choice.** Market data only; he ruled out any access to his real accounts. Yahoo needs no key at all; Finnhub's free tier needs one, stored at `NOVA_DATA_DIR/finnhub_credentials.json` and never in the committed config. What leaves the machine is a ticker or a company name he said out loud. **The privacy wrinkle he was told about and accepted: weather leaks an approximate location, but a stock query leaks what he is invested in.** `finance.enabled: false` disables it.
   - `weather_engine.py` — **added 2026-08-13, Nicholas's explicit choice** after being shown WeatherKit (needs a paid developer account), weather.gov (grid lookup) and browser scraping. Open-Meteo needs **no API key and no account**, so nothing ties a request to him beyond the IP any HTTP call carries. What leaves the machine is an approximate coordinate (rounded to ~100m) or a place name he said aloud.

   Both are used ONLY when he asks that kind of question, touch neither the LLM nor memory, are never stored, and degrade honestly ("I can't get your location", "I couldn't reach the weather service") rather than guessing. `weather.enabled: false` disables the second entirely. **Do not widen further without asking him — this is his privacy line, and he decides where it sits.**
4. **Memory summarization runs on a daemon thread.** It must never block the main pipeline or the voice loop.
5. **RAG loads in the background.** Queries must fail gracefully (return empty string) if the index is not yet ready.
6. **TTS and LLM overlap.** The sentence-chunked streaming pipeline in `nova.py._stream_response` must not be changed to a blocking pattern.
7. **Kokoro model files** (`kokoro-v1.0.onnx`, `voices-v1.0.bin`) live in `nova_backend/` — same files as Jarvis. TTS falls back to macOS `say` if they are missing.
8. **NOVA_DATA_DIR** must be set by the Swift BackendManager so data survives app updates.
9. **The wake phrase is "Nova", NOT "Hey Nova".** Measured over the speakers, 3 trials each: `"Nova, <command>"` fires 6/6 at 0.86-0.99, `"Hey Nova"` fires **0/6 at 0.001-0.211** — the "Hey" prefix destroys the score. With `engine: openwakeword` the `keywords` list is **ignored entirely** (`record_wake` dispatches to `_record_wake_oww` without passing it); it survives only for the transcript fallback. `trigger_level` stays at 2: at 1, "November is cold" false-fires.
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
3. Barge-in over speakers — the canceller now EXISTS (`echo_canceller.py`, measured
   27-36 dB of Nova's voice removed) and `feat/barge-in` is the other half, but
   `stt.echo_cancellation` ships OFF: turning it on also keeps mic frames during
   playback, and that needs his ears on real speakers before it becomes the default.
4. UI overhaul — the HOME tab is done: slots, voice-moved cards, the status
   row, clear/restore, self-refreshing tiles, and the live work surface.
   Remaining: the other tabs (finance, calendar, files, memory, conversations
   all still render generically), and a health screen that needs the phone.
5. iOS app — far back burner, separate on-device architecture

