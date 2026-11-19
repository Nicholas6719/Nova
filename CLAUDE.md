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
    brain.py                 THE BRAIN: learned intent router (see BRAIN.md)
    skills.py                every skill the brain can run without the LLM
    BRAIN.md                 the brain: design, stages, numbers, what's next
    mathskill.py, units.py   arithmetic, percentages, tips, conversions in CODE
    honesty.py               no spoken promise / false "done" from the LLM
    delivery.py              the single door for anything unprompted
    standing.py              "tell me when ..." watches, kept across restarts
    shape.py                 is a sentence SHAPED like a request? (shared)
    turnstats.py             per-turn record: path, skill, timings; "why?"
    world_facts.py           timeless facts: looked up, verified, served (Stage 3)
    wiki.py                  Wikipedia, the one source facts come from
    night_school.py          overnight: fact audit, curiosity, paraphrase learning
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

**`system_audio.py`** + **`Nova/SwiftBackend/SystemAudioTap.swift`** — The speaker mix, so the canceller has a reference for sounds Nova did NOT synthesise. ScreenCaptureKit captures the system mix in the app (it needs a run loop and the Screen Recording grant, neither of which the headless backend has), converts to 16 kHz mono, and sends it over **UDP :8767** — a third port, deliberately separate from 5001/8766 so audio cannot interleave with control, and unreliable on purpose because a late reference frame is worthless while a stalled socket would wedge capture. With the tap live, Kokoro stops feeding a second copy of Nova's voice (already in the mix) and music-pausing turns itself off. **The crash that cost most of the build:** the first version built an `AudioStreamBasicDescription` and passed `[asbd].withUnsafeBufferPointer { $0.baseAddress! }` to `AVAudioFormat` — a pointer into a temporary that is dangling by the time it is read. The garbage format made `AVAudioPCMBuffer.init` raise an Objective-C exception, which Swift cannot catch, so the APP ABORTED rather than degrading. `AVAudioFormat(cmAudioFormatDescription:)` takes the CMFormatDescription directly and has no pointer to outlive. **The delay is MEASURED now, not configured.** `stream_delay_ms` was a constant tuned for Kokoro's path (handoff microseconds before playback); the tap's path is longer, so the far buffer became a RING read at a measured offset instead of a FIFO consumed in step. The offset comes from a normalised cross-correlation of 1ms |amplitude| ENVELOPES over 0-400ms — envelopes because a speaker and a room destroy phase but leave the shape of a syllable intact — and the accepted value is the MEDIAN of nine confident estimates. Measured: an exponential slew still chased outliers, walking 2.5 -> 94.6 -> 16 -> 47ms with cancellation collapsing on every jump, because the filter re-adapts whenever alignment moves; the median holds it at 8-10ms. **Alignment was not the whole problem, and a second stage is what closed it.** AEC3 models the echo path as LINEAR — which is exactly what the own-voice suite's simulated speaker is, hence 27-36 dB — but a real laptop speaker at volume clips and resonates, so much of what reaches the mic was never a linear function of the reference and cannot be subtracted from it. Measured: 1.6-12.7 dB with alignment already correct. `_ResidualSuppressor` works on MAGNITUDES instead: per bin it learns how strongly the reference couples into the mic (learned only where the far end dominates, so his voice never teaches it) and attenuates bins the predicted echo accounts for. Nonlinearity does not defeat a magnitude estimate the way it defeats a subtraction. **Real speaker music through the real mic: 15.2-24.2 dB removed, averaging ~19 — up from ~6.** Gains are smoothed across frequency and time (unsmoothed spectral gain is what makes a suppressor sound like a broken radio) and a silent reference means every gain is 1, so his voice is untouched when nothing is playing — verified, 0.2 dB. **Note the lag estimator does not lock on music** (confidence stays under the floor); the suppressor reaches ~19 dB regardless, because magnitudes tolerate the misalignment a subtractive filter cannot. `audio.system_tap.enabled`.

**`echo_canceller.py`** — Subtracts Nova's own voice from the microphone, so she can be interrupted while speaking. Nova is unusually well placed for this: the hard part of software AEC is a clean reference of what the speaker is playing, and **Nova synthesises her own speech**, so Kokoro hands over that signal before it is ever played. Measured on real Kokoro output against real speech: **27-36 dB of her voice removed, his attenuated only 1.5-3.9 dB, and 0.2 dB when she is silent.** Two traps it exists to handle: the APM works in 10 ms frames (160 samples) while the mic delivers 30 ms (480) and the library does NOT reject a wrong size — it degrades silently; and the reference must be consumed in step with incoming mic frames, because playback runs ahead of capture. **Default OFF** (`stt.echo_cancellation`): enabling it also means keeping mic frames during playback, and Nova listens far more often than she speaks. Nothing raises — a failure returns the raw microphone. `feat/barge-in` is the other half.

**`brain.py`** + **`skills.py`** — **The brain**, ported from JARVIS (MiniPCJarvis `docs/BRAIN.md`); full write-up in `nova_backend/BRAIN.md`. A nearest-neighbour classifier over example sentences: each request type is a SKILL (seeds, a slot function that extracts arguments or refuses, a runner, a consequence tier). A sentence is normalised (lowercase, wake word and politeness stripped, numbers → `N`), embedded with **bge-small via fastembed** (local ONNX, ~4ms; model downloaded once to `nova_backend/models/`, gitignored), and matched against ~675 examples. **JARVIS's rule copied exactly:** `confidence = top if margin >= 0.06 else top - (0.06 - margin) * 3`, act at 0.82, "nearly knew" at 0.66 — same embedder, so the numbers still mean what they were tuned to mean. A confident match whose slots extract is a **reflex**: it runs the SAME `_run_*` stage method the regex chain calls (no LLM, so the MLX prompt cache is untouched), 5-45ms in the live process. A guard refusal hands the turn to the runner-up. Otherwise the old regex stages run unchanged behind it. **Slots are asymmetric on purpose:** reads are lenient; writes with no confirmation step (calendar writes, power, lock, app launch by name, web search) stay strict — the module's own regex must accept the sentence; file moves are lenient because file_intents asks before touching disk. **Shape guards** (each from a measured wrong action): a read must be SHAPED like a request ("my next meeting is at three" read his calendar), navigation like navigation ("we should go home soon" went home), and a transport command must name its object ("what's next" skipped his song). A **confident-conversation veto** (`general` skill ≥ 0.82) skips the regex command stages but never the unsupported-action refusal. `confidence.decide(min_tier=)` takes the skill's tier: the brain can raise the transcript bar, never lower it. Storage: `brain_examples` / `brain_rejections` in nova_memory.db, every write committed on a short-lived connection (JARVIS's 2,600-message night was one uncommitted write). `brain.enabled: false` is an instant rollback. **Stage 2 (learning):** the LLM has no tools, so the TEACHER is the regex chain — a sentence the brain fell through on that a regex stage then claimed is learned, only if the skill's own slots run it AND the brain's own top guess already agrees (the regex makes its own wrong calls). "Did you mean <label>?" when nothing claimed it and the brain nearly knew (raw top ≥ 0.66 AND confidence ≥ 0.70 — the second bar is Nova's, because the first alone was wrong on all 6 of his real offers); yes teaches it, no is remembered forever. Phrases taught by voice ("when I say status check, ..."), near-exact match, nothing high-consequence behind a shortcut. "No, I meant X" un-learns the example behind a wrong reflex (never a seed).

**`mathskill.py`** + **`units.py`** — Arithmetic, percentages ("25% of 10%" → 2.5 percent), tips and unit conversion **in code, never by the model** (ported from JARVIS). No `eval`: a recursive-descent parser over a normalised sentence, which accepts it only when every word is part of the sum. Stage 2e-0.

**`honesty.py`** — Nova's LLM runs NO tools, so every LLM sentence promising to watch / remind / follow up, or claiming "I've opened / set / sent", is false. Filtered per sentence in `_stream_response` before it is spoken (streaming untouched): the first becomes the truth, later ones are dropped, history stores what she said. `llm.honesty_filter`.

**`delivery.py`** — **The single door for anything Nova says unprompted** (timers, the proactive monitor, watches). Claims a UNIQUE row and commits it before speaking (never twice, across threads and restarts), hard hourly/daily ceilings for unprompted kinds counted from the table, and over a ceiling or in quiet hours an item is SHOWN, never dropped. What he asked for (timers, watches) is never held. JARVIS rule 5, from its 2,600-message night.

**`standing.py`** — "Tell me when my battery is fully charged / my CPU goes above 90", list, cancel. Rows in `standing_watches`, one-shot, delivered through the door, survive restarts; an already-true condition is answered, not stored. **Local readings only** — a stock or weather watch would poll the network unprompted (invariant 3, his call). Stage 2e-0b.

**`shape.py`** — One definition of "shaped like a request", shared by the brain's skills and the regex read stages (weather, market, tools readings): a request opens with a question word or a verb, or ends in "?"; a statement ("my battery died yesterday", "I asked Nova what the weather was...") is not one; nor is a definition ("what does a CPU actually do"), nor "tell me WHEN ..." (a watch, not a reading of now).

**`turnstats.py`** — A row per turn in `turn_stats`: the path that answered (reflex / regex stage / math / watch / phrase / ask / LLM...), skill, confidence, matched example, time to decide and to first word. The evidence the next round of routing work starts from. Answers "why did you do that?" from the record, and "how often do you answer things by yourself?". Bookkeeping never costs a turn.

**`world_facts.py`** + **`wiki.py`** — **Stage 3: facts that can never change**, ported from JARVIS's fact store (BRAIN_ROADMAP.md realm 1). A general-knowledge question that reaches the LLM stage is looked up on **Wikipedia** first (one request, ~0.5-0.9s, hard 2s timeout) and the sentences that answer it go to the model as reference — measured on the real 3B: right 9/9 with them, 8/9 without ("324 metres" for the Eiffel Tower). After the turn, off the live path, a one-sentence answer is synthesised at temperature 0 and STORED only if every number and nearly every word is in the source, it is not changeable, and it is not a "why". Next time it is answered from the store in ~20ms with no model and no request. **The 3B cannot judge "timeless"** — measured, it called 6 of 8 changeable facts timeless (Apple's CEO, Japan's population) — so a deterministic realm-2 screen decides and the model may only veto. **Serving is guarded by words, not just similarity:** at JARVIS's 0.90 cosine bar "who directed jaws 2" scored 0.937 against "who directed jaws" and "how wide is the mariana trench" 0.943 against "how deep"; a fact is served only when the question kind AND the content words match too (tall = high = height). **Privacy guard before any request:** nothing in the first/second person, no name he has told Nova about, no contact details — counted at the network boundary in the tests. Relevance floor 0.70 (measured: right pages 0.795-0.898, another question's pages ≤0.564). "How do you know that?" cites the article and when it was last checked; "that's wrong" demotes the fact and it is never stored again. Wikipedia rate-limits bursts (429 after ~10 quick requests): `Retry-After` is honoured. Tables `world_facts` / `world_lookups` in nova_memory.db (`facts` is memory.py's). `facts.enabled: false` stops every request.

**`night_school.py`** — What Nova does while he sleeps (JARVIS's night school), **silent** — reported only when he asks "what did you learn last night?". Runs in `facts.night_hours` (01-06), after 45 idle minutes, at most once a day, on mains power only, every model call a background job, stopping between items the moment he is back. Three jobs: **the audit** — Wikipedia revision ids mean one request re-checks every stored fact and an unedited page is confirmed without reading it; an edited page is re-read, and if the answer is no longer in it the model may say CHANGED (demote) but may never confirm (measured: the 3B said SAME against an unrelated sentence) — two strikes demote; **curiosity** (his option C) — general, timeless questions he asked this week that the model answered from memory are looked up and offered to the store, each sent at most once ever, nothing personal or changeable sent; **paraphrase distillation** (local) — sentences a regex stage had to handle are paraphrased and taught to the brain through `brain.learn`'s full bar.

**`ws_server.py`** — Bridge between Python and Swift. HTTP server on :5001, WebSocket server on :8766. Swift connects here to receive state changes, message content, streaming tokens, and **which screen to show** in real time. The current view and its data are held like the current state, so a client that connects — or reconnects after an app relaunch — is told immediately instead of coming up blank.

**`requirements.txt`** — All Python dependencies. Install with `pip install -r requirements.txt`.

---

## Pipeline Routing Order (nova.py — first match wins)

1. System commands — sleep / wake / mute (always intercepted first)
1b. Destructive-action confirmation — a power command's strict yes/no
2. Calendar follow-up offer — answers a "want to hear what's coming up?" with yes/no
2b. Pending file question — "which one?" / "move it to Documents?" answered before routing
2b′. The brain's lessons — an answer to "did you mean X?", a correction of the last
    reflex, teaching / forgetting / listing a phrase, and a taught phrase itself
2b″. "Why did you do that?" / "how often do you answer by yourself?" (turnstats)
2b‴. World-fact lessons — "how do you know that?", "that's wrong" about a fact,
    "what did you learn last night?"
2c. UI navigation (regex, anchored both ends) — go home / the menu / a named screen /
    work mode, plus the home surface: clear home, restore home, move a card to a slot
2d. Pending actuation question — "that would send it, want me to?"
2e-0. Arithmetic / tips / conversions in code (mathskill.py)
2e-0b. Standing watches — "tell me when ...", list, cancel (standing.py)
2e. **THE BRAIN** (`_brain_turn`) — a confident skill match runs directly; a confident
    "this is conversation" skips 2f-8 (never 8b); anything else falls through
2f-8. **Regex command stages, the fallback** (`_regex_commands`), unchanged in order:
    actuation (work mode) → calendar → screen → weather → market → files →
    memory → fast path → tools
8a. "Did you mean <label>?" — nothing claimed it, but the brain nearly knew
8b. Unsupported action — refused honestly, never left to the LLM to fake
8c. World facts — a verified fact answered from the store; otherwise a general-
    knowledge question is looked up on Wikipedia and the answering sentences go
    to the LLM as reference (stored afterwards if timeless, off the live path)
9. RAG context enrichment — query personal documents for context
10. LLM fallback — MLX streaming with sentence-chunked TTS overlap

Every regex stage is still there. One is removed only when the brain matches it on
the corpus AND the `[brain-disagree]` log from real use says it is safe.

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
| `test_routing_corpus.py` | every phrase that has broken Nova still routes right | real code, real memory stage on a throwaway data dir, tools via dry-run `detect()` |
| `test_brain_seeds.py` | no two skills own a canonical seed; every seed executable by its own skill | real skills + real slot extractors |
| `test_brain_intent.py` | brain vs regex on corpus, 5-fold held-out, two paraphrase sets, his real turns; no NEW wrong action | real brain + real embedder, throwaway DB |
| `test_brain_learning.py` | the brain learns, asks, is taught and corrected — and nothing goes wrong silently | real `_handle_turn_impl`, stage runners recorded |
| `test_turnstats.py` | every turn records what answered it; "why?" is answered from that | real `_handle_turn_impl`, stage runners recorded |
| `test_world_facts.py` | facts looked up, kept only if timeless and grounded, never served for another question, nothing personal sent; night school's audit, curiosity, gates | real store/night school/embedder + real pipeline; Wikipedia canned (real captured responses), model scripted |
| `test_audit_regressions.py` | every defect the 2026-11-19 four-way audit of the brain branch proved (wrong arithmetic, false promises let through, a stale "did you mean" that could lock his Mac, names leaking to Wikipedia in the possessive, "shuffle off" turning shuffle on...) stays fixed — each check confirmed failing on the pre-fix commit | real modules + real pipeline, tools as dry runs, no network, no model |
| `world_facts_live.py` | the real 3B answers better with the reference; real lookups, intake, audit; the real process looks up, stores, serves, cites | real model + real Wikipedia + real process (NOVA_TEST_LIVE=1, ~8 requests) |
| `test_honesty.py` | no spoken promise nothing keeps; typed turns stay silent | real `_stream_response`, scripted LLM |
| `test_unprompted.py` | one door for unprompted speech: never twice, ceilings, watches kept across restarts | real delivery/proactive/standing, throwaway DB |
| `brain_live_check.py` | the brain inside the REAL process: reflexes, timings, LLM still answers after, did-you-mean, taught phrase, correction | real process, read-only sentences, silent turns |
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

8. **Score a router on sentences it was not built from.** The brain is seeded
   from the corpus, so the corpus only proves parity. Its real numbers come
   from 5-fold held-out scoring, paraphrase sets written BEFORE the seeds they
   test (v2 stayed sealed until the seeds froze), and his real turns minus any
   that are seeds. And a wrong action is scored apart from a miss: a miss falls
   to conversation; a wrong action does something on his Mac.
9. **A harness that approximates a stage is not testing it.** The corpus
   harness guessed at the memory stage with two regexes and reported 257/257
   while "what's my memory usage" answered "I don't have your memory usage
   stored yet". Run the real stage against a throwaway data dir; probe tools
   with `detect()` (match() in dry mode), never a stubbed match().

**Reporting to Nicholas** — every test report is four sections: what I tested /
how I tested it (always naming the fidelity above) / what succeeded and what
did not / what he needs to do. Plus an explicit "could not verify" line. Never
lead with a pass count.

10. **An app the sweep launches may bring HIS documents with it.** The
    launch/quit test used TextEdit, which restores untitled documents on
    launch — one he once had Nova type into ("Nova wrote this line") came
    back every run, its quit stalled, the check failed, and TextEdit was left
    open on his Mac. The test uses Calculator now (nothing to restore) and
    skips itself if Calculator is already his. Never quit an app holding his
    content to make a test pass.

**When to run what:** targeted suites for a targeted fix; `--quick` before any
merge; `--all` before a merge touching shared plumbing. Do NOT full-sweep every
small change — Nicholas asked for this explicitly and it is faster and sharper.

---

## Invariants — Never Break These

1. **Ports are 5001 (HTTP) and 8766 (WS).** These differ from Jarvis (3000/8765) intentionally so both apps can run simultaneously. Do not change them.
2. **LLM n_ctx stays at 4096.** Do not raise the context window without explicit instruction.
3. **No cloud dependencies — FIVE narrow, documented exceptions.** Nova is local and private: no API keys, and the LLM/STT/TTS/memory NEVER leave the machine. The exceptions:
   - `maps_engine.py` — "how far is the nearest X" and travel times are Apple MapKit network lookups; getting a location fix sends an approximate position to Apple.
   - `spotify_search.py` — **added 2026-08-13, his explicit choice** after being shown the alternatives (hand off to Spotify's own search with no key; automate Spotify's UI). Name→URI lookup only; the words he asked for are what leaves the machine. Playback is local.
   - `market_engine.py` — **added 2026-08-17, his explicit choice.** Market data only; he ruled out any access to his real accounts. Yahoo needs no key at all; Finnhub's free tier needs one, stored at `NOVA_DATA_DIR/finnhub_credentials.json` and never in the committed config. What leaves the machine is a ticker or a company name he said out loud. **The privacy wrinkle he was told about and accepted: weather leaks an approximate location, but a stock query leaks what he is invested in.** `finance.enabled: false` disables it.
   - `weather_engine.py` — **added 2026-08-13, Nicholas's explicit choice** after being shown WeatherKit (needs a paid developer account), weather.gov (grid lookup) and browser scraping. Open-Meteo needs **no API key and no account**, so nothing ties a request to him beyond the IP any HTTP call carries. What leaves the machine is an approximate coordinate (rounded to ~100m) or a place name he said aloud.

   - `wiki.py` (Stage 3: `world_facts.py`, `night_school.py`) — **added 2026-11-19, his explicit choice: option C** of three he was shown (A local only, B lookups only when he asks, C B plus overnight research on his past questions). Wikipedia needs **no key and no account**; the User-Agent names nobody. What leaves the machine: the words of a general-knowledge question he asked; overnight, his own past general-knowledge questions the model answered from memory (each at most once, ever) and the page ids of facts already stored, to re-check them. **Never**: anything in the first or second person, any name he has told Nova about, memory, documents or the conversation — refused before the request, and logged in `world_lookups` so "what did you look up?" is answerable. Stored facts are Wikipedia's, never his. `facts.enabled: false` stops every request; `facts.night_school: false` keeps on-demand lookups and stops the overnight part.

   Each is used ONLY for that kind of question (night school only for what he already asked), none sends his memory, and they degrade honestly ("I can't get your location", "I couldn't reach the weather service") rather than guessing. `weather.enabled: false` disables the second entirely. **Do not widen further without asking him — this is his privacy line, and he decides where it sits.**
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
2. Proactive notifications — BUILT (proactive.py: events about to start, overdue
   reminders), now through the single door (delivery.py). Remaining: anything
   beyond the calendar, and briefs.
3. Barge-in over speakers — the canceller now EXISTS (`echo_canceller.py`, measured
   27-36 dB of Nova's voice removed) and `feat/barge-in` is the other half, but
   `stt.echo_cancellation` ships OFF: turning it on also keeps mic frames during
   playback, and that needs his ears on real speakers before it becomes the default.
4. UI overhaul — the HOME tab is done: slots, voice-moved cards, the status
   row, clear/restore, self-refreshing tiles, and the live work surface.
   Remaining: the other tabs (finance, calendar, files, memory, conversations
   all still render generically), and a health screen that needs the phone.
5. iOS app — far back burner, separate on-device architecture
6. The brain (see `nova_backend/BRAIN.md`): Stages 1-5 are built. Remaining:
   explanations ("why ...") are looked up but never stored; Wikipedia lead
   sections only (no "why is the sky blue" answer in them); a real night of
   night school has not yet run on his Mac; regex stages retire only on
   `[brain-disagree]` evidence from real use.

