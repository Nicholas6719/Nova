# Nova Backend — Architecture & Swift Integration Guide

## Overview

Nova's macOS backend is a Python voice pipeline that communicates with a native
SwiftUI frontend via HTTP + WebSocket. All inference and data stay 100% local.
No user data ever leaves the machine.

---

## File Map

```
nova_backend/
  nova.py           — VoiceAssistant orchestrator (main entry point)
  brain.py          — the brain: learned intent router (BRAIN.md)
  skills.py         — every skill the brain runs without the LLM
  config.json       — All tuneable parameters
  system_prompt.py  — Nova's personality, identity, memory injection
  requirements.txt  — Python dependencies

  stt_engine.py     — STT: faster-whisper + webrtcvad VAD + mic recording
  llm_engine.py     — LLM: MLX Llama streaming inference (Apple Silicon)
  tts_engine.py     — TTS: Kokoro ONNX primary + macOS say fallback
  memory.py         — SQLite: facts / episodes / conversations / user profile
  rag.py            — ChromaDB: local document retrieval (~/Documents)
  tools.py          — macOS tools: app launch, volume, battery, search, screenshot
  file_manager.py   — Filesystem engine: search, read, move/copy/rename, guards
  file_intents.py   — File NL dispatch: strict detection, disambiguation, confirm
  ws_server.py      — HTTP :5001 + WebSocket :8766 for SwiftUI communication
```

---

## Pipeline (routing order — first match wins)

```
Wake word ("Nova", openWakeWord)
  → STT (faster-whisper, beam 5) → confidence gate (confidence.py; the brain's
    skill tier can raise its bar, never lower it)
  → [1]  System commands            sleep / wake / mute
  → [1b] Power confirmation         strict yes/no
  → [2]  Calendar follow-up         "want to hear what's coming up?"
  → [2b] Pending file question      "which one?" / "move it to Documents?"
  → [2b‴] World-fact lessons       "how do you know that?", "that's wrong",
                                    "what did you learn last night?"
  → [2c] UI navigation (regex)      home / menu / screens / home surface
  → [2d] Pending actuation question "that would send it, want me to?"
  → [2e] THE BRAIN                  brain.py + skills.py: a confident skill runs
                                    directly (no LLM); "surely conversation"
                                    skips 2f-8; otherwise fall through
  → [2f-8] Regex command stages     the fallback, unchanged: actuation →
                                    calendar → screen → weather → market →
                                    files → memory → fast path → tools
  → [8b] Unsupported action         refused honestly, never faked
  → [8c] World facts                world_facts.py: a verified fact from the
                                    store; else a general question is looked
                                    up on Wikipedia (wiki.py) and the answering
                                    sentences go to the LLM as reference
  → [9]  RAG context enrichment
  → [10] LLM (MLX streaming, carried prompt cache) → sentence chunks → Kokoro TTS
```

A reflex (2e) and a regex match (2f-8) end in the SAME `_run_*` stage method in
nova.py, so what happens after a decision cannot differ by who made it. See
BRAIN.md for the brain itself.

---

## Communication Protocol (Python ↔ Swift)

### HTTP endpoints

| Method | Path             | Description                        |
|--------|------------------|------------------------------------|
| GET    | /api/status      | Health check + current state       |
| GET    | /api/messages    | Message history (last 50)          |
| POST   | /api/message     | Send text from UI to pipeline      |
| POST   | /api/mute        | Toggle mute from UI                |

### WebSocket messages (:8766)

```json
{"type": "state",   "state": "idle|listening|thinking|speaking"}
{"type": "message", "role": "user|assistant", "content": "..."}
{"type": "token",   "token": "..."}
```

---

## Swift side

The migration plan that used to sit here (which Swift files to add, gut or keep)
was completed and then superseded by the UI overhaul: ContentView, ChatViewModel,
SpeechManager and SpeechRecognizer no longer exist. CLAUDE.md, "Swift Side —
Current State", is the current description.

---

## Key Invariants

1. All inference is local. No cloud API calls in the default configuration.
2. All user data stays in ~/Library/Application Support/Nova/. Never transmitted.
3. Ports 5001/8766 avoid collision with Jarvis (3000/8765).
4. LLM n_ctx is fixed at 4096. Do not raise without testing on target hardware.
5. Memory summarization runs on a daemon thread. It must never block the pipeline.
6. RAG index loads in the background. Queries skip gracefully if not yet ready.
7. Wake-word detection uses transcript-based fallback if openwakeword is not installed.
8. TTS falls back to macOS say if Kokoro ONNX model files are missing.

---

## First Run

```bash
cd nova_backend
pip install -r requirements.txt

# Copy Kokoro model files from Jarvis (same files):
cp /path/to/Jarvis/kokoro-v1.0.onnx .
cp /path/to/Jarvis/voices-v1.0.bin .

python nova.py
```

The MLX model (~2GB) downloads automatically on first run.

---

## What's Next

Calendar, reminders, timers and passive memory learning are built (CLAUDE.md
has the current state). The active roadmap is the brain's Stages 2-5 in
BRAIN.md, then CLAUDE.md "Not Built Yet".
