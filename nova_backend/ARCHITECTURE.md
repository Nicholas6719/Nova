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
  config.json       — All tuneable parameters
  system_prompt.py  — Nova's personality, identity, memory injection
  requirements.txt  — Python dependencies

  stt_engine.py     — STT: faster-whisper + webrtcvad VAD + mic recording
  llm_engine.py     — LLM: MLX Llama streaming inference (Apple Silicon)
  tts_engine.py     — TTS: Kokoro ONNX primary + macOS say fallback
  memory.py         — SQLite: facts / episodes / conversations / user profile
  rag.py            — ChromaDB: local document retrieval (~/Documents)
  tools.py          — macOS tools: app launch, volume, battery, search, screenshot
  ws_server.py      — HTTP :5001 + WebSocket :8766 for SwiftUI communication
```

---

## Pipeline (routing order — first match wins)

```
Wake word
  → STT (faster-whisper + webrtcvad VAD)
  → [1] System commands      (sleep / wake / mute)
  → [2] Calendar follow-up    (answers "want to hear what's coming up?")
  → [3] Calendar intents      (read / create / complete / delete / update)
  → [4] Memory intents        (remember / recall / update / forget)
  → [5] Fast-path intents     (greeting / date / time / repeat)
  → [6] Tool intents          (open app / volume / battery / search / screenshot)
  → [7] RAG context enrichment
  → [8] LLM (MLX streaming) → sentence chunking → TTS (Kokoro ONNX)
```

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

## Swift Changes Required (Nova.xcodeproj)

The Swift UI layer (ContentView.swift, ChatViewModel.swift) is kept.
The Core/ backend (NovaEngineCore, LLMClient, etc.) is removed.
A new BackendManager is added.

### Files to add to Nova.xcodeproj

**SwiftBackend/BackendManager.swift**
- Finds the Python executable and nova_backend/nova.py
- Launches `python nova.py` as a subprocess
- Sets NOVA_DATA_DIR environment variable
- Polls /api/status to confirm startup
- Restarts on crash

**SwiftBackend/NovaAPIClient.swift**
- HTTP client: GET /api/status, GET /api/messages, POST /api/message
- WebSocket client connecting to :8766
- Publishes state changes and messages via Combine / @Published

### Files to strip from Nova.xcodeproj

Remove or gut (replace with HTTP/WS calls):
- Core/NovaEngineCore.swift
- Core/LLMClient.swift
- Core/IntentDetector.swift
- Core/MathRouter.swift
- Core/NovaPersonality.swift
- Memory/ (replaced by Python memory.py)
- Tools/ (replaced by Python tools.py)

Keep (adapt to receive data from BackendManager):
- ContentView.swift
- Features/Chat/ChatViewModel.swift  (adapt to use NovaAPIClient)
- Features/Chat/Message.swift
- Voice/SpeechManager.swift         (TTS now handled by Python — may simplify)
- Voice/SpeechRecognizer.swift      (STT now handled by Python — may simplify)

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

## What's Next (Phase 2+)

- Calendar integration (EventKit, same approach as Jarvis)
- Reminder / timer support
- Proactive notifications (upcoming events, due reminders)
- Deeper memory: infer preferences from conversation patterns
- LLM-powered memory summarization (replace naive concatenation)
- iOS: independent on-device app (separate architecture — no Python)
