#!/usr/bin/env python3
"""
Environment guard — can Nova's engines still load?

Run this after ANY dependency change, no exceptions.

Why it exists: installing a vision package pulled `huggingface-hub` down two
major versions, `mlx_lm` stopped importing, and Nova's LLM was dead. Nothing
in the behaviour suites noticed, because they all import mlx_lm at module load
and would simply have crashed with a stack trace nobody was reading. The damage
was found by chance.

Everything here is cheap: imports and a one-token generation. It does not
listen, speak, or touch the network beyond the model cache.

Run:  python tests/verify_environment.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
BACKEND = TESTS_DIR.parent
sys.path.insert(0, str(BACKEND))
os.chdir(BACKEND)

PASS = FAIL = 0
FAILURES: list[str] = []


def check(cond, label, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  OK    {label}")
    else:
        FAIL += 1
        FAILURES.append(label)
        print(f"  FAIL  {label}")
    if detail:
        print(f"        {str(detail)[:200]}")


def try_import(name: str, label: str) -> None:
    try:
        __import__(name)
        check(True, label)
    except Exception as exc:
        check(False, label, f"{type(exc).__name__}: {exc}")


def main() -> int:
    print("=" * 72)
    print("ENVIRONMENT GUARD")
    print("=" * 72)
    print(f"  interpreter: {sys.executable}")

    print("\n-- core engines --")
    try_import("mlx_lm", "mlx_lm imports (the LLM)")
    try_import("faster_whisper", "faster_whisper imports (STT)")
    try_import("kokoro_onnx", "kokoro_onnx imports (TTS)")
    try_import("openwakeword", "openwakeword imports (wake word)")
    try_import("webrtcvad", "webrtcvad imports (voice activity)")
    try_import("sounddevice", "sounddevice imports (audio I/O)")

    print("\n-- feature dependencies --")
    try_import("PIL", "pillow imports (image dimensions)")
    try_import("pypdf", "pypdf imports (PDF reading)")
    try_import("docx", "python-docx imports (Word reading)")
    try_import("chromadb", "chromadb imports (RAG)")
    try_import("Quartz", "pyobjc Quartz imports (windows, screen)")
    try_import("Vision", "pyobjc Vision imports (screen OCR)")
    try_import("EventKit", "pyobjc EventKit imports (calendar)")

    print("\n-- model files present --")
    for fname, label in (("nova.onnx", "wake model"),
                         ("kokoro-v1.0.onnx", "Kokoro voice model"),
                         ("voices-v1.0.bin", "Kokoro voices")):
        path = BACKEND / fname
        check(path.is_file(), f"{label} on disk ({fname})",
              "" if path.is_file() else f"missing: {path}")

    print("\n-- the LLM actually generates --")
    try:
        import json
        from llm_engine import LLMEngine
        cfg = json.load(open(BACKEND / "config.json"))
        t0 = time.monotonic()
        eng = LLMEngine(cfg["llm"])
        load_s = time.monotonic() - t0
        out = eng.generate("You are a test.", [],
                           "Reply with the single word: ready.",
                           temperature=0.0, max_tokens=8)
        check(bool(out.strip()), "MLX produces a real token",
              f"loaded in {load_s:.1f}s, said {out.strip()[:40]!r}")
    except Exception as exc:
        check(False, "MLX produces a real token", f"{type(exc).__name__}: {exc}")

    print("\n" + "=" * 72)
    print(f"  {PASS}/{PASS + FAIL}")
    if FAILURES:
        print("\n  BROKEN — do not ship until these load:")
        for f in FAILURES:
            print(f"    ✗ {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
