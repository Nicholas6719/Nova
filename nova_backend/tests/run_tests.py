#!/usr/bin/env python3
"""
Nova test runner — one command, for Nicholas or for Claude.

    python nova_backend/tests/run_tests.py --smoke      # does it start and answer?
    python nova_backend/tests/run_tests.py --env        # did a pip install break it?
    python nova_backend/tests/run_tests.py --routing    # phrases that broke it before
    python nova_backend/tests/run_tests.py --loop       # conversation state machine
    python nova_backend/tests/run_tests.py --full       # everything, incl. real system
    python nova_backend/tests/run_tests.py --quick      # every fast suite (no audio)

Each suite prints its own detail; this prints the summary and, importantly, a
list of what could NOT be verified — the microphone-dependent behaviour that no
automated run can prove.

Suites that need the ports (:5001/:8766) refuse to run while Nova is open; quit
the app first.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

TESTS = Path(__file__).resolve().parent


# ── Which Python? ─────────────────────────────────────────────────────────────
# Packages are installed PER INTERPRETER. This Mac has several python3s and only
# one of them has mlx_lm, whisper and kokoro. Running the suites under whatever
# python happened to be on PATH would test an environment Nova never uses — and
# the environment guard exists precisely because a pip install once broke Nova's
# LLM. A guard pointed at the wrong interpreter is worse than none: it reports
# green while the real one is broken.
#
# So the runner resolves the SAME interpreter the app does. This list, its
# order, and the NOVA_PYTHON override mirror locatePython() in
# BackendManager.swift — if that changes, change this with it.
_CANDIDATES = (
    "/opt/homebrew/Caskroom/miniforge/base/bin/python3",   # conda/miniforge
    "/opt/homebrew/bin/python3",                           # Homebrew (Apple silicon)
    "/usr/local/bin/python3",                              # Homebrew (Intel)
    "/usr/bin/python3",                                    # macOS system
)


def _can_import_backend_deps(path: str) -> bool:
    """Same validation the app uses: can this interpreter import mlx_lm?"""
    try:
        return subprocess.run([path, "-c", "import mlx_lm"],
                              capture_output=True, timeout=60).returncode == 0
    except Exception:
        return False


def resolve_backend_python() -> tuple[str, str]:
    """(interpreter, how_it_was_chosen). Raises SystemExit if none will do.

    Unlike the app, this REFUSES to fall back to an interpreter without the
    dependencies. The app falls back so the backend can start and surface its
    own import error; a test run that silently checks the wrong environment
    would just be lying.
    """
    override = os.environ.get("NOVA_PYTHON", "").strip()
    if override:
        if not os.access(override, os.X_OK):
            sys.exit(f"NOVA_PYTHON is set to {override!r}, which is not executable.")
        return override, "NOVA_PYTHON override"

    seen = []
    for path in _CANDIDATES:
        if not os.access(path, os.X_OK):
            continue
        seen.append(path)
        if _can_import_backend_deps(path):
            return path, "same interpreter the app uses"

    on_path = shutil.which("python3")
    if on_path and on_path not in seen:
        seen.append(on_path)
        if _can_import_backend_deps(on_path):
            return on_path, "python3 on PATH"

    sys.exit(
        "No interpreter with Nova's dependencies was found.\n"
        "  Tried: " + ", ".join(seen or ["(none executable)"]) + "\n"
        "  Every one of these is missing mlx_lm, so none of them is the one\n"
        "  Nova runs under. Install the requirements into the right interpreter,\n"
        "  or set NOVA_PYTHON=/path/to/python to point at it."
    )


PY, PY_REASON = resolve_backend_python()

# name -> (file, description, needs_audio, needs_ports)
SUITES = {
    "env":     ("verify_environment.py",
                "engines import and MLX generates", False, False),
    "routing": ("test_routing_corpus.py",
                "every phrase that has broken Nova before", False, False),
    "loop":    ("test_conversation_loop.py",
                "conversation state machine (real loop, scripted mic)", False, False),
    "cache":   ("test_prompt_cache.py",
                "the prompt cache changed speed and nothing else", False, False),
    "rag":     ("test_rag_relevance.py",
                "Nova only quotes a document when it has one", False, False),
    "tts":     ("test_tts_chunking.py",
                "Nova starts speaking sooner and says the same words", False, False),
    "wake":    ("test_wake_capture.py",
                "the wake word gives him time, and loops never reach the LLM", False, False),
    "weather": ("test_weather.py",
                "weather answers are real numbers, and steal nothing else", False, False),
    "music":   ("test_music.py",
                "play-by-name works and shadows no transport command", False, False),
    "views":   ("test_views.py",
                "voice navigation reaches the right screen and fakes nothing", False, False),
    "conf":    ("test_confidence.py",
                "Nova acts only when sure enough for what it costs", False, False),
    "echo":    ("test_echo_cancellation.py",
                "Nova's own voice is removed from the mic, his is not", False, False),
    "smoke":   ("smoke_launch.py",
                "the REAL process starts and answers a turn", True, True),
    "full":    ("test_full_sweep.py",
                "every subsystem against real system state", True, False),
}

# Things no automated suite can establish. Printed after every run so a green
# result is never mistaken for full coverage.
CANNOT_VERIFY = [
    "wake-word detection with Nicholas's actual voice "
    "(macOS synthetic speech does not drive the model)",
    "speech quality — whether Nova sounds right through the speakers",
    "barge-in over speakers with REAL speakers in a real room — the echo "
    "suite measures the canceller against a simulated speaker path, not his desk",
    "music transport unless a player is already running",
    "anything TCC-gated inside NovaOS.app — screen recording and location are "
    "granted to the app bundle, not to this interpreter",
]


def run(name: str) -> tuple[str, int, float]:
    fname, desc, _, _ = SUITES[name]
    print("\n" + "─" * 72, flush=True)
    print(f"▶ {name}  —  {desc}", flush=True)
    print("─" * 72, flush=True)
    t0 = time.monotonic()
    # -u: without it the child's output is block-buffered and lands out of
    # order relative to these headers, which makes a failure hard to attribute.
    proc = subprocess.run([PY, "-u", str(TESTS / fname)])
    return name, proc.returncode, time.monotonic() - t0


def main() -> int:
    ap = argparse.ArgumentParser(add_help=True)
    for key in SUITES:
        ap.add_argument(f"--{key}", action="store_true", help=SUITES[key][1])
    ap.add_argument("--quick", action="store_true",
                    help="every fast suite: env, routing, loop, wake, cache, rag, tts, weather, music, views, echo (no audio)")
    ap.add_argument("--all", action="store_true", help="every suite")
    args = ap.parse_args()

    if args.all:
        chosen = list(SUITES)
    elif args.quick:
        chosen = ["env", "routing", "loop", "wake", "cache", "rag", "tts",
                  "weather", "music", "views", "conf", "echo"]
    else:
        chosen = [k for k in SUITES if getattr(args, k)]
    if not chosen:
        ap.print_help()
        print("\nNothing selected. --quick is the usual one.")
        return 2

    print(f"interpreter: {PY}", flush=True)
    print(f"             ({PY_REASON})", flush=True)
    # Compare the RESOLVED binaries: miniforge's `python` and `python3` are the
    # same file, and warning about that would be noise.
    if os.path.realpath(PY) != os.path.realpath(sys.executable):
        print(f"  note: you invoked {sys.executable},", flush=True)
        print( "        which is NOT the interpreter Nova runs under. The suites", flush=True)
        print( "        were run under the one above, so the results still apply.", flush=True)

    results = [run(name) for name in chosen]

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    failed = []
    for name, code, secs in results:
        mark = "PASS" if code == 0 else ("SKIPPED" if code == 2 else "FAIL")
        if code not in (0, 2):
            failed.append(name)
        print(f"  {mark:<8} {name:<9} {secs:5.1f}s   {SUITES[name][1]}")

    print("\n  COULD NOT VERIFY (no automated run can prove these):")
    for item in CANNOT_VERIFY:
        print(f"    • {item}")

    if failed:
        print(f"\n  {len(failed)} suite(s) failed: {', '.join(failed)}")
        return 1
    print("\n  All selected suites passed. That means the checks above held —")
    print("  not that Nova works. The list above is still unproven.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
