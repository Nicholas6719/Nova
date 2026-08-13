#!/usr/bin/env python3
"""
Nova test runner — one command, for Nicholas or for Claude.

    python nova_backend/tests/run_tests.py --smoke      # does it start and answer?
    python nova_backend/tests/run_tests.py --env        # did a pip install break it?
    python nova_backend/tests/run_tests.py --routing    # phrases that broke it before
    python nova_backend/tests/run_tests.py --loop       # conversation state machine
    python nova_backend/tests/run_tests.py --full       # everything, incl. real system
    python nova_backend/tests/run_tests.py --quick      # env + routing + loop (no audio)

Each suite prints its own detail; this prints the summary and, importantly, a
list of what could NOT be verified — the microphone-dependent behaviour that no
automated run can prove.

Suites that need the ports (:5001/:8766) refuse to run while Nova is open; quit
the app first.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

TESTS = Path(__file__).resolve().parent
PY = sys.executable

# name -> (file, description, needs_audio, needs_ports)
SUITES = {
    "env":     ("verify_environment.py",
                "engines import and MLX generates", False, False),
    "routing": ("test_routing_corpus.py",
                "every phrase that has broken Nova before", False, False),
    "loop":    ("test_conversation_loop.py",
                "conversation state machine (real loop, scripted mic)", False, False),
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
    "barge-in over speakers (needs acoustic echo cancellation; parked)",
    "music transport unless a player is already running",
    "anything TCC-gated inside NovaOS.app — screen recording and location are "
    "granted to the app bundle, not to this interpreter",
]


def run(name: str) -> tuple[str, int, float]:
    fname, desc, _, _ = SUITES[name]
    print("\n" + "─" * 72)
    print(f"▶ {name}  —  {desc}")
    print("─" * 72)
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
                    help="env + routing + loop (fast, no audio)")
    ap.add_argument("--all", action="store_true", help="every suite")
    args = ap.parse_args()

    if args.all:
        chosen = list(SUITES)
    elif args.quick:
        chosen = ["env", "routing", "loop"]
    else:
        chosen = [k for k in SUITES if getattr(args, k)]
    if not chosen:
        ap.print_help()
        print("\nNothing selected. --quick is the usual one.")
        return 2

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
