#!/usr/bin/env python3
"""
His real turns that still went wrong after the brain branch's audit, through
the REAL pipeline (stage runners recorded, tools stubbed at the action).

The routing half lives in adversarial_phrases.txt (test_routing_corpus);
this covers what only the whole pipeline can show: a request at the end of
a complaint, the sign-off misheard, and the Notes app versus a file.

Run:  python tests/test_real_turns.py
"""
from __future__ import annotations

import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR))
sys.path.insert(0, str(TESTS_DIR.parent))

import test_routing_corpus as T          # noqa: E402
import test_brain_learning as L          # noqa: E402

PASS = FAIL = 0
FAILURES: list[str] = []


def check(cond, label, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {label}")
    else:
        FAIL += 1
        FAILURES.append(f"{label}  {detail}")
        print(f"  FAIL  {label}  {detail}")


def main() -> int:
    va = T.build_assistant()
    va.work_mode = False
    with T.NoSideEffects(va):
        rec = L.Recorder(va)
        ran = []
        real_match = va.tools.match

        def spy(text):
            from tools import _DRY
            if not getattr(_DRY, "active", False):  # a real (stubbed-subprocess) run, not a dry probe
                ran.append(text)
            return real_match(text)
        va.tools.match = spy
        print("── a request at the end of a complaint")
        ran.clear()
        calls, said = rec.turn("You didn't close the brave browser, please close the brave browser")
        check("close the brave browser" in ran, "the request after 'please' is carried out", f"ran {ran}")
        ran.clear()
        calls, said = rec.turn("I had a long day, please tell me a joke")
        check("tell me a joke" not in ran and ("llm", "I had a long day, please tell me a joke") in calls,
              "a tail Nova cannot do is not split off: the whole thing is conversation", f"{calls}")
        ran.clear()
        calls, said = rec.turn("thanks for that, could you open spotify")
        check("open spotify" in ran, "'..., could you open spotify' opens Spotify", f"ran {ran} calls {calls}")

        print("\n── the sign-off as Whisper hears it")
        import nova
        check(nova.find_signoff("Nova, got to sleep")[1] is not None, "'Nova, got to sleep' ends the conversation")
        check(nova.find_signoff("I got to sleep at 3am last night")[1] is None, "...a story about sleep does not")

        print("\n── the Notes app, not a file called notes")
        for t in ("open notes", "open the notes app", "open my notes"):
            check(va.files.detect_intent(t) is None, f"files does not claim {t!r}")
        for t in ("open my resume", "open the budget spreadsheet", "open the notes document"):
            check(va.files.detect_intent(t) is not None, f"files still claims {t!r}")

    print(f"\n  {PASS}/{PASS + FAIL} checks passed")
    for x in FAILURES:
        print(f"    ✗ {x}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
