#!/usr/bin/env python3
"""
Every turn leaves a record of what answered it and how fast — and "why did you
do that?" is answered from that record, never invented.

Fidelity: the REAL _handle_turn_impl (same recorder harness as
test_brain_learning: stage runners and the LLM captured, a throwaway data dir),
the real brain, the real turnstats table.

Run:  python tests/test_turnstats.py
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR))
sys.path.insert(0, str(TESTS_DIR.parent))

import test_routing_corpus as T      # noqa: E402  (throwaway NOVA_DATA_DIR first)
import test_brain_learning as L      # noqa: E402

PASS = FAIL = 0
FAILURES: list[str] = []


def check(cond, label, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {label}")
    else:
        FAIL += 1
        FAILURES.append(f"{label} {detail}")
        print(f"  FAIL  {label} {detail}")


def main() -> int:
    va = T.build_assistant()
    with T.NoSideEffects(va):
        rec = L.Recorder(va)
        va._init_standing()                 # the watch stage, as in production
        b = L.fresh_brain(va, "turnstats")
        cases = [("what time is it", "reflex"), ("what's 10 plus 10", "math"),
                 ("tell me when my cpu goes above 99 percent", "watch"),
                 ("tell me about the history of jazz", None),
                 ("go home", "view")]
        for text, want in cases:
            rec.turn(text)
            row = va._last_turn or {}
            if want is None:
                check(row.get("path") in ("llm", "conversation"),
                      f"{text!r} recorded as conversation", f"{row}")
            else:
                check(row.get("path") == want, f"{text!r} recorded as {want}", f"{row}")
        check(all(r is not None for r in [va._last_turn.get("decide_ms")]),
              "the time to decide is measured")

        rec.turn("what time is it")
        last = dict(va._last_turn)
        calls, said = rec.turn("why did you do that")
        check(said and "tell you the time" in said[-1] and "already know" in said[-1],
              "'why did you do that?' explains the reflex from the record", f"{said}")
        check(va._last_turn == last, "asking why does not overwrite what is being explained")

        calls, said = rec.turn("what's 25% of 10%")
        calls, said = rec.turn("why did you do that")
        check(said and "arithmetic" in said[-1], "...and the arithmetic path", f"{said}")

        calls, said = rec.turn("how often do you answer things by yourself")
        check(said and "handled" in said[-1] and "myself" in said[-1],
              "a brain report counts what she handled herself", f"{said}")

        db = Path(va.turnstats.db_path)
        with sqlite3.connect(db) as c:
            n = c.execute("SELECT COUNT(*) FROM turn_stats").fetchone()[0]
            paths = {r[0] for r in c.execute("SELECT path FROM turn_stats")}
        check(n >= 9, f"every turn is a row ({n})")
        check({"reflex", "math", "watch", "view", "explain"} <= paths, f"paths recorded: {sorted(paths)}")

        # bookkeeping never costs the turn
        va.turnstats.db_path = Path("/nonexistent/dir/x.db")
        calls, said = rec.turn("what time is it")
        check(said and said[-1].startswith("It's"), "a stats write that fails does not cost the turn")

    print(f"\n  {PASS}/{PASS + FAIL}")
    for f in FAILURES:
        print(f"    ✗ {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
