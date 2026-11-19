#!/usr/bin/env python3
"""
The brain's seeds, checked before anything is embedded or learned.

Ported from JARVIS tests/seed_collisions.py, plus two Nova checks.

1. NO TWO SKILLS OWN THE SAME CANONICAL SEED. Seeds are normalised before they
   are embedded, so a seed can silently become a different sentence than the
   one written down. On JARVIS, adding "no more for now" to the sleep skill
   made it own the canonical form of every correction ("no, I meant ..."), and
   corrections put him to sleep at confidence 1.00. Brain.load() resolves a
   clash with "first skill listed wins" and says nothing — this is the loud
   failure.

2. EVERY SEED IS EXECUTABLE BY ITS OWN SKILL. A seed the skill's own slots
   refuse teaches the router to send that phrasing somewhere that will then
   decline it — a guard is not an outcome (BRAIN.md mistake #6). Checked with
   actuation's work-mode gate open, since that gate is about the situation,
   not the sentence.

3. HELD-OUT OVERLAP IS REPORTED. The paraphrase sets share a few short
   canonical forms with the seeds by construction ("can you repeat that" and
   "repeat that" are the same request once politeness is stripped);
   test_brain_intent excludes those sentences from scoring and says how many.

Run:  python tests/test_brain_seeds.py
"""
from __future__ import annotations

import collections
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR))
sys.path.insert(0, str(TESTS_DIR.parent))

import test_routing_corpus as T      # noqa: E402  (throwaway NOVA_DATA_DIR first)

FAILS: list[str] = []


def main() -> int:
    import brain as B
    import skills as S

    print("=" * 72)
    print("BRAIN SEEDS")
    print("=" * 72)

    # 1 — collisions
    owners: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
    for sk in S.SKILLS:
        for seed in sk.seeds:
            owners[B._norm(seed)].append((sk.name, seed))
    clashes = {c: v for c, v in owners.items() if len({n for n, _ in v}) > 1}
    total = sum(len(sk.seeds) for sk in S.SKILLS)
    print(f"  {total} seeds across {len(S.SKILLS)} skills -> {len(owners)} canonical forms")
    for canon, v in clashes.items():
        FAILS.append(f"CLASH {canon!r}: " + ", ".join(f"{n} <- {s!r}" for n, s in v))

    # names and stages are what the rest of the system keys on
    stages = {"view", "actuation", "calendar", "screen", "weather", "market",
              "files", "memory", "fastpath", "tools", "llm"}
    names = [sk.name for sk in S.SKILLS]
    for n, c in collections.Counter(names).items():
        if c > 1:
            FAILS.append(f"duplicate skill name {n}")
    for sk in S.SKILLS:
        if sk.stage not in stages:
            FAILS.append(f"{sk.name}: unknown stage {sk.stage!r}")
        if sk.tier not in ("low", "medium", "high"):
            FAILS.append(f"{sk.name}: unknown tier {sk.tier!r}")
        if sk.run is None and sk.name != "general":
            FAILS.append(f"{sk.name}: no runner")
        if len(sk.seeds) < 2 and sk.name != "general":
            FAILS.append(f"{sk.name}: fewer than two seeds")

    # 2 — every seed executable by its own skill
    va = T.build_assistant()
    unexec = []
    with T.NoSideEffects(va):
        va.work_mode = True
        for sk in S.SKILLS:
            if sk.run is None:
                continue
            for seed in sk.seeds:
                try:
                    ok = sk.slots(va, seed) is not None
                except Exception as exc:
                    ok = False
                    seed = f"{seed} (raised {exc})"
                if not ok:
                    unexec.append(f"{sk.name} <- {seed!r}")
    for u in unexec:
        FAILS.append(f"seed refused by its own skill: {u}")

    # 3 — held-out overlap, REPORTED. Short commands collide by construction
    # once politeness and the wake word are stripped; test_brain_intent
    # excludes those sentences from scoring rather than pretending.
    for name in ("brain_paraphrases.txt", "brain_paraphrases_v2.txt"):
        f = TESTS_DIR / name
        if not f.exists():
            continue
        hits = [line.split("==>")[0].strip() for line in f.read_text().splitlines()
                if "==>" in line and not line.startswith("#")
                and B._norm(line.split("==>")[0].strip()) in owners]
        print(f"  {name}: {len(hits)} sentence(s) share a canonical form with a "
              f"seed (excluded from its score)")

    if FAILS:
        print(f"\n  {len(FAILS)} FAILURE(S):")
        for f in FAILS:
            print(f"    ✗ {f}")
        return 1
    print("\n  ALL PASS: no cross-skill clashes, every seed executable by its own skill.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
