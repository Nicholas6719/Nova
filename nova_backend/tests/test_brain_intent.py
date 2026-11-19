#!/usr/bin/env python3
"""
What he SAYS, against where Nova sends it — the brain versus the regex chain.

Ported in spirit from JARVIS tests/test_intent.py, whose header is the reason
this exists: testing a router with its own seed phrases proves only that the
seeds are spelled right. So the brain is scored four ways:

  corpus       the routing corpus (adversarial_phrases.txt). The brain is
               SEEDED from it, so this measures parity, not understanding.
  held-out     5-fold: each fifth of the corpus is removed from the seeds and
               scored by a brain that never saw it. The honest corpus number.
  paraphrases  tests/brain_paraphrases.txt — new sentences written before any
               skill existed. Scored, never tuned against.
  real         his own sentences from Nova's history, labelled by hand. Lives
               in NOVA_DATA_DIR (the repo is public), so it runs only on his
               Mac; sentences that are also seeds are excluded from its score.

Each sentence is routed three ways:
  regex   the chain as it stands (tests/test_routing_corpus.py route_of)
  brain   production order with the regex stages 2e-8 REMOVED: system
          commands, the 2c view regex, then the brain, else conversation
  hybrid  what ships: the brain first, the regex chain behind it

A WRONG ACTION IS WORSE THAN A MISS, and is counted apart from one. A miss
falls to conversation (or an honest "I can't"); a wrong action does the wrong
thing on his Mac while he watches. The gate:
  - hybrid >= regex on the corpus (Stage 1's done-criterion), and
  - hybrid adds NO wrong action that regex did not make, on any set.

Hermetic: a throwaway NOVA_DATA_DIR (set by test_routing_corpus on import),
the brain's own database inside it, and tools probed only by dry-run detect.

Run:  python tests/test_brain_intent.py          (add -v for every sentence)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR))
sys.path.insert(0, str(TESTS_DIR.parent))

import test_routing_corpus as T          # noqa: E402  (sets NOVA_DATA_DIR first)

VERBOSE = "-v" in sys.argv
ACTING = {"tools", "files", "calendar", "screen", "view", "market", "actuation",
          "memory", "fastpath", "weather"}
QUIET = {"llm", "unsupported"}            # neither does anything on his Mac


def load_file(path: Path) -> list[tuple[str, str]]:
    out = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "==>" not in line:
            continue
        phrase, _, expected = line.partition("==>")
        expected = expected.strip()
        if expected and expected != "?":
            out.append((phrase.strip(), expected))
    return out


def real_path() -> Path:
    # NOT the throwaway data dir: his real one, read-only, where the file lives.
    return (Path.home() / "Library" / "Application Support" / "Nova"
            / "brain_real_utterances.txt")


_EMBEDS: dict = {}      # one embedding per distinct sentence across all brains


def make_brain(va, exclude: set | None = None, tag: str = "brain"):
    """A fresh brain over the real SKILLS, optionally with some sentences
    removed from every skill's seeds (for held-out scoring)."""
    import copy
    import brain as B
    import skills as S
    skills = S.SKILLS
    if exclude:
        drop = {B._norm(x) for x in exclude}
        skills = []
        for sk in S.SKILLS:
            sk2 = copy.copy(sk)
            sk2.seeds = [s for s in sk.seeds if B._norm(s) not in drop]
            skills.append(sk2)
    db = Path(os.environ["NOVA_DATA_DIR"]) / f"{tag}.db"
    if db.exists():
        db.unlink()
    b = B.Brain(db, skills, va.config, va=va, embed_cache=_EMBEDS)
    b.load()
    return b


def brain_route(va, brain, text: str, hybrid: bool) -> tuple[str, str]:
    """(stage, detail). Production order: system commands, the 2c view regex,
    the brain, then (hybrid) the regex chain or (brain) conversation."""
    import nova as nova_mod
    if nova_mod.find_signoff(text)[1] is not None:
        lead = nova_mod._content_before_sleep(text, nova_mod.find_signoff(text)[0])
        return ("signoff" if not lead else "signoff+content"), ""
    if va.views.detect_intent(text) is not None:
        return "view", "regex-2c"
    d = brain.decide(text, va)
    if d is not None:
        return d.skill.stage, f"{d.skill.name}@{d.confidence:.2f}"
    if brain.conversation:
        # Confidently conversation: the regex command stages are skipped, the
        # unsupported-action refusal is not (see nova.py).
        m = brain.last_match or {}
        return ("unsupported" if nova_mod._is_unsupported_action(text) else "llm",
                f"conversation@{m.get('confidence')}")
    if hybrid:
        return T.route_of(va, text), "fallback"
    m = brain.last_match or {}
    return "llm", f"none ({m.get('skill')}@{m.get('confidence')})"


def ok(expected: str, got: str) -> bool:
    if expected == "signoff":
        return got.startswith("signoff")
    if expected == "no-signoff":
        return not got.startswith("signoff")
    return got == expected


def kind(expected: str, got: str) -> str:
    if ok(expected, got):
        return "ok"
    if got in ACTING:
        return "WRONG"            # did something, and not what he meant
    return "miss"


def score(va, brain, data, label: str, results: dict) -> dict:
    rows = {"regex": [], "brain": [], "hybrid": []}
    detail = []
    for phrase, expected in data:
        r = T.route_of(va, phrase)
        b, bd = brain_route(va, brain, phrase, hybrid=False)
        h, hd = brain_route(va, brain, phrase, hybrid=True)
        rows["regex"].append(kind(expected, r))
        rows["brain"].append(kind(expected, b))
        rows["hybrid"].append(kind(expected, h))
        detail.append((phrase, expected, r, b, bd, h, hd))
    summary = {k: (v.count("ok"), v.count("miss"), v.count("WRONG"), len(v))
               for k, v in rows.items()}
    results[label] = (summary, detail)
    return summary


def fmt(s) -> str:
    okc, miss, wrong, n = s
    return f"{okc:>3}/{n:<3} ({100 * okc / max(n, 1):5.1f}%)  miss {miss:>2}  WRONG {wrong:>2}"


def main() -> int:
    t0 = time.time()
    va = T.build_assistant()
    corpus = T.load_corpus()
    para = load_file(TESTS_DIR / "brain_paraphrases.txt")
    real_all = load_file(real_path())

    results: dict = {}
    fails: list[str] = []
    with T.NoSideEffects(va):
        brain = make_brain(va)
        seed_canon = set(brain.canonical_seeds())
        import brain as B
        real = [(p, e) for p, e in real_all if B._norm(p) not in seed_canon]

        # A held-out sentence whose canonical form IS a seed is not held out
        # ("can you repeat that" and the seed "repeat that" both normalise to
        # "repeat that"). Excluded from the held-out sets' scores, and counted.
        para_all = para
        para = [(p, e) for p, e in para if B._norm(p) not in seed_canon]
        v2_all = load_file(TESTS_DIR / "brain_paraphrases_v2.txt")
        v2 = [(p, e) for p, e in v2_all if B._norm(p) not in seed_canon]
        excluded = {"paraphrases": len(para_all) - len(para),
                    "paraphrases-v2": len(v2_all) - len(v2),
                    "real": len(real_all) - len(real)}

        score(va, brain, corpus, "corpus", results)
        score(va, brain, para, "paraphrases", results)
        if v2 and "--sealed" not in sys.argv:
            score(va, brain, v2, "paraphrases-v2", results)
        if real:
            score(va, brain, real, "real", results)

        # 5-fold held-out over the corpus
        folds = [corpus[i::5] for i in range(5)]
        agg = {"regex": [0, 0, 0, 0], "brain": [0, 0, 0, 0], "hybrid": [0, 0, 0, 0]}
        held_detail = []
        for i, fold in enumerate(folds):
            fb = make_brain(va, exclude={p for p, _ in fold}, tag=f"fold{i}")
            tmp: dict = {}
            s = score(va, fb, fold, "fold", tmp)
            held_detail += tmp["fold"][1]
            for k in agg:
                for j in range(4):
                    agg[k][j] += s[k][j]
        results["held-out"] = ({k: tuple(v) for k, v in agg.items()}, held_detail)

    # ── report ──
    print("=" * 78)
    print("BRAIN vs REGEX — where each sentence ends up")
    print("=" * 78)
    print(f"  brain: {brain.example_count} examples, {len(set(brain._skills))} skills, "
          f"threshold {brain.threshold}  (built in {time.time() - t0:.1f}s)")
    for label in ("corpus", "held-out", "paraphrases", "paraphrases-v2", "real"):
        if label not in results:
            print(f"\n  {label:<12} (not run on this machine)")
            continue
        summary, detail = results[label]
        ex = excluded.get(label)
        print(f"\n  {label}" + (f"   ({ex} excluded: canonical form is a seed)" if ex else ""))
        for k in ("regex", "brain", "hybrid"):
            print(f"    {k:<7} {fmt(summary[k])}")
        # the gates
        rx, hy = summary["regex"], summary["hybrid"]
        new_wrong = [d for d in detail
                     if kind(d[1], d[5]) == "WRONG" and kind(d[1], d[2]) != "WRONG"]
        if label == "corpus" and hy[0] < rx[0]:
            fails.append(f"corpus: hybrid {hy[0]} < regex {rx[0]}")
        for d in new_wrong:
            fails.append(f"{label}: NEW WRONG ACTION  {d[0]!r}  expected {d[1]}, "
                         f"hybrid {d[5]} ({d[6]}), regex {d[2]}")
        if VERBOSE or label != "corpus":
            for phrase, exp, r, b, bd, h, hd in detail:
                if VERBOSE or not (ok(exp, r) and ok(exp, h)):
                    flag = ("+" if ok(exp, h) and not ok(exp, r) else
                            "-" if ok(exp, r) and not ok(exp, h) else
                            "=" if ok(exp, h) else "x")
                    print(f"      {flag} {phrase[:52]:<52} exp {exp:<10} "
                          f"regex {r:<10} hybrid {h:<10} {hd}")
    # "Did you mean ...?" on CONVERSATION is an interruption, not help. With
    # JARVIS's raw ask bar alone, all 6 offers on his real turns were wrong.
    with T.NoSideEffects(va):
        va.brain = brain
        print("\n  did you mean ...? on sentences that are conversation:")
        for label, data in (("corpus", corpus), ("real", real_all),
                            ("paraphrases", para_all), ("paraphrases-v2", v2_all)):
            if not data:
                continue
            bad = []
            for p, e in data:
                if e != "llm" or va.views.detect_intent(p) is not None:
                    continue
                if brain.decide(p, va) is None and brain.unsure \
                        and not brain.conversation and T.route_of(va, p) in QUIET:
                    bad.append(f"{p!r} -> {brain.unsure['skill']}")
            n = sum(1 for _, e in data if e == "llm")
            print(f"    {label:<15} {len(bad)} of {n}" + ("" if not bad else ":"))
            for x in bad:
                print(f"      {x}")
            if bad and label in ("corpus", "real"):
                fails.append(f"{label}: 'did you mean' asked on conversation {bad}")
    print("\n  legend: + brain fixed it   - brain broke it   x both wrong")
    if fails:
        print("\n  GATE FAILURES:")
        for f in fails:
            print(f"    ✗ {f}")
        return 1
    print("\n  GATES PASS: hybrid >= regex on the corpus, and no new wrong actions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
