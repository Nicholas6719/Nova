#!/usr/bin/env python3
"""
Stage 2 of the brain: it learns, it asks when it nearly knows, he can teach it
phrases, and he can correct it — and none of that can go wrong silently.

Driven through the REAL _handle_turn_impl, so every stage ordering question is
answered by the pipeline Nova runs, not by a copy of it (rule 1). Side effects
are stubbed at the stage runners — the points where a decision turns into an
action — and at subprocess (rule 2): what gets recorded is WHICH runner was
called with WHAT, so a test sees where a sentence ended up (BRAIN.md mistake
#6: a guard is not an outcome).

Every rule tested here came from a JARVIS mistake (docs/BRAIN.md):
  #1  an uncommitted write held SQLite's write lock for the life of the process
  #2  stored rows are canonical — a validator that cannot read "N" deleted
      every confirmed example on the next boot
  #4  the model's (here: the regex's) choice is not ground truth — learn only
      what the skill could execute, never a bare answer
  #5  the runner-up / near miss must be kept, because it is the question to ask

Hermetic: a throwaway NOVA_DATA_DIR and brain database; nothing is spoken,
launched or written outside it.

Run:  python tests/test_brain_learning.py
"""
from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR))
sys.path.insert(0, str(TESTS_DIR.parent))

import test_routing_corpus as T      # noqa: E402  (throwaway NOVA_DATA_DIR first)
import test_brain_intent as I        # noqa: E402

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
    return bool(cond)


class Recorder:
    """Stub every point where a routing decision becomes an action."""

    RUNNERS = ("_run_view", "_run_actuation", "_run_calendar", "_run_screen",
               "_run_weather", "_run_market", "_run_files")

    def __init__(self, va):
        self.va = va
        self.calls: list[tuple] = []
        self.said: list[str] = []
        for name in self.RUNNERS:
            setattr(va, name, self._mk(name))
        va._finish_tool = lambda resp: self.calls.append(("_finish_tool", resp))
        va._respond = lambda text: self.said.append(text)
        va._stream_response = lambda text, rag_context="": self.calls.append(("llm", text))
        va.set_state = lambda *a, **k: None
        va._rag_ready = False

    def _mk(self, name):
        def run(intent, text):
            self.calls.append((name, intent, text))
        return run

    def turn(self, text):
        self.calls.clear()
        self.said.clear()
        self.va._handle_turn_impl(text)
        return list(self.calls), list(self.said)


def fresh_brain(va, tag, exclude=()):
    b = I.make_brain(va, exclude=set(exclude), tag=tag)
    va.brain = b
    return b


def reload_brain(va, old):
    """A new Brain over the SAME database: what the next start of Nova sees."""
    import brain as B
    b = B.Brain(old.db_path, old.skills, va.config, va=va, embed_cache=I._EMBEDS)
    b.load()
    va.brain = b
    return b


def main() -> int:
    va = T.build_assistant()
    with T.NoSideEffects(va):
        rec = Recorder(va)

        # ── 1. learn() refuses what it must refuse ───────────────────────────
        print("\n── learn(): the bar")
        b = fresh_brain(va, "learn-bar")
        check(not b.learn("yes", "tools.music_play"), "a bare answer is never learned")
        check(not b.learn("thanks", "fast.greet"), "a thank-you is never learned")
        check(not b.learn("can you please find the document I was working on with "
                          "Sarah last Tuesday about the quarterly budget numbers",
                          "files.find"), "more than 14 words is never learned")
        check(not b.learn("hello there friend", "memory.save"),
              "a sentence the skill cannot execute is never learned")
        check(not b.learn("what's on my calendar today", "tools.battery"),
              "a sentence that is another skill's seed is never learned")
        check(not b.learn("tell me a joke", "fast.time"),
              "never learned when the brain is sure it is conversation")
        check(not b.learn("my battery died on the way to work", "tools.battery"),
              "a regex false positive is not learned (the brain's own guess disagrees)")
        check(b.learn("set a timer for 25 minutes", "tools.timer_set"),
              "an executable new phrasing IS learned")
        check(not b.learn("set a timer for 25 minutes", "tools.timer_set"),
              "learning the same thing twice stores it once")

        # ── 2. persistence, and JARVIS mistake #2 ────────────────────────────
        print("\n── persistence across a restart")
        b2 = reload_brain(va, b)
        check("set a timer for N minutes" in b2._texts,
              "a learned row with a number placeholder survives the next load",
              "(validator must probe N with a stand-in value)")
        d = b2.decide("set a timer for 40 minutes", va)
        check(d is not None and d.skill.name == "tools.timer_set",
              "...and now routes as a reflex", f"got {d and d.skill.name}")

        # ── 3. JARVIS mistake #1: no lingering write lock ────────────────────
        print("\n── SQLite: every write committed")
        with sqlite3.connect(b2.db_path) as db:
            db.execute("INSERT INTO brain_examples (ts, text, skill, source, embedding) "
                       "VALUES (0, 'stale seed row', 'tools.cpu', 'seed', x'00')")
        b3 = reload_brain(va, b2)           # drops the stale seed on load
        try:
            other = sqlite3.connect(b3.db_path, timeout=0.2)
            other.execute("BEGIN IMMEDIATE")
            other.execute("CREATE TABLE IF NOT EXISTS lock_probe (x)")
            other.commit()
            other.close()
            locked = False
        except sqlite3.OperationalError as exc:
            locked = "locked" in str(exc)
        check(not locked, "another writer can take the lock right after a load that "
                          "dropped rows (the 2,600-message bug)")
        b3.reject("some sentence", "tools.cpu")
        b3.learn("set a timer for 55 seconds", "tools.timer_set")
        try:
            other = sqlite3.connect(b3.db_path, timeout=0.2)
            other.execute("BEGIN IMMEDIATE")
            other.commit()
            other.close()
            locked = False
        except sqlite3.OperationalError:
            locked = True
        check(not locked, "...and right after learn() and reject()")

        # ── 4. learning from the regex fallback, end to end ──────────────────
        print("\n── the regex chain teaches the brain")
        candidates = ["find the screenshot from yesterday", "what reminders do I have",
                      "how far is the nearest CVS", "put the invoice in Documents",
                      "what's in my downloads folder", "what is Apple stock trading at"]
        learned_one = None
        for sent in candidates:
            b = fresh_brain(va, "teach-" + sent[:10].replace(" ", "_"), exclude=[sent])
            if b.decide(sent, va) is not None:
                continue                      # the brain already knows it: no lesson
            calls, said = rec.turn(sent)
            claimed = [c for c in calls if c[0] != "llm"]
            if not claimed:
                continue
            row = next((i for i, t in enumerate(b._texts) if t == __import__("brain")._norm(sent)), None)
            if row is not None:
                learned_one = (sent, b._skills[row], b._sources[row], b)
                break
        check(learned_one is not None, "a sentence the brain fell through on and a regex "
                                       "stage claimed is learned",
              f"(none of {len(candidates)} candidates)")
        if learned_one:
            sent, skill, source, b = learned_one
            check(source == "learned", f"...stored as source=learned ({skill})")
            d = b.decide(sent, va)
            check(d is not None and d.skill.name == skill,
                  "...and the next time it is a reflex", f"got {d and d.skill.name}")

        # ── 5. "did you mean ...?" ───────────────────────────────────────────
        print("\n── did you mean ...?")
        b = fresh_brain(va, "dym")
        pool = [p for p, _ in I.load_file(TESTS_DIR / "brain_paraphrases.txt")] + \
               [p for p, _ in I.load_file(TESTS_DIR / "brain_paraphrases_v2.txt")]
        asked = None
        for sent in pool:
            if b.decide(sent, va) is None and b.unsure and not b.conversation \
                    and va._regex_route(sent) == "llm":
                asked = (sent, b.unsure["skill"])
                break
        check(asked is not None, "some sentence produces a near miss worth asking about")
        if asked:
            sent, skill = asked
            calls, said = rec.turn(sent)
            check(said and said[-1].startswith("Did you mean"),
                  f"a near miss asks instead of guessing: {sent!r}", f"said {said}")
            check(not any(c[0] == "llm" for c in calls), "...and does not go to the LLM")
            calls, said = rec.turn("yes")
            ran = [c for c in calls if c[0] != "llm"]
            check(ran or said, "a yes runs the skill", f"calls {calls} said {said}")
            check(b.was_rejected(sent, skill) is False, "...is not a rejection")
            d = b.decide(sent, va)
            check(d is not None and d.skill.name == skill,
                  "...and teaches the brain: next time it is a reflex",
                  f"got {d and d.skill.name}")
            src = b._sources[b._texts.index(__import__("brain")._norm(sent))] \
                if __import__("brain")._norm(sent) in b._texts else None
            check(src == "user", "...stored as source=user (his word)", f"got {src}")

            # the no path, on a second near miss
            b = fresh_brain(va, "dym-no")
            sent2 = None
            for s2 in pool:
                if b.decide(s2, va) is None and b.unsure and not b.conversation \
                        and va._regex_route(s2) == "llm":
                    sent2, sk2 = s2, b.unsure["skill"]
                    break
            if sent2:
                rec.turn(sent2)
                calls, said = rec.turn("no")
                check(b.was_rejected(sent2, sk2), "a no is remembered")
                calls, said = rec.turn(sent2)
                check(not (said and said[-1].startswith("Did you mean")) or sk2 not in said[-1],
                      "...the same wrong guess is not offered again", f"said {said}")
                b2 = reload_brain(va, b)
                check(b2.was_rejected(sent2, sk2), "...even after a restart")
            calls, said = rec.turn("tell me about the history of jazz")
            check(not (said and said[-1].startswith("Did you mean")),
                  "conversation is never answered with 'did you mean'")

        # ── 6. phrases he teaches by voice ───────────────────────────────────
        print("\n── taught phrases")
        b = fresh_brain(va, "teach")
        calls, said = rec.turn("when I say quiet time, turn the volume down and pause the music")
        check(said and said[-1].startswith("Got it. When you say quiet time"),
              "a phrase can be taught", f"said {said}")
        check(len(b.commands()) == 1 and len(b.commands()[0]["steps"]) == 2,
              "...with both steps stored", f"{b.commands()}")
        calls, said = rec.turn("quiet time")
        tools_runs = [c for c in calls if c[0] == "_finish_tool"]
        check(len(tools_runs) == 2, "saying it runs every step", f"calls {calls} said {said}")
        calls, said = rec.turn("it's quiet time at the library on sundays")
        check(not [c for c in calls if c[0] == "_finish_tool"],
              "a sentence that merely CONTAINS the phrase does not run it")
        calls, said = rec.turn("when I say goodnight, shut down my mac")
        check(said and "won't" in said[-1] and len(b.commands()) == 1,
              "something that cannot be undone is never put behind a shortcut",
              f"said {said}")
        calls, said = rec.turn("when I say party mode, juggle three oranges")
        check(said and "don't know how" in said[-1], "a step Nova cannot do is refused",
              f"said {said}")
        calls, said = rec.turn("what phrases have I taught you")
        check(said and "quiet time" in said[-1], "he can ask what he has taught", f"said {said}")
        b2 = reload_brain(va, b)
        check(b2.match_command("quiet time") is not None, "a taught phrase survives a restart")
        calls, said = rec.turn("forget the phrase quiet time")
        check(not va.brain.commands(), "a taught phrase can be forgotten", f"said {said}")

        # ── 7. corrections ───────────────────────────────────────────────────
        print("\n── corrections")
        b = fresh_brain(va, "correct")
        b.learn("crank the tunes", "tools.music_play", source="user", va=va)
        d = b.decide("crank the tunes", va)
        if check(d is not None and d.skill.name == "tools.music_play",
                 "setup: a taught example is a reflex"):
            calls, said = rec.turn("crank the tunes")
            calls, said = rec.turn("no, I meant turn the music up")
            check("crank the tunes" not in b._texts,
                  "'no, I meant ...' un-learns the example behind the wrong reflex")
            check(b.was_rejected("crank the tunes", "tools.music_play"),
                  "...and that sentence will not go to that skill again")
            check(any(c[0] == "_finish_tool" for c in calls),
                  "...and the corrected request is carried out", f"calls {calls}")
        calls, said = rec.turn("what's my cpu")
        seeds_before = sum(1 for s in b._sources if s == "seed")
        calls, said = rec.turn("that's wrong")
        check(sum(1 for s in b._sources if s == "seed") == seeds_before,
              "a correction never deletes a SEED")
        check(b.was_rejected("what's my cpu", "tools.cpu"),
              "...but the corrected sentence is kept away from that skill")
        check(said and said[-1].startswith("Sorry"), "...and Nova asks what he meant",
              f"said {said}")
        calls, said = rec.turn("no, I meant the other one")
        check(not any(x.startswith("Sorry") for x in said),
              "a 'no, I meant' with nothing left to correct is not treated as one",
              f"said {said} calls {calls}")

    print(f"\n  {PASS}/{PASS + FAIL} checks passed")
    if FAILURES:
        print("\n  FAILURES:")
        for f in FAILURES:
            print(f"    ✗ {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
