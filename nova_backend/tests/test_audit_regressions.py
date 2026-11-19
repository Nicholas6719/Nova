#!/usr/bin/env python3
"""
Every defect the 2026-11-19 audit of feat/brain proved, kept so none comes back.

Four independent reviews (router, pipeline, Stage 4 modules, Stage 3 + the
intent matchers) each proved defects by running the code; every check below is
one of them, written against the behaviour that was wrong. Each was confirmed
FAILING on the pre-fix commit (ae7173b) before it was fixed (rule 6).

Fidelity: real modules and the real _handle_turn_impl (stage runners recorded,
as in test_brain_learning); tools only ever as detect() dry runs; a throwaway
NOVA_DATA_DIR; no network (Wikipedia canned); no model.

Run:  python tests/test_audit_regressions.py
"""
from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
import threading
import time
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR))
sys.path.insert(0, str(TESTS_DIR.parent))

import test_routing_corpus as T          # noqa: E402  (throwaway NOVA_DATA_DIR first)
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
    return bool(cond)


def tmpdb(name: str) -> Path:
    return Path(tempfile.mkdtemp(prefix="nova-audit-")) / name


def said(r):
    return (r or {}).get("said")


def main() -> int:
    # ── arithmetic (Stage 4) ─────────────────────────────────────────────────
    print("── arithmetic: never a wrong number")
    import mathskill as M
    for text, want in [
        ("what is two point one five times 2", "2.15 times 2 is 4.3."),
        ("what's three point one four times two", "3.14 times 2 is 6.28."),
        ("what's 7 point zero five plus 1", "7.05 plus 1 is 8.05."),
        ("what is one two three plus 1", "123 plus 1 is 124."),
        ("what's nineteen eighty four plus 1", "1,984 plus 1 is 1,985."),
        ("what's twenty twenty five minus 2000", "2,025 minus 2,000 is 25."),
        ("what is a hundred and five plus 2", "105 plus 2 is 107."),
        ("what's 1 divided by 50000", "1 divided by 50,000 is 0.00002."),
        ("what's minus 3 squared", "Minus 3 to the power of 2 is -9."),
        ("what is minus 7 mod 3", "Minus 7 mod 3 is 2."),
    ]:
        check(said(M.slots(text)) == want, f"{text!r}", f"got {said(M.slots(text))!r}")
    r = M.slots("what is 10 to the power of 16")
    check(r and "e+" not in said(r), "a huge result is never spoken in exponent notation", said(r))
    for text in ("What is 9/11?", "what is 24/7", "what is 4x4", "set the volume to 50 percent"):
        check(M.slots(text) is None, f"not a sum: {text!r}", f"{said(M.slots(text))}")
    try:
        M.slots("what is minus 8 to the power of 0.5")
        check(True, "a complex power never raises")
    except Exception as exc:
        check(False, "a complex power never raises", repr(exc))
    r = M.slots("my bill was $50 and tax was 8 percent, how much should I tip")
    check(r and "8 percent tip" not in said(r), "the tax rate is not the tip rate", said(r))
    r = M.slots("what's a 120 percent tip on $10")
    check(r and said(r).startswith("A 120 percent tip"), "a three-digit rate is read whole", said(r))
    r = M.slots("what's 0 c in k")
    check(r and "kelvin" in said(r), "'k' beside a temperature is kelvin", said(r))

    # ── honesty (Stage 4) ────────────────────────────────────────────────────
    print("\n── honesty: the common false promises, and no true sentence replaced")
    import honesty as H
    for s in ["I'll set a reminder for 3pm.", "I'll add that to your calendar.",
              "Let me set a timer for you.", "I'll open it for you.",
              "I'll send that message to Sarah.", "I'll play some jazz for you.",
              "Reminder set for 5pm.", "Done! Your alarm is set for 7.",
              "I'm setting a reminder now.", "I've taken care of it.", "Consider it done."]:
        check(H.unbacked(s) is not None, f"caught: {s!r}")
    for s in ["I'll let you know what I think: it's a good plan.",
              "I'll tell you if I remember the name.", "I've started to think you're right.",
              "I have called it a day too.", "Let me know if you need anything.",
              "Let me think about that."]:
        check(H.unbacked(s) is None, f"left alone: {s!r}", f"{H.unbacked(s)}")

    # ── notes (Stage 4) ──────────────────────────────────────────────────────
    print("\n── forgetting a note forgets THAT note")
    from memory import NovaMemory
    mem = NovaMemory(tmpdb("mem.db"))
    mem.add_note("I parked on level 3")
    check(mem.forget_note("I parked on level 4") == 0, "a different number is not the same note")
    mem.add_note("I like tea")
    check(mem.forget_note("I don't like tea") == 0, "a negation is not the same note")
    mem.add_note("my gym class is on friday")
    check(mem.forget_note("my gym class is on monday") == 0, "a different day is not the same note")
    check(mem.forget_note("I parked on level 3") == 1, "...and the right note still goes")

    # ── standing watches (Stage 4) ───────────────────────────────────────────
    print("\n── watches: the direction comes from the reading")
    import standing as S
    lv = {"battery": 80.0, "cpu": 40.0, "memory": 50.0}
    w = S.StandingWatches(tmpdb("w.db"), lambda m: lv.get(m), lambda k, t, key: None)
    r = w.handle("let me know when my battery is at 20 percent")
    check(r and "drops to 20" in r, "'battery is at 20' from 80 watches it falling", r)
    r = w.handle("tell me when my cpu gets to 10")
    check(r and "drops below 10" in r, "'cpu gets to 10' from 40 watches it falling", r)
    check(S.parse("tell me if my cpu goes above 1000") is None, "1000 is not read as 100")
    check(not S.claims("what are you watching on netflix"), "a Netflix question is not about watches")
    n = len(w.active())
    w.handle("let me know when my battery is at 20 percent")
    check(len(w.active()) == n, "the same watch twice is kept once")

    # ── request shape ────────────────────────────────────────────────────────
    print("\n── request shape: statements are not requests; terse requests are")
    import shape
    for t in ("battery is at like five percent lol", "my battery died yesterday",
              "the weather has been so weird this year"):
        check(not shape.asks(t), f"statement: {t!r}")
    for t in ("apple stock price", "boston weather", "the weather this weekend", "battery percentage"):
        check(shape.asks(t), f"terse request: {t!r}")

    # ── Stage 3 pure functions ───────────────────────────────────────────────
    print("\n── world facts: privacy, dimensions, realm 2, grounding")
    import world_facts as W
    check(W.private_reason("when is Sarah's birthday", {"Sarah"}) is not None,
          "a stored name is caught in the possessive")
    check(W.private_reason("when is mom's birthday", set()) is not None, "'mom' is his family")
    check(W.content_key("what city is the eiffel tower in") !=
          W.content_key("what country is the eiffel tower in"), "city and country are different questions")
    check(W.content_key("who was the wife of henry viii") !=
          W.content_key("who was the first wife of henry viii"), "'first' is part of the question")
    for q in ("what is the most populous country", "highest grossing film of all time",
              "how many films has spielberg directed", "what is the gdp of japan"):
        check(W.changeable(q), f"changeable: {q!r}")
    check(not W.changeable("how many legs does a spider have"), "...and a spider's legs are not")
    check(not W.grounded("The Titanic had three funnels.", "The Titanic had four funnels."),
          "'three' is not grounded in 'four'")
    check(not W.grounded("Buzz Aldrin was the first person on the Moon.",
                         "Buzz Aldrin was the second person to walk on the Moon."),
          "'first' is not grounded in 'second'")
    f = W.WorldFacts(tmpdb("wf.db"), {}, embedder=None, search=lambda q: {"ok": False, "error": "x"})
    f._log_lookup("Who wrote Hamlet", "night", "found")
    check(f.looked_up("who wrote hamlet"), "a question sent once is never sent again, whatever its case")
    import wiki
    f2 = W.WorldFacts(tmpdb("wf2.db"), {}, embedder=None)
    wiki._not_before = time.time() + 60
    try:
        r = f2.reference("who painted the mona lisa")
    finally:
        wiki._not_before = 0.0
    check(r.get("reason") == "rate limited" and not f2.looked_up("who painted the mona lisa"),
          "while Wikipedia says wait, nothing is sent and nothing is logged as looked up")

    # ── the assistant: tools, market, maps, router, pipeline ─────────────────
    va = T.build_assistant()
    va.work_mode = False
    with T.NoSideEffects(va):
        print("\n── matchers (dry runs)")
        for t in ("which planet is closest to the sun", "what's the closest country to australia",
                  "how far is alpha centauri", "how far is the nearest black hole",
                  "I forgot to lock my computer yesterday", "how do I lock my screen",
                  "I want to make a song about growing up", "is something on fire"):
            check(va.tools.detect(t) is None, f"tools does not claim: {t!r}", f"{va.tools.detect(t)}")
        for t in ("lock my mac", "turn the music down", "how far is the nearest cvs"):
            check(va.tools.detect(t) is not None, f"tools still claims: {t!r}")
        check(va.market._subject("what's Sarah at") is None, "'what's Sarah at' is not a ticker lookup")
        check((va.market._subject("what's Nvidia trading at") or "").lower() == "nvidia",
              "'what's Nvidia trading at' names Nvidia", f"{va.market._subject('what is Nvidia trading at')}")

        print("\n── the router")
        b = L.fresh_brain(va, "audit-router")
        rec = L.Recorder(va)
        ran = []
        orig_match = va.tools.match
        va.tools.match = lambda text: (ran.append(text), "ok")[1]
        d = b.decide("shuffle off", va)
        if d is not None:
            d.skill.run(va, "shuffle off", d.args)
        check(not ran or ran[-1] != "turn on shuffle", "'shuffle off' never runs 'turn on shuffle'",
              f"ran {ran}")
        va.tools.match = orig_match          # detect() dry-runs match(): never stubbed below
        for t in ("skip back a song", "skip ahead in this song"):
            d = b.decide(t, va)
            check(d is None or d.skill.name != "tools.music_next", f"{t!r} is not 'next track'",
                  f"{d and d.skill.name}")

        for t in ("minimize it", "take it over"):
            d = b.decide(t, va)
            check(d is None or d.skill.name != "view.work", f"{t!r} does not enter work mode",
                  f"{d and d.skill.name}")
        b.decide("volume all the way down", va)
        check(not b.unsure or b.unsure["skill"] != "tools.volume_up",
              "'volume all the way down' never offers 'turn the volume up'", f"{b.unsure}")
        b.learn("my calendar for tomorrow?", "cal.upcoming", source="user", va=va)
        b2 = L.reload_brain(va, b)
        check("my calendar for tomorrow" in b2._texts, "a 'yes, that's what I meant' survives a restart")
        b2.learn("how's the weather looking out there today", "weather", source="learned", va=va)
        import copy
        cfg = copy.deepcopy(va.config)
        cfg.setdefault("weather", {})["enabled"] = False
        saved = va.config
        va.config = cfg
        import brain as B
        import skills as SK
        b3 = B.Brain(b2.db_path, b2.skills, cfg, va=va, embed_cache=None)
        b3.load()
        va.config = saved
        check("how's the weather looking out there today" in b3._texts,
              "one start with weather off does not delete learned weather examples")
        b3.learn("is it chilly outside right now", "weather", source="night_school", va=va)
        if "is it chilly outside right now" in b3._texts:
            i = b3._texts.index("is it chilly outside right now")
            m = {"text": b3._texts[i], "skill": "weather", "source": "night_school"}
            check(b3.unlearn(m) == "weather", "a night-school example can be un-learned")
        else:
            check(False, "setup: a night-school example is learned")
        va.brain = b3
        b3.decide("what's my cpu", va)
        mine = b3.last_match
        t = threading.Thread(target=lambda: b3.decide("tell me a joke", va))
        t.start()
        t.join()
        check(b3.last_match is mine, "another thread's decide() never changes this turn's verdict")
        va.work_mode = False
        seen = []
        real_detect = va.actuation.detect_intent

        def spy(text, assume_work_mode=False):
            seen.append(va.work_mode)
            return real_detect(text, assume_work_mode=assume_work_mode)
        va.actuation.detect_intent = spy
        SK._act_probe(va, "type hello world")
        va.actuation.detect_intent = real_detect
        check(seen and seen[0] is False, "probing a typing sentence never switches work mode on")

        print("\n── the pipeline")
        rec = L.Recorder(va)
        va.brain = b3
        locked = []
        va._brain_offer = {"text": "lock it", "skill": "tools.lock", "args": {},
                           "ts": time.time() - 3600}
        b3.by_name["tools.lock"].run = lambda va_, t_, a_: locked.append(t_)
        calls, out = rec.turn("right, what's on my calendar today")
        check(not locked, "an hour-old 'did you mean lock' is not answered by 'right, ...'",
              f"ran {locked}")
        va._brain_offer = {"text": "lock it", "skill": "tools.lock", "args": {}, "ts": time.time()}
        rec.turn("go home")
        check(va._brain_offer is None, "any other answer retires the offer")

        calls, out = rec.turn("when I say I'm tired, I mean it")
        check(not any("don't know how" in x for x in out), "'when I say I'm tired, I mean it' is not a lesson",
              f"{out}")
        steps_run = []
        va._run_view = lambda intent, text: steps_run.append(("view", text))
        va._run_step("go home")
        va._run_step("what's 15 percent of 80")
        check(steps_run and steps_run[0][0] == "view", "a taught 'go home' step goes home", f"{steps_run}")
        check(any("15 percent of 80" in x for x in rec.said), "a taught arithmetic step is worked out",
              f"{rec.said}")

        import turnstats as TS
        for t in ("what have you learned about me", "why that restaurant closed down"):
            check(not TS.wants_report(t) and not TS.wants_explanation(t), f"a real question: {t!r}")
        check(not W.wants_source("are you sure it's going to rain tomorrow"),
              "'are you sure it's going to rain' is a weather question")

        slow = []

        def slow_respond(text):
            time.sleep(0.4)
            slow.append(text)
        va._respond = slow_respond
        va.brain = None
        va._handle_turn_impl("remember that my favorite color is blue")
        row = va._last_turn
        check(row and row["path"].startswith("regex:") and (row["decide_ms"] or 0) < 300,
              "a regex turn's decide time excludes the spoken answer", f"{row}")
        va._respond = lambda text: rec.said.append(text)

        class _Boom:
            ready = True

            def match_command(self, text):
                raise RuntimeError("embedding failed")
            by_name = {}

            def decide(self, *a, **k):
                return None
        va.brain = _Boom()
        va._brain_offer = None
        rec.said.clear()
        try:
            va._handle_turn_impl("what's 2 plus 2")
            check(any("4" in x for x in rec.said), "a failing brain lesson costs the lesson, never the turn",
                  f"{rec.said}")
        except Exception as exc:
            check(False, "a failing brain lesson costs the lesson, never the turn", repr(exc))
        va.brain = None

        silent_seen = []
        va._handle_turn_impl = lambda text: silent_seen.append((text, va._silent_turn))
        va._run_turn("typed one", True)
        va._run_turn("spoken one", False)
        check(silent_seen == [("typed one", True), ("spoken one", False)],
              "typed or spoken travels with each turn", f"{silent_seen}")

        print("\n── private names are collected from every kind of fact")
        va2 = T.build_assistant()
        va2.memory.upsert_fact("relationship", "parrot", "Kiwi", source="llm")
        va2.memory.upsert_fact("family", "sister", "sarah", source="llm")
        va2.memory.add_note("Remember that Jake's birthday is in May")
        names = {n.lower() for n in va2._private_names()}
        check({"kiwi", "sarah", "jake's"} & names >= {"kiwi", "sarah"} and
              any(n.startswith("jake") for n in names), "Kiwi, sarah and Jake are protected",
              f"{sorted(names)}")

    print(f"\n  {PASS}/{PASS + FAIL} checks passed")
    if FAILURES:
        print("\n  FAILURES:")
        for x in FAILURES:
            print(f"    ✗ {x}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
