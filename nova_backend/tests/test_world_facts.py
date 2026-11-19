#!/usr/bin/env python3
"""
Stage 3: world facts and night school — Nova looks things up, keeps only what
can never change, never serves one question's fact for another, never sends
anything personal, and checks what she knows while he sleeps.

Fidelity:
  - REAL world_facts.WorldFacts, night_school.NightSchool, wiki.sentences and
    the real embedder (bge-small, local);
  - Wikipedia CANNED: tests/fixtures/wiki_canned.json holds real responses
    captured once (2026-11-19). Every network call is counted, so "this never
    left the machine" is a measured statement;
  - the language model SCRIPTED where a check is about the rules around it
    (does an ungrounded answer get stored?), because those must hold whatever
    the model says. tests/world_facts_live.py runs the real model and the real
    Wikipedia (NOVA_TEST_LIVE=1) — model behaviour is measured there;
  - the pipeline through the REAL _handle_turn_impl (rule 1), stage runners
    recorded, as in test_brain_learning.

Hermetic: a throwaway NOVA_DATA_DIR. Nothing is spoken; nothing leaves the
machine.

Run:  python tests/test_world_facts.py
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR))
sys.path.insert(0, str(TESTS_DIR.parent))

import test_routing_corpus as T          # noqa: E402  (throwaway NOVA_DATA_DIR first)
import test_brain_learning as L          # noqa: E402
import world_facts as W                  # noqa: E402
import night_school as N                 # noqa: E402
import wiki                              # noqa: E402

PASS = FAIL = 0
FAILURES: list[str] = []
CANNED = json.loads((TESTS_DIR / "fixtures" / "wiki_canned.json").read_text())


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


class FakeWiki:
    """Canned Wikipedia. Records every query that would have left the machine."""

    def __init__(self):
        self.sent: list[str] = []
        self.answer_for: dict[str, str] = {}   # question -> canned key
        self.fail: str | None = None

    def search(self, q):
        self.sent.append(q)
        if self.fail:
            return {"ok": False, "error": self.fail}
        key = self.answer_for.get(q.lower(), q.lower())
        pages = CANNED.get(key)
        if pages is None:
            return {"ok": True, "pages": []}
        return {"ok": True, "pages": json.loads(json.dumps(pages))}


def embedder():
    from brain import EMBED_MODEL, MODEL_DIR
    from fastembed import TextEmbedding
    if not hasattr(embedder, "m"):
        embedder.m = TextEmbedding(EMBED_MODEL, cache_dir=str(MODEL_DIR), local_files_only=True)
    return embedder.m


def scripted(answers: dict | None = None, timeless: str = "YES", compare: str = "UNCLEAR"):
    """A stand-in model: synthesis returns answers[question], the timeless check
    `timeless`, the audit comparison `compare`. Records every call."""
    calls = []

    def gen(system, user, temperature, max_tokens):
        calls.append((system, user))
        if system == W.SYNTH_SYSTEM:
            q = user.split("\n", 1)[0].replace("Question: ", "").strip()
            return (answers or {}).get(q, "UNKNOWN")
        if system == W.TIMELESS_SYSTEM:
            return timeless
        if system == W.COMPARE_SYSTEM:
            return compare
        return ""
    gen.calls = calls
    return gen


def store(tag, wikifake=None, names=None, cfg=None):
    import os
    db = Path(os.environ["NOVA_DATA_DIR"]) / f"facts-{tag}.db"
    if db.exists():
        db.unlink()
    wf = wikifake or FakeWiki()
    f = W.WorldFacts(db, {"facts": cfg or {}}, embedder=embedder,
                     names=(lambda: set(names or ())), search=wf.search)
    return f, wf


GOOD = {
    "how tall is the eiffel tower": "The Eiffel Tower is 330 metres tall.",
    "who wrote pride and prejudice": "Jane Austen wrote Pride and Prejudice.",
    "who directed jaws": "Steven Spielberg directed the 1975 film Jaws.",
    "what year did the titanic sink": "The Titanic sank on 15 April 1912.",
    "how deep is the mariana trench": "The maximum known depth of the Mariana Trench is 10,935 meters.",
    "why is the sky blue": "The sky appears blue due to a phenomenon called sky radiation.",
}


def main() -> int:
    # ── 1. what counts as a factual question ─────────────────────────────────
    print("── which questions are looked up at all")
    for t in ["how tall is the eiffel tower", "Nova, who wrote Pride and Prejudice?",
              "when did the berlin wall fall", "is a tomato a fruit", "how do magnets work",
              "what is the capital of australia", "who is the ceo of apple"]:
        check(W.is_factual_question(t), f"factual: {t!r}")
    for t in ["what's my sister's name", "how are you", "who made you", "what's the weather today",
              "what should I eat tonight", "tell me a joke", "what's the best pizza in town",
              "what time is it", "what's up", "what do you think about mondays",
              "what's the score of the game", "what's the price of gold right now",
              "where did I put my keys", "what's our plan for tomorrow"]:
        check(not W.is_factual_question(t), f"not looked up: {t!r}")

    # ── 2. privacy: nothing personal leaves the machine ──────────────────────
    print("\n── the privacy guard (counted at the network boundary)")
    f, wf = store("privacy", names={"Sarah", "Biscuit", "Coppola"})
    for t in ["who is sarah", "how old is biscuit", "who was the first coppola",
              "what's my dog's breed", "who is nicholas.coppola@example.com",
              "what is 555-123-4567"]:
        r = f.reference(t)
        check(not r["ok"], f"refused: {t!r}", r.get("reason", ""))
    check(wf.sent == [], "...and not one of them reached Wikipedia", f"sent {wf.sent}")
    f.reference("who wrote pride and prejudice")
    check(wf.sent == ["who wrote pride and prejudice"],
          "a general question is sent as asked, wake word and punctuation removed", f"{wf.sent}")
    off, wf_off = store("off", cfg={"enabled": False})
    check(not off.reference("how tall is the eiffel tower")["ok"] and wf_off.sent == [],
          "facts.enabled=false: no request at all")

    # ── 3. reference: the relevance gate ─────────────────────────────────────
    print("\n── looking up: only sentences that answer the question")
    f, wf = store("ref")
    r = f.reference("how tall is the eiffel tower")
    check(r["ok"] and "330 metres" in r["context"], "the answering sentence is in the reference",
          r.get("context", "")[:120])
    check(r["ok"] and r["sources"] and r["sources"][0]["title"] == "Eiffel Tower"
          and r["sources"][0]["revid"] > 0, "sources keep the page id and revision")
    r = f.reference("who directed jaws")
    check(r["ok"] and "Steven Spielberg" in r["context"],
          "'who directed jaws' keeps the Spielberg sentence (JARVIS's 0.80 floor dropped it)")
    check("John D." not in r.get("context", "") or "Hancock" in r.get("context", ""),
          "a sentence is never cut at an initial ('John D.')")
    wf.answer_for["how tall is the eiffel tower"] = "who directed jaws"
    r = f.reference("how tall is the eiffel tower")
    check(not r["ok"] and r.get("reason") == "nothing relevant",
          "another question's pages are refused, not handed to the model", f"{r.get('best')}")
    wf.fail = "rate limited"
    r = f.reference("how deep is the mariana trench")
    check(not r["ok"], "Wikipedia unavailable: no reference, no exception")

    # ── 4. the store: what may be kept ───────────────────────────────────────
    print("\n── intake: grounded, timeless, never a 'why', model can only veto")
    f, wf = store("intake")
    gen = scripted(GOOD)
    fid = f.consider(f.reference("how tall is the eiffel tower"), gen)
    check(fid is not None, "a grounded, timeless answer is stored")
    check(f.consider(f.reference("what's the height of the eiffel tower"), gen) is None,
          "the same fact asked another way is not stored twice")
    bad = scripted({"how deep is the mariana trench": "The Mariana Trench is 11,034 metres deep."})
    check(f.consider(f.reference("how deep is the mariana trench"), bad) is None,
          "a number that is not in the source is never stored")
    inv = scripted({"who directed jaws": "Jaws was directed by Steven Spielberg and George Lucas."})
    check(f.consider(f.reference("who directed jaws"), inv) is None,
          "...nor a name that is not in the source")
    check(f.consider(f.reference("who wrote pride and prejudice"), scripted({})) is None,
          "UNKNOWN from the model stores nothing")
    check(f.consider(f.reference("who wrote pride and prejudice"), scripted(GOOD, timeless="NO"))
          is None, "the model saying NO vetoes")
    check(f.consider(f.reference("why is the sky blue"), scripted(GOOD)) is None,
          "an explanation ('why ...') is looked up but never stored")
    wf.answer_for["who is the ceo of eiffel tower"] = "how tall is the eiffel tower"
    yes = scripted({"who is the ceo of eiffel tower": "The Eiffel Tower is 330 metres tall."})
    check(f.consider(f.reference("who is the ceo of eiffel tower"), yes) is None,
          "a changeable question is never stored, whatever the model says")
    for q in ["who is the ceo of apple", "how many moons does jupiter have",
              "who won the last world cup", "what is the newest iphone",
              "who holds the record for the 100 metres", "what is the population of japan"]:
        check(W.changeable(q), f"changeable by rule, not by the model: {q!r}")

    # ── 5. serving: the word guard ───────────────────────────────────────────
    print("\n── serving: never one question's fact for another")
    for q in GOOD:
        if q != "why is the sky blue":
            f.consider(f.reference(q), scripted(GOOD))
    sent_before = len(wf.sent)
    for q, want in [("how tall is the eiffel tower", "330"),
                    ("what's the height of the eiffel tower", "330"),
                    ("how high is the eiffel tower", "330"),
                    ("who was the author of pride and prejudice", "Austen"),
                    ("who was the director of jaws", "Spielberg"),
                    ("Nova, how tall is the Eiffel Tower?", "330")]:
        got = f.lookup(q)
        check(got is not None and want in got["answer"], f"served: {q!r}",
              f"{got and got['answer']}")
    check(len(wf.sent) == sent_before, "...without a single request")
    for q in ["how old is the eiffel tower", "where is the eiffel tower",
              "when was pride and prejudice written", "who directed jaws 2",
              "who starred in jaws", "how wide is the mariana trench",
              "how tall is the empire state building", "when did the lusitania sink",
              "who is the ceo of the eiffel tower",
              # Present-tense "who" is changeable by rule ("who's the director
              # of the FBI" has the same shape), so this paraphrase is a
              # deliberate miss: it gets a live lookup instead.
              "who's the author of pride and prejudice"]:
        got = f.lookup(q)
        check(got is None, f"NOT served: {q!r}", f"{got and got['answer']}")

    # ── 6. disputes ──────────────────────────────────────────────────────────
    print("\n── 'that's wrong' wins")
    f.dispute("how tall is the eiffel tower")
    check(f.lookup("how tall is the eiffel tower") is None, "a disputed fact is no longer served")
    check(f.consider(f.reference("how tall is the eiffel tower"), scripted(GOOD)) is None,
          "...and is not stored again later")

    # ── 7. wiki.py: rate limits and sentences ────────────────────────────────
    print("\n── wiki.py")
    import urllib.request
    real_open = urllib.request.urlopen
    hits = []

    class _H(dict):
        def get(self, k, d=None):
            return super().get(k, d)

    def fake_429(req, timeout=0):
        hits.append(req.full_url)
        raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests",
                                     _H({"Retry-After": "30"}), None)
    try:
        urllib.request.urlopen = fake_429
        wiki._not_before = 0.0
        r1 = wiki.search("how tall is the eiffel tower")
        r2 = wiki.search("who directed jaws")
    finally:
        urllib.request.urlopen = real_open
        wiki._not_before = 0.0
    check(not r1["ok"] and r1["error"] == "rate limited", "a 429 is a clean failure")
    check(not r2["ok"] and len(hits) == 1, "...and Retry-After is honoured: no second request",
          f"{len(hits)} requests")
    check(wiki.USER_AGENT and "@" not in wiki.USER_AGENT and "Coppola" not in wiki.USER_AGENT
          and "github" not in wiki.USER_AGENT.lower(), "the User-Agent identifies nobody")
    s = wiki.sentences("Jaws is a 1975 film. Its first director, John D. Hancock, left. "
                       "It cost about $9 million to make. The U.S. release came first.")
    check(s == ["Jaws is a 1975 film.", "Its first director, John D. Hancock, left.",
                "It cost about $9 million to make.", "The U.S. release came first."],
          "sentences survive initials and abbreviations", f"{s}")

    # ── 8. night school ──────────────────────────────────────────────────────
    print("\n── night school: the audit")
    f, wf = store("night")
    for q in ("how tall is the eiffel tower", "who directed jaws", "what year did the titanic sink",
              "who wrote pride and prejudice"):
        f.consider(f.reference(q), scripted(GOOD))
    facts = {x["question"]: x for x in f.active()}
    check(len(facts) == 4, "setup: four stored facts", f"{list(facts)}")
    eiffel, jaws = facts["how tall is the eiffel tower"], facts["who directed jaws"]
    titanic, pride = facts["what year did the titanic sink"], facts["who wrote pride and prejudice"]

    state = {"rev": {}, "extract": {}, "gone": set(), "requests": 0}
    for x in facts.values():
        pid = x["sources"][0]["pageid"]
        state["rev"][pid] = x["sources"][0]["revid"]

    def pages(ids, with_extract=True):
        state["requests"] += 1
        out = {}
        for pid in ids:
            if pid in state["gone"]:
                continue
            out[pid] = {"revid": state["rev"].get(pid, 1), "title": "",
                        "extract": state["extract"].get(pid, ""), "url": ""}
        return {"ok": True, "pages": out}

    here = {"v": False}
    gen = scripted(GOOD, compare="SAME")
    ns = N.NightSchool(f.db_path, {"facts": {"recheck_days": 0, "request_spacing_s": 0}}, f, gen,
                       here=lambda: here["v"], pages=pages, sleep=lambda s: None)
    rep = ns.run()
    check(rep["confirmed"] == 4 and state["requests"] == 1 and not gen.calls,
          "unchanged pages: all confirmed in ONE request, without the model",
          f"{rep} requests {state['requests']} model {len(gen.calls)}")

    e_pid = eiffel["sources"][0]["pageid"]
    j_pid = jaws["sources"][0]["pageid"]
    t_pid = titanic["sources"][0]["pageid"]
    p_pid = pride["sources"][0]["pageid"]
    state["rev"][e_pid] += 1
    state["extract"][e_pid] = "The Eiffel Tower is a wrought-iron tower. The tower is 330 metres tall."
    state["rev"][j_pid] += 1
    state["extract"][j_pid] = "Jaws is a 1975 film. Universal released it to over 450 screens."
    state["rev"][t_pid] += 1
    state["extract"][t_pid] = "The Titanic sank on 14 April 1913 according to a revised account."
    state["gone"].add(p_pid)
    gen_changed = scripted(GOOD, compare="SAME")

    def gen2(system, user, t, m):
        if system == W.COMPARE_SYSTEM and "Titanic" in user:
            return "CHANGED"
        return gen_changed(system, user, t, m)
    gen2.calls = gen_changed.calls
    ns._generate = gen2
    rep = ns.run(force=True)
    check(f.get(eiffel["id"])["status"] == "active"
          and f.get(eiffel["id"])["sources"][0]["revid"] == state["rev"][e_pid],
          "an edited page that still says it: confirmed, new revision recorded")
    check(f.get(jaws["id"])["strikes"] == 1 and f.get(jaws["id"])["status"] == "active",
          "an edited page that no longer says it: the model's SAME does NOT confirm (a strike)",
          f"{f.get(jaws['id'])}")
    check(f.get(titanic["id"])["status"] == "demoted", "CHANGED demotes at once")
    check(f.get(pride["id"])["strikes"] == 1, "a page that is gone is a strike")
    rep = ns.run(force=True)
    check(f.get(jaws["id"])["status"] == "demoted" and f.get(pride["id"])["status"] == "demoted",
          "two strikes demote: distrust is the default")
    check(f.lookup("who directed jaws") is None, "a demoted fact is never served")

    print("\n── night school: curiosity (his own questions, safely)")
    import os
    import sqlite3
    import turnstats as TS
    f, wf = store("curious", names={"Sarah"})
    TS.TurnStats(f.db_path)
    rows = [("how tall is the eiffel tower", "llm"), ("who directed jaws", "conversation"),
            ("who is sarah", "llm"), ("what's my sister's favourite film", "llm"),
            ("who is the ceo of apple", "llm"), ("what should I eat", "llm"),
            ("how deep is the mariana trench", "regex:tools"),
            ("what year did the titanic sink", "llm"), ("who wrote pride and prejudice", "llm"),
            ("how tall is the eiffel tower", "llm")]
    with sqlite3.connect(f.db_path) as db:
        for text, path in rows:
            db.execute("INSERT INTO turn_stats (ts, text, path) VALUES (?,?,?)",
                       (time.time() - 3600, text, path))
    cand = N.NightSchool(f.db_path, {"facts": {}}, f, scripted(GOOD), here=lambda: False).candidates()
    check(set(cand) == {"how tall is the eiffel tower", "who directed jaws",
                        "what year did the titanic sink", "who wrote pride and prejudice"},
          "candidates: only general, timeless questions the model answered from memory",
          f"{cand}")
    ns = N.NightSchool(f.db_path, {"facts": {"curiosity_budget": 3, "request_spacing_s": 0}}, f,
                       scripted(GOOD), here=lambda: False, sleep=lambda s: None)
    rep = ns.run()
    check(len(wf.sent) == 3 and len(rep["looked_up"]) == 3, "the nightly budget is kept",
          f"sent {wf.sent}")
    check(not any(w in " ".join(wf.sent).lower() for w in ("sarah", "sister", "ceo", "eat")),
          "nothing personal, nothing changeable was sent", f"{wf.sent}")
    check(len(rep["stored"]) == 3, "what was learned is in the report", f"{rep['stored']}")
    ns._set_meta("last_run", "0")
    rep2 = ns.run()
    check(len(wf.sent) == 4 and wf.sent.count("how tall is the eiffel tower") == 1,
          "a question is never sent twice, across nights", f"{wf.sent}")
    said = N.spoken(ns.last_report())
    check(said.startswith("Last night") and "Wikipedia" in said,
          "'what did you learn last night?' is answered from the record", said)
    check(N.spoken(None).startswith("I haven't had a night school"), "...and honestly before the first")

    print("\n── night school: when it runs, and stopping for him")
    nsw = N.NightSchool(f.db_path, {"facts": {"night_hours": [1, 6], "idle_minutes": 45}}, f,
                        scripted(), here=lambda: False, on_power=lambda: False)
    import datetime
    at = lambda h: datetime.datetime(2026, 11, 20, h, 30).timestamp()
    nsw._set_meta("last_run", "0")
    check(nsw.why_not(at(14), idle_for=99999) == "outside the night window", "not in the afternoon")
    check(nsw.why_not(at(3), idle_for=60) == "he was here recently", "not while he's around")
    check(nsw.why_not(at(3), idle_for=99999) == "on battery", "not on battery")
    nsw._on_power = lambda: True
    check(nsw.why_not(at(3), idle_for=99999) is None, "3am, idle, on power: yes")
    nsw._set_meta("last_run", str(at(3) - 3600))
    check(nsw.why_not(at(4), idle_for=99999) == "already ran", "not twice a night")
    f2, wf2 = store("back")
    with sqlite3.connect(f2.db_path) as db:
        db.executescript(TS._SCHEMA)
        for q in ("how tall is the eiffel tower", "who directed jaws", "who wrote pride and prejudice"):
            db.execute("INSERT INTO turn_stats (ts, text, path) VALUES (?,?,?)",
                       (time.time() - 60, q, "llm"))
    back = {"n": 0}

    def he_returns():
        back["n"] += 1
        return back["n"] > 2
    ns = N.NightSchool(f2.db_path, {"facts": {"request_spacing_s": 0}}, f2, scripted(GOOD),
                       here=he_returns, sleep=lambda s: None)
    rep = ns.run()
    check(rep["stopped"] == "he came back" and len(wf2.sent) < 3,
          "he comes back: it stops between items", f"{rep['stopped']} sent {wf2.sent}")

    print("\n── night school: paraphrase distillation goes through brain.learn's bar")
    taught = []
    f3, _ = store("distill")
    with sqlite3.connect(f3.db_path) as db:
        db.executescript(TS._SCHEMA)
        db.execute("INSERT INTO turn_stats (ts, text, path) VALUES (?,?,?)",
                   (time.time() - 60, "what's on my calendar today", "regex:calendar"))
    para = lambda s, u, t, m: ("1. what do I have going on today\n- anything on my schedule today\n"
                               "\"what's on my calendar today\"")
    ns = N.NightSchool(f3.db_path, {"facts": {}}, f3, para, here=lambda: False,
                       learn=lambda p, sk: taught.append((p, sk)) or True,
                       acts=lambda p: p.startswith("what's on my calendar"),
                       skill_for=lambda t: "cal.today")
    rep = ns.run()
    check(taught == [("what do I have going on today", "cal.today"),
                     ("anything on my schedule today", "cal.today")],
          "new phrasings are offered to learn(); one the router already gets is skipped",
          f"{taught}")

    # ── 9. the pipeline ──────────────────────────────────────────────────────
    print("\n── the pipeline (real _handle_turn_impl)")
    va = T.build_assistant()
    with T.NoSideEffects(va):
        rec = L.Recorder(va)
        refs = []
        va._stream_response = lambda text, rag_context="", reference="": \
            (rec.calls.append(("llm", text)), refs.append(reference))
        va._submit_job = lambda fn, wait=False, priority=None: fn()
        va._llm_generate = lambda s, u, t=0.0, m=120: scripted(GOOD)(s, u, t, m)
        f, wf = store("pipeline", names={"Sarah"})
        va.facts = f

        calls, said = rec.turn("how tall is the eiffel tower")
        check(("llm", "how tall is the eiffel tower") in calls and "330 metres" in refs[-1],
              "a new question: looked up, the model answers WITH the reference",
              f"calls {calls}")
        check(f.lookup("how tall is the eiffel tower") is not None,
              "...and afterwards it is a stored fact")
        check(va._last_turn and va._last_turn["path"] == "wiki", "...recorded as path 'wiki'")
        calls, said = rec.turn("how do you know that")
        check(said and "Wikipedia" in said[-1] and "Eiffel Tower" in said[-1],
              "'how do you know that?' names the article", f"{said}")

        n = len(wf.sent)
        calls, said = rec.turn("what's the height of the eiffel tower")
        check(said == ["The Eiffel Tower is 330 metres tall."] and not calls and len(wf.sent) == n,
              "asked again another way: answered from the store, no model, no request",
              f"said {said} calls {calls}")
        check(va._last_turn["path"] == "fact", "...recorded as path 'fact'")
        calls, said = rec.turn("how do you know that")
        check(said and "last checked it today" in said[-1], "...and its receipt says when it was checked",
              f"{said}")
        calls, said = rec.turn("that's wrong")
        check(said and "stopped trusting" in said[-1] and f.lookup("how tall is the eiffel tower") is None,
              "'that's wrong' after a fact: it is no longer served", f"{said}")

        n = len(wf.sent)
        calls, said = rec.turn("who is sarah")
        check(len(wf.sent) == n and ("llm", "who is sarah") in calls and refs[-1] == "",
              "a name he told Nova about: answered without a lookup", f"sent {wf.sent[n:]}")
        calls, said = rec.turn("how do you know that")
        check(said and "didn't look that one up" in said[-1],
              "...and Nova says so if he asks how she knows", f"{said}")
        calls, said = rec.turn("what's my cpu")
        calls, said = rec.turn("how do you know that")
        check(said and not calls and "Wikipedia" not in said[-1],
              "after a command, 'how do you know that?' is explained from the record, not the model",
              f"said {said} calls {calls}")
        calls, said = rec.turn("how far is the moon from earth")
        check(not any(c[0] == "_finish_tool" for c in calls) and ("llm", "how far is the moon from earth") in calls,
              "'how far is the moon from earth' is a question, not a MapKit search", f"{calls}")

        print("\n── regression: a correction only ever corrects the LAST thing")
        b = L.fresh_brain(va, "facts-correct")
        d = b.decide("what's my cpu", va)
        if check(d is not None, "setup: 'what's my cpu' is a reflex"):
            rec.turn("what's my cpu")
            rec.turn("who wrote pride and prejudice")
            calls, said = rec.turn("that's wrong")
            check(not b.was_rejected("what's my cpu", d.skill.name),
                  "'that's wrong' about an ANSWER does not un-learn the reflex before it",
                  f"said {said}")
        va.facts = None
        calls, said = rec.turn("how tall is the eiffel tower")
        check(("llm", "how tall is the eiffel tower") in calls,
              "facts unavailable: the turn is answered exactly as before Stage 3")

    print(f"\n  {PASS}/{PASS + FAIL} checks passed")
    if FAILURES:
        print("\n  FAILURES:")
        for x in FAILURES:
            print(f"    ✗ {x}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
