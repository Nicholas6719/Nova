#!/usr/bin/env python3
"""
Stage 3 against the REAL model, the REAL Wikipedia and the REAL process.

test_world_facts.py proves the rules around the model hold whatever it says.
This measures what the model actually says (testing rule 4: never trust an
assumption about a 3B), and whether the whole thing works inside nova.py.

Part 1 — answers with and without the reference (real 3B, canned pages, no
         network): does handing the model Wikipedia's sentences make its
         answers right where its memory was wrong?
Part 2 — intake and night school in-process (real 3B, real Wikipedia, a
         throwaway database): lookups, storage, the overnight audit.
Part 3 — the real process (NOVA_DATA_DIR throwaway, typed silent turns over
         HTTP): a question is looked up, then served from the store, its
         receipt is real, a changeable one is never kept.

Network: about eight Wikipedia requests in total, spaced out (Wikipedia
rate-limits bursts). Questions are generic — nothing of his is sent.

Run:  NOVA_TEST_LIVE=1 python tests/world_facts_live.py
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
ROOT = TESTS_DIR.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

HTTP = "http://127.0.0.1:5001"
CANNED = json.loads((TESTS_DIR / "fixtures" / "wiki_canned.json").read_text())
EXPECT = {
    "how tall is the eiffel tower": r"\b330\b",
    "who wrote pride and prejudice": r"austen",
    "who directed jaws": r"spielberg",
    "what year did the titanic sink": r"1912",
    "how deep is the mariana trench": r"10,?935|10,?9\d\d",
    "what is the capital of australia": r"canberra",
    "who painted the mona lisa": r"leonardo|da vinci",
    "when did the berlin wall fall": r"1989",
    "how many legs does a spider have": r"\beight\b|\b8\b",
}
fails: list[str] = []


def check(cond, label, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}  {detail if not cond else ''}")
    if not cond:
        fails.append(label)
    return bool(cond)


def part1(llm, cfg, f):
    print("\n── Part 1: the real model, with and without the reference")
    from system_prompt import build_system_prompt
    cur = {}
    f._search = lambda q: {"ok": True, "pages": CANNED[cur["q"]]}
    with_ok = without_ok = 0
    for q, rx in EXPECT.items():
        cur["q"] = q
        ref = f.reference(q)
        outs = {}
        for name, ctx in (("with", ref.get("context", "")), ("without", "")):
            sp = build_system_prompt(cfg, memory_context="", reference_context=ctx)
            outs[name] = llm.stream(system_prompt=sp, history=[], user_message=q,
                                    on_token=lambda t: None)
        w = bool(re.search(rx, outs["with"], re.I))
        wo = bool(re.search(rx, outs["without"], re.I))
        with_ok += w
        without_ok += wo
        print(f"   {'✓' if w else '✗'} with / {'✓' if wo else '✗'} without  {q}")
        if not w:
            print(f"        with:    {outs['with'][:140]!r}")
        if not wo:
            print(f"        without: {outs['without'][:140]!r}")
        if re.search(r"wikipedia|according to|the reference|sentences", outs["with"], re.I):
            print(f"        ! mentions its source: {outs['with'][:140]!r}")
    print(f"   right WITH the reference {with_ok}/{len(EXPECT)}, WITHOUT {without_ok}/{len(EXPECT)}")
    check(with_ok >= without_ok and with_ok >= len(EXPECT) - 1,
          "with Wikipedia's sentences the model is right at least as often",
          f"{with_ok} vs {without_ok}")


def part2(gen, f):
    print("\n── Part 2: real Wikipedia + real model, in-process")
    import wiki
    import night_school as N
    f._search = None
    for q in ("who painted the mona lisa", "who is the ceo of apple"):
        t = time.time()
        ref = f.reference(q)
        ms = (time.time() - t) * 1000
        check(ref.get("ok"), f"looked up: {q!r} in {ms:.0f}ms", ref.get("reason", ""))
        fid = f.consider(ref, gen) if ref.get("ok") else None
        if "ceo" in q:
            check(fid is None, "...a changeable answer is never stored")
        else:
            got = f.get(fid) if fid else None
            check(got and re.search(EXPECT[q], got["answer"], re.I),
                  f"...stored: {got and got['answer']!r}")
        time.sleep(8)
    if wiki.backing_off():
        print("   (rate limited; skipping the audit)")
        return
    ns = N.NightSchool(f.db_path, {"facts": {"recheck_days": 0, "request_spacing_s": 8}}, f, gen,
                       here=lambda: False)
    rep = ns.run(force=True)
    check(rep["audited"] >= 1 and rep["confirmed"] == rep["audited"],
          "the overnight audit re-checks real pages and confirms an unchanged fact",
          json.dumps({k: rep[k] for k in ("audited", "confirmed", "changed", "unclear", "stopped")}))
    print(f"   report: {N.spoken(ns.last_report())}")


def get(path, timeout=3.0):
    with urllib.request.urlopen(HTTP + path, timeout=timeout) as r:
        data = json.loads(r.read())
    return data.get("messages", []) if path == "/api/messages" else data


def post(path, body, timeout=60.0):
    req = urllib.request.Request(HTTP + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def part3():
    print("\n── Part 3: the real process")
    data = tempfile.mkdtemp(prefix="nova-factslive-")
    log_path = Path(data) / "backend.log"
    env = dict(os.environ, NOVA_DATA_DIR=data, NOVA_PARENT_PID=str(os.getpid()),
               PYTHONUNBUFFERED="1")
    logf = open(log_path, "w")
    proc = subprocess.Popen([sys.executable, "-u", str(ROOT / "nova.py")], cwd=ROOT,
                            env=env, stdout=logf, stderr=subprocess.STDOUT)
    db = Path(data) / "nova_memory.db"
    try:
        for _ in range(240):
            time.sleep(1)
            if "Nova ready" in log_path.read_text(errors="ignore"):
                break
        else:
            check(False, "Nova started")
            return
        time.sleep(6)                                   # the brain loads in the background

        def turn(text, wait_s=40):
            n = len(get("/api/messages"))
            post("/api/message", {"content": text, "silent": True})
            for _ in range(int(wait_s * 4)):
                time.sleep(0.25)
                msgs = get("/api/messages")
                if len(msgs) >= n + 2 and msgs[-1].get("role") == "assistant":
                    time.sleep(0.5)
                    break
            row = sqlite3.connect(db).execute(
                "SELECT path, decide_ms, first_ms FROM turn_stats ORDER BY id DESC LIMIT 1").fetchone()
            msgs = get("/api/messages")
            return (msgs[-1]["content"] if msgs else ""), row

        said, row = turn("how tall is the eiffel tower")
        check(row and row[0] == "wiki" and re.search(r"\b330\b", said),
              f"a new question is looked up and answered: {said[:90]!r}", f"{row}")
        print(f"     first word after {row and row[2] and int(row[2])}ms")
        for _ in range(60):
            time.sleep(0.5)
            if sqlite3.connect(db).execute("SELECT COUNT(*) FROM world_facts").fetchone()[0]:
                break
        stored = sqlite3.connect(db).execute("SELECT question, answer FROM world_facts").fetchall()
        check(stored, f"...and stored afterwards, off the live path: {stored}")
        said, row = turn("what's the height of the eiffel tower")
        check(row and row[0] == "fact", f"asked another way: answered from the store: {said!r}",
              f"{row}")
        if row and row[2]:
            print(f"     decided in {row[1]:.0f}ms, first word after {row[2]:.0f}ms")
        said, row = turn("how do you know that")
        check("Wikipedia" in said and "Eiffel" in said, f"the receipt is real: {said!r}")
        time.sleep(8)
        said, row = turn("who is the ceo of apple")
        n = sqlite3.connect(db).execute(
            "SELECT COUNT(*) FROM world_facts WHERE question LIKE '%ceo%'").fetchone()[0]
        time.sleep(5)
        check(row and row[0] == "wiki" and n == 0,
              f"a changeable question is looked up, never kept: {said[:90]!r}", f"{row}")
        said, row = turn("what did you learn last night")
        check(said.startswith("I haven't had a night school"), f"night school, honestly: {said!r}")
        sent = sqlite3.connect(db).execute("SELECT question, origin FROM world_lookups").fetchall()
        print(f"     everything that left the machine: {sent}")
        check(all(o == "asked" for _, o in sent), "nothing was sent that he did not ask")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        logf.close()
        print(f"     log: {log_path}")


def main() -> int:
    if os.environ.get("NOVA_TEST_LIVE") != "1":
        print("skipped: set NOVA_TEST_LIVE=1 (uses the network and the model)")
        return 0
    import nova
    import world_facts as W
    from llm_engine import LLMEngine
    from brain import EMBED_MODEL, MODEL_DIR
    from fastembed import TextEmbedding
    cfg = nova.load_config()
    llm = LLMEngine(cfg["llm"])
    gen = lambda s, u, t, m: llm.generate(system_prompt=s, history=[], user_message=u,
                                          temperature=t, max_tokens=m)
    emb = TextEmbedding(EMBED_MODEL, cache_dir=str(MODEL_DIR), local_files_only=True)
    f = W.WorldFacts(Path(tempfile.mkdtemp()) / "live.db", cfg, embedder=lambda: emb)
    parts = os.environ.get("NOVA_LIVE_PARTS", "123")
    if "1" in parts:
        part1(llm, cfg, f)
    if "2" in parts:
        part2(gen, f)
    del llm
    if "3" in parts:
        part3()
    print(f"\n  {'ALL PASS' if not fails else str(len(fails)) + ' FAILED: ' + ', '.join(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
