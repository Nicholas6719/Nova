#!/usr/bin/env python3
"""
The brain inside the REAL process — not a harness.

Starts nova.py the way BackendManager does (a child process, its own
NOVA_DATA_DIR), waits for the brain to finish loading, then sends typed turns
over HTTP with silent=true (answered in text, nothing spoken) and reads the
process log to see WHICH path answered each one and how fast.

Every sentence is READ-ONLY on purpose: a reflex runs the real handler, so
"turn it up" would really change his volume. Nothing here moves a file,
changes a setting, opens an app or touches the network except the market
quote (Yahoo — invariant 3's market exception, a ticker and nothing else).

What it proves that the unit suites cannot:
  - _init_brain runs at real startup and loads in the background
  - reflexes answer through the real _respond path, in milliseconds
  - a fall-through still reaches the LLM afterwards and answers (the reflex
    path never touches MLX, so the prompt cache is not disturbed)
  - the [brain-disagree] log fires when the two routers differ

Run:  python tests/brain_live_check.py
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HTTP = "http://127.0.0.1:5001"

# (sentence, expected route) — "reflex:<skill>" or "llm"
TURNS = [
    ("what time is it", "reflex:fast.time"),
    ("how hard is my processor working", "reflex:tools.cpu"),
    ("what's my battery at", "reflex:tools.battery"),
    ("hello there", "reflex:fast.greet"),
    ("remember that I like green tea", "reflex:memory.note"),
    ("what do you know about me", "reflex:memory.meta"),
    ("is the Dow up or down", "reflex:market.overview"),
    ("tell me a joke about computers", "llm"),
    ("what's the date", "reflex:fast.date"),
    ("explain what a black hole is in one sentence", "llm"),
    # ── Stage 2, inside the real process (all read-only) ──
    ("what does my day look like", "ask:cal.today"),     # did you mean ...?
    ("yes", "confirmed:cal.today"),                      # runs it, learns it
    ("what does my day look like", "reflex:cal.today"),  # now a reflex
    ("when I say status check, what's my battery at and what time is it",
     "taught"),
    ("status check", "phrase"),
    ("what time is it", "reflex:fast.time"),
    ("that's wrong", "corrected"),
    # ── Stage 4 (read-only) ──
    ("what's 25% of 10%", "math"),
    ("My dinner was $95.82 before tax. How much should I tip?", "math"),
    ("tell me when my cpu goes above 99 percent", "watch"),
    ("what are you watching", "watch"),
    ("cancel all watches", "watch"),
]


def get(path: str, timeout: float = 3.0):
    with urllib.request.urlopen(HTTP + path, timeout=timeout) as r:
        return json.loads(r.read())


def post(path: str, body: dict):
    req = urllib.request.Request(HTTP + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read() or b"{}")


def main() -> int:
    data = tempfile.mkdtemp(prefix="nova-brainlive-")
    log_path = Path(data) / "backend.log"
    env = dict(os.environ, NOVA_DATA_DIR=data, NOVA_PARENT_PID=str(os.getpid()),
               PYTHONUNBUFFERED="1")
    logf = open(log_path, "w")
    proc = subprocess.Popen([sys.executable, "-u", str(ROOT / "nova.py")], cwd=ROOT,
                            env=env, stdout=logf, stderr=subprocess.STDOUT)
    fails: list[str] = []
    try:
        t0 = time.time()
        while time.time() - t0 < 180:
            if proc.poll() is not None:
                print(log_path.read_text()[-3000:])
                print("backend exited during startup")
                return 1
            if "brain loaded:" in log_path.read_text():
                try:
                    get("/api/status")
                    break
                except Exception:
                    pass
            time.sleep(0.5)
        else:
            print("timed out waiting for the backend and the brain")
            return 1
        txt = log_path.read_text()
        m = re.search(r"brain loaded: (\d+) examples across (\d+) skills", txt)
        print(f"  backend up in {time.time() - t0:.1f}s; {m.group(0) if m else '?'}")
        emb = re.search(r"embedded (\d+) new seed\(s\) in ([\d.]+)s", txt)
        if emb:
            print(f"  first run: {emb.group(0)}")

        for sentence, expect in TURNS:
            mark = len(log_path.read_text())
            post("/api/message", {"content": sentence, "silent": True})
            # wait for the turn to finish: the state returns to idle after it
            deadline = time.time() + 90
            seg = ""
            while time.time() < deadline:
                seg = log_path.read_text()[mark:]
                if "[brain" in seg and "[state] → idle" in seg.split("[brain", 1)[1]:
                    break
                if "[nova]" in seg and time.time() > deadline - 85:
                    # brain lessons answer before routing: no [brain] decide line
                    if "→ idle" in seg or "[nova]" in seg:
                        time.sleep(0.5)
                        seg = log_path.read_text()[mark:]
                        break
                time.sleep(0.2)
            route = re.search(r"\[brain\] (?:reflex (\S+)@[\d.]+.*?\((\d+)ms\)|"
                              r"(?:fallthrough|conversation)[^\n]*?(\d+)ms\))", seg)
            try:
                msgs = get("/api/messages").get("messages", [])
                last = next((m["content"] for m in reversed(msgs)
                             if m.get("role") == "assistant"), "")
            except Exception:
                last = ""
            answer = re.match(r"(.*)", last) if last else None
            ask = re.search(r"\[brain\] asking: did you mean (\S+)\?", seg)
            conf = re.search(r"\[brain\] confirmed: .* -> (\S+)", seg)
            phrase = re.search(r"\[brain\] taught phrase", seg)
            corrected = re.search(r"\[brain\] corrected:", seg)
            if "[math]" in seg:
                got, ms = "math", "-"
            elif re.search(r"\[watch\]|I'll tell you when|watching|stopped all|stopped watching", last):
                got, ms = "watch", "-"
            elif conf:
                got, ms = f"confirmed:{conf.group(1)}", "-"
            elif corrected:
                got, ms = "corrected", "-"
            elif phrase:
                got, ms = "phrase", "-"
            elif "Got it. When you say" in last:
                got, ms = "taught", "-"
            elif ask:
                got, ms = f"ask:{ask.group(1)}", route.group(3) if route else "?"
            elif route and route.group(1):
                got, ms = f"reflex:{route.group(1)}", route.group(2)
            elif route:
                got, ms = "llm", route.group(3)
            else:
                got, ms = "?", "?"
            ok = got == expect
            if not ok:
                fails.append(f"{sentence!r}: expected {expect}, got {got}")
            dis = re.search(r"\[brain-disagree\][^\n]*", seg)
            print(f"  {'PASS' if ok else 'FAIL'}  {sentence:<46} {got:<26} "
                  f"brain {ms}ms")
            print(f"        -> {(answer.group(1) if answer else '(no answer logged)')[:90]}")
            if dis:
                print(f"        {dis.group(0)[:110]}")
        errors = [l for l in log_path.read_text().splitlines()
                  if "Traceback" in l or "[brain] decide failed" in l
                  or "failed:" in l and "[brain]" in l]
        for e in errors:
            fails.append(f"log: {e}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        logf.close()
    print(f"\n  log: {log_path}")
    if fails:
        print("\n  FAILURES:")
        for f in fails:
            print(f"    ✗ {f}")
        return 1
    print("\n  ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
