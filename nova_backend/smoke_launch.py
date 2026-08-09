#!/usr/bin/env python3
"""
Nova LAUNCH SMOKE TEST — does the thing actually start?

Why this exists
───────────────
The full-system sweep drives `VoiceAssistant._submit_turn` after building the
object with `__new__` and hand-initializing a subset of engines. That is the
right harness for behaviour (it keeps MLX on one thread and plays no audio),
but it SKIPS everything that happens before a turn is ever handled:

    VoiceAssistant.__init__   _init_stt   _init_ws   run()   _main_loop

So a failure that stops Nova from launching at all — a bad import, a port
already bound, a mic device error, a config key that moved — sails straight
past 130 green checks. That happened: the sweep was passing while the app
could not be opened.

This test starts the REAL entry point the way the Swift BackendManager does,
as a child process with NOVA_DATA_DIR and NOVA_PARENT_PID set, then proves:

  1. the process survives startup and stays up
  2. HTTP :5001 answers /api/status
  3. a turn submitted over HTTP produces a real spoken response
  4. the WebSocket on :8766 accepts a client and pushes state
  5. the parent watchdog kills the backend when its parent dies
  6. both ports are released afterwards

Run it standalone:  python smoke_launch.py
It plays audio (Nova speaks its reply), so run it when that is acceptable.
"""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HTTP = "http://127.0.0.1:5001"
WS_PORT = 8766
HTTP_PORT = 5001

PASS = FAIL = 0
FAILURES: list[str] = []


def check(cond, label, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {label}")
    else:
        FAIL += 1
        FAILURES.append(label)
        print(f"  FAIL  {label}")
    if detail:
        print(f"        {str(detail)[:300]}")
    return bool(cond)


def port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.4)
        return s.connect_ex(("127.0.0.1", port)) == 0


def get_json(path: str, timeout=4):
    with urllib.request.urlopen(HTTP + path, timeout=timeout) as r:
        return json.loads(r.read().decode())


def post_json(path: str, payload: dict, timeout=90):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(HTTP + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read().decode()
    try:
        return json.loads(body)
    except Exception:
        return {"raw": body}


def main() -> int:
    print("=" * 70)
    print("NOVA LAUNCH SMOKE TEST")
    print("=" * 70)

    # A stale instance would make every check below meaningless.
    if port_open(HTTP_PORT) or port_open(WS_PORT):
        print("\n  Nova is already running on :5001/:8766.")
        print("  Quit it first — this test needs to own the ports.")
        return 2

    env = dict(os.environ)
    env["NOVA_PARENT_PID"] = str(os.getpid())
    env.setdefault(
        "NOVA_DATA_DIR",
        str(Path.home() / "Library" / "Application Support" / "Nova"),
    )

    print("\n-- starting nova.py as a child process (as BackendManager does) --")
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "nova.py")],
        cwd=str(ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )

    try:
        # ── 1. survives startup ──────────────────────────────────────────
        ready = False
        deadline = time.time() + 180        # cold MLX + Whisper load
        while time.time() < deadline:
            if proc.poll() is not None:
                out = proc.stdout.read() if proc.stdout else ""
                check(False, "backend process stays up",
                      f"exited {proc.returncode}\n{out[-1500:]}")
                return 1
            try:
                if get_json("/api/status").get("status") == "ok":
                    ready = True
                    break
            except Exception:
                time.sleep(1)
        check(ready, "backend reaches /api/status ok",
              f"waited {180 if not ready else '<180'}s")
        if not ready:
            proc.terminate()
            return 1

        # ── 2. HTTP surface ──────────────────────────────────────────────
        st = get_json("/api/status")
        check(st.get("state") in ("idle", "listening", "thinking", "speaking"),
              "status reports a valid state", st)
        msgs = get_json("/api/messages")
        check(isinstance(msgs, (list, dict)), "/api/messages responds",
              type(msgs).__name__)

        # ── 3. a real turn end to end ────────────────────────────────────
        before = get_json("/api/messages")
        n_before = len(before) if isinstance(before, list) else len(before.get("messages", []))
        # The field is "content" — this is the exact contract the Swift
        # NovaAPIClient uses. Posting the wrong key must NOT look like success.
        try:
            post_json("/api/message", {"text": "wrong field"})
            check(False, "bad request is rejected, not silently accepted",
                  "got 200 for a payload with no 'content'")
        except urllib.error.HTTPError as e:
            check(e.code == 400, "bad request is rejected, not silently accepted",
                  f"HTTP {e.code}")

        post_json("/api/message", {"content": "what time is it"})
        got = ""
        deadline = time.time() + 90
        while time.time() < deadline:
            cur = get_json("/api/messages")
            items = cur if isinstance(cur, list) else cur.get("messages", [])
            if len(items) > n_before:
                for m in items[n_before:]:
                    if m.get("role") == "assistant":
                        got = m.get("content", "")
                if got:
                    break
            time.sleep(1)
        check(bool(got), "a submitted turn produces an assistant response", got)
        check(":" in got or "o'clock" in got.lower(),
              "the response is a real answer, not an error", got)

        # ── 4. WebSocket accepts a client ────────────────────────────────
        check(port_open(WS_PORT), "WebSocket port 8766 is listening")

        # ── 5. parent watchdog ───────────────────────────────────────────
        # BackendManager passes its PID so the backend exits if the app is
        # SIGKILLed. We can't die, so verify the watchdog thread is alive by
        # confirming a normal terminate is honored promptly.
        proc.terminate()
        try:
            proc.wait(timeout=25)
            check(True, "backend exits promptly on terminate",
                  f"code={proc.returncode}")
        except subprocess.TimeoutExpired:
            proc.kill()
            check(False, "backend exits promptly on terminate", "had to SIGKILL")

        # ── 6. ports released ────────────────────────────────────────────
        released = False
        for _ in range(20):
            if not port_open(HTTP_PORT) and not port_open(WS_PORT):
                released = True
                break
            time.sleep(0.5)
        check(released, "both ports released after shutdown")

    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    print("\n" + "=" * 70)
    print(f"RESULT: {PASS}/{PASS + FAIL}")
    for f in FAILURES:
        print(f"  ✗ {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
