#!/usr/bin/env python3
"""
Routing corpus — every phrase that has ever broken Nova, replayed.

Reads adversarial_phrases.txt and checks each one still routes where it should.
This is the regression net: a phrase only earns a place here by having caused a
real, user-visible failure, so a green run means none of them have come back.

Detection ONLY. Nothing here executes a command, launches an app, or touches
the filesystem — `tools.match()` performs as it matches, so the side-effectful
handlers are stubbed for the duration.

Run:  python tests/test_routing_corpus.py
"""
from __future__ import annotations

import os
import re
import sys
import threading
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
BACKEND = TESTS_DIR.parent
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(TESTS_DIR))
os.chdir(BACKEND)

PASS = FAIL = 0
FAILURES: list[str] = []


def check(cond, label, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        FAILURES.append(f"{label}  ({detail})" if detail else label)
    return bool(cond)


def load_corpus() -> list[tuple[str, str]]:
    out = []
    path = TESTS_DIR / "adversarial_phrases.txt"
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "==>" not in line:
            continue
        phrase, _, expected = line.partition("==>")
        out.append((phrase.strip(), expected.strip()))
    return out


class NoSideEffects:
    """Neutralise the SIDE EFFECTS, not the decision logic.

    tools.match() performs as it matches — probing routing with "open spotify"
    really launched Spotify. The first version of this stubbed whole handlers,
    which was worse: it reported a match where the real code returns None
    (`_resolve_app("downloads")` is None, so "close downloads" correctly falls
    through to unsupported). Stubbing the verdict makes the test lie.

    So only `subprocess.run` and `time.sleep` are replaced, in the modules that
    reach the machine. Every regex, alias lookup and installed-app scan runs for
    real; nothing launches, quits, or types.
    """

    MODULES = ("tools", "browser_control", "maps_engine", "file_manager")

    class _Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def __init__(self, va):
        self.va = va
        self.saved: dict = {}

    def __enter__(self):
        import importlib
        for name in self.MODULES:
            try:
                mod = importlib.import_module(name)
            except Exception:
                continue
            if hasattr(mod, "subprocess"):
                self.saved[(name, "subprocess.run")] = mod.subprocess.run
                mod.subprocess.run = lambda *a, **k: NoSideEffects._Result()
            if hasattr(mod, "time"):
                self.saved[(name, "time.sleep")] = mod.time.sleep
                mod.time.sleep = lambda *a, **k: None
        return self

    def __exit__(self, *exc):
        import importlib
        for (name, what), fn in self.saved.items():
            mod = importlib.import_module(name)
            if what == "subprocess.run":
                mod.subprocess.run = fn
            else:
                mod.time.sleep = fn


def build_assistant():
    import nova as nova_mod
    from nova import VoiceAssistant
    va = VoiceAssistant.__new__(VoiceAssistant)
    va.config = nova_mod.load_config()
    va._init_state()                     # the REAL initializer, never a copy

    class _LLM:
        def generate(self, *a, **k):
            raise AssertionError("routing detection must not call the LLM")
    va.llm = _LLM()
    va._init_memory()
    va._init_tools()
    va._init_calendar()
    va._init_files()
    va._init_screen()
    return va


def route_of(va, text: str) -> str:
    """Which stage claims this utterance, in pipeline order."""
    import nova as nova_mod

    if nova_mod.find_signoff(text)[1] is not None:
        lead = nova_mod._content_before_sleep(text, nova_mod.find_signoff(text)[0])
        return "signoff" if not lead else "signoff+content"
    if va.calendar.detect_intent(text) is not None:
        return "calendar"
    if va.screen.detect_intent(text) is not None:
        return "screen"
    if va.files.detect_intent(text) is not None:
        return "files"
    if _memory_would_fire(va, text):
        return "memory"
    if va._fast_path(text) is not None:
        return "fastpath"
    try:
        if va.tools.match(text) is not None:
            return "tools"
    except Exception:
        # A handler that blew up on stubbed output still CLAIMED the phrase,
        # which is what routing is about.
        return "tools"
    if nova_mod._ACTION_REQUEST_RE.match(text):
        return "unsupported"
    return "llm"


def _memory_would_fire(va, text: str) -> bool:
    low = text.lower().strip()
    if re.match(r"^\s*(remember|forget)\b", low):
        return True
    m = va._RECALL_RE.search(low)
    return bool(m) and va.tools.match(text) is None


def main() -> int:
    corpus = load_corpus()
    print("=" * 72)
    print(f"ROUTING CORPUS — {len(corpus)} phrases that have broken Nova before")
    print("=" * 72)

    va = build_assistant()
    with NoSideEffects(va):
        for phrase, expected in corpus:
            got = route_of(va, phrase)
            if expected == "signoff":
                ok = got.startswith("signoff")
            elif expected == "no-signoff":
                ok = not got.startswith("signoff")
            else:
                ok = got == expected
            check(ok, f"{expected:<12} {phrase[:58]}", f"got {got}")

    print(f"\n  {PASS}/{PASS + FAIL} phrases route correctly")
    if FAILURES:
        print("\n  REGRESSIONS:")
        for f in FAILURES:
            print(f"    ✗ {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
