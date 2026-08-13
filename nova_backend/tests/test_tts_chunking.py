#!/usr/bin/env python3
"""
TTS chunking — Nova may start speaking sooner, but it must say the SAME words.

Nothing is audible until the first chunk is generated AND synthesised, so a
long opening sentence is a long silence. Nova now speaks the opening CLAUSE of
a long first sentence instead of waiting for the full stop.

The thing that would be embarrassing is not slowness, it is a dropped or
duplicated word. So the guarantee here is that the spoken chunks reassemble
into exactly the reply, on every phrasing shape — and that only the FIRST
chunk is ever split, because a mid-response split buys nothing (once audio is
playing, synthesis runs well ahead of the speaker).

Fidelity: the REAL _stream_response from nova.py, driven with a scripted LLM
and a capturing TTS. No audio is played.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path as _Path

TESTS_DIR = _Path(__file__).resolve().parent
BACKEND = str(TESTS_DIR.parent)
sys.path.insert(0, BACKEND)
sys.path.insert(0, str(TESTS_DIR))
os.chdir(BACKEND)

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
        print(f"        {detail}")


def section(t):
    print("\n" + "=" * 70)
    print(t)
    print("=" * 70)


import nova as nova_mod
from nova import VoiceAssistant, _clean_for_tts


def build(reply: str, split_first: bool = True):
    """Real VoiceAssistant state; scripted LLM; TTS captured, never played."""
    va = VoiceAssistant.__new__(VoiceAssistant)
    va.config = nova_mod.load_config()
    va.config["tts"]["split_first_clause"] = split_first
    # The REAL initializer — a hand-copied field list is how this harness once
    # drifted from __init__ and hid a crash on every utterance.
    va._init_state()

    spoken: list[str] = []

    class _TTS:
        def speak(self, text): spoken.append(text)
        def wait_until_done(self, timeout=None): pass
        def is_speaking(self): return False
    va.tts = _TTS()

    class _Mem:
        def add_turn(self, *a, **k): pass
        def get_context_for_llm(self, *a, **k): return ""
        def get_recent_turns(self, *a, **k): return []
    va.memory = _Mem()

    class _WS:
        def send_message(self, *a, **k): pass
        def broadcast_state(self, *a, **k): pass
        def stream_token(self, *a, **k): pass
    va.ws = _WS()
    va.set_state = lambda s: None

    class _LLM:
        def stream(self, system_prompt, history, user_message, on_token):
            # token-by-token, the way mlx-lm actually delivers it
            for tok in re.findall(r"\S+\s*", reply):
                on_token(tok)
            return reply
    va.llm = _LLM()
    return va, spoken


REPLIES = [
    # long opening sentence WITH a clause break — the case this optimises
    "The shortest war in history was between Britain and Zanzibar, and it lasted "
    "about forty minutes. Rather one sided.",
    "Do you have a specific craving in mind, or would you like some general ideas?",
    # no comma at all — must behave exactly as before
    "I'm functioning within normal parameters.",
    "I don't have real-time access to current weather conditions.",
    # comma too early to be worth splitting
    "Yes, that works.",
    "Well, I can look into it for you later today if you want.",
    # several sentences
    "Paris. It's the capital of France. Anything else?",
    # comma AFTER the first sentence ends — must not be used as a first split
    "Sure thing. Later, when you have a moment, we can look at it.",
    # single short reply
    "Paris.",
    # question with a list-like clause structure
    "You have three things today, a dentist appointment, and two meetings.",
]


# ══════════════════════════════════════════════════════════════════════════
section("THE WORDS ARE UNCHANGED  (chunks must reassemble into the reply)")
# ══════════════════════════════════════════════════════════════════════════
for reply in REPLIES:
    va, spoken = build(reply, split_first=True)
    va._stream_response("anything")
    got = " ".join(s.strip() for s in spoken)
    want = _clean_for_tts(reply)
    # join/normalise whitespace the same way before comparing
    norm = lambda s: re.sub(r"\s+", " ", s).strip()
    check(norm(got) == norm(want), f"reassembles: {reply[:46]!r}",
          "" if norm(got) == norm(want)
          else f"want={norm(want)[:100]!r}\n        got ={norm(got)[:100]!r}")


# ══════════════════════════════════════════════════════════════════════════
section("SPLITTING IS LIMITED TO THE FIRST CHUNK")
# ══════════════════════════════════════════════════════════════════════════
long_reply = ("The shortest war in history was between Britain and Zanzibar, and it "
              "lasted about forty minutes. It happened in 1896, and the ultimatum "
              "expired at nine, so it ended quickly.")
va, spoken = build(long_reply, split_first=True)
va._stream_response("anything")
check(len(spoken) >= 2, "a long opening sentence is split", f"{len(spoken)} chunks")
check(spoken[0].endswith(","), "the first chunk ends at the clause break",
      f"first chunk: {spoken[0]!r}")

# every chunk after the first must be a whole sentence (no interior commas used
# as split points) — i.e. it ends with terminal punctuation
tail_bad = [s for s in spoken[1:] if not re.search(r"[.!?][\"')\]]?$", s.strip())]
check(not tail_bad, "later chunks are whole sentences, never clauses",
      "" if not tail_bad else f"these ended mid-sentence: {tail_bad}")


# ══════════════════════════════════════════════════════════════════════════
section("NO SPLIT WHERE IT WOULD NOT HELP")
# ══════════════════════════════════════════════════════════════════════════
va, spoken = build("Yes, that works.", split_first=True)
va._stream_response("anything")
check(len(spoken) == 1, "a 1-word opening clause is not split off",
      f"chunks: {spoken}")

va, spoken = build("I'm functioning within normal parameters.", split_first=True)
va._stream_response("anything")
check(len(spoken) == 1, "a sentence with no clause break stays whole",
      f"chunks: {spoken}")

va, spoken = build("Sure thing. Later, when you have a moment, we can look.",
                   split_first=True)
va._stream_response("anything")
check(spoken[0] == "Sure thing.",
      "a comma in a LATER sentence never splits the first chunk",
      f"first chunk: {spoken[0]!r}")


# ══════════════════════════════════════════════════════════════════════════
section("THE CONFIG SWITCH REALLY DISABLES IT")
# ══════════════════════════════════════════════════════════════════════════
va, spoken = build(long_reply, split_first=False)
va._stream_response("anything")
check(not spoken[0].endswith(","), "with split_first_clause false, no clause split",
      f"first chunk: {spoken[0][:70]!r}")
got = re.sub(r"\s+", " ", " ".join(s.strip() for s in spoken)).strip()
want = re.sub(r"\s+", " ", _clean_for_tts(long_reply)).strip()
check(got == want, "disabled path still says the whole reply")


# ══════════════════════════════════════════════════════════════════════════
section("RESULT")
# ══════════════════════════════════════════════════════════════════════════
print(f"\n  {PASS}/{PASS + FAIL}")
for f in FAILURES:
    print(f"    ✗ {f}")
sys.exit(1 if FAIL else 0)
