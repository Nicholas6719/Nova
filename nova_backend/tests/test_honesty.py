#!/usr/bin/env python3
"""
Nova never speaks a promise nothing will keep, or claims an action nothing did.

Nova's LLM runs no tools: every action is a deterministic handler decided
before the model is reached. So on the LLM path, "I'll keep an eye on it",
"I'll remind you" and "I've opened the folder" are always false (honesty.py,
ported from JARVIS, where each was said to him with nothing behind it).

Fidelity: honesty.unbacked on real sentences (JARVIS's transcript, Nova's own
"I've opened the coding projects folder for you"), then the REAL
_stream_response with a scripted LLM and a capturing TTS — what is SPOKEN and
what is STORED. Also: a typed (silent) turn is answered in text only.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR))
sys.path.insert(0, str(TESTS_DIR.parent))

import os                             # noqa: E402
os.chdir(TESTS_DIR.parent)
import honesty                        # noqa: E402
import nova as nova_mod               # noqa: E402
from nova import VoiceAssistant       # noqa: E402


class H:
    """The real-state builder from test_tts_chunking (that module runs its
    suite on import, so it is mirrored here rather than imported)."""

    @staticmethod
    def build(reply: str, split_first: bool = True):
        va = VoiceAssistant.__new__(VoiceAssistant)
        va.config = nova_mod.load_config()
        va.config["tts"]["split_first_clause"] = split_first
        va._init_state()                       # the REAL initializer
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
                for tok in re.findall(r"\S+\s*", reply):
                    on_token(tok)
                return reply
        va.llm = _LLM()
        return va, spoken

PASS = FAIL = 0
FAILURES: list[str] = []


def check(cond, label, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {label}")
    else:
        FAIL += 1
        FAILURES.append(f"{label} {detail}")
        print(f"  FAIL  {label} {detail}")


UNBACKED = [
    "I'll keep an eye on it.",                                 # JARVIS 2026-09-30
    "I'll alert you right away when a new call comes in.",     # JARVIS 2026-11-08
    "I'll start by scanning the project folder.",              # JARVIS 2026-10-27
    "Once that's done, I'll review the results.",
    "I've opened the coding projects folder for you.",         # Nova, CLAUDE.md
    "I've set a reminder for 5 PM.",
    "Sure, I'll remind you tomorrow morning.",
    "I'll let you know when it starts raining.",
    "I have gone ahead and added it to your calendar.",
    "I'll follow up on that later.",
    "I'll look into it and get back to you.",
]
HONEST = [
    "I'll tell you a joke about computers.",
    "I think you'll really like that movie.",
    "I've always loved jazz, especially Coltrane.",
    "I have a question for you.",
    "I'll remember that.",
    "Would you like me to set a reminder?",
    "You can set a timer by asking me.",
    "I've heard that one before.",
    "I have to say, that's impressive.",
    "Let me know if you want more ideas.",
    "I'll be honest, I don't know.",
]


def main() -> int:
    print("── unbacked sentences are caught")
    for s in UNBACKED:
        check(honesty.unbacked(s) is not None, f"caught: {s!r}")
    print("\n── ordinary sentences are left alone")
    for s in HONEST:
        check(honesty.unbacked(s) is None, f"kept:   {s!r}", f"-> {honesty.unbacked(s)}")

    print("\n── inside the real _stream_response")
    reply = ("Sure thing. I'll keep an eye on it and let you know. "
             "I'll also remind you tonight. Anything else?")
    va, spoken = H.build(reply, split_first=False)
    stored = {}
    va.memory.add_turn = lambda role, content: stored.__setitem__(role, content)
    va._stream_response("let me know if the price drops")
    said = " ".join(spoken)
    check(honesty.CANT_WATCH in said, "the first promise is replaced by the truth", said)
    check("keep an eye" not in said and "remind you tonight" not in said,
          "no promise is spoken", said)
    check(said.count(honesty.CANT_WATCH) == 1, "one correction, not a chorus", said)
    check("Sure thing." in said and "Anything else?" in said,
          "everything else is spoken as written", said)
    check(stored.get("assistant", "").find("keep an eye") == -1
          and honesty.CANT_WATCH in stored.get("assistant", ""),
          "what is STORED is what she said (no promise left in history)",
          stored.get("assistant"))

    reply = "The capital of France is Paris. It has been since the tenth century."
    va, spoken = H.build(reply, split_first=False)
    va._stream_response("what's the capital of France")
    check(" ".join(spoken).replace("  ", " ") == reply, "an honest reply is untouched",
          " ".join(spoken))

    print("\n── a typed turn is answered in text, not aloud")
    va, spoken = H.build("Paris is the capital of France.", split_first=False)
    va._silent_turn = True
    sent = []
    va.ws.send_message = lambda role, content: sent.append((role, content))
    va._stream_response("capital of france")
    check(spoken == [], "nothing is spoken on a silent turn", f"spoke {spoken}")
    check(sent and sent[-1][1].startswith("Paris"), "...but the answer reaches the app",
          f"{sent}")

    va, spoken = H.build("I'll keep an eye on it.", split_first=False)
    va.config["llm"]["honesty_filter"] = False
    va._stream_response("watch it")
    check(spoken == ["I'll keep an eye on it."], "honesty_filter=false is an instant rollback",
          f"{spoken}")

    print(f"\n  {PASS}/{PASS + FAIL}")
    if FAILURES:
        for f in FAILURES:
            print(f"    ✗ {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
