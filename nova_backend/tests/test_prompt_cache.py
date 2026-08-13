#!/usr/bin/env python3
"""
Prompt-cache tests — the guarantee is that speed changed and NOTHING ELSE did.

A prompt cache that is fast and subtly wrong is worse than no cache: it would
answer the next turn from a context the model never actually saw, and nothing
would raise. So the headline check here is not the speed-up, it is that the
cached path returns text IDENTICAL to the uncached path at temperature 0.

Fidelity: real LLMEngine, real model, real MLX. Slow on purpose (~1 min) —
the whole point is exercising the thing Nova actually runs.
"""
from __future__ import annotations

import json
import os
import sys
import time
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


from system_prompt import build_system_prompt
from memory import NovaMemory
from llm_engine import LLMEngine, _ENDS_SENTENCE

config = json.loads((_Path(BACKEND) / "config.json").read_text())
DATA_DIR = _Path(os.environ.get(
    "NOVA_DATA_DIR", _Path.home() / "Library/Application Support/Nova")).expanduser()

memory_ctx = ""
history: list[dict] = []
try:
    _mem = NovaMemory(DATA_DIR / "nova_memory.db")
    memory_ctx = _mem.get_context_for_llm()
    history = _mem.get_recent_turns(n=10)
except Exception as exc:               # a fresh machine has no DB yet
    print(f"  (no memory DB: {exc}; testing with empty context)")

llm_cfg = dict(config["llm"])
llm_cfg["temperature"] = 0.0     # greedy — any text difference is a REAL difference
llm_cfg["max_tokens"] = 80

print("Loading the real model…", flush=True)
engine = LLMEngine(llm_cfg)

SYS = build_system_prompt(config, memory_context=memory_ctx, rag_context="")

TURNS = [
    "what do you think about the weather today",
    "give me a good movie recommendation",
    "how are you doing",
    "tell me something interesting",
    "explain why the sky is blue",
]


def diff_report(want: str, got: str) -> str:
    """Full text plus where they diverge.

    Truncating this to 90 characters once hid a real mismatch behind two
    strings that looked identical, and diagnosing it needed a separate script.
    A mismatch here is rare and serious, so print everything.
    """
    i = next((k for k, (a, b) in enumerate(zip(want, got)) if a != b),
             min(len(want), len(got)))
    return (f"diverges at char {i}\n"
            f"        want: {want!r}\n"
            f"        got : {got!r}\n"
            f"        want@{i}: {want[max(0, i - 40):i + 60]!r}\n"
            f"        got @{i}: {got[max(0, i - 40):i + 60]!r}")


def stream(msg, system=SYS):
    """One turn through the real engine; returns (text, time_to_first_token)."""
    t0 = time.perf_counter()
    first = {}

    def on_token(_tok):
        first.setdefault("t", time.perf_counter() - t0)

    text = engine.stream(system_prompt=system, history=history,
                         user_message=msg, on_token=on_token)
    return text, first.get("t", float("nan"))


# ══════════════════════════════════════════════════════════════════════════
section("BASELINE — cache disabled (what Nova did before)")
# ══════════════════════════════════════════════════════════════════════════
engine._cache_enabled = False
engine._drop_cache()
baseline, base_ttft = {}, []
for m in TURNS:
    text, ttft = stream(m)
    baseline[m] = text
    base_ttft.append(ttft)
    print(f"  {ttft:.3f}s  {m[:44]}")


# ══════════════════════════════════════════════════════════════════════════
section("SAME WORDS — the cache must not change one thing Nova says")
# ══════════════════════════════════════════════════════════════════════════
engine._cache_enabled = True
engine._drop_cache()
engine.warm(SYS)
cached_ttft = []
for m in TURNS:
    text, ttft = stream(m)
    cached_ttft.append(ttft)
    check(text == baseline[m], f"identical answer: {m[:40]}",
          "" if text == baseline[m] else diff_report(baseline[m], text))


# ══════════════════════════════════════════════════════════════════════════
section("STILL CORRECT WHEN THE PROMPT CHANGES UNDERNEATH IT")
# ══════════════════════════════════════════════════════════════════════════
# The system prompt carries a clock and live memory, so it is DIFFERENT on most
# turns. The cache must notice and reprocess, not serve the stale prefix.
alt_sys = build_system_prompt(config, memory_context="He has a dog named Rex.",
                              rag_context="Some retrieved document text.")
q = "what should I do this evening"

engine._cache_enabled = False
engine._drop_cache()
want, _ = stream(q, system=alt_sys)

engine._cache_enabled = True
engine._drop_cache()
engine.warm(SYS)                       # warmed on the OLD system prompt
got, _ = stream(q, system=alt_sys)
check(want == got, "a changed system prompt is answered correctly anyway",
      "" if want == got else diff_report(want, got))


# ══════════════════════════════════════════════════════════════════════════
section("A FAILURE MID-GENERATION MUST NOT POISON THE NEXT TURN")
# ══════════════════════════════════════════════════════════════════════════
# If a turn dies partway, the cache holds tokens that _cache_ids does not
# describe. Left alone that silently corrupts every later answer.
engine._drop_cache()
engine.warm(SYS)


class Boom(Exception):
    pass


try:
    engine.stream(system_prompt=SYS, history=history, user_message=TURNS[0],
                  on_token=lambda _t: (_ for _ in ()).throw(Boom()))
except Boom:
    pass
except Exception as exc:
    print(f"        (raised {type(exc).__name__} instead of Boom)")

after, _ = stream(TURNS[1])
check(after == baseline[TURNS[1]], "the turn after a mid-stream failure is correct",
      "" if after == baseline[TURNS[1]] else diff_report(baseline[TURNS[1]], after))

check(engine._cache_ids == [] or engine._cache is not None,
      "cache and its token list never disagree")


# ══════════════════════════════════════════════════════════════════════════
section("THE CONFIG SWITCH REALLY DISABLES IT")
# ══════════════════════════════════════════════════════════════════════════
off_engine_cfg = dict(llm_cfg)
off_engine_cfg["prompt_cache"] = False
engine._cache_enabled = bool(off_engine_cfg.get("prompt_cache", True))
engine._drop_cache()
check(engine.warm(SYS) == 0, "warm() is a no-op when prompt_cache is false")
_text, _t = stream(TURNS[2])
check(engine._cache is None, "no cache is built when prompt_cache is false")
check(_text == baseline[TURNS[2]], "disabled path still answers correctly")


# ══════════════════════════════════════════════════════════════════════════
section("THE LENGTH BUDGET IS INDEPENDENT OF THE CACHE")
# ══════════════════════════════════════════════════════════════════════════
# These are unrelated switches, and they were entangled: the uncached path had
# its own copy of the generation loop, so the word budget was only applied when
# the cache happened to be ON. Turning the cache off silently made Nova
# long-winded again.
budget_cfg = dict(llm_cfg)
budget_cfg["soft_max_words"] = 25
budget_cfg["max_tokens"] = 200
probe = LLMEngine(budget_cfg)
LONG_Q = "explain in detail why the sky is blue"

probe._cache_enabled = False
probe._drop_cache()
off_text = probe.stream(system_prompt=SYS, history=history,
                        user_message=LONG_Q, on_token=lambda _t: None)
probe._cache_enabled = True
probe._drop_cache()
probe.warm(SYS)
on_text = probe.stream(system_prompt=SYS, history=history,
                       user_message=LONG_Q, on_token=lambda _t: None)

check(off_text == on_text,
      "the same words come out with the cache off and on",
      f"off={len(off_text.split())}w on={len(on_text.split())}w")
for label, txt in (("cache off", off_text), ("cache on", on_text)):
    n = len(txt.split())
    check(n < 200, f"the budget is enforced with {label} ({n} words)",
          "" if n < 200 else f"ran to the token limit: {txt[:110]!r}")
    check(_ENDS_SENTENCE.search(txt.strip()) is not None,
          f"…and it stops on a complete sentence with {label}",
          f"ends: {txt.strip()[-60:]!r}")


# ══════════════════════════════════════════════════════════════════════════
section("SPEED  (informational — correctness above is the guarantee)")
# ══════════════════════════════════════════════════════════════════════════
avg = lambda x: sum(x) / len(x)
print(f"  time to first token, cache OFF : {avg(base_ttft):.3f}s")
print(f"  time to first token, cache ON  : {avg(cached_ttft):.3f}s")
print(f"  saved                          : {avg(base_ttft) - avg(cached_ttft):.3f}s per turn")
check(avg(cached_ttft) < avg(base_ttft),
      "the cache is actually faster than no cache",
      f"{avg(base_ttft):.3f}s -> {avg(cached_ttft):.3f}s")


# ══════════════════════════════════════════════════════════════════════════
section("RESULT")
# ══════════════════════════════════════════════════════════════════════════
print(f"\n  {PASS}/{PASS + FAIL}")
for f in FAILURES:
    print(f"    ✗ {f}")
sys.exit(1 if FAIL else 0)
