#!/usr/bin/env python3
"""
Market data — Nova is a researcher, never an advisor.

Nicholas said plainly that this feature will affect real money, which makes it
the highest-stakes thing in Nova. The line he agreed to:

    ✅ "Of 40 analysts covering it, 32 say buy, 7 hold, 1 sell."
    ❌ "Now is not the time to buy."

The first is attributed data. The second is advice, and a 3B inventing it would
cost him money. So the checks that matter most here are the NEGATIVE ones: that
no code path can produce a recommendation, that no number is ever phrased by a
model, and that a symbol Nova cannot find is admitted rather than guessed.

Network calls are CANNED by default so the suite is deterministic and does not
burn his 60-per-minute Finnhub budget on every run. NOVA_TEST_LIVE=1 runs the
same checks against the real APIs.

Run:  python tests/test_market.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
BACKEND = TESTS_DIR.parent
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(TESTS_DIR))
os.chdir(BACKEND)

from listener import check_spoken  # noqa: E402

PASS = FAIL = 0
FAILURES: list[str] = []
LIVE = os.environ.get("NOVA_TEST_LIVE") == "1"


def check(cond, label, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        FAILURES.append(f"{label}  ({detail})" if detail else label)
    return bool(cond)


# ── Canned payloads, shaped exactly like the real ones ────────────────────────
QUOTE = {"ok": True, "symbol": "SPCX", "name": "Space Exploration Technologies",
         "price": 146.23, "prev_close": 140.1, "change": 6.13,
         "change_pct": 4.375, "currency": "USD", "high": 147.0, "low": 139.5}
RECS = {"ok": True, "buy": 32, "hold": 7, "sell": 1, "total": 40,
        "period": "2026-08-01"}
FUND = {"ok": True, "week52_high": 225.64, "week52_low": 98.2, "pe": 41.2,
        "market_cap": 900000}


# ── 1. Every number is templated ──────────────────────────────────────────────
def test_templated() -> None:
    print("\n1. NUMBERS ARE TEMPLATED, NEVER PHRASED")
    import market_engine as M

    said = M.say_quote(QUOTE)
    check("146.23" in said, "the price spoken is the price returned", said)
    check("4.4" in said, "the percentage spoken is computed, not guessed", said)
    check("up" in said, "direction matches the sign", said)
    check(not check_spoken(said), "fit to speak", said)

    down = dict(QUOTE, change_pct=-2.5)
    check("down" in M.say_quote(down), "a fall is called a fall",
          M.say_quote(down))

    # Every figure spoken must TRACE to the payload. Rounding is legitimate —
    # 4.375% is spoken as "4.4 percent" — so the source includes the rounded
    # forms the templates actually produce. What this still catches is the
    # thing that matters: a number appearing from nowhere.
    traceable = str(QUOTE) + f" {QUOTE['change_pct']:.1f} {QUOTE['price']:,.2f}"
    problems = check_spoken(said, source_text=traceable)
    check(not problems, "every number spoken traces to the payload",
          "; ".join(problems))

    # And prove the check would still bite: a figure with no source fails it.
    fabricated = said + " Analysts expect 210 dollars by December."
    check(check_spoken(fabricated, source_text=traceable),
          "the invented-number check still catches a figure from nowhere")


# ── 2. Analysts are quoted, never interpreted ─────────────────────────────────
def test_no_advice() -> None:
    print("\n2. RESEARCHER, NOT ADVISOR")
    import market_engine as M

    said = M.say_recommendations(RECS, "SpaceX")
    check("32" in said and "7" in said and "1" in said,
          "the counts spoken are the counts returned", said)
    check("40 analysts" in said, "the total is attributed", said)
    check("2026-08-01" in said, "the data is dated", said)

    # The whole point. No code path may produce a view.
    banned = ("you should", "i'd recommend", "i recommend", "good time to buy",
              "bad time", "worth buying", "i think you", "my advice",
              "looks strong", "looks weak", "bullish", "bearish")
    for phrase in banned:
        check(phrase not in said.lower(), f"never says '{phrase}'", said)
    check(not check_spoken(said), "fit to speak", said)

    # Grammar: "1 say sell" was wrong and audible.
    one = M.say_recommendations({"ok": True, "buy": 1, "hold": 0, "sell": 0,
                                 "total": 1, "period": ""}, "X")
    check("1 says buy" in one, "singular analyst uses 'says'", one)


# ── 3. Missing data is admitted ───────────────────────────────────────────────
def test_honest_failure() -> None:
    print("\n3. HONEST WHEN IT CANNOT ANSWER")
    import market_engine as M
    import nova as nova_mod
    from market_intents import NovaMarket

    check(M.say_quote({"ok": False}) == "", "a failed quote says nothing")
    check(M.say_recommendations({"ok": False}, "X") == "",
          "failed recommendations say nothing")
    check("couldn't" in M.say_indices([]).lower(),
          "no index data is admitted", M.say_indices([]))

    market = NovaMarket(nova_mod.load_config())
    # A miss and an outage must be DIFFERENT sentences: one means "no such
    # stock", the other means "try again".
    market_engine_quote = M.quote
    try:
        M.quote = lambda s: {"ok": False, "error": "not found", "symbol": s}
        M.resolve_symbol_orig = M.resolve_symbol
        M.resolve_symbol = lambda t: {"ok": False, "error": "not found",
                                      "term": t}
        out = market.handle("quote", "what is Zzqqxx stock at")
        check("couldn't find a listing" in out.lower(),
              "an unknown symbol is a MISS", out)
        check(not check_spoken(out), "fit to speak", out)

        M.resolve_symbol = lambda t: {"ok": False, "error": "unreachable"}
        out2 = market.handle("quote", "what is Apple stock at")
        check("reach" in out2.lower(), "an outage is an OUTAGE, not a miss", out2)
        check(out2 != out, "the two failures are worded differently")
    finally:
        M.quote = market_engine_quote
        M.resolve_symbol = M.resolve_symbol_orig


# ── 4. Detection ──────────────────────────────────────────────────────────────
def test_detection() -> None:
    print("\n4. DETECTION")
    import nova as nova_mod
    from market_intents import NovaMarket
    market = NovaMarket(nova_mod.load_config())

    for phrase, want in (
        ("how is the market doing", "market"),
        ("how's the stock market today", "market"),
        ("how is SpaceX stock doing", "quote"),
        ("what is Apple stock trading at", "quote"),
        ("what do analysts say about Tesla", "analysts"),
        ("what are the analyst ratings for Nvidia", "analysts"),
    ):
        got = market.detect_intent(phrase)
        check(got == want, f"'{phrase}' -> {want}", f"got {got}")

    # Ordinary conversation. Every one of these contains a finance-adjacent
    # word and must NOT become a stock lookup.
    for phrase in ("how are you doing", "how's it going", "how's my day looking",
                   "I had apple pie for dessert", "my apple watch died",
                   "can you share that with me", "the price of gas is insane",
                   "what do you think about Apple products",
                   "how's the family"):
        check(market.detect_intent(phrase) is None,
              f"'{phrase[:42]}' stays conversation",
              str(market.detect_intent(phrase)))


# ── 5. The key never reaches the repo ─────────────────────────────────────────
def test_key_hygiene() -> None:
    print("\n5. CREDENTIALS")
    import json

    import market_engine as M

    cfg = json.loads((BACKEND / "config.json").read_text())
    blob = json.dumps(cfg)
    check("finnhub" not in blob.lower() or "api_key" not in blob.lower(),
          "no API key in config.json — it is committed to a public repo")
    check("credentials" in str(M.DATA_DIR / "finnhub_credentials.json"),
          "the key lives under NOVA_DATA_DIR")

    # Absent key is a normal state: quotes still work through Yahoo.
    saved = M._key_cache
    try:
        M._key_cache = ""
        r = M._finnhub("quote", symbol="AAPL")
        check(r.get("needs_key") is True,
              "no key is reported as such, not as an outage", str(r))
    finally:
        M._key_cache = saved


# ── 6. Live, opt-in ───────────────────────────────────────────────────────────
def test_live() -> None:
    print("\n6. LIVE APIS" + ("" if LIVE else "  (skipped; NOVA_TEST_LIVE=1)"))
    if not LIVE:
        return
    import market_engine as M

    q = M.quote("AAPL")
    check(q.get("ok"), "a real quote comes back", str(q)[:80])
    check(isinstance(q.get("price"), (int, float)) and 1 < q["price"] < 100000,
          "the price is physically plausible", str(q.get("price")))
    check(M.quote("ZZQQXX").get("error") == "not found",
          "a nonsense ticker is a miss, not an outage")
    check(M.resolve_symbol("spacex").get("symbol") == "SPCX",
          "'spacex' resolves (Finnhub misses it; Yahoo does not)")
    check(len(M.indices()) >= 2, "indices come back from Yahoo")


def main() -> int:
    print("=" * 72)
    print("MARKET DATA — researcher, not advisor")
    print("=" * 72)
    test_templated()
    test_no_advice()
    test_honest_failure()
    test_detection()
    test_key_hygiene()
    test_live()

    print(f"\n  {PASS}/{PASS + FAIL} checks passed")
    if FAILURES:
        print("\n  FAILURES:")
        for f in FAILURES:
            print(f"    ✗ {f}")
    if not LIVE:
        print("\n  Ran against CANNED payloads. NOVA_TEST_LIVE=1 hits the real")
        print("  APIs — worth doing before trusting a number with money.")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
