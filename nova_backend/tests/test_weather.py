#!/usr/bin/env python3
"""
Weather — and the location plumbing it depends on.

Weather is the one deliberate widening of CLAUDE.md invariant 3 beyond MapKit
(Open-Meteo, chosen by Nicholas 2026-08-13: no API key, no account). Two things
have to hold:

  1. it must not STEAL ordinary conversation — a loose matcher here would do
     to weather what the memory intent once did to "what's my calendar"
  2. it must never INVENT a forecast. Every spoken number is templated from the
     response; the 3B is not asked to phrase a temperature, because a wrong one
     sounds exactly like a right one to someone listening.

Fidelity: real intent code and real formatting against canned API payloads
(no network needed). The live-network checks run only if open-meteo is
reachable, and are skipped rather than failed when it is not.
"""
from __future__ import annotations

import json
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


import weather_engine as we
from weather_intents import NovaWeather

config = json.loads((_Path(BACKEND) / "config.json").read_text())
w = NovaWeather(config)


# ══════════════════════════════════════════════════════════════════════════
section("IT ANSWERS WEATHER QUESTIONS")
# ══════════════════════════════════════════════════════════════════════════
WEATHER = {
    "what's the weather": "now",
    "what's the weather today": "now",
    "what's the temperature": "now",
    "how hot is it": "now",
    "how cold is it outside": "now",
    "do I need an umbrella": "now",
    "is it snowing": "now",
    "is it going to rain tomorrow": "tomorrow",
    "weather for tomorrow": "tomorrow",
    "what's the forecast this week": "week",
    "what's the weather like for the next few days": "week",
    "what's the weather in Boston": "now",
}
for phrase, want in WEATHER.items():
    got = w.detect_intent(phrase)
    check(got == want, f"{phrase[:44]!r} -> {want}", "" if got == want else f"got {got!r}")


# ══════════════════════════════════════════════════════════════════════════
section("IT STEALS NOTHING ELSE")
# ══════════════════════════════════════════════════════════════════════════
NOT_WEATHER = [
    "what time is it",
    "tell me something interesting",
    "how are you doing",
    "open spotify and play some music",
    "play some rain sounds",
    "search for rain sounds on youtube",
    "remind me to call mom at five",
    "what's on my calendar today",
    "what does my finance degree roadmap say",
    "how far is the nearest coffee shop",
    "do you know who Jarvis is",
    "what's the meaning of life",
    "delete the snow photos",
    "move my rainy day playlist to Documents",
]
for phrase in NOT_WEATHER:
    got = w.detect_intent(phrase)
    check(got is None, f"not weather: {phrase[:44]!r}", "" if got is None else f"claimed {got!r}")


# ══════════════════════════════════════════════════════════════════════════
section("PLACE NAMES, AND THINGS THAT ONLY LOOK LIKE THEM")
# ══════════════════════════════════════════════════════════════════════════
for phrase, want in [("what's the weather in Boston", "Boston"),
                     ("forecast for New York", "New York"),
                     ("weather in San Francisco", "San Francisco"),
                     ("what's the weather", None),
                     ("what's the weather today", None),
                     ("is it going to rain tomorrow", None),
                     ("what's the weather for tonight", None)]:
    got = w._named_place(phrase)
    check(got == want, f"place in {phrase[:40]!r} -> {want!r}",
          "" if got == want else f"got {got!r}")


# ══════════════════════════════════════════════════════════════════════════
section("EVERY SPOKEN NUMBER COMES FROM THE RESPONSE")
# ══════════════════════════════════════════════════════════════════════════
CANNED = {
    "ok": True,
    "current": {"temp": 71.4, "feels_like": 71.9, "humidity": 55,
                "wind": 6.0, "code": 2},
    "days": [
        {"code": 2, "high": 78.2, "low": 61.1, "rain_chance": 10},
        {"code": 61, "high": 70.0, "low": 58.0, "rain_chance": 80},
        {"code": 0, "high": 75.0, "low": 60.0, "rain_chance": 0},
    ],
}
now = we.say_current(CANNED)
check("71" in now, "the current temperature is the one returned", now)
check("78" in now and "61" in now, "today's high and low are the ones returned", now)
check("10 percent" not in now, "a 10% rain chance is not worth mentioning", now)
check("feels like" not in now.lower(),
      "'feels like' is omitted when it matches the temperature", now)

tom = we.say_day(CANNED, 1, "tomorrow")
check("70" in tom and "58" in tom, "tomorrow's numbers are the ones returned", tom)
check("80 percent" in tom, "a high rain chance IS mentioned", tom)
check("raining" in tom, "the condition is described in words", tom)

hot = dict(CANNED)
hot["current"] = {"temp": 91.0, "feels_like": 103.0, "humidity": 70,
                  "wind": 3.0, "code": 0}
check("feels like 103" in we.say_current(hot).lower(),
      "'feels like' IS said when it differs sharply", we.say_current(hot))


# ══════════════════════════════════════════════════════════════════════════
section("NOTHING UNSPEAKABLE, NOTHING INVENTED")
# ══════════════════════════════════════════════════════════════════════════
spoken = [now, tom, we.say_current(CANNED, "Boston, Massachusetts"),
          we.say_day(CANNED, 2, "the day after")]
for s in spoken:
    check(not re.search(r"[*_`#—–]|\bhttps?://|\d+\.\d+", s),
          f"speakable: {s[:52]!r}",
          "" if not re.search(r"[*_`#—–]|\bhttps?://|\d+\.\d+", s)
          else "contains markdown, a URL, or an unrounded decimal")
    check("nicholas" not in s.lower(),
          f"no third person: {s[:44]!r}")


# ══════════════════════════════════════════════════════════════════════════
section("FAILURE IS ADMITTED, NOT GUESSED")
# ══════════════════════════════════════════════════════════════════════════
check(we.describe_code(None) == "unsettled", "an unknown code degrades to a word")
check(we.describe_code(99) == "thunderstorms with hail", "a known code is described")

empty = we.say_current({"ok": True, "current": {}, "days": []})
check("no current temperature" in empty.lower(),
      "a response with no temperature says so", empty)
check(we.say_day(CANNED, 6, "Saturday").startswith("I don't have"),
      "a day beyond the forecast says so", we.say_day(CANNED, 6, "Saturday"))

# No location and no place named -> decline, and suggest what DOES work.
import maps_engine
_real = maps_engine.current_location
maps_engine.current_location = lambda *a, **k: None
try:
    out = w.handle("what's the weather", "now")
finally:
    maps_engine.current_location = _real
check("can't get your location" in out.lower(), "no location is admitted plainly", out)
check("boston" in out.lower(), "…and it offers the thing that still works", out)


# ══════════════════════════════════════════════════════════════════════════
section("THE KILL SWITCH")
# ══════════════════════════════════════════════════════════════════════════
off = NovaWeather({**config, "weather": {"enabled": False}})
check(off.detect_intent("what's the weather") is None,
      "weather.enabled false stops Nova answering weather at all")


# ══════════════════════════════════════════════════════════════════════════
section("LOCATION IS REQUESTED ON DEMAND, NOT ONLY AT LAUNCH")
# ══════════════════════════════════════════════════════════════════════════
# The app used to volunteer a fix exactly once at launch, in a POST that raced
# the backend's own startup and was dropped. Nova then had no coordinate for
# the whole session.
import time as _time

maps_engine._location_cache.update({"at": 0.0, "coord": None, "denied": False})
asked = {"n": 0}


def _fake_request():
    asked["n"] += 1
    # Answer the way the app does, via the real entry point.
    maps_engine.set_location_from_app(
        {"available": True, "lat": 42.2792, "lon": -71.4161, "accuracy_m": 65})


maps_engine.set_location_requester(_fake_request)
coord = maps_engine.current_location()
check(asked["n"] == 1, "a missing fix causes the app to be asked exactly once",
      f"asked {asked['n']} times")
check(coord is not None, "and the fix that comes back is used", f"{coord is not None}")

# A second question inside the TTL must reuse it, not ask again.
before = asked["n"]
maps_engine.current_location()
check(asked["n"] == before, "a fresh fix is reused rather than re-requested")

# A denial must not turn into repeated prompting.
maps_engine._location_cache.update({"at": 0.0, "coord": None, "denied": True})
before = asked["n"]
check(maps_engine.current_location() is None, "a denied location returns nothing")
check(asked["n"] == before, "…and the app is NOT asked again after a denial")

maps_engine._location_cache.update({"at": 0.0, "coord": None, "denied": False})
maps_engine.set_location_requester(None)


# ══════════════════════════════════════════════════════════════════════════
section("LIVE (skipped when open-meteo is unreachable)")
# ══════════════════════════════════════════════════════════════════════════
geo = we.geocode("Boston")
if not geo.get("ok"):
    print("  SKIPPED — could not reach the geocoding service")
else:
    check(abs(geo["lat"] - 42.36) < 1.0 and abs(geo["lon"] + 71.06) < 1.0,
          "Boston geocodes to roughly the right place", f"{geo['name']}")
    live = we.fetch(geo["lat"], geo["lon"])
    check(live.get("ok"), "a real forecast comes back")
    if live.get("ok"):
        t = (live.get("current") or {}).get("temp")
        check(t is not None and -60 < float(t) < 140,
              "the temperature is physically plausible", f"{t}F")
        said = we.say_current(live, geo["name"])
        check(len(said) < 300 and said.endswith("."),
              "the spoken answer is short and a complete sentence", said)
    check(not we.geocode("zzzqqqnotaplace").get("ok"),
          "a nonsense place is not silently resolved")


# ══════════════════════════════════════════════════════════════════════════
section("RESULT")
# ══════════════════════════════════════════════════════════════════════════
print(f"\n  {PASS}/{PASS + FAIL}")
for f in FAILURES:
    print(f"    ✗ {f}")
sys.exit(1 if FAIL else 0)
