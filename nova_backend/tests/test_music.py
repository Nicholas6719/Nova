#!/usr/bin/env python3
"""
Music control, including playing something BY NAME.

Spotify's desktop app exposes six AppleScript commands and none of them is
`search`; `play track` needs a URI. So "play some rain sounds" used to fall all
the way through to the unsupported-action guard and get an honest refusal. Nova
now resolves the name to a URI first (see spotify_search.py) and plays it
locally.

The risk in adding that is not the search — it is SHADOWING. A greedy "play X"
matcher turns "play the next song" into a search for a song called "the next
song". So most of this file is about what must NOT become a named request.

Fidelity: the REAL `_match_music` router and the REAL parsing. Only the things
with side effects are stubbed — AppleScript (which would launch and control
Spotify) and the network call. Per the harness rule: stub side effects, never
verdicts.
"""
from __future__ import annotations

import os
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


from tools import NovaTools
import spotify_search

# ── A player we can watch, that never touches the real Spotify ───────────────
commands: list[str] = []


class Rig(NovaTools):
    """Real router, real regexes; AppleScript captured instead of executed."""

    def __init__(self):
        self.state = {"player_state": "playing",
                      "name": "Weightless", "artist": "Marconi Union"}
        self.launched = False

    # every AppleScript call funnels through _osa
    def _osa(self, script):
        commands.append(script)
        # Order matters: the now-playing script contains BOTH "name of
        # current track" and "player state", so the more specific one has to
        # be checked first or this stub returns a bare state with no
        # delimiter and _now_playing silently gives up.
        if "name of current track" in script:
            return True, f"{self.state['name']}||{self.state['artist']}||{self.state['player_state']}"
        if "player state" in script:
            return True, self.state["player_state"]
        if "sound volume" in script:
            return True, "55"
        if "player position" in script or "duration" in script:
            return True, "42"
        return True, ""

    def _running_player(self, launch_if_none=False):
        if launch_if_none:
            self.launched = True
        return "Spotify"


def run(phrase):
    commands.clear()
    rig = Rig()
    return rig._match_music(phrase.lower()), rig


# ══════════════════════════════════════════════════════════════════════════
section("TRANSPORT STILL WINS  (nothing below may become a search)")
# ══════════════════════════════════════════════════════════════════════════
# Each of these must be handled by its own branch. The tell is that no
# `play track "spotify:...` command is ever issued.
TRANSPORT = [
    "play", "play music", "play the music", "resume", "keep playing",
    "pause", "pause the music", "stop the music",
    "next", "skip", "next song", "skip the track",
    "play the next song", "play the previous song", "play the last track",
    "previous song", "go back a song",
    "what's playing", "what song is this", "who sings this",
    "restart the song", "restart this song", "play it again", "play that again",
    "shuffle the music", "turn off shuffle", "repeat on",
    "turn up the music volume", "set the music volume to 40",
    "skip ahead 30 seconds", "jump back 10 seconds",
    "how much longer is this song",
]
for phrase in TRANSPORT:
    resp, _ = run(phrase)
    searched = any("play track" in c for c in commands)
    check(resp is not None and not searched,
          f"handled without a search: {phrase[:42]!r}",
          "" if resp is not None and not searched
          else (f"resp={resp!r} searched={searched}"))


# ══════════════════════════════════════════════════════════════════════════
section("\"JUST PUT MUSIC ON\" NEEDS NO CREDENTIALS  (and never did)")
# ══════════════════════════════════════════════════════════════════════════
# This is what Nicholas actually asks for most, and what he had with the
# assistant he used before: say the words, Spotify opens and plays whatever
# context it was last in. Six of these used to fall through to "I can't do that
# one yet", which is the opposite of true. None of them touches the network.
JUST_PLAY = [
    "play music", "play", "play the music", "start the music", "start some music",
    "resume", "keep playing", "play me some music",
    "put on some music", "put on music", "throw on some music",
    "get some music going", "turn some music on",
    "play something", "play anything", "music please",
]
_saved_conf = spotify_search.is_configured
spotify_search.is_configured = lambda: (_ for _ in ()).throw(
    AssertionError("the plain 'play music' path must never consult Spotify search"))
for phrase in JUST_PLAY:
    try:
        resp, rig = run(phrase)
    except AssertionError as exc:
        check(False, f"no search needed: {phrase[:40]!r}", str(exc))
        continue
    searched = any("play track" in c for c in commands)
    check(resp is not None and not searched,
          f"plays with no credentials: {phrase[:40]!r}",
          "" if resp is not None and not searched else f"resp={resp!r}")
spotify_search.is_configured = _saved_conf


# ══════════════════════════════════════════════════════════════════════════
section("'PLAY IT AGAIN' RESTARTS, IT DOES NOT SEARCH")
# ══════════════════════════════════════════════════════════════════════════
# Deliberately NOT bare "restart" or "start over": "let's start over" is in
# the adversarial corpus and must reach the LLM, and bare "restart" could mean
# restarting the Mac. A music word or a pronoun is required.
for phrase in ("play it again", "play that again", "again", "restart this song"):
    resp, _ = run(phrase)
    restarted = any("player position to 0" in c for c in commands)
    check(restarted, f"{phrase!r} restarts the track", f"resp={resp!r}")


# ══════════════════════════════════════════════════════════════════════════
section("NAMED REQUESTS ARE PARSED CORRECTLY")
# ══════════════════════════════════════════════════════════════════════════
rig = Rig()
for phrase, want_q, want_kind in [
    ("play some rain sounds", "rain sounds", None),
    ("play bohemian rhapsody", "bohemian rhapsody", None),
    ("put on some jazz", "jazz", None),
    ("throw on some blues", "blues", None),
    ("play the album abbey road", "abbey road", "album"),
    ("play a playlist of lo-fi beats", "lo-fi beats", "playlist"),
    ("play the beatles on spotify", "beatles", None),
    ("play taylor swift", "taylor swift", None),
]:
    got = rig._named_request(phrase)
    ok = got is not None and got[0] == want_q and got[1] == want_kind
    check(ok, f"{phrase[:40]!r} -> {want_q!r}", "" if ok else f"got {got!r}")

# The article must not chew into the following word.
for phrase in ("play something", "play anything", "play some music"):
    got = rig._named_request(phrase)
    check(got is None, f"not a name: {phrase!r}", "" if got is None else f"got {got!r}")


# ══════════════════════════════════════════════════════════════════════════
section("PLAYING BY NAME: VERIFIED, NEVER CLAIMED")
# ══════════════════════════════════════════════════════════════════════════
_real_search = spotify_search.search
_real_conf = spotify_search.is_configured

spotify_search.is_configured = lambda: True
spotify_search.search = lambda q, prefer=None, market="US": {
    "ok": True, "uri": "spotify:track:TESTURI", "kind": "track",
    "label": "Weightless by Marconi Union"}

resp, rig = run("play some rain sounds")
check(any('play track "spotify:track:TESTURI"' in c for c in commands),
      "the resolved URI is handed to the desktop app")
check(resp and "Weightless" in resp, "it reports what is actually playing", f"{resp!r}")

# The player refusing to start must NOT be reported as success.
commands.clear()
rig = Rig()
rig.state["player_state"] = "paused"
resp = rig._match_music("play some rain sounds")
check(resp is not None and "wouldn't start" in resp,
      "a player that never started is admitted, not claimed", f"{resp!r}")

# Read the ACTUAL track back, not the search label.
commands.clear()
rig = Rig()
rig.state.update({"name": "Rainfall", "artist": "Ambient Co"})
resp = rig._match_music("play some rain sounds")
check(resp and "Rainfall" in resp and "Weightless" not in resp,
      "the spoken answer comes from the player, not the search result", f"{resp!r}")


# ══════════════════════════════════════════════════════════════════════════
section("FAILURE IS ADMITTED, WITH SOMETHING ACTIONABLE")
# ══════════════════════════════════════════════════════════════════════════
# `open spotify:search:...` LAUNCHES SPOTIFY. Writing this test without
# stubbing it did exactly that on a machine where Spotify was closed — the same
# "tools.match() executes as it matches" trap that once launched Spotify during
# a routing probe. Every path below can reach it, so the stub wraps all of them.
import subprocess as _sp
_orig_run = _sp.run
opened: list = []


class _FakeCompleted:
    returncode = 0


def _no_launch(*a, **k):
    opened.append(list(a[0]) if a and isinstance(a[0], (list, tuple)) else a)
    return _FakeCompleted()


_sp.run = _no_launch
spotify_search.search = lambda q, prefer=None, market="US": {
    "ok": False, "reason": "not_found"}
resp, _ = run("play some rain sounds")
check(resp and "couldn't find" in resp.lower(), "a miss says so", f"{resp!r}")
check(resp and "playlist" in resp.lower(),
      "…and explains that personal playlists aren't searchable this way",
      f"{resp!r}")

# An unreachable API should not end in an apology either: fall back to the
# thing that works without the network being helpful.
opened.clear()
spotify_search.search = lambda q, prefer=None, market="US": {
    "ok": False, "reason": "unreachable"}
resp, _ = run("play some rain sounds")
check(resp and "opened a search" in resp.lower(),
      "an unreachable API falls back to opening the search", f"{resp!r}")
check(any("spotify:search:" in " ".join(o) for o in opened if o),
      "…and it really is a search URI", f"{opened!r}")

# With no credentials a NAMED request must not nag him about API keys. It does
# the useful thing the desktop app can do unaided: opens that search in
# Spotify, and says plainly that it stopped one step short of playing.
spotify_search.is_configured = lambda: False
opened.clear()
resp, _ = run("play some rain sounds")
check(resp and "client id" not in resp.lower() and "secret" not in resp.lower(),
      "no credentials does NOT nag him about API keys", f"{resp!r}")
check(any("spotify:search:" in " ".join(o) for o in opened if o),
      "…it opens the search in his Spotify instead", f"{opened!r}")
check(resp and "opened a search" in resp.lower(),
      "…and says so, without implying it started playing", f"{resp!r}")

spotify_search.search = _real_search
spotify_search.is_configured = _real_conf
check(_sp.run is _no_launch, "the launch stub stayed in place for every path above")


# ══════════════════════════════════════════════════════════════════════════
section("NOTHING IS SPOKEN THAT SHOULDN'T BE")
# ══════════════════════════════════════════════════════════════════════════
import re as _re
spotify_search.is_configured = lambda: True
spotify_search.search = lambda q, prefer=None, market="US": {
    "ok": True, "uri": "spotify:track:X", "kind": "track", "label": "A Song by An Artist"}
for phrase in ("play some rain sounds", "what's playing", "pause the music",
               "turn up the music volume"):
    resp, _ = run(phrase)
    if not resp:
        continue
    check(not _re.search(r"[*_`#—–]|spotify:track:|https?://", resp),
          f"speakable: {resp[:48]!r}",
          "" if not _re.search(r"[*_`#—–]|spotify:track:|https?://", resp)
          else "leaked a URI, markdown, or a URL")
spotify_search.search = _real_search
spotify_search.is_configured = _real_conf


_sp.run = _orig_run          # never leave the real one stubbed


# ══════════════════════════════════════════════════════════════════════════
section("RESULT")
# ══════════════════════════════════════════════════════════════════════════
print(f"\n  {PASS}/{PASS + FAIL}")
for f in FAILURES:
    print(f"    ✗ {f}")
sys.exit(1 if FAIL else 0)
