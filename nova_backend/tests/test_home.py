#!/usr/bin/env python3
"""
The home surface — what appears, what leaves, and where it sits.

Home is the screen Nicholas looks at when he is not asking anything, so almost
all of its behaviour is about ABSENCE: the greeting that goes, the status row
that goes with it, the Now Playing card that is only there while something is
playing. Absence is exactly the kind of thing that regresses silently, because
nothing on screen is wrong — there is just less of it than there should be.

Proves, with the real `NovaViews`, the real `panels`, and the real
`NovaWSServer`:
  * the greeting and the status row appear at launch and leave TOGETHER on his
    first sentence — the bug that shipped was half of that pair being
    conditional and the other half not
  * asking about CPU brings the row back, and it settles away again
  * Now Playing is absent unless something is actually playing
  * cards land in their slots, a move SWAPS rather than evicts, and the layout
    survives a restart
  * clear home empties the surface and restore home brings it back
  * a home render never blocks — it is on a ticker, and 350ms of AppleScript
    per redraw was the original design and the wrong one
  * an unchanged world sends NOTHING, so the ticker cannot churn the app
  * a live step list advances, and a failure stays on screen
  * the web search readback is templated and fit to speak

Side effects are stubbed, verdicts never are (rule 2): the music lookup and the
browser are replaced, the routing, payload building, slot arithmetic and
spoken text are all the shipping code.

Run:  python tests/test_home.py
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import tempfile
import time
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
BACKEND = TESTS_DIR.parent
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(TESTS_DIR))
os.chdir(BACKEND)

# The saved layout lives in NOVA_DATA_DIR. Point it at a temp dir BEFORE views
# is imported so a test run can never rearrange his real home screen.
_TMP = tempfile.mkdtemp(prefix="nova-home-test-")
os.environ["NOVA_DATA_DIR"] = _TMP

from listener import check_spoken  # noqa: E402

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


class CapturingWS:
    """The real NovaWSServer with only the transport replaced."""

    def __init__(self):
        from ws_server import NovaWSServer
        self.server = NovaWSServer(http_port=0, ws_port=0,
                                   on_text_message=lambda _t, _s=False: None)
        self.sent: list[dict] = []
        self.server._ws_broadcast = self.sent.append

    def __getattr__(self, name):
        return getattr(self.server, name)


class FakeTools:
    """The music lookup and the status readings, without the OS.

    `status_row` returns the same SHAPE the real one does, because what is
    under test is whether home shows it, not whether `top` parses.
    """

    def __init__(self):
        self.track = None          # (title, artist) or None
        self.charging = False

    def current_track_for_panel(self):
        return self.track

    def status_row(self, max_age: float = 5.0):
        return [
            {"label": "CPU", "value": "18%", "pct": 0.18, "alert": False},
            {"label": "Memory", "value": "11.2 GB", "pct": 0.62, "alert": False},
            {"label": "Battery", "value": "84%", "pct": 0.84,
             "flag": "charging" if self.charging else "", "alert": False},
        ]


class FakeMarket:
    """His watchlist without the network. Same shape market_intents returns."""

    def home_block(self):
        import panels as P
        return P.items([{"title": "AAPL", "detail": "305.59", "meta": "-0.11%"},
                        {"title": "NVDA", "detail": "225.01", "meta": "+1.20%"}],
                       title="Markets")


class FakeAssistant:
    def __init__(self, tools):
        self.tools = tools
        self.market = FakeMarket()
        self._last_response = ""

    def set_work_mode(self, on, reason=""):
        pass


def build(track=None):
    """Real NovaViews, real config, stubbed data sources.

    Tiles are primed on background threads, so this waits for them rather than
    racing them — a test that reads a tile before it has ever been built is
    testing the empty state by accident.
    """
    import nova as nova_mod
    from views import NovaViews
    tools = FakeTools()
    tools.track = track
    ws = CapturingWS()
    views = NovaViews(nova_mod.load_config(), ws=ws, assistant=FakeAssistant(tools))
    deadline = time.time() + 8
    while time.time() < deadline:
        if all(t._at for t in views._tiles.values()):
            break
        time.sleep(0.05)
    return views, ws, tools


def slots_of(payload):
    return {b.get("card"): b.get("slot") for b in payload["blocks"] if b.get("card")}


def has_status(payload):
    return any(b.get("slot") == "status" for b in payload["blocks"])


# ── 1. The greeting and the status row leave together ────────────────────────
def test_greeting_and_status():
    print("\n1. THE GREETING, AND WHAT LEAVES WITH IT")
    views, _ws, _tools = build()

    launch = views._home_payload()
    check(launch["title"] != "", "launch greets him", launch["title"])
    check(has_status(launch), "launch shows the status row")

    # The shipped bug: the greeting was conditional and his NAME was not, so
    # "NICHOLAS" sat under the orb for the rest of the session.
    views.assistant._last_response = "Sure."
    after = views._home_payload()
    check(after["title"] == "", "greeting is gone after his first sentence",
          repr(after["title"]))
    check(not has_status(after), "status row leaves WITH the greeting")

    # ...and the trigger is the WAKE WORD, not her reply. Measured in the
    # running app on the old signal: the welcome was still under the orb while
    # Nova was already listening to him.
    import nova as nova_mod
    src = pathlib.Path(nova_mod.__file__).read_text()
    wake = src[src.index("wake_detected = self.stt.record_wake"):]
    wake = wake[:wake.index("# \u2500\u2500 Mic health")]
    check("_begin_conversation" in wake,
          "the wake word retires the greeting, not the reply")
    # Typing never passes the wake word, so it has to count too.
    respond = src[src.index("def _respond(self, text: str)"):]
    respond = respond[:respond.index("self.memory.add_turn")]
    check("_begin_conversation" in respond, "typing retires it as well")

    views.assistant._conversation_started = True
    woke = views._home_payload()
    check(woke["title"] == "", "greeting is gone once the conversation starts")
    check(not has_status(woke), "and the status row goes with it")
    views.assistant._conversation_started = False

    row = [b for b in launch["blocks"] if b.get("slot") == "status"]
    check(row and row[0]["kind"] == "metrics", "the row is metrics, not a card")
    labels = [m["label"] for m in row[0]["metrics"]] if row else []
    check(labels == ["CPU", "Memory", "Battery"], "row reads CPU, memory, battery",
          str(labels))


# ── 2. Asking brings it back, and it settles ─────────────────────────────────
def test_status_recall():
    print("\n2. ASKING ABOUT CPU BRINGS THE ROW BACK")
    views, _ws, _tools = build()
    views.assistant._last_response = "Sure."
    check(not has_status(views._home_payload()), "row is away before he asks")

    views.recall_system()
    check(has_status(views._home_payload()), "row is back after he asks")

    # HELD while she is still talking. A timer armed when the handler ran
    # would already be most of the way down by the time he heard the number.
    time.sleep(0.8)
    check(has_status(views._home_payload()),
          "row is held while Nova is still speaking")

    # The clock starts when she stops.
    views.settle_system(0.6)
    check(has_status(views._home_payload()), "still up the moment she stops")
    time.sleep(1.0)
    check(not has_status(views._home_payload()), "row settles away on its own")

    # Asking twice must not let the first countdown close the second window.
    views.recall_system()
    views.settle_system(0.5)
    views.recall_system()
    time.sleep(0.8)
    check(has_status(views._home_payload()),
          "a second question re-holds the row past the first countdown")
    views.settle_system(0.3)
    time.sleep(0.7)
    check(not has_status(views._home_payload()), "and then settles once")


# ── 3. Now Playing only when something is playing ────────────────────────────
def test_now_playing():
    print("\n3. NOW PLAYING ONLY WHEN SOMETHING IS PLAYING")
    views, _ws, tools = build(track=None)
    check("playing" not in slots_of(views._home_payload()),
          "no card when nothing is playing")

    # He started the music himself. The tile notices without being asked.
    tools.track = ("Midnight City", "M83")
    views._tiles["playing"]._at = 0          # let it go stale
    deadline = time.time() + 5
    while time.time() < deadline and "playing" not in slots_of(views._home_payload()):
        time.sleep(0.05)
    cards = slots_of(views._home_payload())
    check("playing" in cards, "card appears on its own when he starts a song")

    block = [b for b in views._home_payload()["blocks"] if b.get("card") == "playing"]
    if check(bool(block), "the card carries the track"):
        item = block[0]["items"][0]
        check(item["title"] == "Midnight City" and item["detail"] == "M83",
              "title and artist are the real ones", str(item))

    # The detection underneath must not go stale. NSWorkspace's running-app
    # list is maintained by run-loop notifications, and the backend has no run
    # loop — so it only ever knows what was running at startup, and Now Playing
    # never appeared for music NOVA started. It looked right in every test.
    import tools as tools_mod
    tsrc = pathlib.Path(tools_mod.__file__).read_text()
    detect = tsrc[tsrc.index("def any_player_running"):]
    detect = detect[:detect.index("\n    def ")]
    # The docstring NAMES NSWorkspace to explain why it is not used, so check
    # the code rather than the prose.
    body = detect.split('"""')[-1]
    check("NSWorkspace" not in body,
          "player detection asks the kernel, not a run-loop snapshot")
    check("pgrep" in body, "...specifically pgrep")
    # And every AppleScript is bounded: an unbounded one wedges the tile thread
    # forever and that card silently stops updating for the life of the process.
    osa = tsrc[tsrc.index("def _osa("):]
    osa = osa[:osa.index("\n    def ")]
    check("timeout" in osa, "AppleScript calls are bounded")

    tools.track = None
    views._tiles["playing"]._at = 0
    deadline = time.time() + 5
    while time.time() < deadline and "playing" in slots_of(views._home_payload()):
        time.sleep(0.05)
    check("playing" not in slots_of(views._home_payload()),
          "card leaves when the music stops")


# ── 4. Slots, moving, swapping, and surviving a restart ──────────────────────
def test_slots():
    print("\n4. SLOTS, AND MOVING A CARD BY VOICE")
    import views as V
    views, _ws, tools = build(track=("Song", "Artist"))

    cards = slots_of(views._home_payload())
    check(cards.get("market") == "L1", "markets start top left", str(cards))
    check(cards.get("playing") == "L2", "now playing starts middle left")
    check(cards.get("weather") == "R1" or "weather" not in cards,
          "weather starts top right")

    said = views.handle("move_card", "move the now playing to the bottom right")
    check(views.slots["playing"] == "R3", "moved to the bottom right",
          views.slots["playing"])
    # allow_action_claim: she really did move it, and verified the slot.
    why = check_spoken(said, allow_action_claim=True)
    check(not why, "the confirmation is fit to speak", f"{said} -> {why}")

    # A slot holds one card, so a collision must SWAP — evicting the occupant
    # would silently lose him a card he never mentioned.
    before_market = views.slots["market"]
    views.handle("move_card", "put markets bottom right")
    check(views.slots["market"] == "R3", "markets took the bottom right")
    check(views.slots["playing"] == before_market,
          "now playing took markets' old slot — nothing was evicted",
          views.slots["playing"])

    # A layout he set by voice should still be true next week.
    saved = json.loads((Path(_TMP) / "home_layout.json").read_text())
    check(saved.get("market") == "R3", "the layout is on disk", str(saved))
    reloaded = V.NovaViews(views.config, ws=None,
                           assistant=FakeAssistant(tools))
    check(reloaded.slots["market"] == "R3", "and survives a restart",
          str(reloaded.slots))

    # A file move shares the verb and must never reach the home grid.
    for phrase in ("move my resume to Downloads", "put the invoice in Documents"):
        check(views.detect_intent(phrase) is None,
              f"'{phrase}' is not a card move")


# ── 5. Clear and restore ─────────────────────────────────────────────────────
def test_clear_restore():
    print("\n5. CLEAR HOME, RESTORE HOME")
    views, _ws, _tools = build(track=("Song", "Artist"))
    check(len(views._home_payload()["blocks"]) > 0, "home has cards to begin with")

    said = views.handle("clear_home", "clear home")
    check(views._home_payload()["blocks"] == [], "clear home empties the surface")
    check(views._home_payload()["title"] == "",
          "and does not greet him over an empty screen")
    why = check_spoken(said, allow_action_claim=True)
    check(not why, "clear is fit to speak", f"{said} -> {why}")

    said = views.handle("restore_home", "restore home")
    check(len(views._home_payload()["blocks"]) > 0, "restore home brings it back")
    why = check_spoken(said, allow_action_claim=True)
    check(not why, "restore is fit to speak", f"{said} -> {why}")

    # "Everything" has to mean everything. By the time he says this the
    # greeting is long gone, so the status row would otherwise stay hidden and
    # restore would quietly bring back less than it claims.
    check(has_status(views._home_payload()),
          "restore brings the status row back too")


# ── 6. Home must never block, and must not churn ─────────────────────────────
def test_render_cost():
    print("\n6. A RENDER COSTS NOTHING, AND SILENCE SENDS NOTHING")
    views, ws, _tools = build()
    views._home_payload()                       # warm

    t0 = time.time()
    for _ in range(200):
        views._home_payload()
    per_ms = (time.time() - t0) * 1000 / 200
    # The blocking version measured ~570ms per render (352 music + 218
    # calendar). Home is redrawn on a ticker, so this is the guarantee.
    check(per_ms < 5.0, f"render is {per_ms:.2f}ms, well under 5ms",
          f"{per_ms:.2f}ms")

    views.current = "home"
    ws.sent.clear()
    # A tile callback already pushed during build, so `_last_sent` holds the
    # current screen. Clear it to test the dedupe from a known start rather
    # than from whatever the priming happened to leave behind.
    views._last_sent = None
    views.refresh_home()
    first = len(ws.sent)
    for _ in range(10):
        views.refresh_home()
    check(first == 1, "the first refresh sends", str(first))
    check(len(ws.sent) == 1, "an unchanged world sends nothing more",
          f"{len(ws.sent)} sends")

    # An ANSWER is on screen: the ticker must not push home over the top of it.
    ws.server._view = "weather"
    views.current = "weather"
    ws.sent.clear()
    views.refresh_home()
    check(ws.sent == [], "the ticker leaves an answer alone")

    # ...and it must decide that from what the app is actually SHOWING, not
    # from views.current. Those are two copies of one fact and nova.py writes
    # the other one; when they drift, home goes stale for the life of the
    # process and nothing says why.
    ws.server._view = "home"
    views.current = "weather"          # drifted
    views._last_sent = None
    ws.sent.clear()
    views.refresh_home()
    check(len(ws.sent) == 1,
          "home still refreshes when views.current has drifted off home")
    views.current = "home"


# ── 7. Showing the work ──────────────────────────────────────────────────────
def test_progress():
    print("\n7. THE LIVE STEP LIST")
    import panels as P
    pushes: list[dict] = []
    prog = P.Progress(pushes.append, "Searching the web",
                      ["Opening the browser", "Loading the page",
                       "Reading the results"], detail="spider-man")

    def states(payload):
        block = [b for b in payload["blocks"] if b["kind"] == "steps"][0]
        return [s["state"] for s in block["steps"]]

    prog.start()
    check(states(pushes[-1]) == ["running", "pending", "pending"],
          "the whole sequence is visible from the first frame",
          str(states(pushes[-1])))
    prog.advance()
    check(states(pushes[-1]) == ["done", "running", "pending"], "it advances")
    prog.finish(P.items([{"title": "Spider-Man", "meta": "imdb.com"}]))
    check(states(pushes[-1]) == ["done", "done", "done"], "and completes")
    check(any(b["kind"] == "items" for b in pushes[-1]["blocks"]),
          "findings arrive under the steps")

    # A failure must leave the list up: a step that went red says more than a
    # screen that went blank.
    pushes.clear()
    prog2 = P.Progress(pushes.append, "Searching the web", ["Opening", "Reading"])
    prog2.start()
    prog2.fail("The browser wouldn't open.")
    check(states(pushes[-1])[0] == "failed", "the failing step is marked")
    check(any(b["kind"] == "note" for b in pushes[-1]["blocks"]),
          "and says why, on screen")

    # No sink (the routing harness) must be a no-op, not a crash.
    quiet = P.Progress(None, "x", ["a"])
    quiet.start(); quiet.advance(); quiet.finish()
    check(True, "a progress list with nowhere to go is harmless")


# ── 8. What she says about a search ──────────────────────────────────────────
def test_search_readback():
    print("\n8. THE SEARCH READBACK IS TEMPLATED, NOT GENERATED")
    from tools import NovaTools
    results = [
        {"title": "Spider-Man in film", "host": "en.wikipedia.org"},
        {"title": "Spider-Man: Brand New Day (2026)", "host": "imdb.com"},
        {"title": "Spider-Man | Marvel", "host": "marvel.com"},
    ]
    said = NovaTools._speak_results("spider-man", results)
    check("wikipedia" in said, "it names the source as a name, not a domain", said)
    check(".org" not in said and ".com" not in said,
          "no domains are read aloud", said)
    why = check_spoken(said)
    check(not why, "fit to speak", f"{said} -> {why}")

    # A page title is untrusted text and can be enormous.
    huge = [{"title": "x" * 400, "host": "example.com"}]
    said = NovaTools._speak_results("q", huge)
    check(len(said) < 160, "an absurd page title is cut down", str(len(said)))
    why = check_spoken(said)
    check(not why, "and still fit to speak", f"{said} -> {why}")

    check(NovaTools._speak_results("q", [{"title": "Only", "host": ""}]),
          "a result with no host still produces a sentence")


def main() -> int:
    print("=" * 72)
    print("HOME SURFACE — what appears, what leaves, and where it sits")
    print("=" * 72)

    test_greeting_and_status()
    test_status_recall()
    test_now_playing()
    test_slots()
    test_clear_restore()
    test_render_cost()
    test_progress()
    test_search_readback()

    print(f"\n  {PASS}/{PASS + FAIL} checks passed")
    if FAILURES:
        print("\n  FAILURES:")
        for f in FAILURES:
            print(f"    ✗ {f}")
    print("\n  NOT PROVEN HERE: how any of this LOOKS. That a card flies to its "
          "new\n  slot, that the row is legible at the bottom of his screen, and "
          "that\n  the orb stays centred are all his eyes, not this file.")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
