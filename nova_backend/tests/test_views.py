#!/usr/bin/env python3
"""
View protocol — the screen Nova is showing, and how it gets there.

This is the foundation the whole UI overhaul stands on: every panel, every
destination, and the home and menu screens all ride on `{"type": "view"}`.
Getting it wrong is rework everywhere, so it is tested on its own rather than
only through the routing corpus.

Proves, with the real `NovaViews` and the real `NovaWSServer`:
  * navigation phrases reach a view, and sentences that merely CONTAIN
    navigation words do not
  * arriving at a view broadcasts it, with data, and updates what a
    newly-connected client is told
  * a view that is not built yet says so out loud and shows NOTHING — no empty
    panel pretending to work
  * the menu is generated from the registry, so it cannot drift from what
    actually exists
  * every spoken reply obeys the listener rules

No sockets are opened: the WS broadcast is captured, since what matters here is
the payload, not asyncio.

Run:  python tests/test_views.py
"""
from __future__ import annotations

import json
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


def check(cond, label, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        FAILURES.append(f"{label}  ({detail})" if detail else label)
    return bool(cond)


class CapturingWS:
    """The real NovaWSServer with the socket send replaced.

    Deliberately NOT a hand-written stub: `send_view` must be the shipping one,
    because the thing under test is what it puts on the wire and what it stores
    for the next client to connect.
    """

    def __init__(self):
        from ws_server import NovaWSServer
        self.server = NovaWSServer(http_port=0, ws_port=0,
                                   on_text_message=lambda _t: None)
        self.sent: list[dict] = []
        self.server._ws_broadcast = self.sent.append   # only the transport

    def __getattr__(self, name):
        return getattr(self.server, name)


def build_views():
    import nova as nova_mod
    from views import NovaViews
    ws = CapturingWS()
    return NovaViews(nova_mod.load_config(), ws=ws), ws


# ── 1. Detection ──────────────────────────────────────────────────────────────
NAVIGATES = {
    "go home": "home",
    "nova go home": "home",
    "take me home": "home",
    "go back home": "home",
    "return home": "home",
    "home screen": "home",
    "show me the menu": "menu",
    "open the menu": "menu",
    "what can you do": "menu",
    "go to finance": "finance",
    "take me to markets": "finance",
    "go to the weather": "weather",
    "show me the finance screen": "finance",
    "open the memory panel": "memory",
    "pull up my past conversations": None,   # "my" is not navigational
    "go to conversation history": "conversations",
    "switch to calendar": "calendar",
}

# Ordinary English that happens to contain a navigation word. Every one of these
# would have been stolen by a looser matcher.
DOES_NOT_NAVIGATE = (
    "I'm going home at five",
    "I go home every friday",
    "home is where the heart is",
    "what's on the menu for dinner",
    "the menu at that place is huge",
    "I want to go to italy someday",
    "remind me to call mom when I get home",
    "show me my calendar",
    "what's the weather at home",
    "find the menu pdf",
    "we should go to the beach",
    "can you show me how to do that",
    "I need to go to the store",
)


def test_detection(views) -> None:
    print("\n1. DETECTION")
    for phrase, expected in NAVIGATES.items():
        got = views.detect_intent(phrase)
        check(got == expected, f"'{phrase}' -> {expected}", f"got {got}")
    for phrase in DOES_NOT_NAVIGATE:
        got = views.detect_intent(phrase)
        check(got is None, f"'{phrase[:44]}' stays conversation", f"got {got}")


# ── 2. The broadcast ──────────────────────────────────────────────────────────
def test_broadcast(views, ws) -> None:
    print("\n2. BROADCAST")
    ws.sent.clear()
    spoken = views.handle("home")

    view_msgs = [m for m in ws.sent if m.get("type") == "view"]
    check(len(view_msgs) == 1, "arriving at a view sends exactly one view message",
          f"sent {len(ws.sent)}")
    if view_msgs:
        msg = view_msgs[0]
        check(msg.get("view") == "home", "view name is on the wire", str(msg)[:80])
        check("data" in msg, "payload carries a data object", str(msg)[:80])
        check(json.dumps(msg) is not None, "payload is JSON-serialisable")

    check(ws.server._view == "home",
          "server remembers the view for the next client to connect",
          ws.server._view)
    check(views.current == "home", "engine tracks where it is", views.current)
    check(bool(spoken.strip()), "arriving says something", repr(spoken))


# ── 3. Honest degradation ─────────────────────────────────────────────────────
def test_unbuilt_views(views, ws) -> None:
    print("\n3. VIEWS THAT DO NOT EXIST YET")
    from views import VIEWS

    for name in ("finance", "health"):
        ws.sent.clear()
        before = ws.server._view
        spoken = views.handle(name)

        check(not [m for m in ws.sent if m.get("type") == "view"],
              f"'{name}' shows NO panel while unbuilt", f"sent {ws.sent}")
        check(ws.server._view == before,
              f"'{name}' does not move the UI off the current screen")
        check(len(spoken.strip()) > 8, f"'{name}' explains itself", repr(spoken))
        check("yet" in spoken.lower(),
              f"'{name}' says it is not ready, rather than failing silently",
              repr(spoken))

    live = [v.name for v in VIEWS.values() if v.is_live]
    check("home" in live and "menu" in live,
          "home and menu are live in phase 1", str(live))


# ── 4. The menu is generated, not hand-written ────────────────────────────────
def test_menu_payload(views, ws) -> None:
    print("\n4. MENU CONTENT")
    from views import VIEWS

    ws.sent.clear()
    views.handle("menu")
    msgs = [m for m in ws.sent if m.get("type") == "view"]
    check(bool(msgs), "menu broadcasts")
    if not msgs:
        return

    data = msgs[0]["data"]
    sections = data.get("sections", [])
    check(len(sections) >= 2, "menu has both destinations and things to ask",
          str(len(sections)))

    dest_names = {i["name"] for s in sections for i in s["items"] if "name" in i}
    for name in VIEWS:
        if name == "home":
            continue
        check(name in dest_names, f"menu lists the '{name}' destination")

    # An unbuilt destination must be marked, or the menu promises what it can't do.
    flat = [i for s in sections for i in s["items"]]
    unbuilt = [i for i in flat if i.get("available") is False]
    check(len(unbuilt) >= 2, "menu marks unbuilt destinations as unavailable",
          str(unbuilt))
    check(all(i.get("note") for i in unbuilt),
          "each unavailable destination explains why", str(unbuilt))


# ── 5. Listener rules on everything spoken ────────────────────────────────────
def test_spoken(views) -> None:
    print("\n5. FIT TO SPEAK")
    from views import VIEWS
    for name in VIEWS:
        spoken = views.handle(name)
        problems = check_spoken(spoken)
        check(not problems, f"'{name}' is fit to speak",
              "; ".join(problems) + f" | {spoken[:70]}" if problems else "")
        check(len(spoken) < 120,
              f"'{name}' is brief — the panel answers, the voice acknowledges",
              f"{len(spoken)} chars")


def main() -> int:
    print("=" * 72)
    print("VIEW PROTOCOL — navigation, broadcast, and honest degradation")
    print("=" * 72)

    views, ws = build_views()
    test_detection(views)
    test_broadcast(views, ws)
    test_unbuilt_views(views, ws)
    test_menu_payload(views, ws)
    test_spoken(views)

    print(f"\n  {PASS}/{PASS + FAIL} checks passed")
    if FAILURES:
        print("\n  FAILURES:")
        for f in FAILURES:
            print(f"    ✗ {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
