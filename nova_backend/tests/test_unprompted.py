#!/usr/bin/env python3
"""
Everything Nova says that he did not just ask for — through one door, never
twice, never a flood, and never dropped.

JARVIS's rule 5, from a night of ~2,600 repeated messages: claim before
speaking, de-duplicate everything, hard ceilings. Plus Nova's standing watches
("tell me when my battery is charged"), which are only honest if they are
kept, once, across restarts.

Fidelity: the real delivery.Delivery, proactive.ProactiveMonitor and
standing.StandingWatches over a throwaway SQLite file; speech and the screen
captured; readings scripted. No audio, no real calendar, no real battery.

Run:  python tests/test_unprompted.py
"""
from __future__ import annotations

import sqlite3
import sys
import tempfile
import threading
import time
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR.parent))

import delivery as D          # noqa: E402
import proactive as P         # noqa: E402
import standing as S          # noqa: E402

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


def door(db, **cfg):
    spoken, shown = [], []
    d = D.Delivery(db, {"delivery": cfg}, speak=spoken.append, show=shown.append)
    return d, spoken, shown


def lock_free(db) -> bool:
    try:
        c = sqlite3.connect(str(db), timeout=0.2)
        c.execute("BEGIN IMMEDIATE")
        c.commit()
        c.close()
        return True
    except sqlite3.OperationalError:
        return False


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="nova-unprompted-"))

    print("── the door: never twice")
    db = tmp / "a.db"
    d, spoken, shown = door(db)
    check(d.deliver("event_soon", "Standup starts in 5 minutes.", key="event:standup:9:00") == "spoken",
          "an unprompted item is spoken")
    check(d.deliver("event_soon", "Standup starts in 4 minutes.", key="event:standup:9:00") == "duplicate",
          "the same THING is not said again, even reworded")
    d2, spoken2, _ = door(db)
    check(d2.deliver("event_soon", "Standup starts in 3 minutes.", key="event:standup:9:00") == "duplicate"
          and not spoken2, "...not even after a restart")
    check(lock_free(db), "no write lock is left behind")

    print("\n── claim before speaking, under contention")
    db = tmp / "b.db"
    d, spoken, _ = door(db)
    threads = [threading.Thread(target=d.deliver,
                                args=("reminder", "Your reminder was due.", "reminder:x:1"))
               for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    check(len(spoken) == 1, "twelve threads, one sentence", f"spoke {len(spoken)}")

    print("\n── ceilings: quiet is not silent")
    db = tmp / "c.db"
    d, spoken, shown = door(db, max_per_hour=2, max_per_day=5)
    out = [d.deliver("proactive", f"thing {i}", key=f"p:{i}") for i in range(4)]
    check(out == ["spoken", "spoken", "shown", "shown"], "over the hourly ceiling, held to the screen",
          f"{out}")
    check(len(spoken) == 2 and {"thing 2", "thing 3"} <= set(shown),
          "...and NOT dropped: it is shown", f"spoken {spoken} shown {shown}")
    check(d.deliver("timer", "Your 5 minute timer is up.") == "spoken",
          "something he ASKED for is never held by the ceiling")
    d2, _, _ = door(db, max_per_hour=2)
    check(d2.hold_reason() is not None, "the count survives a restart (it lives in the table)")

    db = tmp / "q.db"
    h = time.localtime().tm_hour
    d, spoken, shown = door(db, quiet_hours=[h, (h + 1) % 24])
    check(d.deliver("event_soon", "Lunch starts in 10 minutes.", key="e:l") == "shown"
          and not spoken, "inside quiet hours an unprompted item is shown, not spoken")
    check(d.deliver("watch", "Your battery is fully charged.", key="w:1") == "spoken",
          "...but a watch he set still speaks")

    print("\n── the proactive monitor keeps 'never twice' across restarts")
    db = tmp / "p.db"
    d, spoken, _ = door(db)

    class _A:
        is_muted = False
        is_awake = True
        work_mode = False
        tts = None

    def monitor():
        said = []
        m = P.ProactiveMonitor({"proactive": {"min_gap_seconds": 0}}, _A(),
                               lambda text, key=None, kind="proactive":
                               (said.append(text), d.deliver(kind, text, key=key, claimed=True)),
                               claim=lambda key, kind, text: d.claim(key, kind, text))
        m._candidates = lambda now: [("event:Standup:2026-11-19T09:00", "Standup starts in 5 minutes.")]
        return m
    m1 = monitor()
    got1 = m1.next_announcement(with_key=True)
    if got1:
        m1._announce(got1[2], key=got1[0], kind=got1[1])
    m2 = monitor()                                   # Nova restarted
    got2 = m2.next_announcement(with_key=True)
    check(got1 is not None and got2 is None and spoken.count("Standup starts in 5 minutes.") == 1,
          "a meeting announced before a restart is not announced again after it",
          f"first {got1} second {got2} spoken {spoken}")

    print("\n── standing watches: parsing")
    for text, want in [
        ("tell me when my battery is fully charged", ("battery_full", ">=", 100.0)),
        # no direction in the words: decided from the reading when it is set
        ("let me know when the battery hits 80", ("battery", "auto", 80.0)),
        ("warn me if my battery drops below 15 percent", ("battery", "<=", 15.0)),
        ("tell me if my battery is low", ("battery", "<=", 20.0)),
        ("let me know if my cpu goes above 90 percent", ("cpu", ">=", 90.0)),
        ("tell me when the processor calms down", ("cpu", "<=", 30.0)),
        ("alert me when memory usage goes over 85", ("memory", ">=", 85.0)),
    ]:
        w = S.parse(text)
        check(w is not None and (w["metric"], w["op"], w["value"]) == want, f"watch: {text!r}", f"{w}")
    for text in ["tell me about my battery", "let me know what you think",
                 "tell me when you're ready", "my battery is fully charged",
                 "tell me when my meeting starts", "let me know if it rains tomorrow"]:
        check(S.parse(text) is None, f"not a watch: {text!r}")

    print("\n── standing watches: kept, once, across restarts")
    db = tmp / "w.db"
    d, spoken, shown = door(db, max_per_hour=0)       # ceiling 0: only requested kinds speak
    level = {"battery": 60.0, "cpu": 20.0, "memory": 50.0}
    deliver = lambda kind, text, key: d.deliver(kind, text, key=key)
    w = S.StandingWatches(db, lambda m: level.get(m), deliver)
    r = w.handle("tell me when my battery is fully charged")
    check(r and r.startswith("Okay. I'll tell you when your battery is fully charged"),
          "setting a watch is confirmed in words that are now true", r)
    check(w.check_once() == 0 and not spoken, "nothing is said while it is not true")
    w2 = S.StandingWatches(db, lambda m: level.get(m), deliver)   # restart
    check(len(w2.active()) == 1, "the watch survives a restart")
    level["battery"] = 100.0
    check(w2.check_once() == 1 and spoken == ["Your battery is fully charged."],
          "it fires when it becomes true — through the door, despite a ceiling of 0",
          f"{spoken}")
    w3 = S.StandingWatches(db, lambda m: level.get(m), deliver)   # restart again
    check(w3.check_once() == 0 and len(spoken) == 1, "a fired watch never fires again")
    r = w3.handle("let me know when the battery goes above 80")
    check(r == "It's already at 100 percent." and not w3.active(),
          "a condition already true is answered now, not stored", r)
    r = w3.handle("let me know when the battery hits 80")
    check(r and "drops to 80" in r and len(w3.active()) == 1,
          "'hits 80' from 100 means falling to 80 (direction from the reading)", r)
    r = w3.handle("let me know when the battery hits 80")
    check(r and r.startswith("I'm already watching") and len(w3.active()) == 1,
          "the same watch twice is stored once", r)
    w3.cancel(None)
    w3.handle("warn me if my cpu goes above 90")
    w3.handle("tell me when memory usage goes over 85")
    r = w3.handle("what are you watching")
    check(r and "CPU goes above 90" in r and "memory" in r, "he can ask what is being watched", r)
    r = w3.handle("stop watching my cpu")
    check(r and r.startswith("Done") and len(w3.active()) == 1, "a watch can be cancelled", r)
    r = w3.handle("cancel all watches")
    check(not w3.active(), "...or all of them", r)
    w4 = S.StandingWatches(db, lambda m: None, deliver)
    r = w4.handle("tell me when my battery is fully charged")
    check(r and "can't read it right now" in r and len(w4.active()) == 1,
          "an unreadable metric is said honestly, and still watched", r)
    check(lock_free(db), "no write lock is left behind")

    print(f"\n  {PASS}/{PASS + FAIL}")
    if FAILURES:
        for f in FAILURES:
            print(f"    ✗ {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
