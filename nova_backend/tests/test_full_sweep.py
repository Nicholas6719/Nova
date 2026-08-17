#!/usr/bin/env python3
"""
NOVA FULL-SYSTEM SWEEP — every capability built to date.

Drives the REAL pipeline: va._submit_turn(text, wait=True) on the nova-llm
worker thread, tts.speak monkeypatched to capture. No audio, MLX thread
locality preserved.

Rules this harness follows (learned the hard way):
  * Assert on the VALUE, never on "something happened".
  * Verify against real system state (EventKit, SQLite, filesystem, osascript).
  * Clean up ALL residue and PROVE it.
  * Anything the user owns (browser windows) is created fresh and closed BY ID.
    Never by index, never with a count-based loop.
  * The memory DB is WAL — clean with DELETE + wal_checkpoint, not file copies.
  * Never leave an invented fact behind, especially a safety-relevant one.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from pathlib import Path as _Path
TESTS_DIR = _Path(__file__).resolve().parent
BACKEND = str(TESTS_DIR.parent)
sys.path.insert(0, BACKEND)
sys.path.insert(0, str(TESTS_DIR))
os.chdir(BACKEND)

MARKER = "novafull" + os.urandom(3).hex()
HOME = Path.home()

PASS = FAIL = 0
FAILURES: list[tuple[str, str]] = []
UNVERIFIED: list[str] = []
SECTION = ""


def check(cond, label, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {label}")
    else:
        FAIL += 1
        FAILURES.append((f"[{SECTION}] {label}", detail))
        print(f"  FAIL  {label}")
    if detail:
        print(f"        {str(detail)[:400]}")
    return bool(cond)


def unverified(what):
    UNVERIFIED.append(what)
    print(f"  ----  UNVERIFIED: {what}")


def section(t):
    global SECTION
    SECTION = t
    print("\n" + "=" * 74)
    print(t)
    print("=" * 74)


def osa(script, timeout=15):
    try:
        r = subprocess.run(["osascript", "-e", script], capture_output=True,
                           text=True, timeout=timeout)
        return r.returncode == 0, (r.stdout or "").strip()
    except Exception as exc:
        return False, str(exc)


# ══════════════════════════════════════════════════════════════════════════
section("BOOT")
# ══════════════════════════════════════════════════════════════════════════
import nova as nova_mod
from nova import VoiceAssistant

spoken: list[str] = []
tokens: list[str] = []


# The mic, the speakers and the sockets are the only things a sweep fakes.
# Everything else is the real engine.
_FAKED = {"stt", "tts", "ws"}


def boot():
    va = VoiceAssistant.__new__(VoiceAssistant)
    va.config = nova_mod.load_config()
    # Use the REAL state initializer — see test_conversation_loop.py.
    va._init_state()
    # Derived from _REQUIRED_ENGINES rather than hand-listed, so adding a
    # routing stage cannot leave this harness behind. It just did: `weather`
    # was added to the router and to _REQUIRED_ENGINES, and this list still
    # said screen — every turn in the sweep died on `no attribute 'weather'`.
    # That is the same shape as the `screen` bug that once hid behind 130
    # green checks, only pointed the other way.
    for name in VoiceAssistant._REQUIRED_ENGINES:
        if name in _FAKED:
            continue
        getattr(va, f"_init_{name}")()
    va._init_rag()          # optional, so not in _REQUIRED_ENGINES

    class _STT:
        """There is no microphone in a sweep. Present so the router's engine
        check passes; every phrase is fed in as text, never recorded."""
        def record_command(self, *a, **k): return None
        def transcribe(self, *a, **k): return ""
        def record_wake(self, *a, **k): return False
    va.stt = _STT()

    class _TTS:
        def speak(self, t): spoken.append(t)
        def wait_until_done(self, timeout=0): pass
        def is_speaking(self): return False
        def stop_and_flush(self): pass
    va.tts = _TTS()

    class _WS:
        def send_message(self, *a, **k): pass
        def broadcast_state(self, *a, **k): pass
        def stream_token(self, t, *a, **k): tokens.append(t)
        def start(self, *a, **k): pass
        def stop(self, *a, **k): pass
    va.ws = _WS()
    va.set_state = lambda s: None
    # The product's own startup guard, run against the harness's object: if an
    # engine the router dereferences is missing, fail HERE with a clear message
    # instead of as an AttributeError on every single phrase.
    va._verify_engines()
    return va


t0 = time.monotonic()
va = boot()
print(f"  booted in {time.monotonic() - t0:.1f}s")
CFG = va.config


from listener import check_spoken

# Every response Nova produces during the sweep, checked against the listener
# rules regardless of which feature was under test. A path spoken aloud by the
# file layer is caught by the calendar tests too.
LISTENER_PROBLEMS: list[tuple[str, str, list]] = []


def say(text):
    spoken.clear()
    tokens.clear()
    va._submit_turn(text, wait=True)
    out = " ".join(spoken).strip()
    # Deterministic handlers DO perform actions and verify them, so an action
    # claim from them is truthful; only flag the rest.
    problems = check_spoken(out, allow_action_claim=True)
    if problems:
        LISTENER_PROBLEMS.append((text, out, problems))
    return out


check(va.llm is not None, "LLM engine loaded")
check(va.memory is not None, "memory loaded")
check(va.tools is not None, "tools loaded")
check(va.calendar is not None, "calendar loaded")
check(va.files is not None, "files loaded")
check(va.screen is not None, "screen awareness loaded")

# Baselines to restore
BASE_VOL = None
ok, v = osa("output volume of (get volume settings)")
if ok:
    BASE_VOL = int(v)
print(f"  baseline system volume: {BASE_VOL}")

DB_PATH = HOME / "Library" / "Application Support" / "Nova" / "nova_memory.db"


def db_facts():
    con = sqlite3.connect(str(DB_PATH))
    try:
        return con.execute("SELECT category, key, value FROM facts").fetchall()
    finally:
        con.close()


BASE_FACTS = set(db_facts())
print(f"  baseline facts in DB: {len(BASE_FACTS)}")


# ══════════════════════════════════════════════════════════════════════════
section("1. FAST PATH  (greeting / date / time / repeat)")
# ══════════════════════════════════════════════════════════════════════════
from datetime import datetime

now = datetime.now()
out = say("hello")
check(any(g in out.lower() for g in ("morning", "afternoon", "evening")),
      "greeting is time-of-day correct", out)
check(CFG["user"]["address_as"] in out, "greeting uses his name", out)

out = say("what time is it")
h12 = now.hour % 12 or 12
check(f"{h12}:" in out, "time matches the real clock", f"expected {h12}: | {out}")

out = say("what day is it")
check(now.strftime("%A") in out and str(now.year) in out,
      "date matches the real date", f"expected {now.strftime('%A')} {now.year} | {out}")

prev = out
out = say("what did you just say")
check(prev.split(".")[0][:20] in out or "today is" in out.lower(),
      "repeat returns the previous response", out)


# ══════════════════════════════════════════════════════════════════════════
section("2. SYSTEM STATS & CONTROL")
# ══════════════════════════════════════════════════════════════════════════
ok, real_batt = osa("do shell script \"pmset -g batt | grep -o '[0-9]*%' | head -1\"")
out = say("what's my battery at")
m = re.search(r"(\d{1,3})\s*(?:percent|%)", out)
check(m is not None, "battery reports a real percentage", out)
if m and ok and real_batt:
    diff = abs(int(m.group(1)) - int(real_batt.strip("%")))
    check(diff <= 3, "battery matches pmset", f"nova={m.group(1)} pmset={real_batt}")

out = say("how much RAM am I using")
check(re.search(r"\d", out) and ("gb" in out.lower() or "percent" in out.lower()),
      "RAM reports real numbers", out)

out = say("how's my disk space")
check(re.search(r"\d", out), "disk reports real numbers", out)

out = say("what's my CPU at")
check(re.search(r"\d", out), "CPU reports real numbers", out)

# Volume — set, verify against the real system, restore.
out = say("set the volume to 30")
ok, v = osa("output volume of (get volume settings)")
check(ok and int(v) == 30, "volume ACTUALLY changed to 30", f"system says {v} | {out}")
out = say("what's my volume")
check("30" in out, "volume query reads back the real value", out)
if BASE_VOL is not None:
    osa(f"set volume output volume {BASE_VOL}")
    ok, v = osa("output volume of (get volume settings)")
    check(ok and int(v) == BASE_VOL, "volume RESTORED to baseline", f"{v} vs {BASE_VOL}")

out = say("what's my screen brightness")
check(bool(out) and ("percent" in out.lower() or "can't" in out.lower()
                     or "bright" in out.lower()),
      "brightness answers or declines honestly", out)

out = say("turn on do not disturb")
check("can't" in out.lower() and "control cent" in out.lower(),
      "DND declines honestly with a real alternative", out)

# Power — must ASK, never act.
out = say("shut down my mac")
check(va._tool_confirm is not None, "power arms a confirmation instead of acting")
check("?" in out or "sure" in out.lower() or "confirm" in out.lower(),
      "power asks first", out)
out = say("no")
check(va._tool_confirm is None and "cancel" in out.lower(),
      "power cancels safely on anything but yes", out)

out = say("what apps are running")
check("claude" in out.lower() or "xcode" in out.lower(),
      "running apps names REAL running apps", out)
out = say("what app is in front")
check(bool(out) and len(out) < 200, "frontmost app answers", out)

out = say("what mac am I on")
check(re.search(r"m\d|macbook|mac\b|apple", out.lower()),
      "system info reports the real machine", out)


# ══════════════════════════════════════════════════════════════════════════
section("3. TIMERS")
# ══════════════════════════════════════════════════════════════════════════
out = say("set a timer for 45 minutes")
check("45" in out and "minute" in out.lower(), "timer accepts a duration", out)
out = say("how much time is left")
check("44" in out or "45" in out, "timer status reports real remaining time", out)
out = say("cancel my timers")
check("cancel" in out.lower(), "timers cancel", out)
out = say("how much time is left")
check("no " in out.lower() or "not" in out.lower() or "don't" in out.lower(),
      "no timers remain after cancel", out)
check(len(va.tools._timers) == 0, "timer registry is empty",
      f"{list(va.tools._timers)}")


# ══════════════════════════════════════════════════════════════════════════
section("4. MEMORY")
# ══════════════════════════════════════════════════════════════════════════
out = say("remember that my favorite programming language is Swift")
check("swift" in out.lower() or "got it" in out.lower() or "remember" in out.lower(),
      "explicit remember acknowledges", out)
facts = {(c, k, v) for c, k, v in db_facts()} - BASE_FACTS
check(any("swift" in str(v).lower() for _, _, v in facts),
      "fact ACTUALLY written to the SQLite DB", f"new rows: {facts}")

out = say("what's my favorite programming language")
check("swift" in out.lower(), "recall returns the stored VALUE", out)

# Correction must SUPERSEDE, not duplicate.
say("actually my favorite programming language is Python")
rows = [r for r in db_facts() if "programming" in str(r[1]).lower()
        or "language" in str(r[1]).lower()]
check(len(rows) == 1, "correction supersedes (exactly one canonical row)", f"{rows}")
check(any("python" in str(r[2]).lower() for r in rows),
      "correction stored the NEW value", f"{rows}")

out = say("forget my favorite programming language")
rows = [r for r in db_facts() if "programming" in str(r[1]).lower()
        or "language" in str(r[1]).lower()]
check(not rows, "forget actually deletes the row", f"{rows}")

# Tool queries must NOT be hijacked by memory recall (old BUG 1).
out = say("what's my battery at")
check("percent" in out.lower() or "%" in out,
      "memory does not hijack 'what's my battery'", out)

# Grounding guard on the reconciler.
from fact_reconciler import _is_grounded
check(not _is_grounded("walks his dog every morning at 6", "I go running every morning at 6"),
      "reconciler rejects the fabricated dog")
check(_is_grounded("runs every morning at 6", "I go running every morning at 6"),
      "reconciler accepts a grounded fact")


# ══════════════════════════════════════════════════════════════════════════
section("5. CALENDAR & REMINDERS  (real EventKit)")
# ══════════════════════════════════════════════════════════════════════════
import calendar_reminders as cr

try:
    base_rem = cr.get_all_reminders()
    ek_ok = True
except Exception as exc:
    base_rem = []
    ek_ok = False
    check(False, "EventKit reachable", str(exc))

if ek_ok:
    check(True, "EventKit reachable", f"{len(base_rem)} existing reminders")

    out = say("what's on my calendar today")
    check(bool(out) and "error" not in out.lower(), "calendar read answers", out)
    check("nicholas has" not in out.lower() and "he has" not in out.lower(),
          "calendar read speaks in second person", out)

    out = say("what reminders do I have")
    check(bool(out), "reminder read answers", out)
    check(not re.search(r"be sure to|make sure to|critical|don't forget to|important that",
                        out.lower()),
          "reminder read adds no invented advice or urgency", out)

    title = f"{MARKER} test reminder"
    out = say(f"remind me to {MARKER} test reminder tomorrow at 3 PM")
    time.sleep(1.0)
    after = cr.get_all_reminders()
    made = [r for r in after if MARKER in str(r.get("title", "")).lower()
            or MARKER in str(r.get("title", ""))]
    check(bool(made), "reminder ACTUALLY created in Reminders.app",
          f"{[r.get('title') for r in after[:5]]}")
    if made:
        due = str(made[0].get("due_iso") or made[0].get("due") or "")
        check("15:00" in due or "3:00 PM" in due or "T15" in due,
              "reminder due time is 3 PM, not garbled", f"due={due}")
        out = say(f"delete the reminder {MARKER} test reminder")
        time.sleep(1.0)
        left = [r for r in cr.get_all_reminders()
                if MARKER in str(r.get("title", ""))]
        check(not left, "reminder ACTUALLY deleted", f"left={[r.get('title') for r in left]}")

    # Time-range parsing (the old zero-length-event bug).
    for phrase, want in (("I'm working Saturday from 7 to 5", (7, 17)),
                         ("meeting from 9 to 5", (9, 17)),
                         ("lunch from 1 to 3", (13, 15))):
        got = va.calendar._parse_time_range(phrase)
        got_hours = (got[0][0], got[1][0]) if got else None
        check(got_hours == want,
              f"time range parsed: {phrase!r}", f"got={got_hours} want={want}")
else:
    unverified("calendar/reminders (EventKit unreachable in this process)")


# ══════════════════════════════════════════════════════════════════════════
section("6. FILE MANAGEMENT")
# ══════════════════════════════════════════════════════════════════════════
import file_manager as fm

created: list[Path] = []


def mk(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    created.append(path)
    return path


budget = mk(HOME / "Desktop" / f"{MARKER}-budget.txt",
            "Q3 budget\nRent 1200\nGroceries 400\nTransit 90\nTotal 1690\n")
notes = mk(HOME / "Documents" / f"{MARKER}-notes.md",
           "# Roof project\nCall the contractor on Tuesday.\nGet three quotes.\n")

# Folder listing must come from the REAL filesystem, never the LLM.
import file_manager as _fm
_docs = _fm.list_folder(HOME / "Documents")
out = say("what's in my documents folder")
if _docs["n_files"] == 0 and _docs["n_folders"] == 0:
    check("empty" in out.lower(), "empty folder reported as empty", out)
else:
    m_f = re.search(r"(\d+)\s+folder", out)
    m_x = re.search(r"(\d+)\s+file", out)
    check(m_f and int(m_f.group(1)) == _docs["n_folders"],
          "folder COUNT matches the real filesystem",
          f"nova={m_f.group(1) if m_f else None} real={_docs['n_folders']} | {out}")
    check(m_x and int(m_x.group(1)) == _docs["n_files"],
          "file COUNT matches the real filesystem",
          f"nova={m_x.group(1) if m_x else None} real={_docs['n_files']} | {out}")
out = say("what's in my downloads folder")
_dl = _fm.list_folder(HOME / "Downloads")
check(("empty" in out.lower()) == (_dl["n_files"] == 0 and _dl["n_folders"] == 0),
      "downloads listing matches reality", f"real={_dl['n_folders']}d/{_dl['n_files']}f | {out}")

out = say(f"find my {MARKER} budget file")
check(MARKER in out.lower() or "budget" in out.lower(), "find locates the file", out)
check("desktop" in out.lower(), "find names the right folder", out)

out = say(f"what's in my {MARKER} budget file")
check(any(w in out.lower() for w in ("rent", "grocer", "budget", "transit")),
      "read/summarize is grounded in the real contents", out)
nums = set(re.findall(r"\b\d{3,}\b", out))
bad = nums - {"1200", "400", "90", "1690"}
check(not bad, "summary invents no numbers", f"unexpected: {sorted(bad)} | {out}")

# Move with confirmation + pronoun follow-up.
out = say(f"move the {MARKER} budget file to Documents")
check("?" in out, "move ASKS before touching the disk", out)
out = say("yes")
moved = HOME / "Documents" / f"{MARKER}-budget.txt"
check(moved.exists() and not (HOME / "Desktop" / f"{MARKER}-budget.txt").exists(),
      "file ACTUALLY moved on disk", f"exists={moved.exists()}")
created.append(moved)

# Copy never overwrites.
ok2, msg = fm.copy_file(str(moved), str(HOME / "Documents"))
check(not ok2, "copy refuses to overwrite an existing destination", msg)

# Protected paths.
check(fm.is_protected_path(os.path.join(BACKEND, "config.json")),
      "Nova's own config is a protected path")
ok3, msg = fm.move_file(os.path.join(BACKEND, "config.json"), str(HOME / "Desktop"))
check(not ok3, "filesystem call REFUSES to move Nova's own files", msg)

# No delete, ever.
out = say(f"delete the {MARKER} notes file")
check("can't" in out.lower() or "won't" in out.lower() or "don't" in out.lower(),
      "Nova declines to delete files", out)
check(notes.exists(), "the file still exists after a delete request")

# Rename.
out = say(f"rename the {MARKER} notes file to {MARKER}-roof")
if "?" in out:
    out = say("yes")
renamed = HOME / "Documents" / f"{MARKER}-roof.md"
check(renamed.exists(), "rename ACTUALLY applied on disk", f"{out}")
if renamed.exists():
    created.append(renamed)

# Errors must not speak filesystem paths.
broken = mk(HOME / "Documents" / f"{MARKER}-corrupt.docx", "not a real docx")
out = say(f"summarize the {MARKER} corrupt document")
check("/users/" not in out.lower(), "spoken error leaks no absolute path", out)


# ══════════════════════════════════════════════════════════════════════════
section("7. SCREEN AWARENESS")
# ══════════════════════════════════════════════════════════════════════════
import screen_awareness as sa

front = sa.frontmost_app()
out = say("what app am I in")
check(front and front.lower() in out.lower(), "names the REAL frontmost app",
      f"expected {front} | {out}")

before_shots = set(glob.glob(os.path.join(tempfile.gettempdir(), "nova_screen_*")))
out = say("what's on my screen")
check(bool(out), "screen describe answers", out)
check(not re.search(r"\b(keyboard|mouse|webcam|dog|cat)s?\b", out.lower()),
      "invents no physical objects", out)
after_shots = set(glob.glob(os.path.join(tempfile.gettempdir(), "nova_screen_*")))
check(after_shots == before_shots, "no screenshot residue",
      f"{sorted(after_shots - before_shots)}")


# ══════════════════════════════════════════════════════════════════════════
section("8. BROWSER  (fresh window, closed BY ID)")
# ══════════════════════════════════════════════════════════════════════════
import browser_control as bc

browser_was_running = bc._is_running("Brave Browser")
print(f"  Brave already running: {browser_was_running}")

out = say("where am I")
if not browser_was_running:
    check("browser" in out.lower() or "not" in out.lower() or "open" in out.lower(),
          "declines honestly when no browser is running", out)

# Launch Brave ourselves so every window is OURS. Close BY ID only.
win_id = None
ours_is_front = False
subprocess.run(["open", "-a", "Brave Browser", "--args", "--no-startup-window"],
               check=False)
time.sleep(4)
ok, wid = osa('tell application "Brave Browser" to set w to make new window\n'
              'tell application "Brave Browser" to return id of w', timeout=25)
if ok and wid.strip().isdigit():
    win_id = wid.strip()
    print(f"  created OUR window id={win_id}")

    ok2, _ = osa(f'tell application "Brave Browser" to set URL of active tab of '
                 f'window id {win_id} to "https://example.com"', timeout=25)
    time.sleep(3)

    # CRITICAL. browser_control always drives `window 1` — the FRONT window —
    # while this test verifies against `window id {win_id}`. When Brave was not
    # already running those were the same window and everything passed. When
    # HIS Brave is already open and in front, they are not: "close this tab" and
    # "go back" were operating on HIS window while the assertions read ours.
    # That is a destructive test acting on a real page, not just a wrong result.
    #
    # So: bring ours to the front, and REFUSE to run the live steps unless
    # window 1 really is ours.
    osa(f'tell application "Brave Browser" to set index of window id {win_id} to 1',
        timeout=15)
    osa('tell application "Brave Browser" to activate', timeout=15)
    time.sleep(1.5)
    okf, front_id = osa('tell application "Brave Browser" to return id of window 1',
                        timeout=15)
    ours_is_front = okf and front_id.strip() == win_id

if win_id and not ours_is_front:
    unverified("browser live tests — could not bring OUR window to the front, "
               "and browser_control drives window 1, so running them would act "
               "on his page")

if win_id and ours_is_front:
    out = say("where am I")
    check("example" in out.lower(), "'where am I' reads the REAL page", out)

    out = say("open a new tab")
    check("tab" in out.lower(), "new tab responds", out)
    ok3, cnt = osa(f'tell application "Brave Browser" to return (count of tabs of window id {win_id})')
    check(ok3 and int(cnt) >= 2, "a tab was ACTUALLY opened", f"tabs={cnt}")

    out = say("close this tab")
    ok4, cnt2 = osa(f'tell application "Brave Browser" to return (count of tabs of window id {win_id})')
    check(ok4 and int(cnt2) < int(cnt), "tab ACTUALLY closed", f"{cnt} -> {cnt2}")

    out = say("scroll down")
    check("scroll" in out.lower() or "javascript" in out.lower() or not out.strip() == "",
          "scroll responds (or explains the JS setting)", out)

    out = say("go back")
    check(bool(out), "back navigation responds", out)

if not win_id:
    unverified(f"browser live tests (could not create a window: {wid})")


# ══════════════════════════════════════════════════════════════════════════
section("9. MAPS  (degrades honestly without location)")
# ══════════════════════════════════════════════════════════════════════════
out = say("how far is the nearest coffee shop")
check(bool(out), "maps question answers something", out)
check("location" in out.lower() or "mile" in out.lower() or "minute" in out.lower()
      or "can't" in out.lower() or "directions" in out.lower(),
      "maps answers with distance OR degrades honestly", out)
check("i think" not in out.lower(), "maps does not guess a distance", out)


# ══════════════════════════════════════════════════════════════════════════
section("10. MUSIC  (no player running — honest path)")
# ══════════════════════════════════════════════════════════════════════════
spotify_running = bool(osa('tell application "System Events" to (name of processes) contains "Spotify"')[1] == "true")
music_running = bool(osa('tell application "System Events" to (name of processes) contains "Music"')[1] == "true")
print(f"  Spotify running: {spotify_running} | Music running: {music_running}")

out = say("what's playing")
if not spotify_running and not music_running:
    check(re.search(r"neither|not running|nothing.*playing|no player|isn'?t running",
                    out.lower()) is not None,
          "declines honestly when no player is running", out)
    unverified("music transport (play/pause/skip) — needs a running player")
else:
    check(bool(out), "what's playing answers", out)


# ══════════════════════════════════════════════════════════════════════════
section("11. APPS & FOLDERS")
# ══════════════════════════════════════════════════════════════════════════
check(va.tools._resolve_app("clawed") == "Claude", "STT alias 'clawed' -> Claude")
check(va.tools._resolve_app("vs code") == "Visual Studio Code", "alias 'vs code'")
check(va.tools._resolve_app("chrome") == "Google Chrome", "alias 'chrome'")

# Remember which Finder windows were HIS before we open one, so cleanup can
# close only ours. Nicholas asked for this: the sweep used to leave a Downloads
# window sitting open every single run.
_finder_before = set()
_ok, _ids = osa('tell application "Finder" to return id of every window')
if _ok:
    _finder_before = {w.strip() for w in _ids.split(",") if w.strip()}
_textedit_before = osa('tell application "System Events" to '
                       '(name of processes) contains "TextEdit"')[1]

out = say("open my downloads folder")
check("download" in out.lower(), "Finder folder opens", out)
time.sleep(1)

# Guarantee a COLD launch so we test the real path, not "already open".
osa('quit app "TextEdit"'); time.sleep(2)
out = say("open textedit")
check("opening" in out.lower(), "cold launch reports opening (not 'already open')", out)
# It must not claim success unless a window is really on screen.
time.sleep(1)
from tools import NovaTools as _NT
_w = _NT._on_screen_windows("TextEdit")
check(_w is None or _w > 0, "launch produced a REAL on-screen window", f"windows={_w} | {out}")
out2 = say("open textedit")
check("already open" in out2.lower(), "re-opening a running app says so honestly", out2)
time.sleep(2)
running = osa('tell application "System Events" to (name of processes) contains "TextEdit"')[1]
check(running == "true", "TextEdit ACTUALLY launched", f"running={running}")
out = say("quit textedit")
time.sleep(2)
running = osa('tell application "System Events" to (name of processes) contains "TextEdit"')[1]
check(running == "false", "TextEdit ACTUALLY quit", f"running={running}")


# ══════════════════════════════════════════════════════════════════════════
section("12. ROUTING COLLISION MATRIX  (10 stages, first match wins)")
# ══════════════════════════════════════════════════════════════════════════
# Each phrase must reach the intended stage and NO other.
cases = [
    ("what's on my calendar today",      "calendar"),
    ("what reminders do I have",         "calendar"),
    ("what are all my reminders",        "calendar"),
    ("what's on my screen",              "screen"),
    ("what app am I in",                 "screen"),
    ("find my resume",                   "files"),
    ("what's in my budget file",         "files"),
    ("what's my battery at",             "tools"),
    ("what's my volume",                 "tools"),
    ("open spotify",                     "tools"),
    ("what's my favorite color",         "memory"),
    ("remember that I like tea",         "memory"),
    ("what time is it",                  "fastpath"),
]


class _NoSideEffects:
    """tools.match() executes as it matches, so probing the routing matrix
    with phrases like 'open spotify' actually LAUNCHED Spotify and left it
    running. Stub the executors while we probe routing."""

    _STUBS = ("_open_app", "_quit_app", "_open_folder", "_web_search",
              "_screenshot", "_minimize", "_lock_screen", "_request_power",
              "_volume_set", "_volume_adjust", "_mute_audio", "_brightness_set",
              "_brightness_step", "_start_timer", "_cancel_timers")

    def __enter__(self):
        self.saved = {}
        for name in self._STUBS:
            if hasattr(va.tools, name):
                self.saved[name] = getattr(va.tools, name)
                setattr(va.tools, name, lambda *a, **k: "(stubbed)")
        self.saved_music = va.tools._match_music
        self.saved_browser = va.tools._match_browser
        self.saved_maps = va.tools._match_maps
        va.tools._match_music = lambda low: "(stubbed)" if re.search(
            r"\b(music|song|track|spotify|playing|pause|play)\b", low) else None
        va.tools._match_browser = lambda low: None
        va.tools._match_maps = lambda low: None
        return self

    def __exit__(self, *exc):
        for name, fn in self.saved.items():
            setattr(va.tools, name, fn)
        va.tools._match_music = self.saved_music
        va.tools._match_browser = self.saved_browser
        va.tools._match_maps = self.saved_maps


def route_of(text):
    if va.calendar.detect_intent(text) is not None:
        return "calendar"
    if va.screen.detect_intent(text) is not None:
        return "screen"
    if va.files.detect_intent(text) is not None:
        return "files"
    if va._handle_memory_intent.__name__ and _mem_would_fire(text):
        return "memory"
    if va._fast_path(text) is not None:
        return "fastpath"
    if va.tools.match(text) is not None:
        return "tools"
    return "llm"


def _mem_would_fire(text):
    # Non-destructive probe of the memory stage's regexes.
    low = text.lower()
    return bool(re.match(r"^\s*(remember|forget)\b", low)) or \
        bool(re.search(r"what(?:'?s| is) my ([\w][\w ]*?)[\s.?]*$", low)
             and not va.tools.match(text))


_apps_before_matrix = {
    app: osa(f'tell application "System Events" to (name of processes) contains "{app}"')[1]
    for app in ("Spotify", "TextEdit", "Music", "Brave Browser")
}

with _NoSideEffects():
    for phrase, want in cases:
        got = route_of(phrase)
        check(got == want, f"routes to {want}: {phrase!r}", f"got={got}")

# Ordinary conversation must reach the LLM, not be hijacked.
with _NoSideEffects():
    # Launch/search verbs used to match ANYWHERE in a sentence, so talking about
    # comics produced "I couldn't find an app called getting into them".
    for phrase in ("tell me a joke", "how are you doing", "what do you think about jazz",
                   "explain how photosynthesis works",
                   "I haven't seen the comics but I wanted to start getting into them",
                   "I want to start working out more",
                   "I usually run errands on saturday",
                   "I need to find a new job",
                   "let's start over",
                   "I need to run to the store"):
        got = route_of(phrase)
        check(got == "llm", f"falls through to the LLM: {phrase!r}", f"got={got}")


for app, was in _apps_before_matrix.items():
    now_running = osa(f'tell application "System Events" to (name of processes) contains "{app}"')[1]
    check(now_running == was, f"routing matrix launched nothing new: {app}",
          f"before={was} after={now_running}")


# ══════════════════════════════════════════════════════════════════════════
section("13. LLM FALLBACK + TTS CLEANLINESS")
# ══════════════════════════════════════════════════════════════════════════
out = say("in one sentence, what is a black hole")
check(len(out) > 20, "LLM answers a general question", out)
check("—" not in out and "*" not in out and "#" not in out,
      "no markdown or em dashes in spoken output (invariant 10)", out)
check(not re.match(r"^\s*[-•\d]+[.)]\s", out), "no list formatting", out)


# ══════════════════════════════════════════════════════════════════════════
section("14. VOICE COMPONENTS  (wake / STT / TTS, no mic needed)")
# ══════════════════════════════════════════════════════════════════════════
import numpy as np

# --- TTS: synthesize to audio, verify real samples come out ---
try:
    from tts_engine import TTSEngine
    tts = TTSEngine(CFG["tts"], mic_gate=va.mic_gate)
    audio = None
    kok = getattr(tts, "_primary", None)
    if kok is not None and hasattr(kok, "create"):
        voice = CFG["tts"].get("voice", "af_nova")
        try:
            audio = kok.create("Testing Nova speech synthesis.", voice=voice,
                               speed=float(CFG["tts"].get("speed", 1.0)))
        except Exception as exc:
            check(False, "Kokoro synthesis", str(exc)[:160])
    if audio is not None:
        arr = np.asarray(audio[0] if isinstance(audio, tuple) else audio, dtype=float)
        check(arr.size > 1000 and float(np.abs(arr).max()) > 0.01,
              "TTS synthesizes real non-silent audio",
              f"samples={arr.size} peak={float(np.abs(arr).max()):.3f}")
    else:
        unverified("TTS synthesis (no reachable synth method)")
except Exception as exc:
    check(False, "TTS engine loads", str(exc)[:200])

# --- Wake word: score a synthetic "Nova" and silence ---
try:
    from wake_openwakeword import OpenWakeWordDetector
    wc = CFG.get("wake_word", {})
    det = OpenWakeWordDetector(
        model_path=str(Path(BACKEND) / "nova.onnx"),
        threshold=float(wc.get("oww_threshold", 0.5)),
        trigger_level=int(wc.get("oww_trigger_level", 2)),
    )
    check(True, "wake model loads", "nova.onnx")
    silence = np.zeros(16000 * 2, dtype=np.int16)
    fired = False
    for i in range(0, len(silence) - 1280, 1280):
        if det.process(silence[i:i + 1280].tobytes()):
            fired = True
    check(not fired, "wake word does NOT fire on silence")

    wav = Path(tempfile.gettempdir()) / f"{MARKER}_nova.wav"
    subprocess.run(["say", "-v", "Samantha", "-o", str(wav),
                    "--data-format=LEI16@16000", "Nova"], check=False, timeout=30)
    if wav.exists():
        import wave
        with wave.open(str(wav)) as w:
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        det2 = OpenWakeWordDetector(
            model_path=str(Path(BACKEND) / "nova.onnx"),
            threshold=float(wc.get("oww_threshold", 0.5)),
            trigger_level=int(wc.get("oww_trigger_level", 2)))
        peak = 0.0
        hit = False
        for i in range(0, len(pcm) - 1280, 1280):
            r = det2.process(pcm[i:i + 1280].tobytes())
            peak = max(peak, getattr(det2, "last_score", 0.0) or 0.0)
            hit = hit or bool(r)
        # macOS `say` voices do NOT drive this model (measured: peak 0.001-0.017
        # across Samantha/Alex, bare and padded). It was live-verified at
        # 0.93-1.00 on his real voice through a running fan. Lowering the
        # threshold to make a synthetic clip pass would degrade real accuracy,
        # so this stays honestly unverified rather than faked green.
        if hit or peak > 0.3:
            check(True, "wake model responds to a synthetic 'Nova'", f"peak={peak:.3f}")
        else:
            unverified(f"wake-word POSITIVE detection (macOS `say` peaks {peak:.3f}; "
                       "needs his real voice — live-verified 0.93-1.00 previously)")
        wav.unlink(missing_ok=True)
    else:
        unverified("wake-word positive test (could not synthesize audio)")
except Exception as exc:
    check(False, "wake word detector loads", str(exc)[:200])

# --- STT: transcribe synthesized speech ---
try:
    wav = Path(tempfile.gettempdir()) / f"{MARKER}_stt.wav"
    subprocess.run(["say", "-v", "Samantha", "-o", str(wav),
                    "--data-format=LEI16@16000",
                    "what is on my calendar today"], check=False, timeout=30)
    if wav.exists():
        from faster_whisper import WhisperModel
        model = WhisperModel(CFG["stt"].get("model", "base.en"),
                             device="cpu", compute_type="int8")
        segs, _ = model.transcribe(str(wav))
        text = " ".join(sg.text for sg in segs).strip().lower()
        check("calendar" in text, "STT transcribes real speech", repr(text))
        wav.unlink(missing_ok=True)
    else:
        unverified("STT (could not synthesize audio)")
except Exception as exc:
    check(False, "STT transcription", str(exc)[:200])


# ══════════════════════════════════════════════════════════════════════════
section("15. SYSTEM COMMANDS  (sleep / wake / mute)")
# ══════════════════════════════════════════════════════════════════════════
out = say("mute yourself")
check(va.is_muted is True, "mute actually sets the flag", f"muted={va.is_muted} | {out}")
out = say("unmute")
check(va.is_muted is False, "unmute clears the flag", f"muted={va.is_muted} | {out}")

out = say("go to sleep")
# Sleep is ONE level: it returns to wake mode via _return_to_wake, it does not
# clear is_awake (there is deliberately no way to get stuck asleep).
check(va._return_to_wake is True, "sleep returns Nova to wake mode",
      f"_return_to_wake={va._return_to_wake} | {out}")
check("nova" in out.lower(), "sleep tells him how to get back", out)
va._return_to_wake = False


# ══════════════════════════════════════════════════════════════════════════
section("16. PASSIVE LEARNING  (wake-mode reconciliation)")
# ══════════════════════════════════════════════════════════════════════════
before = set(db_facts())
va._session_turns = [
    {"role": "user", "content": "I just started taking guitar lessons on Thursdays"},
    {"role": "assistant", "content": "That's great."},
]
try:
    va._end_conversation()
    va._llm_queue.join()
    time.sleep(1)
    learned = set(db_facts()) - before
    check(bool(learned), "reconciler learned something from the conversation",
          f"{learned}")
    joined = " ".join(f"{k} {v}" for _, k, v in learned).lower()
    check(("guitar" in joined or "lesson" in joined) if learned else False,
          "the learned fact is GROUNDED in what was actually said", f"{learned}")
    check("piano" not in joined and "violin" not in joined,
          "reconciler invented no instrument", f"{learned}")
except Exception as exc:
    check(False, "reconciliation runs", str(exc)[:200])


# ══════════════════════════════════════════════════════════════════════════
section("17. RAG")
# ══════════════════════════════════════════════════════════════════════════
if va._rag is not None:
    try:
        res = va._rag.query("budget", n_results=2)
        check(isinstance(res, str), "RAG query returns a string (never raises)",
              f"{type(res).__name__}, len={len(res) if isinstance(res, str) else 'n/a'}")
    except Exception as exc:
        check(False, "RAG query fails gracefully", str(exc)[:200])
else:
    check(True, "RAG absent — queries degrade to empty context by design")


# ══════════════════════════════════════════════════════════════════════════
section("CLEANUP")
# ══════════════════════════════════════════════════════════════════════════
# Finder — close ONLY the windows this run opened, by id. Same discipline as
# the browser: never touch a window that was his.
_ok, _ids = osa('tell application "Finder" to return id of every window')
_ours = ([w.strip() for w in _ids.split(",")
          if w.strip() and w.strip() not in _finder_before] if _ok else [])
for _wid in _ours:
    osa(f'tell application "Finder" to close window id {_wid}', timeout=15)
if _ours:
    time.sleep(0.5)
    _ok2, _after = osa('tell application "Finder" to return id of every window')
    _left = ([w.strip() for w in _after.split(",")
              if w.strip() and w.strip() not in _finder_before] if _ok2 else [])
    check(not _left, f"Finder windows we opened are closed ({len(_ours)})",
          f"still open: {_left}")

# TextEdit — the cold-launch test starts it. Quit it ONLY if it was not his.
if _textedit_before == "false":
    osa('quit app "TextEdit"', timeout=20)
    time.sleep(1)
    _te = osa('tell application "System Events" to '
              '(name of processes) contains "TextEdit"')[1]
    check(_te == "false", "TextEdit quit (it was not running before)",
          f"running={_te}")

# Browser — close ONLY our window, BY ID.
if not browser_was_running:
    # WE launched Brave, and it holds only our window, so quitting the app is
    # both correct and the only thing that reliably works: `close window id`
    # silently fails on Brave (this once destroyed a real window of his), and
    # `tell application ... to quit` trips Brave's warn-before-quit with -128.
    # `quit app "X"` goes through cleanly.
    for _ in range(3):
        osa('quit app "Brave Browser"', timeout=25)
        time.sleep(2)
        still = osa('tell application "System Events" to (name of processes) contains "Brave Browser"')[1]
        if still == "false":
            break
    check(still == "false", "Brave quit (it was not running before)", f"running={still}")
elif win_id:
    # His browser was ALREADY running: never quit it. Close only our window,
    # by id, once — and report honestly if Brave refuses.
    osa(f'tell application "Brave Browser" to close window id {win_id}', timeout=20)
    time.sleep(1)
    still = osa(f'tell application "Brave Browser" to return (exists window id {win_id})')[1]
    check(still != "true", "our browser window closed BY ID", f"exists={still}")

# Files
removed = 0
for p in created:
    try:
        if p.exists():
            p.unlink()
            removed += 1
    except OSError:
        pass
stray = []
for root in ("Desktop", "Downloads", "Documents", "Pictures"):
    stray += glob.glob(str(HOME / root / f"*{MARKER}*"))
check(not stray, f"zero file residue (removed {removed})", f"{stray}")

# Memory
now_facts = set(db_facts())
extra = now_facts - BASE_FACTS
if extra:
    con = sqlite3.connect(str(DB_PATH))
    try:
        for c, k, v in extra:
            con.execute("DELETE FROM facts WHERE category=? AND key=?", (c, k))
        con.commit()
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        con.close()
final_extra = set(db_facts()) - BASE_FACTS
check(not final_extra, "zero memory residue (DB back to baseline)", f"{final_extra}")

# Volume
if BASE_VOL is not None:
    osa(f"set volume output volume {BASE_VOL}")
    ok, v = osa("output volume of (get volume settings)")
    check(ok and int(v) == BASE_VOL, "volume at baseline", f"{v}")

# Reminders
if ek_ok:
    left = [r for r in cr.get_all_reminders() if MARKER in str(r.get("title", ""))]
    check(not left, "zero reminder residue", f"{[r.get('title') for r in left]}")

# Screenshots / timers
check(not glob.glob(os.path.join(tempfile.gettempdir(), "nova_screen_*")),
      "zero screenshot residue")
check(len(va.tools._timers) == 0, "zero timers left running")


# ══════════════════════════════════════════════════════════════════════════
section("LISTENER  (rules applied to EVERY response above)")
# ══════════════════════════════════════════════════════════════════════════
check(not LISTENER_PROBLEMS,
      f"every spoken response was fit to speak ({len(LISTENER_PROBLEMS)} problem(s))",
      "" if not LISTENER_PROBLEMS else
      " || ".join(f"{p[0][:28]!r}: {'; '.join(p[2])}" for p in LISTENER_PROBLEMS[:6]))
for asked, said, problems in LISTENER_PROBLEMS[:8]:
    print(f"    asked : {asked[:70]}")
    print(f"    said  : {said[:110]}")
    print(f"    issue : {'; '.join(problems)}")


# ══════════════════════════════════════════════════════════════════════════
section("RESULT")
# ══════════════════════════════════════════════════════════════════════════
print(f"\n  {PASS}/{PASS + FAIL} passed")
if FAILURES:
    print(f"\n  {len(FAILURES)} FAILURES:")
    for label, detail in FAILURES:
        print(f"    ✗ {label}")
        if detail:
            print(f"      {str(detail)[:220]}")
if UNVERIFIED:
    print(f"\n  {len(UNVERIFIED)} UNVERIFIED (needs conditions this run lacked):")
    for u in UNVERIFIED:
        print(f"    - {u}")
sys.exit(1 if FAIL else 0)
