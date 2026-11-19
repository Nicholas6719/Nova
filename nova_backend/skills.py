"""
Nova's skills — everything the brain (brain.py) can do without the LLM.

One Skill per kind of request. Each has:
  seeds  — example phrasings. Most come from tests/adversarial_phrases.txt,
           the routing corpus: every line there is a sentence that once broke
           Nova, which makes it exactly the hard material a router needs.
  slots  — (va, text) -> args, or None to refuse. These REUSE the extraction
           half of the existing *_intents.py modules (file tokens and
           destinations, the weather place, the market subject, the memory
           regexes). A refusal hands the turn to the runner-up skill.
  run    — (va, text, args): the SAME stage method the regex chain calls
           (nova.py "Stage runners"), so a reflex and a regex match can never
           behave differently after the decision.
  tier   — confidence.py consequence (low | medium | high).

THE PRINCIPLE FOR SLOTS, AND WHY IT IS ASYMMETRIC
The brain may widen how Nova UNDERSTANDS a question. It does not, in Stage 1,
widen what makes Nova CHANGE something unconfirmed:
  - reads (calendar, weather, screen, market, folder listings, system stats,
    what's playing) are lenient — the embedding decides, the handler answers;
  - writes with no confirmation step (calendar writes, power, lock, launching
    or quitting an app by a name the regex did not parse, a web search
    query) are STRICT — the skill acts only on sentences the module's own
    regex already accepts, so the brain changes nothing about them except
    which stage gets asked first;
  - file move/copy/rename are lenient on the verb, because file_intents asks
    "move it to Documents?" and nothing on disk changes without a spoken yes;
  - actuation is strict AND work-mode gated, exactly as before.
Stage 2's learning (confirmed "did you mean" answers) is how writes widen.

TOOLS
tools.match() decides and acts in one pass with no intent names to reuse, so
argument-free tool skills carry a `command`: the canonical phrase in tools'
own vocabulary ("pause the music"). The skill runs tools.match(command), which
is precisely what the regex path runs for the canonical phrase. Skills that
need an argument from the sentence (a volume level, an app name, a search
query, a place) are strict: they run only when tools.detect(text) — the same
match() in dry mode — would claim the sentence itself.
"""
from __future__ import annotations

import re
from typing import Optional

from brain import Skill

LOW, MEDIUM, HIGH = "low", "medium", "high"


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════
def _intent_or(module_detect, text: str, default: str, allowed: set) -> str:
    """The module's own sub-intent when its regex has one in `allowed`, else
    the skill's default. The regex is the more specific parser when it
    speaks; the brain only decides which module is asked."""
    try:
        got = module_detect(text)
    except Exception:
        got = None
    return got if got in allowed else default


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z']+", (text or "").lower())


# A READ IS A REQUEST, AND REQUESTS HAVE A SHAPE. "my next meeting is at three"
# sat 0.89 from "when's my next meeting" and read his calendar back to him.
# CLAUDE.md's rule for tools — command words that are also ordinary English must
# be SHAPED like commands — applies to the brain's reads for the same reason: a
# question opens with a question word or a verb, or ends in a question mark; a
# statement about the same subject opens with "my", "I", "the"...
_ASK_OPENERS = {
    "what", "what's", "whats", "when", "when's", "where", "where's", "which", "who",
    "who's", "how", "how's", "hows", "do", "does", "did", "am", "is", "are", "was",
    "any", "anything", "read", "show", "tell", "list", "check", "give", "get",
    "pull", "will", "would", "should", "have", "has", "got", "open", "find",
    "look", "let", "lets", "let's", "describe", "summarize", "say", "go", "take",
    "bring", "turn", "make", "set", "put", "start", "play", "pause", "skip",
    "stop", "resume", "mute", "unmute", "dim", "brighten", "lower", "raise",
    "crank", "throw", "close", "quit", "launch", "search", "google", "navigate",
    "cancel", "clear", "hide", "restore", "shuffle", "repeat", "next", "previous",
    "volume", "louder", "quieter", "battery", "cpu", "weather", "forecast",
    "hello", "hi", "hey", "good", "morning", "afternoon", "evening", "sorry",
    "one", "come", "remember", "forget", "actually", "or", "and", "so", "ok",
    "okay", "now", "also", "just", "please", "nova", "can", "could", "need",
    "full", "max", "reload", "refresh", "new", "minimize", "lock", "screenshot",
    "capture", "grab", "locate", "move", "copy", "rename", "drop", "drag",
    "stick", "shove", "file", "delete", "trash", "duplicate", "change", "use",
}


def _asks(text: str) -> bool:
    t = (text or "").strip()
    if t.endswith("?"):
        return True
    w = _light_words(t)
    return bool(w) and w[0] in _ASK_OPENERS


def _asking_def(fn):
    return _asking(fn)


def _asking(slots_fn):
    """Wrap a read skill's slots: a statement is refused before extraction."""
    def slots(va, text):
        return slots_fn(va, text) if _asks(text) else None
    return slots


# ═══════════════════════════════════════════════════════════════════════════════
# Views — navigation and the home surface (stage "view")
# ═══════════════════════════════════════════════════════════════════════════════
def _view_alias(va, text: str) -> Optional[dict]:
    import views as V
    if not _nav_shaped(text):
        return None
    low = (text or "").lower()
    # Longest alias first, so "browser activity" beats "browser".
    for alias, name in sorted(V._ALIAS_TO_VIEW.items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"\b{re.escape(alias)}\b", low):
            if name in ("home", "menu"):
                return None            # their own skills
            return {"intent": name}
    return None


# NAVIGATION WORDS ARE ORDINARY ENGLISH — the reason views.py anchors its
# regexes at both ends. The brain's lenient match needs the same discipline in
# another form: a navigation request is SHAPED like one. Measured before these
# guards: "we should go home soon" went home, "what's on the menu for dinner"
# opened the menu, and "minimize Xcode" (an app!) entered work mode.
_NAV_VERBS = {"go", "take", "bring", "back", "show", "open", "return", "head",
              "get", "pull", "switch", "jump", "home", "let's", "lets"}


def _light_words(text: str) -> list[str]:
    from brain import _light
    return _light(text).split()


def _nav_shaped(text: str) -> bool:
    w = _light_words(text)
    return bool(w) and (w[0] in _NAV_VERBS or len(w) <= 2)


def _home_slots(va, text):
    return {"intent": "home"} if _nav_shaped(text) and "home" in _light_words(text) else None


_CAPABILITY_Q = re.compile(
    r"\bwhat\s+(?:else\s+)?(?:can|could)\s+you\s+(?:do|show|help)|"
    r"\bwhat\s+are\s+you\s+capable\b|\b(?:list|tell me)\s+your\s+capabilities\b|"
    r"\bwhat\s+can\s+you\s+help\b", re.I)


def _menu_slots(va, text):
    w = _light_words(text)
    if _CAPABILITY_Q.search(text or "") or (w and w[0] in _NAV_VERBS and
                                            ("menu" in w or "capabilities" in w)):
        return {"intent": "menu"}
    return None


_WORK_WORDS = {"work", "with", "me", "mode", "take", "over", "minimize", "go", "to",
               "puck", "let's", "lets", "get", "start", "shrink", "down", "and",
               "us", "together", "it", "lets", "time", "working", "lets", "now"}


def _work_slots(va, text):
    w = _light_words(text)
    return {"intent": "work"} if w and all(x in _WORK_WORDS for x in w) else None


def _home_surface_slots(intent: str, verbs: set):
    def slots(va, text):
        w = _light_words(text)
        return {"intent": intent} if w and w[0] in verbs else None
    return slots


def _run_view(intent: str):
    return lambda va, t, a: va._run_view(a.get("intent", intent), t)


def _move_slots(va, text: str) -> Optional[dict]:
    import views as V
    return {"intent": "move_card"} if V._move_match(text) else None


VIEW_SKILLS = [
    Skill("view.home", "view", [
        "go home", "nova go home", "take me home", "return home", "go back home",
        "home screen", "back to home", "show me the home screen",
        "go back to the home screen", "head home",
    ], slots=_home_slots, run=_run_view("home"), label="go to the home screen"),
    Skill("view.menu", "view", [
        "show me the menu", "open the menu", "show the menu", "what can you do",
        "what can you show me", "go to the menu", "what are you capable of",
        "list your capabilities", "what can you help me with",
    ], slots=_menu_slots, run=_run_view("menu"), label="show you the menu"),
    Skill("view.work", "view", [
        "work with me", "nova work with me", "work mode", "take over",
        "minimize", "go to puck mode", "let's get to work", "start work mode",
        "shrink down and work with me",
    ], slots=_work_slots, run=_run_view("work"), label="switch to work mode"),
    Skill("view.goto", "view", [
        "go to finance", "take me to finance", "show me the finance screen",
        "open the memory panel", "go to calendar", "go to browser activity",
        "go to system", "go to health", "go to automation", "go to browser",
        "show me the conversations screen", "open the files screen",
        "pull up the system screen", "go to the markets screen",
    ], slots=_view_alias, run=_run_view("menu"), label="open that screen"),
    Skill("view.clear_home", "view", [
        "clear home", "nova clear home", "clear the home screen",
        "clear my screen", "clear home again", "hide all the cards",
        "strip the home screen back", "just show the orb",
    ], slots=_home_surface_slots("clear_home", {"clear", "hide", "strip", "empty", "just"}),
        run=_run_view("clear_home"), label="clear the home screen"),
    Skill("view.restore_home", "view", [
        "restore home", "bring back the cards", "put back everything",
        "restore the home screen", "show the cards again", "bring everything back",
        "put the cards back", "put home back the way it was", "bring the home cards back",
    ], slots=_home_surface_slots("restore_home", {"restore", "bring", "put", "show"}),
        run=_run_view("restore_home"), label="bring the home cards back"),
    Skill("view.move_card", "view", [
        "move music to the left", "move now playing bottom right",
        "move the now playing to the bottom right", "put markets top left",
        "send the stocks to the middle right", "stick the weather in the bottom left",
        "move markets to the bottom right", "move the music widget to the right side",
    ], slots=_move_slots, run=_run_view("move_card"), label="move that card"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Actuation — typing and clicking (stage "actuation"). STRICT and work-mode only.
# ═══════════════════════════════════════════════════════════════════════════════
def _act_slots(intent: str):
    def slots(va, text):
        got = va.actuation.detect_intent(text) if _work_mode(va) else _act_probe(va, text)
        return {"intent": got} if got == intent else None
    return slots


def _work_mode(va) -> bool:
    return bool(getattr(va, "work_mode", False))


def _act_probe(va, text: str) -> Optional[str]:
    """Outside work mode actuation's detector declines everything by design.
    For the brain's load-time "could this skill run this sentence?" question
    (brain._executable) the SENTENCE is what is being judged, so ask the
    detector as if in work mode. decide() still requires work mode via gate."""
    saved = getattr(va, "work_mode", False)
    try:
        va.work_mode = True
        return va.actuation.detect_intent(text)
    finally:
        va.work_mode = saved


def _run_act(va, t, a):
    va._run_actuation(a["intent"], t)


def _scroll_slots(va, text):
    if _work_mode(va):
        return {"via": "actuation"} if va.actuation.detect_intent(text) == "scroll" else None
    if va.tools.detect(text) is not None:
        return {"via": "tools"}
    # judged as a sentence (brain._executable): either path will do
    return {"via": "actuation"} if _act_probe(va, text) == "scroll" else None


def _run_scroll(va, t, a):
    if a.get("via") == "actuation" and _work_mode(va):
        va._run_actuation("scroll", t)
    else:
        _run_tool_text(va, t, a)


ACT_SKILLS = [
    Skill("act.type", "actuation", [
        "type hello there", "type hello from nova", "type yes I do",
        "type this is a test", "write thanks for the update",
    ], slots=_act_slots("type"), run=_run_act, tier=MEDIUM, gate=_work_mode,
        label="type that"),
    Skill("act.key", "actuation", [
        "press enter", "press escape", "hit tab", "hit return",
    ], slots=_act_slots("key"), run=_run_act, tier=MEDIUM, gate=_work_mode,
        label="press that key"),
    Skill("act.click", "actuation", [
        "click the send button", "send it", "click the file menu", "click save",
        "click on the search field",
    ], slots=_act_slots("click"), run=_run_act, tier=MEDIUM, gate=_work_mode,
        label="click that"),
    # ONE scroll skill, routed by the situation. In work mode actuation scrolls
    # whatever is in front; outside it the browser does (tools). Two skills
    # would own the same sentence ("scroll down") — the seed-collision test
    # caught exactly that — and the words cannot say which mode he is in.
    Skill("act.scroll", "actuation", [
        "scroll down", "scroll up", "scroll down more",
    ], slots=_scroll_slots, run=_run_scroll, label="scroll"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Calendar and reminders (stage "calendar")
# ═══════════════════════════════════════════════════════════════════════════════
_CAL_READS = {"read_today", "read_upcoming", "read_rest_of_week",
              "read_reminders", "read_reminders_lookahead"}
_CAL_WRITES = {"create_event", "create_reminder", "copy_reminder",
               "complete_reminder", "delete_reminder", "update_reminder",
               "delete_event"}


def _cal_read_slots(default: str):
    def slots(va, text):
        import calendar_intents as C
        low = (text or "").lower()
        # The module's own contract: a file word with no calendar word is a
        # file request ("what's in my budget file" must never read a calendar).
        if C._FILE_WORD_RE.search(low) and not C._CALENDAR_WORD_RE.search(low):
            return None
        if C._FILE_EXT_RE.search(low):
            return None
        return {"intent": _intent_or(va.calendar.detect_intent, text, default, _CAL_READS)}
    return _asking(slots)


def _cal_write_slots(va, text):
    got = va.calendar.detect_intent(text)
    return {"intent": got} if got in _CAL_WRITES else None


def _run_cal(va, t, a):
    va._run_calendar(a["intent"], t)


CAL_SKILLS = [
    Skill("cal.today", "calendar", [
        "what's on my calendar today", "show me my calendar",
        "what's going on today", "what's my day like", "do I have any meetings today",
        "am I busy today", "what appointments do I have today", "is my afternoon free",
        "what's on the agenda today", "anything on the calendar this morning",
        "what's my calendar today", "what's on my calendar for today",
        "is there anything on my calendar", "what's my schedule today",
        "what do I have today", "do I have anything scheduled today",
        "read me my calendar", "what's all my calendar today",
    ], slots=_cal_read_slots("read_today"), run=_run_cal, label="read your calendar"),
    Skill("cal.upcoming", "calendar", [
        "what's my schedule at home tomorrow", "what's coming up this week",
        "what's on for tomorrow", "do I have meetings tomorrow", "what's my week like",
        "am I free tomorrow afternoon", "what's on the calendar for the weekend",
        "what's my next event", "when's my next meeting",
        "what's on my calendar this week", "what do I have tomorrow",
        "anything coming up", "what does the rest of my week look like",
        "what's on my schedule tomorrow",
    ], slots=_cal_read_slots("read_upcoming"), run=_run_cal,
        label="read what's coming up"),
    Skill("cal.reminders", "calendar", [
        "what reminders do I have", "what are all my reminders",
        "what's on my reminders list", "what do I need to do today",
        "what tasks do I have", "read me my to do list",
        "what are my reminders", "do I have any reminders",
        "do I have any reminders today", "what reminders do I have for today",
        "read me my reminders", "list my reminders",
    ], slots=_cal_read_slots("read_reminders"), run=_run_cal,
        label="read your reminders"),
    Skill("cal.write", "calendar", [
        "put a reminder in for 6pm", "put in a reminder for six",
        "remind me to call mom when I get home", "stick a reminder in for tomorrow morning",
        "copy my receipts reminder to next Monday", "duplicate the dentist reminder",
        "set a reminder for five to call mom", "create a reminder to call mom at 5",
        "create an event called team sync at 3 pm today",
        "delete the reminder to check the oven", "delete the team sync event",
        "mark the call mom reminder as complete", "move the dentist reminder to friday",
        "remind me to pay rent on the first",
    ], slots=_cal_write_slots, run=_run_cal, tier=MEDIUM,
        label="change your calendar"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Screen awareness (stage "screen")
# ═══════════════════════════════════════════════════════════════════════════════
def _screen_slots(default: str):
    def slots(va, text):
        return {"intent": _intent_or(va.screen.detect_intent, text, default,
                                     {"context", "describe", "describe_window"})}
    return _asking(slots)


def _run_screen(va, t, a):
    va._run_screen(a["intent"], t)


SCREEN_SKILLS = [
    Skill("screen.describe", "screen", [
        "what's on my screen", "read my screen", "read the notes on my screen",
        "what am I looking at right now", "can you read this window",
        "what does this page say", "read what's in front of me",
        "read me the notes on my screen", "tell me what the code on my screen says",
        "what are the errors on my screen", "look at my screen",
        "what does my screen say", "read the window behind you",
        "describe my screen",
    ], slots=_screen_slots("describe"), run=_run_screen,
        label="look at your screen"),
    Skill("screen.context", "screen", [
        "what app am I in", "what app is this", "which app am I using",
        "which application is in the front", "what's the frontmost app",
        "what program am I in right now", "what's in front right now",
    ], slots=_screen_slots("context"), run=_run_screen,
        label="tell you which app you're in"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Weather (stage "weather")
# ═══════════════════════════════════════════════════════════════════════════════
@_asking_def
def _weather_slots(va, text):
    import weather_intents as W
    if not va.config.get("weather", {}).get("enabled", True):
        return None
    # The module's action-verb guard is load-bearing ("play some rain sounds",
    # "put my weather report in Documents"), so it applies here unchanged.
    if W._NOT_WEATHER_RE.search(text or ""):
        return None
    if W._WEEK_RE.search(text or ""):
        return {"intent": "week"}
    if W._TOMORROW_RE.search(text or ""):
        return {"intent": "tomorrow"}
    return {"intent": "now"}


WEATHER_SKILLS = [
    Skill("weather", "weather", [
        "what's the weather", "how's the weather", "show me the weather",
        "what's it like outside", "is it cold out", "is it nice out today",
        "do I need an umbrella today", "will it rain later", "what's the high for tomorrow",
        "is it going to snow this week", "how warm is it outside", "what's the weather in Denver right now",
        "go to the weather", "weather in Boston", "what's the weather at home",
        "what's the weather today", "is it going to rain tomorrow",
        "what's the forecast", "what's the temperature outside",
        "how hot is it outside", "what's the weather this week",
        "what's the weather in New Orleans", "is it raining",
    ], slots=_weather_slots, run=lambda va, t, a: va._run_weather(a["intent"], t),
        label="check the weather"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Market data (stage "market")
# ═══════════════════════════════════════════════════════════════════════════════
def _market_slots(default: str):
    def slots(va, text):
        if not getattr(va.market, "enabled", True):
            return None
        import market_intents as M
        if M._NOT_MARKET_RE.search(text or ""):
            return None
        if default == "market":
            return {"intent": "market"}
        if va.market._subject(text or "") is None:
            return None
        return {"intent": _intent_or(va.market.detect_intent, text, default,
                                     {"quote", "analysts", "news"})}
    return _asking(slots)


def _run_market(va, t, a):
    va._run_market(a["intent"], t)


MARKET_SKILLS = [
    Skill("market.overview", "market", [
        "how is the market doing", "how's the stock market today",
        "how's the Dow doing", "is the S&P up today", "how did the Nasdaq close",
        "are stocks up or down today",
        "how's the stock market looking", "how are stocks doing today",
        "what did the market do", "how are the indexes doing",
    ], slots=_market_slots("market"), run=_run_market, label="check the market"),
    Skill("market.quote", "market", [
        "what is Apple stock trading at", "what's the price of Microsoft stock",
        "how are Apple shares doing", "get me a quote for Netflix",
        "what's Disney's stock price", "how's Tesla doing in the market today",
        "how is SpaceX stock doing", "how is the SpaceX stock doing today",
        "what's Tesla stock at", "how's Google stock doing",
        
    ], slots=_market_slots("quote"), run=_run_market, label="get that quote"),
    Skill("market.analysts", "market", [
        "what are the analyst ratings for Nvidia", "what do analysts say about Tesla",
        "what's the analyst rating on Apple", "is Microsoft a buy according to analysts",
    ], slots=_market_slots("analysts"), run=_run_market,
        label="read the analyst ratings"),
    Skill("market.news", "market", [
        "what's the latest news on Tesla stock", "any news about Apple stock",
        "what's happening with Nvidia stock",
    ], slots=_market_slots("news"), run=_run_market, label="get the stock news"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Files (stage "files")
# ═══════════════════════════════════════════════════════════════════════════════
def _file_guard(text: str) -> bool:
    """file_intents' own refusals, reused: calendar commands, memory
    commands and unambiguous web searches are never file requests."""
    import file_intents as F
    t = (text or "").lower().strip()
    return not (F._CALENDAR_COMMAND_RE.search(t) or F._MEMORY_COMMAND_RE.match(t)
                or F._WEB_SEARCH_COMMAND_RE.match(t))


_FILE_INTENTS = {"file_find", "file_describe", "file_open", "file_move",
                 "file_copy", "file_rename", "file_delete", "list_folder"}


def _file_slots(default: str):
    def slots(va, text):
        import file_intents as F
        if not _file_guard(text):
            return None
        t = (text or "").lower().strip()
        intent = _intent_or(va.files.detect_intent, text, default, _FILE_INTENTS)
        if intent in ("file_move", "file_copy"):
            if not va.files._extract_destination(t):
                return None
        # A rename with no new name is still a rename: file_intents asks
        # "what did you want to rename it to?", which is the right answer.
        if intent == "file_delete":
            return {"intent": intent} if F._DELETE_VERB_RE.search(t) else None
        if intent == "list_folder":
            return {"intent": intent} if _folder_named(t) else None
        followup = va.files._recent_file() is not None and bool(F._PRONOUN_RE.search(t))
        if not followup and not va.files._query_tokens(t, intent):
            return None
        return {"intent": intent}
    return slots


def _folder_named(t: str) -> bool:
    import file_intents as F
    import file_manager as fm
    m = F._LIST_FOLDER_RE.search(t)
    if m:
        spoken = next((g for g in m.groupdict().values() if g), "")
        if fm.resolve_folder(spoken) is not None:
            return True
    return any(fm.resolve_folder(w) is not None
               for w in ("desktop", "documents", "downloads", "pictures")
               if re.search(rf"\b{w}\b", t))


@_asking_def
def _list_slots(va, text):
    if not _file_guard(text):
        return None
    return {"intent": "list_folder"} if _folder_named((text or "").lower()) else None


def _run_files(va, t, a):
    va._run_files(a["intent"], t)


FILE_SKILLS = [
    Skill("files.find", "files", [
        "find my resume", "find the menu pdf", "find the notes file",
        "where did I put my resume", "can you find my lease document",
        "look for the budget spreadsheet", "where's my passport pdf",
        "find the screenshot from yesterday", "search for my resume",
        "where is my resume", "where's the budget spreadsheet",
        "locate the invoice pdf", "find the document called project plan",
    ], slots=_file_slots("file_find"), run=_run_files, label="find that file"),
    Skill("files.open", "files", [
        "open my resume", "open the budget spreadsheet", "open the menu pdf",
        "pull up the invoice file", "open the notes document",
    ], slots=_file_slots("file_open"), run=_run_files, label="open that file"),
    Skill("files.describe", "files", [
        "what's in my budget file", "what does the notes file say",
        "summarize my resume", "read the budget spreadsheet to me",
        "tell me what's in the project plan document",
    ], slots=_file_slots("file_describe"), run=_run_files,
        label="read that file"),
    Skill("files.move", "files", [
        "move my resume to Downloads", "move the reminder pdf to Documents",
        "move the report to Documents", "move the weather notes to the desktop",
        "put my resume in Downloads", "put my weather report in Documents",
        "put the budget pdf on my desktop", "put the budget spreadsheet in Documents",
        "put the invoice in Documents", "drop the receipt in Downloads",
        "drag the upcoming events doc into my Documents folder",
        "file this receipt in the taxes folder", "shove the old screenshots in the archive folder",
        "stick that report in the desktop folder",
    ], slots=_file_slots("file_move"), run=_run_files, tier=MEDIUM,
        label="move that file"),
    Skill("files.copy", "files", [
        "copy the weather forecast spreadsheet to Documents",
        "copy my resume to Downloads", "make a copy of the invoice in Documents",
        "duplicate the budget file into the desktop folder",
    ], slots=_file_slots("file_copy"), run=_run_files, tier=MEDIUM,
        label="copy that file"),
    Skill("files.rename", "files", [
        "rename my resume file", "rename my resume to resume final draft",
        "rename the budget spreadsheet to budget final",
        "change the name of the notes file to meeting notes",
    ], slots=_file_slots("file_rename"), run=_run_files, tier=MEDIUM,
        label="rename that file"),
    Skill("files.list", "files", [
        "what's in my documents folder", "what's in my downloads folder",
        "what's on my desktop", "show me my downloads folder",
        "or what's in my downloads folder", "list what's in my documents",
        "what files are on my desktop",
    ], slots=_list_slots, run=_run_files, label="list that folder"),
    Skill("files.delete", "files", [
        "delete my resume file", "trash the old invoice pdf",
        "delete the budget spreadsheet",
    ], slots=_file_slots("file_delete"), run=_run_files,
        label="delete that file"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Memory (stage "memory") — the regexes in nova.py ARE the extractor
# ═══════════════════════════════════════════════════════════════════════════════
def _mem_slots(kind: str):
    def slots(va, text):
        t = text or ""
        if kind == "save":
            return {} if va._SAVE_RE.search(t) else None
        if kind == "recall":
            m = va._RECALL_RE.search(t)
            if not m:
                return None
            key = m.group(1).strip()
            # A system question with nothing stored is a TOOL question.
            if va.memory.find_fact(key) is None and va._is_system_query(key):
                return None
            return {}
        if kind == "note":
            return {} if (not va._SAVE_RE.search(t) and va._NOTE_SAVE_RE.match(t)) else None
        if kind == "unnote":
            return {} if (not va._FORGET_RE.search(t) and va._NOTE_FORGET_RE.match(t)) else None
        if kind == "update":
            return {} if va._UPDATE_RE.search(t) else None
        if kind == "forget":
            return {} if va._FORGET_RE.search(t) else None
        if kind == "meta":
            low = t.lower()
            return {} if any(p in low for p in va._META_RECALL) else None
        return None
    return slots


def _run_mem(va, t, a):
    resp = va._handle_memory_intent(t)
    if resp is not None:
        va._respond(resp)
    else:   # the slots said yes and the handler said no: never go silent
        va._respond("I'm not sure what you'd like me to remember.")


MEMORY_SKILLS = [
    Skill("memory.save", "memory", [
        "remember that my favorite color is blue", "remember my favorite movie is Toy Story",
        "remember that my wife's name is Anna", "remember my gym is Planet Fitness",
    ], slots=_mem_slots("save"), run=_run_mem, label="remember that"),
    Skill("memory.note", "memory", [
        "remember that I like tea", "remember that I put my tax documents in Documents every April",
        "remember I put the budget spreadsheet in Downloads", "remember that I'm allergic to cats",
        "keep in mind that I work late on Thursdays", "remember that I parked on level 3",
    ], slots=_mem_slots("note"), run=_run_mem, label="remember that"),
    Skill("memory.unnote", "memory", [
        "forget that I put my resume in Downloads", "forget that I like tea",
        "forget that I'm allergic to cats", "forget about the parking spot",
    ], slots=_mem_slots("unnote"), run=_run_mem, label="forget that"),
    Skill("memory.recall", "memory", [
        "what's my favorite color", "who is my favorite superhero",
        "what's my car", "what's my address",
        "who's my favorite superhero", "what's my favorite movie",
        "who's my doctor", "what's my wife's name",
    ], slots=_mem_slots("recall"), run=_run_mem, label="tell you what I remember"),
    Skill("memory.update", "memory", [
        "actually my favorite movie is Iron Man", "actually it's blue",
        "actually my gym is Gold's",
    ], slots=_mem_slots("update"), run=_run_mem, label="update that"),
    Skill("memory.forget", "memory", [
        "forget my favorite color", "forget that my gym is Planet Fitness",
        "forget my wife's name",
    ], slots=_mem_slots("forget"), run=_run_mem, label="forget that"),
    Skill("memory.meta", "memory", [
        "what do you know about me", "what do you remember about me",
        "what have I told you",
    ], slots=_mem_slots("meta"), run=_run_mem, label="tell you what you've told me"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Fast path — greeting / time / date / repeat (stage "fastpath")
# ═══════════════════════════════════════════════════════════════════════════════
_GREET = {"hi", "hello", "hey", "morning", "afternoon", "evening", "howdy",
          "sup", "yo", "good", "there", "nova", "again"}


def _greet_slots(va, text):
    from brain import _light
    words = _light(text).split()
    # Only a greeting, nothing after it: "hi, can you find my resume" is a
    # file request that happens to start politely.
    name = va.config["user"]["address_as"].lower()
    if not words or len(words) > 4 or any(w not in _GREET | {name} for w in words):
        return None
    return {}


def _clock_slots(want: str):
    def slots(va, text):
        low = (text or "").lower()
        if re.search(r"\btimes\b|\btimer\b|\btime\s+left\b|\bhow\s+many\s+times\b", low):
            return None
        has_time = bool(re.search(r"\btime\b|\bclock\b", low))
        has_date = bool(re.search(r"\bdate\b|\bday\b|\btoday\b", low))
        if want == "time":
            return {} if has_time else None
        if want == "date":
            return {} if has_date else None
        return {} if has_time and has_date else None
    return slots


def _run_fast(kind: str):
    def run(va, t, a):
        import nova as N
        from datetime import datetime
        now = datetime.now()
        if kind == "greet":
            va._respond(f"{N._time_of_day_greeting()}, {va.config['user']['address_as']}.")
        elif kind == "time":
            va._respond(f"It's {N._spoken_time(now)}.")
        elif kind == "date":
            va._respond(f"Today is {N._spoken_date(now)}.")
        elif kind == "datetime":
            va._respond(f"Today is {N._spoken_date(now)}, and it's {N._spoken_time(now)}.")
        elif kind == "repeat":
            va._respond(va._last_response or "I haven't said anything yet.")
    return run


FAST_SKILLS = [
    Skill("fast.greet", "fastpath", [
        "hello", "hi nova", "hey", "good morning", "good evening", "nova hey",
        "hello nova", "morning",
        "hey there", "hi",
    ], slots=_greet_slots, run=_run_fast("greet"), label="say hello"),
    Skill("fast.time", "fastpath", [
        "what time is it", "what's the time", "tell me the time",
        "what time is it right now", "got the time",
        "what time was it", "current time", "do you know what time it is",
    ], slots=_clock_slots("time"), run=_run_fast("time"), label="tell you the time"),
    Skill("fast.date", "fastpath", [
        "what's the date", "what day is it", "what is today",
        "what's the date today", "what's today", "what day is today",
    ], slots=_clock_slots("date"), run=_run_fast("date"), label="tell you the date"),
    Skill("fast.datetime", "fastpath", [
        "what's the date and what time is it", "what's the date and time",
        "tell me the date and the time",
    ], slots=_clock_slots("both"), run=_run_fast("datetime"),
        label="tell you the date and time"),
    Skill("fast.repeat", "fastpath", [
        "what did you just say", "say that again", "repeat that", "come again",
        "sorry I missed that", "can you say that again", "one more time please",
        "what did you say", "repeat yourself",
    ], run=_run_fast("repeat"), label="repeat that"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Tools (stage "tools")
# ═══════════════════════════════════════════════════════════════════════════════
def _tool_fixed(command: str):
    """An argument-free tool: run tools.match on the canonical phrase."""
    def run(va, t, a):
        resp = va.tools.match(command)
        if resp is None:        # the canonical phrase no longer matches: say so
            va._respond("I couldn't do that one.")
        else:
            va._finish_tool(resp)
    return run


def _tool_strict(va, text):
    """Arguments come from the sentence and tools has no separate extractor,
    so act only where tools itself would claim the sentence (dry run)."""
    return {} if va.tools.detect(text) is not None else None


def _run_tool_text(va, t, a):
    resp = va.tools.match(t)
    if resp is None:
        va._respond("I couldn't do that one.")
    else:
        va._finish_tool(resp)


# THE OBJECT MUST BE NAMED. CLAUDE.md's tools history is nine matchers that
# fired on innocent sentences because the command word is also ordinary
# English: "in memory of my grandfather" read his RAM, "what's next" skipped
# his song, "refresh my memory on that" reloaded his browser. A fixed-command
# skill therefore acts on a sentence only if tools itself would claim it
# (dry-run detect) OR the sentence names what the command acts on.
def _names(pattern: str):
    rx = re.compile(pattern, re.I)
    def slots(va, text):
        if va.tools.detect(text) is not None:
            return {}
        if rx.search(text or "") and _asks(text):
            return {}
        return None
    return slots


_MUSIC = r"\b(?:music|songs?|tracks?|spotify|playlist|tunes?|album|playback)\b"
_GUARDS = {
    "tools.battery": r"\b(?:battery|charge|charged|charging|plugged)\b",
    "tools.cpu": r"\b(?:cpu|processor)\b",
    "tools.memory": r"\b(?:ram|memory)\b(?!.*\bremember)",
    "tools.disk": r"\b(?:disk|storage|drive|space)\b",
    "tools.stats": r"\b(?:mac|computer|laptop|system)\b",
    "tools.running_apps": r"\b(?:apps?|applications?|programs?)\b",
    "tools.system_info": r"\b(?:mac|computer|machine)\b",
    "tools.volume_query": r"\b(?:volume|loud)\b",
    "tools.volume_up": r"\b(?:volume|louder|it up|sound up)\b",
    "tools.volume_down": r"\b(?:volume|quieter|softer|it down|sound down)\b",
    "tools.volume_max": r"\bvolume\b",
    "tools.mute": r"\b(?:sound|audio|speakers?|volume|mac)\b",
    "tools.unmute": r"\bunmute\b",
    "tools.brightness_up": r"\b(?:bright\w*|screen|display)\b",
    "tools.brightness_down": r"\b(?:dim\w*|screen|display|brightness)\b",
    "tools.timer_status": r"\btimers?\b",
    "tools.timer_cancel": r"\b(?:timers?|alarms?)\b",
    "tools.music_play": _MUSIC,
    "tools.music_pause": _MUSIC,
    "tools.music_next": _MUSIC,
    "tools.music_previous": _MUSIC,
    "tools.now_playing": r"\b(?:playing|song|track|music|sings|singing|tune)\b",
    "tools.shuffle": _MUSIC,
    "tools.repeat": _MUSIC,
    "tools.repeat_off": _MUSIC,
    "tools.reload": r"\b(?:page|tab|browser|site|website)\b",
    "tools.forward": r"\b(?:page|browser)\b",
    "tools.new_tab": r"\btab\b",
    "tools.close_tab": r"\btab\b",
    "tools.where_am_i": r"\b(?:page|site|website|url)\b",
    "tools.list_tabs": r"\btabs\b",
}


def _T(name, seeds, command=None, tier=LOW, label=""):
    """A tool skill: fixed command if given, else strict on the sentence."""
    if command:
        guard = _GUARDS.get(name)
        slots = _names(guard) if guard else _tool_strict
        return Skill(name, "tools", seeds, slots=slots, run=_tool_fixed(command),
                     tier=tier, label=label or name.split(".", 1)[1].replace("_", " "))
    return Skill(name, "tools", seeds, slots=_tool_strict, run=_run_tool_text,
                 tier=tier, label=label or name.split(".", 1)[1].replace("_", " "))


TOOL_SKILLS = [
    # ── system readings ──
    _T("tools.battery", ["what's my battery at", "how's my battery", "battery level",
                         "how much battery is left", "is my mac plugged in", "battery percentage",
                         "how much charge do I have", "what's my battery"],
       command="what's my battery level", label="check your battery"),
    _T("tools.cpu", ["what's my cpu", "what is my cpu", "what's my cpu usage",
                     "how hard is the cpu working", "what's using my processor",
                     "how busy is the cpu", "cpu usage"],
       command="what's my cpu usage", label="check your CPU"),
    _T("tools.memory", ["what's my memory usage", "how much memory am I using",
                        "how much ram is in use", "am I running out of memory",
                        "how much ram do I have", "what's my ram", "what's my memory yet",
                        "how much ram is free"],
       command="how much memory am I using", label="check your memory"),
    _T("tools.disk", ["check my storage", "how much disk space do I have",
                      "how much storage is left", "am I running out of space on my drive",
                      "how much space is left", "how full is my disk",
                      "how much free space is on my mac"],
       command="how much disk space do I have", label="check your storage"),
    _T("tools.stats", ["system stats", "how's my mac doing", "how is my computer doing"],
       command="system stats", label="check your Mac"),
    _T("tools.running_apps", ["what apps are open", "what apps are running",
                              "list my open apps", "what's running"],
       command="what apps are running", label="list your open apps"),
    _T("tools.system_info", ["what mac do I have", "what computer is this", "system info"],
       command="system info", label="tell you about your Mac"),
    # ── volume ──
    _T("tools.volume_query", ["what's my volume", "what's the volume", "what's my volume at",
                              "how loud is it", "what's the volume set to"],
       command="what's the volume", label="check the volume"),
    _T("tools.volume_up", ["turn the volume up", "volume up", "louder",
                           "turn it up a little", "crank it up",
                           "turn up the volume", "raise the volume"],
       command="turn the volume up", label="turn the volume up"),
    _T("tools.volume_down", ["turn the volume down", "volume down", "quieter",
                             "turn it down a little", "make it softer",
                             "lower the volume", "turn down the volume"],
       command="turn the volume down", label="turn the volume down"),
    _T("tools.volume_max", ["full volume", "volume all the way up", "max volume"],
       command="volume max", label="turn the volume all the way up"),
    _T("tools.mute", ["mute the audio", "mute my mac", "mute the speakers"],
       command="mute the sound", label="mute the sound"),
    _T("tools.unmute", ["unmute", "unmute the sound", "unmute my mac"],
       command="unmute the sound", label="unmute the sound"),
    _T("tools.volume_set", ["set the volume to 40", "volume 100%", "put the volume at 20",
                            "volume to 70"], tier=LOW, label="set the volume"),
    # ── brightness ──
    _T("tools.brightness_up", ["make the screen brighter", "brighten the screen",
                               "make the display brighter",
                               "screen brightness up"],
       command="brightness up", label="brighten the screen"),
    _T("tools.brightness_down", ["dim the screen", "make the screen dimmer",
                                 "turn the brightness down", "lower the screen brightness",
                                 "brightness down"],
       command="brightness down", label="dim the screen"),
    _T("tools.brightness_set", ["set brightness to 50", "brightness to 80"],
       label="set the brightness"),
    # ── screenshot, lock, power ──
    # Strict: it writes a PNG to his Desktop ("shove the old screenshots in the
    # archive folder" once took a fresh one).
    _T("tools.screenshot", ["take a screenshot", "grab a screenshot of this",
                            "capture the screen"], label="take a screenshot"),
    _T("tools.lock", ["lock the screen", "lock my mac", "lock it up"],
       tier=MEDIUM, label="lock your Mac"),
    _T("tools.power", ["shut down my mac", "restart my computer", "put the mac to sleep",
                       "reboot the computer"], tier=HIGH, label="power your Mac"),
    # ── timers ──
    _T("tools.timer_set", ["set a timer for 10 seconds", "set a timer for ten minutes",
                           "timer for 5 minutes", "remind me in 20 minutes to check the oven",
                           "give me two minutes"], tier=LOW, label="set a timer"),
    _T("tools.timer_status", ["how much time is left on the timer", "check my timer",
                              "how long left on my timer"],
       command="check my timer", label="check your timer"),
    _T("tools.timer_cancel", ["cancel my timer", "stop the timer", "clear the timers"],
       command="cancel my timer", label="cancel your timer"),
    # ── apps, folders, windows ──
    _T("tools.open_app", ["can you open brave", "launch xcode", "open spotify",
                          "open messages", "open youtube", "nova open youtube",
                          "open safari", "launch the calculator"], label="open that"),
    _T("tools.quit_app", ["close brave browser", "close the brave browser",
                          "quit spotify", "close system settings", "close maps",
                          "nova close the brave browser"], tier=LOW, label="close that app"),
    _T("tools.minimize", ["minimize Xcode", "minimize browser", "nova minimize browser",], label="minimize that"),
    _T("tools.open_folder", ["open my downloads folder", "open my home folder",
                             "open the coding projects folder for me",
                             "open the downloads folder", "show me the applications folder"],
       label="open that folder"),
    # ── music ──
    _T("tools.music_play", ["play music", "play some music", "nova play music",
                            "throw some music on", "start playing music", "let's have some music",
                            "play music please", "resume", "resume the music",
                            "unpause", "nova resume", "start the music",
                            "put on some tunes"],
       command="play music", label="play music"),
    _T("tools.music_pause", ["pause music", "pause the music", "pause", "stop the music",
                             "hold the music"],
       command="pause the music", label="pause the music"),
    _T("tools.music_next", ["next song", "next", "skip this song", "next track",
                            "skip to the next song", "play the next track",
                            "skip the song"],
       command="next song", label="skip the song"),
    _T("tools.music_previous", ["previous song", "go back a track", "play the last song again",
                                "previous track"],
       command="previous song", label="go back a song"),
    _T("tools.now_playing", ["what is playing", "what's playing", "what song is this",
                             "what's this song called", "who's singing this", "what track is playing",
                             "what's this song", "who sings this song",
                             "is music playing", "is anything playing right now"],
       command="what's playing", label="tell you what's playing"),
    _T("tools.shuffle", ["shuffle", "shuffle the music", "turn on shuffle"],
       command="turn on shuffle", label="turn on shuffle"),
    _T("tools.repeat", ["repeat this song", "turn on repeat", "put this song on repeat"],
       command="repeat this song", label="repeat the song"),
    _T("tools.repeat_off", ["turn off repeat", "turn off repeat for this song"],
       command="turn off repeat", label="turn off repeat"),
    _T("tools.music_volume", ["turn the music up", "turn the music down",
                              "make the music quieter", "spotify volume to 50"],
       label="change the music volume"),
    _T("tools.play_named", ["play some jazz", "play Back in Black by AC/DC",
                            "play the Tony Stark workshop playlist", "play Taylor Swift",
                            "play rain sounds"], label="play that"),
    # ── browser ──
    _T("tools.reload", ["refresh", "reload the page", "refresh the browser", "reload this tab"],
       command="reload the page", label="reload the page"),
    _T("tools.back", ["go back", "go back a page in the browser"],
       label="go back a page"),
    _T("tools.forward", ["go forward", "go forward a page"], command="go forward",
       label="go forward a page"),
    _T("tools.new_tab", ["new tab", "open a new tab please"], command="open a new tab",
       label="open a new tab"),
    _T("tools.close_tab", ["close this tab", "close the tab", "shut this tab"],
       command="close this tab", label="close the tab"),
    _T("tools.where_am_i", ["what page am I on", "what site is this", "what's the current page"],
       command="what page am i on", label="tell you what page you're on"),
    _T("tools.list_tabs", ["what tabs do I have open", "list my tabs"],
       command="list my tabs", label="list your tabs"),
    _T("tools.web_search", ["search the web for spider-man", "search for italian restaurants",
                            "search for the best laptop stand", "look up python decorators",
                            "look up who won the game", "google the weather in Denver",
                            "search the web for how to move photos into Documents",
                            "nova search the web for the tallest building in the world"],
       label="search the web for that"),
    _T("tools.site_search", ["search amazon for spiderman", "search Disney Plus for Iron Man",
                             "search youtube for cooking videos"], label="search that site"),
    _T("tools.open_site", ["go to youtube.com", "open github.com", "pull up reddit"],
       label="open that site"),
    # ── maps ──
    _T("tools.maps_eta", ["how far is the nearest CVS", "how far is the nearest coffee shop",
                          "where's the closest grocery store", "how long to get to the airport",
                          "how far away is Walmart",
                          "how long would it take me to get to CVS",
                          "how long will it take for me to get to CVS",
                          "where is the nearest gas station"], label="find how far that is"),
    _T("tools.maps_directions", ["navigate to Boston", "directions to the airport",
                                 "how do I get to Fenway Park"], label="open directions"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# General — NOT a request Nova acts on. Conversation, knowledge, creative work,
# and actions she cannot do. No tool: when it wins, classify() returns None and
# the old chain runs (which still refuses unsupported actions honestly).
# Seeded from every `llm` and `unsupported` line of the routing corpus, so the
# sentences that once looked like commands now sit nearest to "not a command".
# ═══════════════════════════════════════════════════════════════════════════════
def _general_seeds() -> list[str]:
    from pathlib import Path
    path = Path(__file__).resolve().parent / "tests" / "adversarial_phrases.txt"
    out = []
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "==>" not in line:
                continue
            phrase, _, expected = line.partition("==>")
            if expected.strip() in ("llm", "unsupported", "no-signoff"):
                out.append(phrase.strip())
    except OSError:
        pass
    return out + [
        "tell me a joke", "how are you", "what's the meaning of life",
        "explain quantum computing", "write me a poem", "what do you think",
        "who is Spider-Man", "what is the capital of France",
        "I had a long day", "can you hear me", "thank you", "thanks",
        "never mind", "I'm just kidding",
        # Command NOUNS used as TOPICS — the false-positive class CLAUDE.md
        # documents for every regex matcher that ever fired on one.
        "the weather was beautiful yesterday", "I love listening to music",
        "my phone battery never lasts", "I spend way too much time on screens",
        "I got home really late last night", "my calendar is packed this month",
        "stocks are too risky for me", "I need to organize my files someday",
        "my memory is terrible", "that song reminds me of high school",
        "I want to work from home more", "the menu at that restaurant was huge",
        "I need more storage in my garage", "time to get some sleep soon",
        "what's your favorite song", "I listened to a great album today",
        "my computer is getting old", "do you like the rain",
        "who invented the telephone", "how does the stock market work",
    ]


GENERAL = Skill("general", "llm", _general_seeds(), run=None)


SKILLS: list[Skill] = (VIEW_SKILLS + ACT_SKILLS + CAL_SKILLS + SCREEN_SKILLS
                       + WEATHER_SKILLS + MARKET_SKILLS + FILE_SKILLS
                       + MEMORY_SKILLS + FAST_SKILLS + TOOL_SKILLS + [GENERAL])
SKILL_BY_NAME = {s.name: s for s in SKILLS}


# ═══════════════════════════════════════════════════════════════════════════════
# The teacher: which skill a regex stage's claim corresponds to
# ═══════════════════════════════════════════════════════════════════════════════
# Nova's LLM has no tools, so JARVIS's lesson — "the LLM solved this with
# exactly one tool" — has no Nova equivalent. The regex chain behind the brain
# is the teacher instead: a sentence the brain fell through on and a regex stage
# then claimed. This maps that claim to ONE skill, or None when it cannot say
# which (brain.learn still applies its whole bar afterwards).
_CAL_SKILL = {"read_today": "cal.today", "read_upcoming": "cal.upcoming",
              "read_rest_of_week": "cal.upcoming", "read_reminders": "cal.reminders",
              "read_reminders_lookahead": "cal.reminders"}
_FILE_SKILL = {"file_find": "files.find", "file_open": "files.open",
               "file_describe": "files.describe", "file_move": "files.move",
               "file_copy": "files.copy", "file_rename": "files.rename",
               "list_folder": "files.list", "file_delete": "files.delete"}
_MARKET_SKILL = {"market": "market.overview", "quote": "market.quote",
                 "analysts": "market.analysts", "news": "market.news"}
_ACT_SKILL = {"type": "act.type", "key": "act.key", "click": "act.click",
              "scroll": "act.scroll"}
# tools.detect() labels -> the skills that can produce them. A tool lesson is
# taken only when the brain's own best tools guess is in the family the dry run
# names, so a volume sentence can never be learned as a brightness one.
_TOOL_FAMILY = {
    "volume": {"tools.volume_query", "tools.volume_up", "tools.volume_down",
               "tools.volume_max", "tools.mute", "tools.unmute", "tools.volume_set"},
    "brightness": {"tools.brightness_up", "tools.brightness_down", "tools.brightness_set"},
    "timer": {"tools.timer_set", "tools.timer_cancel"},
    "open_app": {"tools.open_app"}, "quit_app": {"tools.quit_app"},
    "minimize": {"tools.minimize"}, "web_search": {"tools.web_search"},
    "spotify_search": {"tools.play_named"},
    "maps_engine.nearest": {"tools.maps_eta"},
    "maps_engine.open_directions": {"tools.maps_directions"},
    "browser_control.open_url": {"tools.open_site", "tools.site_search", "tools.web_search"},
    "browser_control.navigate_history": {"tools.back", "tools.forward"},
    "browser_control.reload_page": {"tools.reload"},
    "browser_control.new_tab": {"tools.new_tab"},
    "browser_control.close_tab": {"tools.close_tab"},
    "browser_control.where_am_i": {"tools.where_am_i"},
    "browser_control.list_tabs": {"tools.list_tabs"},
    "browser_control.scroll": {"act.scroll"},
}


def _memory_kind(va, text: str) -> Optional[str]:
    t = text or ""
    if va._SAVE_RE.search(t):
        return "memory.save"
    if va._NOTE_SAVE_RE.match(t):
        return "memory.note"
    if va._UPDATE_RE.search(t):
        return "memory.update"
    if va._FORGET_RE.search(t):
        return "memory.forget"
    if va._NOTE_FORGET_RE.match(t):
        return "memory.unnote"
    if va._RECALL_RE.search(t):
        return "memory.recall"
    if any(p in t.lower() for p in va._META_RECALL):
        return "memory.meta"
    return None


def _fast_kind(va, text: str) -> Optional[str]:
    low = (text or "").lower()
    if _greet_slots(va, text) is not None:
        return "fast.greet"
    if re.search(r"say that again|repeat (?:that|yourself|it)|come again|what did you (?:just )?say", low):
        return "fast.repeat"
    clock = _clock_slots("both")(va, text) is not None
    if clock:
        return "fast.datetime"
    if _clock_slots("time")(va, text) is not None:
        return "fast.time"
    if _clock_slots("date")(va, text) is not None:
        return "fast.date"
    return None


def skill_for_regex(va, text: str, stage: str, intent) -> Optional[str]:
    if stage == "calendar":
        return _CAL_SKILL.get(intent, "cal.write" if intent in _CAL_WRITES else None)
    if stage == "files":
        return _FILE_SKILL.get(intent)
    if stage == "market":
        return _MARKET_SKILL.get(intent)
    if stage == "weather":
        return "weather"
    if stage == "screen":
        return "screen.context" if intent == "context" else "screen.describe"
    if stage == "actuation":
        return _ACT_SKILL.get(intent)
    if stage == "memory":
        return _memory_kind(va, text)
    if stage == "fastpath":
        return _fast_kind(va, text)
    if stage == "tools":
        family = _TOOL_FAMILY.get(intent or "")
        if not family or va.brain is None:
            return None
        # The brain's own best guess, restricted to tool skills.
        saved = (va.brain.last_match, va.brain.unsure, va.brain.conversation)
        try:
            exclude = {sk.name for sk in SKILLS if sk.name not in family}
            name, _ = va.brain.classify(text, exclude=exclude)
        finally:
            va.brain.last_match, va.brain.unsure, va.brain.conversation = saved
        return name if name in family else None
    return None          # views claim at 2c, ahead of the brain: nothing to learn

