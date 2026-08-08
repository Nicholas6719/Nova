"""
Browser control — Brave / Chrome / Safari.

Improves on the Jarvis project's version in a few ways:
  * Not hardcoded to Brave. Picks whichever supported browser is RUNNING
    (Brave preferred), so it follows the user rather than forcing an app.
  * Back / forward / reload actually work — Chromium's AppleScript exposes
    `go back` natively, which the Jarvis notes assumed was impossible.
  * Tab awareness: list open tabs, switch to one by name, close others.
  * Scroll via injected JavaScript, which needs a browser setting; when it's
    off we say so instead of pretending it worked.

Chromium and Safari differ in their scripting vocabulary, so everything goes
through the small adapter below rather than raw AppleScript at each call site:
    active tab   Chromium: `active tab of window 1`   Safari: `current tab of ...`
    title        Chromium: `title of ...`             Safari: `name of ...`
    back/reload  Chromium: native commands            Safari: JavaScript only
"""

from __future__ import annotations

import logging
import re
import subprocess
import urllib.parse
from typing import Optional

log = logging.getLogger("nova.browser")

CHROMIUM = ("Brave Browser", "Google Chrome", "Arc", "Microsoft Edge")
SAFARI = "Safari"
_PREFERENCE = ("Brave Browser", "Google Chrome", "Safari", "Arc", "Microsoft Edge")

# Spoken site names → URLs. Anything not here still works: a bare domain is
# opened directly, and everything else falls back to a Google search.
SITES: dict[str, str] = {
    "youtube": "https://www.youtube.com", "google": "https://www.google.com",
    "gmail": "https://mail.google.com", "google drive": "https://drive.google.com",
    "google docs": "https://docs.google.com", "google calendar": "https://calendar.google.com",
    "github": "https://github.com", "reddit": "https://www.reddit.com",
    "twitter": "https://x.com", "x": "https://x.com",
    "facebook": "https://www.facebook.com", "instagram": "https://www.instagram.com",
    "amazon": "https://www.amazon.com", "netflix": "https://www.netflix.com",
    "hulu": "https://www.hulu.com", "disney plus": "https://www.disneyplus.com",
    "spotify": "https://open.spotify.com", "wikipedia": "https://www.wikipedia.org",
    "stack overflow": "https://stackoverflow.com", "stackoverflow": "https://stackoverflow.com",
    "linkedin": "https://www.linkedin.com", "chatgpt": "https://chat.openai.com",
    "claude": "https://claude.ai", "hacker news": "https://news.ycombinator.com",
    "twitch": "https://www.twitch.tv", "discord": "https://discord.com/app",
    "notion": "https://www.notion.so", "figma": "https://www.figma.com",
    "ebay": "https://www.ebay.com", "espn": "https://www.espn.com",
    "cnn": "https://www.cnn.com", "bbc": "https://www.bbc.com",
    "new york times": "https://www.nytimes.com", "nytimes": "https://www.nytimes.com",
    "weather": "https://weather.com", "maps": "https://maps.google.com",
    "apple": "https://www.apple.com", "zillow": "https://www.zillow.com",
    "yelp": "https://www.yelp.com", "imdb": "https://www.imdb.com",
    "paypal": "https://www.paypal.com", "venmo": "https://venmo.com",
}

# Proper display names where .title() gets it wrong (spoken aloud).
DISPLAY: dict[str, str] = {
    "youtube": "YouTube", "github": "GitHub", "gmail": "Gmail", "x": "X",
    "chatgpt": "ChatGPT", "stackoverflow": "Stack Overflow",
    "stack overflow": "Stack Overflow", "linkedin": "LinkedIn",
    "nytimes": "the New York Times", "new york times": "the New York Times",
    "hacker news": "Hacker News", "imdb": "IMDb", "espn": "ESPN",
    "cnn": "CNN", "bbc": "the BBC", "ebay": "eBay", "paypal": "PayPal",
    "disney plus": "Disney Plus", "google drive": "Google Drive",
    "google docs": "Google Docs", "google calendar": "Google Calendar",
}

_DOMAIN_RE = re.compile(
    r"^[\w-]+(\.[\w-]+)+(/\S*)?$|^https?://\S+$", re.I)


# ══════════════════════════════════════════════════════════════════════════
# Plumbing
# ══════════════════════════════════════════════════════════════════════════
def _osa(script: str, timeout: float = 15.0) -> tuple[bool, str]:
    try:
        r = subprocess.run(["osascript", "-e", script],
                           capture_output=True, text=True, timeout=timeout)
    except Exception as e:
        return False, str(e)
    return r.returncode == 0, (r.stdout or r.stderr or "").strip()


def _is_running(app: str) -> bool:
    ok, out = _osa(f'tell application "System Events" to '
                   f'(name of processes) contains "{app}"')
    return ok and out == "true"


def _installed(app: str) -> bool:
    from pathlib import Path
    return any(Path(d, f"{app}.app").exists()
               for d in ("/Applications", "/System/Applications"))


def active_browser(launch: bool = False) -> Optional[str]:
    """The running supported browser (preference order), or None.

    Never uses `tell application "X"` to probe — that would LAUNCH the browser.
    """
    for app in _PREFERENCE:
        if _is_running(app):
            return app
    if launch:
        for app in _PREFERENCE:
            if _installed(app):
                subprocess.run(["open", "-a", app], capture_output=True)
                import time
                for _ in range(20):
                    time.sleep(0.5)
                    if _is_running(app):
                        time.sleep(1.0)
                        return app
    return None


def _tab_ref(app: str) -> str:
    return "current tab of window 1" if app == SAFARI else "active tab of window 1"


def _title_prop(app: str) -> str:
    return "name" if app == SAFARI else "title"


def resolve_target(raw: str) -> tuple[str, str]:
    """Spoken target → (url, spoken_label).

    Known site → its URL; bare domain → https://domain; anything else →
    a Google search, so "pull up chocolate chip cookies" still does something
    sensible rather than failing.
    """
    s = re.sub(r"[?.!,]+$", "", (raw or "").strip())
    s = re.sub(r"^(the|a|an)\s+", "", s, flags=re.I)
    s = re.sub(r"\s+(website|site|page|dot com)$", "", s, flags=re.I).strip()
    key = s.lower()
    if key in SITES:
        return SITES[key], DISPLAY.get(key, s.title())
    if _DOMAIN_RE.match(s):
        return (s if s.lower().startswith("http") else f"https://{s}"), s
    return (f"https://www.google.com/search?q={urllib.parse.quote(s)}",
            f"a search for {s}")


# ══════════════════════════════════════════════════════════════════════════
# Actions — each returns a spoken string
# ══════════════════════════════════════════════════════════════════════════
def open_url(url: str, label: str, new_tab: bool = True) -> str:
    app = active_browser(launch=True)
    if not app:
        subprocess.run(["open", url], capture_output=True)   # default browser
        return f"Opening {label}."
    esc = url.replace('"', '\\"')
    if app == SAFARI:
        script = (f'tell application "{app}"\n activate\n'
                  f' if (count of windows) = 0 then\n  make new document with properties {{URL:"{esc}"}}\n'
                  f' else\n  tell window 1 to set current tab to (make new tab with properties {{URL:"{esc}"}})\n'
                  f' end if\nend tell')
    elif new_tab:
        script = (f'tell application "{app}"\n activate\n'
                  f' if (count of windows) = 0 then\n  make new window\n'
                  f'  set URL of active tab of window 1 to "{esc}"\n'
                  f' else\n  tell window 1 to make new tab at end of tabs '
                  f'with properties {{URL:"{esc}"}}\n end if\nend tell')
    else:
        script = (f'tell application "{app}"\n activate\n'
                  f' set URL of active tab of window 1 to "{esc}"\nend tell')
    ok, err = _osa(script)
    return f"Opening {label}." if ok else f"I couldn't open {label}."


def where_am_i() -> str:
    app = active_browser()
    if not app:
        return "No browser is open right now."
    ok, out = _osa(f'tell application "{app}" to return '
                   f'({_title_prop(app)} of {_tab_ref(app)}) & "||" & '
                   f'(URL of {_tab_ref(app)})')
    if not ok or "||" not in out:
        return f"I couldn't read the current page in {app}."
    title, url = out.split("||", 1)
    host = re.sub(r"^www\.", "", urllib.parse.urlparse(url.strip()).netloc)
    title = title.strip()
    return f"You're on {title}" + (f", at {host}." if host else ".")


def navigate_history(direction: str) -> str:
    """Back / forward. Chromium supports these natively; Safari needs JS."""
    app = active_browser()
    if not app:
        return "No browser is open right now."
    if app == SAFARI:
        ok, _ = _osa(f'tell application "Safari" to do JavaScript '
                     f'"history.{direction}()" in current tab of window 1')
        return (f"Going {direction}." if ok else
                "Safari needs 'Allow JavaScript from Apple Events' enabled in "
                "its Develop menu for that.")
    ok, err = _osa(f'tell application "{app}" to tell active tab of window 1 '
                   f'to go {direction}')
    return f"Going {direction}." if ok else f"I couldn't go {direction}."


def reload_page() -> str:
    app = active_browser()
    if not app:
        return "No browser is open right now."
    ref = _tab_ref(app)
    ok, _ = _osa(f'tell application "{app}" to tell {ref} to reload') if app != SAFARI \
        else _osa('tell application "Safari" to set URL of current tab of window 1 '
                  'to (URL of current tab of window 1)')
    return "Reloading the page." if ok else "I couldn't reload the page."


def new_tab() -> str:
    app = active_browser(launch=True)
    if not app:
        return "I couldn't open a browser."
    if app == SAFARI:
        ok, _ = _osa('tell application "Safari"\n activate\n'
                     ' tell window 1 to set current tab to (make new tab)\nend tell')
    else:
        ok, _ = _osa(f'tell application "{app}"\n activate\n'
                     f' tell window 1 to make new tab at end of tabs\nend tell')
    return "Opened a new tab." if ok else "I couldn't open a new tab."


def close_tab() -> str:
    app = active_browser()
    if not app:
        return "No browser is open right now."
    ok, _ = _osa(f'tell application "{app}" to close {_tab_ref(app)}')
    return "Closed the tab." if ok else "I couldn't close the tab."


def list_tabs() -> str:
    app = active_browser()
    if not app:
        return "No browser is open right now."
    ok, out = _osa(f'tell application "{app}" to return '
                   f'{_title_prop(app)} of every tab of window 1')
    if not ok:
        return "I couldn't read your tabs."
    tabs = [t.strip() for t in out.split(",") if t.strip()]
    if not tabs:
        return "You don't have any tabs open."
    n = len(tabs)
    if n <= 5:
        return f"You have {n} tab{'s' if n != 1 else ''} open: " + "; ".join(tabs) + "."
    return (f"You have {n} tabs open. The first few are: "
            + "; ".join(tabs[:5]) + ".")


def switch_tab(query: str) -> str:
    """Focus the first tab whose title or URL contains `query`."""
    app = active_browser()
    if not app:
        return "No browser is open right now."
    ok, out = _osa(f'tell application "{app}" to return '
                   f'{_title_prop(app)} of every tab of window 1')
    if not ok:
        return "I couldn't read your tabs."
    tabs = [t.strip() for t in out.split(",")]
    q = query.lower().strip()
    idx = next((i for i, t in enumerate(tabs, start=1) if q in t.lower()), None)
    if idx is None:
        ok2, urls = _osa(f'tell application "{app}" to return URL of every tab of window 1')
        if ok2:
            idx = next((i for i, u in enumerate(urls.split(","), start=1)
                        if q in u.lower()), None)
    if idx is None:
        return f"I couldn't find a tab matching {query}."
    if app == SAFARI:
        ok3, _ = _osa(f'tell application "Safari"\n activate\n'
                      f' tell window 1 to set current tab to tab {idx}\nend tell')
    else:
        ok3, _ = _osa(f'tell application "{app}"\n activate\n'
                      f' tell window 1 to set active tab index to {idx}\nend tell')
    return f"Switched to {tabs[idx-1].strip()}." if ok3 else "I couldn't switch tabs."


def close_other_tabs() -> str:
    app = active_browser()
    if not app:
        return "No browser is open right now."
    if app == SAFARI:
        ok, _ = _osa('tell application "Safari" to tell window 1 to '
                     'close (every tab whose index is not (index of current tab))')
    else:
        ok, _ = _osa(f'tell application "{app}" to tell window 1 to '
                     f'close (every tab whose id is not (id of active tab))')
    return "Closed the other tabs." if ok else "I couldn't close the other tabs."


_JS_HELP = ("That needs JavaScript from Apple Events enabled: in the browser's "
            "Develop menu, turn on Allow JavaScript from Apple Events.")


def scroll(direction: str, amount: str = "page") -> str:
    """Scroll via injected JS. Browsers block this by default, so verify and
    explain rather than claiming success."""
    app = active_browser()
    if not app:
        return "No browser is open right now."
    px = {"page": "window.innerHeight*0.9", "top": "0", "bottom": "document.body.scrollHeight"}
    if direction == "top":
        js = "window.scrollTo(0,0)"
    elif direction == "bottom":
        js = "window.scrollTo(0,document.body.scrollHeight)"
    else:
        sign = "" if direction == "down" else "-"
        js = f"window.scrollBy(0,{sign}{px['page']})"
    if app == SAFARI:
        ok, err = _osa(f'tell application "Safari" to do JavaScript "{js}" '
                       f'in current tab of window 1')
    else:
        ok, err = _osa(f'tell application "{app}" to tell active tab of window 1 '
                       f'to execute javascript "{js}"')
    if ok:
        return {"top": "Scrolled to the top.", "bottom": "Scrolled to the bottom."}.get(
            direction, f"Scrolled {direction}.")
    return f"I couldn't scroll. {_JS_HELP}"
