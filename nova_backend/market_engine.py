"""
Market data — Nova as a RESEARCHER, never an advisor.

Nicholas is serious about the stock market and said plainly that this feature
will affect real money. That makes it the highest-stakes thing in Nova, and the
whole design follows from one line he agreed to:

    ✅ "Eighteen analysts cover it. Twelve say buy, five hold, one sell."
    ❌ "Now is not the time to buy."

The first is data, sourced and attributed. The second is advice, and if a 3B
invented it he would spend real money on it. So:

  * EVERY spoken number is templated from a real payload. The LLM never phrases
    a price. Same rule as weather_engine, enforced harder, for the same reason:
    a wrong number sounds exactly like a right one to someone listening.
  * An unknown ticker is admitted, never guessed at.
  * A source that is down is admitted, never filled in from the other one
    silently — the caller is told which parts are missing.
  * Nothing here raises. A failed lookup returns an error dict and Nova says
    she could not get it.

TWO SOURCES, because neither is sufficient alone and the split was measured:

    Yahoo    quotes and INDICES, no API key at all. Finnhub's index endpoint
             is paid ("Market data subscription required for CFD indices"),
             so the S&P, Nasdaq and Dow have to come from here.
    Finnhub  symbol search, analyst recommendations, company news, fundamentals
             and earnings. Free tier, personal non-commercial use, ~60/min.
             Tested on his key: recommendations ARE free, which was the part he
             cared most about. Price targets are 403 — paid — and are simply
             absent rather than faked.

INVARIANT 3, fourth exception (his explicit choice). What leaves the machine is
a ticker symbol or a company name he said out loud. The privacy wrinkle he was
told about and accepted: weather leaks an approximate location, but a stock
query leaks WHAT HE IS INVESTED IN. Never the LLM, never memory, never the
transcript. `finance.enabled: false` disables it entirely.

The key lives at NOVA_DATA_DIR/finnhub_credentials.json, never config.json,
which is committed.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

log = logging.getLogger("nova.market")

_TIMEOUT = 6.0
_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"

DATA_DIR = Path(os.environ.get(
    "NOVA_DATA_DIR", Path.home() / "Library" / "Application Support" / "Nova"))

# The indices "how's the market doing" means, and the names Nova says.
INDICES = (("^GSPC", "the S and P 500"),
           ("^IXIC", "the Nasdaq"),
           ("^DJI", "the Dow"))


# ── Plumbing ──────────────────────────────────────────────────────────────────
def _get(url: str, headers: Optional[dict] = None) -> dict:
    """One HTTP GET. Never raises — the caller gets {'ok': False, ...}."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _UA,
                                                   **(headers or {})})
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
            return {"ok": True, "data": json.loads(r.read().decode())}
    except urllib.error.HTTPError as exc:
        # 404 means the symbol does not exist. Reporting that as "I couldn't
        # reach the market" would be a lie, and the wrong lie: he would retry.
        if exc.code == 404:
            return {"ok": False, "error": "not found", "status": 404}
        log.warning(f"market fetch failed: HTTP {exc.code}")
        return {"ok": False, "error": f"http {exc.code}", "status": exc.code}
    except Exception as exc:
        log.warning(f"market fetch failed: {str(exc)[:120]}")
        return {"ok": False, "error": str(exc)[:120]}


_key_cache: Optional[str] = None


def finnhub_key() -> Optional[str]:
    """The Finnhub key, or None. Absent is a normal state, not an error: quotes
    and indices still work through Yahoo without it."""
    global _key_cache
    if _key_cache is not None:
        return _key_cache or None
    path = DATA_DIR / "finnhub_credentials.json"
    try:
        _key_cache = json.loads(path.read_text()).get("api_key", "").strip()
    except Exception:
        _key_cache = ""
    return _key_cache or None


def _finnhub(path: str, **params) -> dict:
    key = finnhub_key()
    if not key:
        return {"ok": False, "error": "no finnhub key", "needs_key": True}
    q = urllib.parse.urlencode({**params, "token": key})
    return _get(f"https://finnhub.io/api/v1/{path}?{q}")


# ── Quotes (Yahoo: works for stocks AND indices, needs no key) ────────────────
def quote(symbol: str) -> dict:
    """Current price and today's move. `ok` False means Nova says so."""
    sym = (symbol or "").strip().upper()
    if not sym:
        return {"ok": False, "error": "no symbol"}

    # 5m over one day: enough points to draw a line, one request, and the
    # same endpoint the price already comes from. The series was always in
    # this response and was being thrown away — a chart that costs no extra
    # network call is the only kind worth having on a screen Nova redraws
    # whenever he looks at it.
    res = _get("https://query1.finance.yahoo.com/v8/finance/chart/"
               f"{urllib.parse.quote(sym)}?interval=5m&range=1d")
    if not res["ok"]:
        if res.get("error") == "not found":
            return {"ok": False, "error": "not found", "symbol": sym}
        return {"ok": False, "error": "unreachable"}
    try:
        meta = res["data"]["chart"]["result"][0]["meta"]
    except (KeyError, IndexError, TypeError):
        # Yahoo answers 200 with an empty result for a symbol that does not
        # exist. That is a MISS, not an outage, and must be said differently.
        return {"ok": False, "error": "not found", "symbol": sym}

    price = meta.get("regularMarketPrice")
    prev = meta.get("chartPreviousClose") or meta.get("previousClose")
    if not isinstance(price, (int, float)):
        return {"ok": False, "error": "not found", "symbol": sym}

    change = pct = None
    if isinstance(prev, (int, float)) and prev:
        change = price - prev
        pct = change / prev * 100

    # The closes, thinned to something a sparkline can use. Gaps are real —
    # Yahoo returns null for minutes with no trade — and are dropped rather
    # than interpolated, because a drawn line between two real points is a
    # shape and a line through an invented one is a claim.
    series: list[float] = []
    try:
        closes = res["data"]["chart"]["result"][0]["indicators"]["quote"][0]["close"]
        clean = [c for c in closes if isinstance(c, (int, float))]
        if len(clean) > 120:
            step = len(clean) / 120
            clean = [clean[int(i * step)] for i in range(120)]
        series = clean
    except (KeyError, IndexError, TypeError):
        series = []

    return {"ok": True, "symbol": meta.get("symbol", sym), "series": series,
            "name": " ".join((meta.get("shortName")
                              or meta.get("longName") or sym).split()),
            "price": price, "prev_close": prev,
            "change": change, "change_pct": pct,
            "currency": meta.get("currency", "USD"),
            "high": meta.get("regularMarketDayHigh"),
            "low": meta.get("regularMarketDayLow")}


def indices() -> list[dict]:
    """The three he means by "how's the market". Failures are simply absent —
    two of three is still a useful answer."""
    out = []
    for sym, spoken in INDICES:
        q = quote(sym)
        if q.get("ok"):
            q["spoken_name"] = spoken
            out.append(q)
    return out


# ── Symbol resolution (Finnhub) ───────────────────────────────────────────────
def resolve_symbol(text: str) -> dict:
    """'apple' -> AAPL. Returns ok False when nothing matches, so Nova can say
    she could not find a listing instead of inventing a ticker."""
    term = (text or "").strip()
    if not term:
        return {"ok": False, "error": "no term"}

    # Already a plausible ticker? Confirm it against a real quote rather than
    # assuming — "IT" and "ALL" are real words and real tickers.
    if term.isupper() and 1 <= len(term) <= 5 and term.isalpha():
        if quote(term).get("ok"):
            return {"ok": True, "symbol": term, "name": term}

    res = _finnhub("search", q=term, exchange="US")
    if not res["ok"]:
        return {"ok": False, "error": res.get("error", "unreachable"),
                "needs_key": res.get("needs_key", False)}

    for row in (res["data"] or {}).get("result", []):
        if row.get("type") != "Common Stock":
            continue
        sym = (row.get("displaySymbol") or row.get("symbol") or "").strip()
        # Skip foreign/derivative listings, which carry a dot suffix.
        if sym and "." not in sym:
            return {"ok": True, "symbol": sym,
                    "name": row.get("description") or sym}

    return _resolve_via_yahoo(term)


def _resolve_via_yahoo(term: str) -> dict:
    """Second resolver, and not a redundant one.

    Finnhub's symbol database misses recent listings: "spacex" returns count 0
    on the US exchange — and that was HIS example. Yahoo returns SPCX. Two
    sources for resolution, mirroring the two sources for the data itself.

    EQUITY only, and no dotted symbols, so a leveraged ETF tracking a company
    ("Tradr 2X Short SpaceX Daily ETF") can never be mistaken for the company.
    """
    res = _get("https://query2.finance.yahoo.com/v1/finance/search"
               f"?q={urllib.parse.quote(term)}&quotesCount=6")
    if not res["ok"]:
        return {"ok": False, "error": "not found", "term": term}
    for row in (res["data"] or {}).get("quotes", []):
        if row.get("quoteType") != "EQUITY":
            continue
        sym = (row.get("symbol") or "").strip()
        if sym and "." not in sym and "-" not in sym:
            name = " ".join((row.get("shortname") or row.get("longname")
                             or sym).split())
            return {"ok": True, "symbol": sym, "name": name}
    return {"ok": False, "error": "not found", "term": term}


# ── What the experts say (Finnhub; free, and the part he cared most about) ────
def recommendations(symbol: str) -> dict:
    res = _finnhub("stock/recommendation", symbol=symbol)
    if not res["ok"]:
        return {"ok": False, "error": res.get("error", "unreachable")}
    rows = res["data"] or []
    if not rows:
        return {"ok": False, "error": "no coverage"}
    latest = rows[0]
    buy = int(latest.get("strongBuy", 0)) + int(latest.get("buy", 0))
    hold = int(latest.get("hold", 0))
    sell = int(latest.get("sell", 0)) + int(latest.get("strongSell", 0))
    total = buy + hold + sell
    if total == 0:
        return {"ok": False, "error": "no coverage"}
    return {"ok": True, "buy": buy, "hold": hold, "sell": sell,
            "total": total, "period": latest.get("period", "")}


def news(symbol: str, limit: int = 5) -> dict:
    today = date.today()
    res = _finnhub("company-news", symbol=symbol,
                   **{"from": str(today - timedelta(days=7)), "to": str(today)})
    if not res["ok"]:
        return {"ok": False, "error": res.get("error", "unreachable")}
    rows = [r for r in (res["data"] or []) if r.get("headline")][:limit]
    if not rows:
        return {"ok": False, "error": "no news"}
    return {"ok": True, "items": [
        {"headline": r.get("headline", ""), "source": r.get("source", ""),
         "url": r.get("url", "")} for r in rows]}


def fundamentals(symbol: str) -> dict:
    res = _finnhub("stock/metric", symbol=symbol, metric="all")
    if not res["ok"]:
        return {"ok": False, "error": res.get("error", "unreachable")}
    m = (res["data"] or {}).get("metric") or {}
    if not m:
        return {"ok": False, "error": "no data"}
    return {"ok": True,
            "week52_high": m.get("52WeekHigh"),
            "week52_low": m.get("52WeekLow"),
            "pe": m.get("peBasicExclExtraTTM"),
            "market_cap": m.get("marketCapitalization")}


# ── Spoken formatting: DETERMINISTIC, always ──────────────────────────────────
# Nothing below is ever phrased by a model. A wrong price sounds exactly like a
# right one, and he is spending real money on what he hears.

def money(v, currency: str = "USD") -> str:
    if not isinstance(v, (int, float)):
        return "an unknown amount"
    return f"{v:,.2f} dollars" if currency == "USD" else f"{v:,.2f}"


def say_quote(q: dict) -> str:
    if not q.get("ok"):
        return ""
    name = q.get("name") or q.get("symbol")
    parts = [f"{name} is at {money(q['price'], q.get('currency','USD'))}"]
    pct = q.get("change_pct")
    if isinstance(pct, (int, float)):
        direction = "up" if pct >= 0 else "down"
        parts.append(f", {direction} {abs(pct):.1f} percent today")
    return "".join(parts) + "."


def say_recommendations(r: dict, name: str) -> str:
    """Attributed, and never turned into a verdict. 'Twelve say buy' is what
    analysts said; 'you should buy' is what Nova must never say."""
    if not r.get("ok"):
        return ""
    def verb(n: int) -> str:
        return "says" if n == 1 else "say"

    bits = []
    if r["buy"]:
        bits.append(f"{r['buy']} {verb(r['buy'])} buy")
    if r["hold"]:
        bits.append(f"{r['hold']} {verb(r['hold'])} hold")
    if r["sell"]:
        bits.append(f"{r['sell']} {verb(r['sell'])} sell")
    listed = ", ".join(bits[:-1]) + (" and " + bits[-1] if len(bits) > 1 else bits[0])
    return (f"Of {r['total']} analysts covering {name}, {listed}"
            + (f", as of {r['period']}." if r.get("period") else "."))


def say_indices(rows: list[dict]) -> str:
    if not rows:
        return "I couldn't get the market right now."
    parts = []
    for q in rows:
        pct = q.get("change_pct")
        if isinstance(pct, (int, float)):
            direction = "up" if pct >= 0 else "down"
            parts.append(f"{q['spoken_name']} is {direction} {abs(pct):.1f} percent")
        else:
            parts.append(f"{q['spoken_name']} is at {q['price']:,.0f}")
    sentence = (", ".join(parts[:-1]) + ", and " + parts[-1]
                if len(parts) > 1 else parts[0])
    # "the S and P 500 is down..." mid-sentence is right, but this IS the
    # sentence, so it gets a capital.
    return sentence[0].upper() + sentence[1:] + "."
