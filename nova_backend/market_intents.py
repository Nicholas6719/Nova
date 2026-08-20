"""
Market NL dispatch — "how is SpaceX doing", "what do analysts say about Apple".

Strict regex, like every other intent module. A finance word must appear, and
the phrasing must be a market question, because "how is my mom doing" and "what
do you think about Apple products" are ordinary conversation.

The spoken answer is assembled from `market_engine`'s templates and NOTHING
here is phrased by a model. That is the whole point: he is spending real money
on what he hears, and a wrong price sounds exactly like a right one.

Nova is a RESEARCHER. She reports what the market and the analysts say, with
attribution. She never forms a view, and there is no code path that could
produce one — the strings are all templates over real payloads.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import market_engine as M
import panels as P

log = logging.getLogger("nova.market.intents")

# "how's the market" — the indices, not a company.
_MARKET_RE = re.compile(
    r"\b(?:how(?:'?s| is| are)\s+(?:the\s+)?(?:market|markets|stock\s+market)|"
    r"what(?:'?s| is)\s+(?:the\s+)?market\s+doing|"
    r"market\s+(?:today|update|summary)|"
    r"how\s+did\s+the\s+market\s+(?:do|close))\b",
    re.I,
)

# A company or ticker question. The finance word is what makes it one.
_STOCK_RE = re.compile(
    r"\b(?:stock|stocks|share|shares|ticker|price)\b"
    r"|\bhow(?:'?s| is| are)\s+(?P<a>[A-Za-z][\w.& ]{1,28}?)\s+"
    r"(?:stock\s+)?(?:doing|trading|performing)\b"
    r"|\bwhat(?:'?s| is)\s+(?P<b>[A-Za-z][\w.& ]{1,28}?)\s+(?:stock\s+)?"
    r"(?:trading\s+at|at|worth)\b",
    re.I,
)

# What the experts say — the thing he cared most about.
_ANALYST_RE = re.compile(
    r"\b(?:analyst|analysts|rating|ratings|recommendation|recommendations|"
    r"price\s+target|experts?)\b",
    re.I,
)

_NEWS_RE = re.compile(r"\b(?:news|headlines?|what(?:'?s| is)\s+happening\s+with)\b",
                      re.I)

# Ordinary conversation that would otherwise be swept up. "Apple" is a fruit and
# a company; "how are you doing" is not a market question.
_NOT_MARKET_RE = re.compile(
    r"\b(?:how\s+are\s+you|how(?:'?s| is)\s+(?:it|your|my|his|her|everything|"
    r"the\s+(?:weather|family|kids|dog|day|food))|"
    r"apple\s+(?:pie|juice|sauce|watch|music|tv|pay|care|store)|"
    r"share\s+(?:it|this|that|with|my\s+screen)|price\s+of\s+(?:gas|milk|eggs))\b",
    re.I,
)

# Words that are never a company, so a stray match cannot become a lookup.
_STOPWORDS = frozenset({
    "the", "my", "your", "our", "this", "that", "it", "everything", "things",
    "today", "tomorrow", "stuff", "work", "life", "school", "weather", "market",
    "markets", "stock", "stocks", "you", "he", "she", "they", "we",
})

# How the company is named inside a question.
_SUBJECT_RE = re.compile(
    r"\b(?:about|on|for|of|with|is|in)\s+(?P<s>[A-Z][\w.&]*(?:\s+[A-Z][\w.&]*)?)"
    r"|\bhow(?:'?s| is| are)\s+(?P<h>[\w.&]+(?:\s+[\w.&]+)?)\s+"
    r"(?:stock\s+)?(?:doing|trading|performing)"
    r"|\b(?P<t>[A-Z]{1,5})\s+(?:stock|shares?|ticker)\b"
    r"|\b(?:stock|shares?|price|ticker)\s+(?:of|for)\s+(?P<p>[\w.&]+(?:\s+[\w.&]+)?)",
    re.I,
)


class NovaMarket:
    """Market questions. Never touches the LLM."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.enabled = bool(config.get("finance", {}).get("enabled", True))
        self.watchlist = list(config.get("finance", {}).get("watchlist", []))
        self.last_panel = None

    # ── Detection ─────────────────────────────────────────────────────────────
    def detect_intent(self, text: str) -> Optional[str]:
        if not self.enabled or not text or not text.strip():
            return None
        t = text.strip()
        if _NOT_MARKET_RE.search(t):
            return None

        if _MARKET_RE.search(t):
            return "market"

        has_subject = self._subject(t) is not None
        if _ANALYST_RE.search(t) and has_subject:
            return "analysts"
        if _NEWS_RE.search(t) and has_subject and _STOCK_RE.search(t):
            return "news"
        if _STOCK_RE.search(t) and has_subject:
            return "quote"
        return None

    def _subject(self, text: str) -> Optional[str]:
        m = _SUBJECT_RE.search(text)
        if not m:
            return None
        raw = next((g for g in m.groupdict().values() if g), "").strip()
        raw = re.sub(r"[^\w.& ]", "", raw).strip()
        if not raw or raw.lower() in _STOPWORDS:
            return None
        # Trim trailing stopwords so "Apple stock" resolves as "Apple".
        words = [w for w in raw.split() if w.lower() not in _STOPWORDS]
        return " ".join(words) or None

    # ── Handling ──────────────────────────────────────────────────────────────
    def handle(self, intent: str, text: str) -> str:
        self.last_panel = None
        try:
            if intent == "market":
                return self._market()
            subject = self._subject(text)
            if not subject:
                return "Which stock did you mean?"
            found = M.resolve_symbol(subject)
            if not found.get("ok"):
                if found.get("error") == "not found":
                    return f"I couldn't find a listing for {subject}."
                return "I couldn't reach market data just now."
            if intent == "analysts":
                return self._analysts(found)
            if intent == "news":
                return self._news(found)
            return self._quote(found)
        except Exception as exc:
            log.exception(f"market intent failed: {exc}")
            return "Something went wrong looking that up."

    # ── The market as a whole ─────────────────────────────────────────────────
    def _market(self) -> str:
        rows = M.indices()
        if not rows:
            return "I couldn't reach market data just now."
        self.last_panel = ("finance", P.panel(
            title="Markets",
            subtitle="Today",
            blocks=[P.items([{
                "title": q["spoken_name"].replace("the ", "").title(),
                "detail": f"{q['price']:,.2f}",
                "meta": self._move(q),
            } for q in rows])] + self._watchlist_blocks()))
        return M.say_indices(rows)

    # ── One company ───────────────────────────────────────────────────────────
    def _quote(self, found: dict) -> str:
        q = M.quote(found["symbol"])
        if not q.get("ok"):
            return (f"I couldn't find a listing for {found['name']}."
                    if q.get("error") == "not found"
                    else "I couldn't reach market data just now.")

        recs = M.recommendations(found["symbol"])
        fund = M.fundamentals(found["symbol"])
        self.last_panel = ("finance", self._company_panel(q, recs, fund))

        spoken = M.say_quote(q)
        # The analyst line rides along when it exists, because "how is it doing"
        # is really asking that too — but only ever as attributed counts.
        if recs.get("ok"):
            spoken += " " + M.say_recommendations(recs, q["name"])
        return spoken

    def _analysts(self, found: dict) -> str:
        recs = M.recommendations(found["symbol"])
        q = M.quote(found["symbol"])
        if not recs.get("ok"):
            if recs.get("error") == "no coverage":
                return f"I don't have any analyst coverage for {found['name']}."
            return "I couldn't reach the analyst data just now."
        self.last_panel = ("finance", self._company_panel(
            q, recs, M.fundamentals(found["symbol"])))
        # Deliberately NOT followed by a view. He asked what they think; that
        # is the answer, and adding "so it looks strong" would be advice.
        return M.say_recommendations(recs, found["name"])

    def _news(self, found: dict) -> str:
        n = M.news(found["symbol"])
        if not n.get("ok"):
            return f"I couldn't find recent news for {found['name']}."
        items = n["items"]
        self.last_panel = ("finance", P.panel(
            title=found["name"], subtitle="Recent news",
            blocks=[P.items([{"title": i["headline"], "meta": i["source"]}
                             for i in items])]))
        # Headlines are quoted, never summarised by a model.
        head = items[0]["headline"].rstrip(".")
        more = len(items) - 1
        return (f"The latest on {found['name']}: {head}."
                + (f" There are {more} more headlines on screen." if more else ""))

    # ── Panels ────────────────────────────────────────────────────────────────
    def _move(self, q: dict) -> str:
        pct = q.get("change_pct")
        if not isinstance(pct, (int, float)):
            return ""
        return f"{'+' if pct >= 0 else ''}{pct:.2f}%"

    def _company_panel(self, q: dict, recs: dict, fund: dict) -> dict:
        blocks = []
        if q.get("ok"):
            blocks.append(P.stat(f"{q['price']:,.2f}", label=q["symbol"],
                                 detail=self._move(q)))
        if recs.get("ok"):
            blocks.append(P.rows([
                ("Buy", str(recs["buy"])),
                ("Hold", str(recs["hold"])),
                ("Sell", str(recs["sell"])),
            ], title=f"Analysts ({recs['total']})"))
        if fund.get("ok"):
            pairs = []
            if isinstance(fund.get("week52_high"), (int, float)):
                pairs.append(("52 week high", f"{fund['week52_high']:,.2f}"))
            if isinstance(fund.get("week52_low"), (int, float)):
                pairs.append(("52 week low", f"{fund['week52_low']:,.2f}"))
            if isinstance(fund.get("pe"), (int, float)):
                pairs.append(("P/E", f"{fund['pe']:,.1f}"))
            if pairs:
                blocks.append(P.rows(pairs))
        # Said out loud too, so the absence is never mistaken for a zero.
        if not recs.get("ok"):
            blocks.append(P.note("No analyst coverage available."))
        return P.panel(title=q.get("name", "") or q.get("symbol", ""),
                       subtitle=q.get("symbol", ""), blocks=blocks)

    def _watchlist_blocks(self) -> list:
        """His own tickers, from config so he edits them without code."""
        if not self.watchlist:
            return []
        rows = []
        for sym in self.watchlist[:12]:
            q = M.quote(sym)
            if q.get("ok"):
                rows.append({"title": q["symbol"], "detail": q["name"][:30],
                             "meta": f"{q['price']:,.2f}   {self._move(q)}"})
        return [P.items(rows, title="Watchlist")] if rows else []

    # ── The home card ─────────────────────────────────────────────────────────
    def home_block(self) -> Optional[dict]:
        """A compact watchlist card for home.

        BLOCKING — one network round trip per ticker — and that is fine,
        because views.py wraps it in a tile that only ever calls this from a
        background thread. It deliberately does NOT cache here: two cache
        layers with different clocks is how the card ends up pinned to
        whatever the inner one happened to hold when the outer one primed.
        """
        if not self.enabled:
            return None
        rows = []
        for sym in self.watchlist[:5]:
            q = M.quote(sym)
            if q.get("ok"):
                rows.append({"title": q["symbol"], "detail": f"{q['price']:,.2f}",
                             "meta": self._move(q)})
        return P.items(rows, title="Markets") if rows else None

    # ── The finance screen ────────────────────────────────────────────────────
    def screen_payload(self) -> dict:
        """What "go to finance" shows: the indices and his watchlist."""
        rows = M.indices()
        blocks = []
        if rows:
            blocks.append(P.items([{
                "title": q["spoken_name"].replace("the ", "").title(),
                "detail": f"{q['price']:,.2f}",
                "meta": self._move(q)} for q in rows], title="Indices"))
        else:
            blocks.append(P.note("I couldn't reach market data just now."))
        blocks += self._watchlist_blocks()
        if not self.watchlist:
            blocks.append(P.note("No watchlist yet. Add tickers to "
                                 "finance.watchlist in the config."))
        return P.panel(title="Markets", subtitle="Today", blocks=blocks)
