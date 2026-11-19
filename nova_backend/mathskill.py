"""Arithmetic in code, never by the model.

Ported from JARVIS (MiniPCJarvis sidecar/brain/mathskill.py, where "what's 17
times 23" took the model seventeen seconds). On Nova the reason is sharper
than speed: CLAUDE.md invariant 3's listener rules forbid invented numbers,
and a wrong number spoken aloud sounds exactly like a right one. His own
turns (Aug 2026) sent "what's 10 plus 10", "what's 25% of 10%", "what's the
square root of 10" and a tip on a $95.82 dinner to the 3B.

Pure text in, a number out. No eval(): the sentence is normalised (number
words to digits, operator words to symbols) and parsed by a small recursive
descent, so nothing he says can run as code. Anything that is not clearly a
sum returns None and routes as before — "set the volume to 50 percent" has a
number and the word percent and is not a sum.
"""
from __future__ import annotations

import math
import re

_ONES = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
         "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
         "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
         "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
         "seventy": 70, "eighty": 80, "ninety": 90}
_SCALE = {"hundred": 100, "thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}

_LEAD = re.compile(
    r"^(?:hey |hi |ok |okay )?(?:nova[,!.]?\s*)?"
    r"(?:(?:can|could|would) you\s+)?(?:please\s+)?"
    r"(?:what(?:'s| is| does)\s+|how much is\s+|calculate\s+|compute\s+|work out\s+|"
    r"tell me\s+|give me\s+)?(?:the\s+(?:answer|result|value)\s+(?:of|to)\s+)?",
    re.I)
_TRAIL = re.compile(r"(?:\s*(?:,|please|nova|equal(?:s)?|make|come to|be|is|equal to|\?|\.|!))+$", re.I)

# Operator words to symbols. Order matters: longer phrases first.
_OPS = [
    (r"\bto the power of\b|\braised to(?: the power of)?\b|\bto the\b(?=\s*\d+(?:st|nd|rd|th)\b)", " ^ "),
    (r"\bthe square root of\b|\bsquare root of\b|\bthe root of\b|\broot of\b|\bsqrt\b", " sqrt "),
    (r"\bsquared\b", " ^ 2 "),
    (r"\bcubed\b", " ^ 3 "),
    (r"\bpercent of\b|% of\b|\bpercentage of\b", " %of "),
    (r"\bhalf of\b|\ba half of\b", " 0.5 * "),
    (r"\ba third of\b", " 0.333333333333 * "),
    (r"\ba quarter of\b", " 0.25 * "),
    (r"\bdouble\b", " 2 * "),
    (r"\btriple\b", " 3 * "),
    (r"\bdivided by\b|\bover\b|\bdivide\b|/|÷", " / "),
    (r"\bmultiplied by\b|\btimes\b|\bmultiply\b|×|\*|(?<=\d)\s*x\s*(?=\d)", " * "),
    (r"\bplus\b|\badded to\b|\badd\b|\+", " + "),
    (r"\bminus\b|\bsubtract\b|\btake away\b|\bless\b|(?<=\d)\s*-\s*(?=\d)", " - "),
    (r"\bmod(?:ulo)?\b|\bremainder of\b", " mod "),
    (r"\bpoint\b", " . "),
]
_ORDINAL = re.compile(r"(\d+)(?:st|nd|rd|th)\b")


_DIGIT = {"zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4",
          "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"}


def _kind(x: str) -> str:
    if x in _ONES:
        return "ones" if _ONES[x] < 10 else "teen"
    if x in _TENS:
        return "tens"
    return "scale"


def _words_to_digits(t: str) -> str:
    """"seventeen times twenty three" -> "17 times 23"; "two hundred and five"
    -> 205; "two point one five" -> "2 point 15"; "nineteen eighty four" -> 1984.

    Two rules the JARVIS port lacked, each a wrong number spoken aloud (audit,
    2026-11-19): words after "point" are DIGITS, one each ("point one five" is
    .15, not .6 - consecutive words used to be ADDED); and a ones/teen word
    after a ones/teen word, or a teen/tens word after a tens word, starts a new
    group read as more digits ("nineteen eighty four" is 1984, "one two three"
    is 123 - they were 103 and 6)."""
    toks = t.split()
    out: list[str] = []
    i = 0
    while i < len(toks):
        w = toks[i]
        if out and out[-1] == "point" and w in _DIGIT:
            digits = ""
            while i < len(toks) and toks[i] in _DIGIT:
                digits += _DIGIT[toks[i]]
                i += 1
            out.append(digits)
            continue
        if w in _ONES or w in _TENS or w in _SCALE or (w == "a" and i + 1 < len(toks)
                                                          and toks[i + 1] in _SCALE):
            groups: list[str] = []
            total, cur, last = 0, 0, None
            j = i
            while j < len(toks):
                x = toks[j]
                if x == "a" and j + 1 < len(toks) and toks[j + 1] in _SCALE:
                    cur, last = 1, "ones"
                elif x in _ONES or x in _TENS:
                    k = _kind(x)
                    if last in ("ones", "teen") or (k in ("teen", "tens") and last == "tens"):
                        groups.append(str(total + cur))
                        total, cur = 0, 0
                    cur += _ONES.get(x, 0) + _TENS.get(x, 0)
                    last = k
                elif x in _SCALE:
                    cur = (cur or 1) * _SCALE[x]
                    if _SCALE[x] >= 1000:
                        total += cur
                        cur = 0
                    last = "scale"
                elif x == "and" and last == "scale" and j + 1 < len(toks) and \
                        (toks[j + 1] in _ONES or toks[j + 1] in _TENS):
                    pass
                else:
                    break
                j += 1
            groups.append(str(total + cur))
            out.append("".join(groups))
            i = j
        else:
            out.append(w)
            i += 1
    return " ".join(out)


_TOKEN = re.compile(r"\d+(?:\.\d+)?|\^|sqrt|%of|mod|[+\-*/()]|\.")


def _tokens(t: str) -> list[str] | None:
    body = _TOKEN.findall(t)
    # Every non-space character must be accounted for: a stray word means this
    # is a sentence about numbers, not a sum.
    rest = _TOKEN.sub("", t).replace(" ", "")
    if rest:
        return None
    # "3 . 5" (from "three point five") -> 3.5
    joined: list[str] = []
    k = 0
    while k < len(body):
        if k + 2 < len(body) and body[k + 1] == "." and body[k].replace(".", "").isdigit() \
                and body[k + 2].isdigit():
            joined.append(f"{body[k]}.{body[k + 2]}")
            k += 3
        elif body[k] == ".":
            return None
        else:
            joined.append(body[k])
            k += 1
    return joined


class _Parser:
    def __init__(self, toks: list[str]) -> None:
        self.t = toks
        self.i = 0

    def peek(self) -> str | None:
        return self.t[self.i] if self.i < len(self.t) else None

    def take(self) -> str:
        v = self.t[self.i]
        self.i += 1
        return v

    def expr(self) -> float:
        v = self.term()
        while self.peek() in ("+", "-"):
            op = self.take()
            r = self.term()
            v = v + r if op == "+" else v - r
        return v

    def term(self) -> float:
        v = self.factor()
        while self.peek() in ("*", "/", "mod", "%of"):
            op = self.take()
            r = self.factor()
            if op == "*":
                v = v * r
            elif op == "/":
                if r == 0:
                    raise ZeroDivisionError
                v = v / r
            elif op == "mod":
                if r == 0:
                    raise ZeroDivisionError
                v = v % r                # floored: "minus 7 mod 3" is 2
            else:                       # a percent of b
                v = v / 100.0 * r
        return v

    def factor(self) -> float:
        # Unary minus binds LOOSER than a power: "minus 3 squared" is -9.
        if self.peek() == "-":
            self.take()
            return -self.factor()
        b = self.base()
        if self.peek() == "^":
            self.take()
            e = self.factor()           # right-associative
            if abs(e) > 64 or abs(b) > 1e6:
                raise OverflowError
            r = b ** e
            if isinstance(r, complex):  # "minus 8 to the power of 0.5"
                raise ValueError
            b = r
        return b

    def base(self) -> float:
        p = self.peek()
        if p is None:
            raise ValueError
        if p == "sqrt":
            self.take()
            v = self.base()
            if v < 0:
                raise ValueError
            return math.sqrt(v)
        if p == "(":
            self.take()
            v = self.expr()
            if self.peek() != ")":
                raise ValueError
            self.take()
            return v
        if re.fullmatch(r"\d+(?:\.\d+)?", p):
            return float(self.take())
        raise ValueError


_SAY = {"+": "plus", "-": "minus", "*": "times", "/": "divided by", "^": "to the power of",
        "mod": "mod", "%of": "percent of", "sqrt": "the square root of", "(": "", ")": ""}


def fmt(v: float) -> str:
    """A number as it should be SAID. Never exponent notation (TTS reads
    "1e+15" letter by letter) and never "0" for something that is not zero:
    four fixed decimals turned 1/50000 into "0" (audit, 2026-11-19)."""
    if v == 0:
        return "0"
    if abs(v) >= 1e15:
        exp = int(math.floor(math.log10(abs(v))))
        mant = f"{v / 10 ** exp:.3g}"
        return f"{mant} times 10 to the power of {exp}"
    if abs(v - round(v)) < 1e-9:
        return f"{int(round(v)):,}"
    if abs(v) < 0.001:
        places = -int(math.floor(math.log10(abs(v)))) + 2      # three significant figures
        return f"{v:.{places}f}".rstrip("0").rstrip(".")
    s = f"{v:,.4f}".rstrip("0").rstrip(".")
    return s


_NOT_SUMS = re.compile(r"(?<![\d.])(?:9/11|24/7|4x4|2x4|50/50|20/20|7/11|4/20|365/24|1x1)(?![\d.])")
_CALC_WORDS = re.compile(r"\b(?:calculate|compute|work\s+out|divided|times|multiplied|over|"
                         r"plus|minus|how\s+much\s+is)\b")


def parse(text: str) -> dict | None:
    """{"expr": "17 times 23", "value": 391.0, "said": "17 times 23 is 391."} or None."""
    # A unit conversion is arithmetic with a table - "how many milliliters in
    # a cup" went to the model's near-miss path and came back as "did you
    # mean render that in 3D". Tried first: its shapes are narrower.
    import units
    conv = units.convert(text)
    if conv:
        return conv
    tip_ = tip(text)
    if tip_:
        return tip_
    t = (text or "").strip().lower()
    # Names, not sums: "what is 9/11", "24/7", "4x4" were answered as division
    # and multiplication (audit). Only when nothing else says "arithmetic".
    if _NOT_SUMS.search(t) and not _CALC_WORDS.search(t):
        return None
    t = _LEAD.sub("", t, count=1)
    t = _TRAIL.sub("", t)
    # "25% of 10%": the FIRST percent is an operator ("percent of"), a percent
    # on the LAST number says the answer is a percentage too (2.5 percent).
    t = re.sub(r"(\d)\s*%\s+of\b", r"\1 percent of", t)
    pct_result = False
    m_pct = re.search(r"(\d)\s*(?:%|percent)\s*$", t)
    if m_pct and re.search(r"percent of|% of", t):
        pct_result = True
        t = t[:m_pct.start(1) + 1]
    t = t.replace("$", "")
    t = _words_to_digits(t)
    for pat, rep in _OPS:
        t = re.sub(pat, rep, t)
    t = _ORDINAL.sub(r"\1", t)          # after the ops: "to the 10th" needs its "th"
    t = re.sub(r"\s+", " ", t).strip()
    if not re.search(r"\d", t):
        return None
    toks = _tokens(t)
    if not toks:
        return None
    if not any(x in ("+", "-", "*", "/", "^", "mod", "%of", "sqrt") for x in toks):
        return None
    numbers = [x for x in toks if re.fullmatch(r"\d+(?:\.\d+)?", x)]
    if not numbers or (len(numbers) < 2 and "sqrt" not in toks and "^" not in toks):
        return None
    p = _Parser(toks)
    try:
        v = p.expr()
    except ZeroDivisionError:
        return {"expr": _spoken(toks), "error": "division by zero",
                "said": "That divides by zero, so there's no answer."}
    except (ValueError, OverflowError, IndexError, TypeError):
        return None
    if p.i != len(toks) or not math.isfinite(v):
        return None
    if abs(v) > 1e18:
        return None
    expr = _spoken(toks)
    if pct_result:
        return {"expr": expr + " percent", "value": v,
                "said": f"{expr} percent is {fmt(v)} percent."}
    return {"expr": expr, "value": v, "said": f"{expr} is {fmt(v)}."}


def _spoken(toks: list[str]) -> str:
    out: list[str] = []
    for x in toks:
        if re.fullmatch(r"\d+(?:\.\d+)?", x):
            out.append(fmt(float(x)))
        else:
            w = _SAY.get(x, x)
            if w:
                out.append(w)
    return " ".join(out)


def slots(text: str) -> dict | None:
    r = parse(text)
    if r and r.get("said"):
        r["said"] = r["said"][:1].upper() + r["said"][1:]     # "The square root..."
    return r


def say(slots_: dict, _res: dict) -> str:
    return str(slots_.get("said") or "I couldn't work that out.")


# ── Tips ───────────────────────────────────────────────────────────────────────
# "My dinner last night was $95.82 before tax. How much should I tip?" went to
# the 3B. A tip is a percentage of a number he said: code, not a guess. With no
# rate named, the usual range is given (the RATE is a custom, the arithmetic
# is not); with one, just that.
_TIP_ASK = re.compile(
    r"\b(?:how\s+much\s+(?:should|do|would)\s+i\s+tip|what(?:'s|\s+is|\s+would\s+be)\s+"
    r"(?:a\s+|the\s+)?(?:good\s+|\d+\s*(?:%|percent)\s+)?tip|calculate\s+(?:a\s+|the\s+)?tip|"
    r"tip\s+(?:on|for)\b|\d+\s*(?:%|percent)\s+tip\b)", re.I)
_MONEY = re.compile(r"\$\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)|(\d+(?:,\d{3})*(?:\.\d{1,2})?)\s*(?:dollars|bucks)\b", re.I)
# The RATE is the percentage attached to the tip ("an 18 percent tip", "tip
# 20 percent"), never just the first one in the sentence: "tax was 8 percent,
# how much should I tip" was answered as an 8 percent tip (audit).
_RATE_AT_TIP = re.compile(
    r"\b(\d{1,3}(?:\.\d+)?)\s*(?:%|percent)\s+tip\b|"
    r"\btip(?:ping)?\s+(?:of\s+|at\s+|about\s+|around\s+)?(\d{1,3}(?:\.\d+)?)\s*(?:%|percent)", re.I)
_ANY_RATE = re.compile(r"\b(\d{1,3}(?:\.\d+)?)\s*(?:%|percent)", re.I)
_TAX = re.compile(r"\b(?:tax|vat|gst|service\s+charge|discount|off)\b", re.I)


def _money(v: float) -> str:
    return f"${v:,.0f}" if abs(v - round(v)) < 0.005 else f"${v:,.2f}"


def _a(n: float) -> str:
    """'an 18 percent tip', 'a 20 percent tip' — by how the number is SAID."""
    i = int(n)
    return "An" if i in (8, 11, 18) or 80 <= i < 90 or 800 <= i < 900 else "A"


def tip(text: str) -> dict | None:
    t = text or ""
    if not _TIP_ASK.search(t):
        return None
    m = _MONEY.search(t)
    if not m:
        return None
    bill = float((m.group(1) or m.group(2)).replace(",", ""))
    if bill <= 0:
        return None
    r = _RATE_AT_TIP.search(t)
    rate_s = (r.group(1) or r.group(2)) if r else None
    if rate_s is None:
        rates = _ANY_RATE.findall(t)
        if len(rates) == 1 and not _TAX.search(t):
            rate_s = rates[0]
    if rate_s is not None:
        rate = float(rate_s)
        amt = bill * rate / 100
        return {"expr": f"{fmt(rate)} percent of {_money(bill)}", "value": amt,
                "said": f"{_a(rate)} {fmt(rate)} percent tip on {_money(bill)} is {_money(amt)}, "
                        f"so {_money(bill + amt)} in all."}
    lo, mid, hi = (bill * x / 100 for x in (15, 18, 20))
    return {"expr": f"a tip on {_money(bill)}", "value": mid,
            "said": f"On {_money(bill)}, 15 percent is {_money(lo)}, 18 percent is "
                    f"{_money(mid)}, and 20 percent is {_money(hi)}."}
