"""Unit conversions, answered instantly and exactly.

Ported from JARVIS (MiniPCJarvis sidecar/brain/units.py), where "how many
milliliters in a US cup" once reached a near-miss path and was asked about as
a hologram. A conversion is arithmetic with a table: instant, deterministic,
and never wrong by a factor of ten the way a language model can be.

Shapes understood ("N" optional, defaults to one):
    how many milliliters (are) in a cup / in N cups
    convert N miles to kilometers
    N miles in kilometers / N miles to km
    what's N celsius in fahrenheit / N degrees f in c
"""
from __future__ import annotations

import re

# canonical name -> (factor to the base unit of its kind, kind)
_UNITS: dict[str, tuple[float, str]] = {
    # length, base metre
    "millimeter": (0.001, "length"), "centimeter": (0.01, "length"), "meter": (1.0, "length"),
    "kilometer": (1000.0, "length"), "inch": (0.0254, "length"), "foot": (0.3048, "length"),
    "yard": (0.9144, "length"), "mile": (1609.344, "length"), "nautical mile": (1852.0, "length"),
    # mass, base gram
    "milligram": (0.001, "mass"), "gram": (1.0, "mass"), "kilogram": (1000.0, "mass"),
    "ounce": (28.349523125, "mass"), "pound": (453.59237, "mass"), "stone": (6350.29318, "mass"),
    "ton": (907184.74, "mass"), "tonne": (1_000_000.0, "mass"),
    # volume, base millilitre (US customary, which is what a kitchen here uses)
    "milliliter": (1.0, "volume"), "liter": (1000.0, "volume"), "teaspoon": (4.92892159375, "volume"),
    "tablespoon": (14.78676478125, "volume"), "fluid ounce": (29.5735295625, "volume"),
    "cup": (236.5882365, "volume"), "pint": (473.176473, "volume"), "quart": (946.352946, "volume"),
    "gallon": (3785.411784, "volume"),
    # time, base second
    "second": (1.0, "time"), "minute": (60.0, "time"), "hour": (3600.0, "time"),
    "day": (86400.0, "time"), "week": (604800.0, "time"),
    # data, base byte
    "byte": (1.0, "data"), "kilobyte": (1000.0, "data"), "megabyte": (1e6, "data"),
    "gigabyte": (1e9, "data"), "terabyte": (1e12, "data"),
    # temperature is affine; handled apart
    "celsius": (1.0, "temp"), "fahrenheit": (1.0, "temp"), "kelvin": (1.0, "temp"),
}

# every way he might say a unit -> canonical
_ALIASES: dict[str, str] = {}
for _name in _UNITS:
    _ALIASES[_name] = _name
    _ALIASES[_name + "s"] = _name
_ALIASES.update({
    "millimetre": "millimeter", "millimetres": "millimeter", "mm": "millimeter",
    "centimetre": "centimeter", "centimetres": "centimeter", "cm": "centimeter",
    "metre": "meter", "metres": "meter", "m": "meter",
    "kilometre": "kilometer", "kilometres": "kilometer", "km": "kilometer", "k": "kilometer",
    "inches": "inch", "in": "inch", "feet": "foot", "ft": "foot", "yd": "yard", "mi": "mile",
    "nautical miles": "nautical mile",
    "mg": "milligram", "g": "gram", "grams": "gram", "kg": "kilogram", "kilo": "kilogram",
    "kilos": "kilogram", "oz": "ounce", "lb": "pound", "lbs": "pound",
    "tons": "ton", "tonnes": "tonne",
    "millilitre": "milliliter", "millilitres": "milliliter", "ml": "milliliter", "mils": "milliliter",
    "litre": "liter", "litres": "liter", "l": "liter",
    "tsp": "teaspoon", "tbsp": "tablespoon", "fl oz": "fluid ounce", "fluid ounces": "fluid ounce",
    "gal": "gallon",
    "sec": "second", "secs": "second", "min": "minute", "mins": "minute", "hr": "hour", "hrs": "hour",
    "kb": "kilobyte", "mb": "megabyte", "gb": "gigabyte", "tb": "terabyte", "gigs": "gigabyte",
    "c": "celsius", "centigrade": "celsius", "degrees celsius": "celsius", "degrees c": "celsius",
    "f": "fahrenheit", "degrees fahrenheit": "fahrenheit", "degrees f": "fahrenheit",
    "k": "kilometer",
})
_ALIASES["kelvin"] = "kelvin"

_PLURAL = {"foot": "feet", "inch": "inches", "celsius": "celsius", "fahrenheit": "fahrenheit",
           "kelvin": "kelvin"}


def _say_unit(name: str, n: float) -> str:
    if abs(n - 1.0) < 1e-9:
        return name
    return _PLURAL.get(name, name + "s")


_UNIT_RE = "|".join(sorted((re.escape(a) for a in _ALIASES), key=len, reverse=True))
_NUM = r"(?P<n>\d+(?:\.\d+)?|a|an|one)?"
_PATTERNS = [
    # how many X (are there) in N Y
    re.compile(r"^how many\s+(?P<to>" + _UNIT_RE + r")\s+(?:are\s+(?:there\s+)?)?in\s+"
               + _NUM + r"\s*(?:us\s+|u\.s\.\s+)?(?P<frm>" + _UNIT_RE + r")$", re.I),
    # convert N X to Y  /  N X to Y  /  N X in Y
    re.compile(r"^(?:convert\s+)?" + _NUM + r"\s*(?:degrees?\s+)?(?P<frm>" + _UNIT_RE
               + r")\s+(?:to|in|into|as)\s+(?:degrees?\s+)?(?P<to>" + _UNIT_RE + r")$", re.I),
]
_LEAD = re.compile(
    r"^(?:hey |hi |ok |okay )?(?:nova[,!.]?\s*)?(?:(?:can|could|would) you\s+)?(?:please\s+)?"
    r"(?:what(?:'s| is| are)\s+|tell me\s+|give me\s+)?", re.I)
_TRAIL = re.compile(r"(?:\s*(?:,|please|nova|\?|\.|!))+$", re.I)


def _temp(v: float, frm: str, to: str) -> float:
    c = v if frm == "celsius" else (v - 32) * 5 / 9 if frm == "fahrenheit" else v - 273.15
    return c if to == "celsius" else c * 9 / 5 + 32 if to == "fahrenheit" else c + 273.15


def fmt(v: float) -> str:
    if abs(v) >= 1e6:
        return f"{v:,.0f}"
    if abs(v - round(v)) < 1e-6:
        return f"{int(round(v)):,}"
    if abs(v) >= 100:
        return f"{v:,.1f}".rstrip("0").rstrip(".")
    if abs(v) >= 1:
        return f"{v:,.2f}".rstrip("0").rstrip(".")
    return f"{v:.3g}"


def convert(text: str) -> dict | None:
    """{'said': '1 cup is about 236.6 milliliters.', 'value': 236.59, ...} or None."""
    t = re.sub(r"\s+", " ", (text or "").strip().lower())
    t = _LEAD.sub("", t, count=1)
    t = _TRAIL.sub("", t)
    for pat in _PATTERNS:
        m = pat.match(t)
        if not m:
            continue
        raw_n = (m.group("n") or "1").strip()
        n = 1.0 if raw_n in ("a", "an", "one") else float(raw_n)
        frm = _ALIASES.get(m.group("frm").strip())
        to = _ALIASES.get(m.group("to").strip())
        if not frm or not to or frm == to:
            return None
        f_fac, f_kind = _UNITS[frm]
        t_fac, t_kind = _UNITS[to]
        if f_kind != t_kind:
            return {"said": f"{_say_unit(frm, 2).capitalize()} and {_say_unit(to, 2)} "
                            f"measure different things.", "error": "different kinds"}
        if f_kind == "temp":
            v = round(_temp(n, frm, to), 1)     # nobody wants 21.11 degrees
            said = f"{fmt(n)} degrees {frm} is {fmt(v)} degrees {to}."
        else:
            v = n * f_fac / t_fac
            exact = abs(v - round(v)) < 1e-9
            said = (f"{fmt(n)} {_say_unit(frm, n)} is {'' if exact else 'about '}"
                    f"{fmt(v)} {_say_unit(to, v)}.")
        return {"expr": f"{fmt(n)} {_say_unit(frm, n)} in {_say_unit(to, 2)}",
                "value": v, "said": said}
    return None
