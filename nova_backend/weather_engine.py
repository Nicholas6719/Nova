"""
Weather — the ONE deliberate widening of Nova's no-cloud rule, after maps.

CLAUDE.md invariant 3 says the LLM, STT, TTS and memory never leave the
machine, and that the MapKit exception must not be widened casually. Nicholas
chose Open-Meteo for this (2026-08-13) after being shown the alternatives:

  - Open-Meteo    no API key, no account, no sign-up; free for non-commercial
                  use under CC BY 4.0            <- chosen
  - weather.gov   US government, no key, but needs a grid-point lookup first
  - WeatherKit    same vendor as the existing MapKit exception, but needs a
                  paid Apple developer account and an entitlement
  - browser scrape  no new service, but brittle, slow, and still leaks location

What actually leaves the machine: an approximate latitude and longitude, or a
place name he said out loud. Nothing else — not the LLM, not memory, not the
transcript. No key, no account, so there is nothing tying the request to him
beyond the IP address any HTTP request carries. It is used only when he asks
about weather, and the answer is not stored.

Everything here is a pure function over HTTP with a hard timeout, and NOTHING
raises: a failed lookup returns an error dict so Nova can say it could not get
the forecast rather than inventing one.
"""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from typing import Optional

log = logging.getLogger("nova.weather")

FORECAST_URL  = "https://api.open-meteo.com/v1/forecast"
GEOCODE_URL   = "https://geocoding-api.open-meteo.com/v1/search"
_TIMEOUT      = 8.0

# WMO weather interpretation codes, in words Nova can say aloud.
_WMO = {
    0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "foggy",
    51: "drizzling", 53: "drizzling", 55: "drizzling",
    56: "freezing drizzle", 57: "freezing drizzle",
    61: "raining lightly", 63: "raining", 65: "raining heavily",
    66: "freezing rain", 67: "freezing rain",
    71: "snowing lightly", 73: "snowing", 75: "snowing heavily",
    77: "snow grains",
    80: "rain showers", 81: "rain showers", 82: "heavy rain showers",
    85: "snow showers", 86: "snow showers",
    95: "thunderstorms", 96: "thunderstorms with hail",
    99: "thunderstorms with hail",
}


def describe_code(code) -> str:
    try:
        return _WMO.get(int(code), "unsettled")
    except (TypeError, ValueError):
        return "unsettled"


def _get(url: str, params: dict) -> dict:
    """One JSON GET. Never raises — returns {"ok": False, "error": ...}."""
    try:
        full = f"{url}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(full, headers={"User-Agent": "Nova/1.0"})
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
            return {"ok": True, "data": json.loads(r.read().decode())}
    except Exception as exc:
        # Do not log the coordinates.
        log.warning(f"weather lookup failed: {type(exc).__name__}")
        return {"ok": False, "error": str(exc)[:120]}


def geocode(place: str) -> dict:
    """Resolve a spoken place name to a coordinate. {"ok", "lat", "lon", "name"}."""
    if not place or not place.strip():
        return {"ok": False, "error": "no place given"}
    res = _get(GEOCODE_URL, {"name": place.strip(), "count": 1,
                             "language": "en", "format": "json"})
    if not res["ok"]:
        return res
    results = (res["data"] or {}).get("results") or []
    if not results:
        return {"ok": False, "error": "not found"}
    top = results[0]
    label = top.get("name", place)
    admin = top.get("admin1")
    return {"ok": True, "lat": top["latitude"], "lon": top["longitude"],
            "name": f"{label}, {admin}" if admin else label}


def fetch(lat: float, lon: float, days: int = 3) -> dict:
    """Current conditions plus a daily forecast. Never raises.

    Fahrenheit and mph because Nova is speaking to someone in Massachusetts;
    timezone=auto so "today" means his today, not UTC's.
    """
    res = _get(FORECAST_URL, {
        "latitude": round(float(lat), 3),      # ~100m; no need to send more
        "longitude": round(float(lon), 3),
        "current": ("temperature_2m,apparent_temperature,relative_humidity_2m,"
                    "precipitation,weather_code,wind_speed_10m"),
        "daily": ("weather_code,temperature_2m_max,temperature_2m_min,"
                  "precipitation_probability_max,sunrise,sunset"),
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
        "timezone": "auto",
        "forecast_days": max(1, min(int(days), 7)),
    })
    if not res["ok"]:
        return res

    d = res["data"] or {}
    cur = d.get("current") or {}
    daily = d.get("daily") or {}

    def day(i: int) -> Optional[dict]:
        try:
            return {
                "code": daily["weather_code"][i],
                "high": daily["temperature_2m_max"][i],
                "low": daily["temperature_2m_min"][i],
                "rain_chance": (daily.get("precipitation_probability_max") or [None])[i],
            }
        except (KeyError, IndexError, TypeError):
            return None

    return {
        "ok": True,
        "current": {
            "temp": cur.get("temperature_2m"),
            "feels_like": cur.get("apparent_temperature"),
            "humidity": cur.get("relative_humidity_2m"),
            "wind": cur.get("wind_speed_10m"),
            "code": cur.get("weather_code"),
        },
        "days": [d for d in (day(i) for i in range(7)) if d],
    }


# ── Spoken formatting (deterministic on purpose) ─────────────────────────────
# The 3B is not asked to phrase these. It has invented figures before — a
# leftover-budget number in file summaries — and a wrong temperature is
# indistinguishable from a right one to someone listening.

def _round(v) -> Optional[int]:
    try:
        return int(round(float(v)))
    except (TypeError, ValueError):
        return None


def say_current(data: dict, place: Optional[str] = None) -> str:
    cur = data.get("current") or {}
    temp = _round(cur.get("temp"))
    feels = _round(cur.get("feels_like"))
    cond = describe_code(cur.get("code"))
    where = f" in {place}" if place else ""

    if temp is None:
        return f"I got the forecast{where} but no current temperature."

    out = f"It's {temp} degrees and {cond}{where}."
    if feels is not None and abs(feels - temp) >= 4:
        out += f" Feels like {feels}."
    today = (data.get("days") or [None])[0]
    if today:
        hi, lo = _round(today.get("high")), _round(today.get("low"))
        if hi is not None and lo is not None:
            out += f" Today's high is {hi}, low {lo}."
        chance = _round(today.get("rain_chance"))
        if chance is not None and chance >= 30:
            out += f" There's a {chance} percent chance of rain."
    return out


def say_day(data: dict, index: int, label: str,
            place: Optional[str] = None) -> str:
    days = data.get("days") or []
    if index >= len(days):
        return f"I don't have the forecast for {label} yet."
    d = days[index]
    hi, lo = _round(d.get("high")), _round(d.get("low"))
    cond = describe_code(d.get("code"))
    where = f" in {place}" if place else ""
    if hi is None or lo is None:
        return f"{label.capitalize()}{where} looks {cond}."
    out = f"{label.capitalize()}{where} looks {cond}, with a high of {hi} and a low of {lo}."
    chance = _round(d.get("rain_chance"))
    if chance is not None and chance >= 30:
        out += f" There's a {chance} percent chance of rain."
    return out
