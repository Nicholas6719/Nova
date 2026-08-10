"""
Maps engine — location, nearby search, and travel time via Apple MapKit.

WHY A SUBPROCESS
────────────────────────────────────────────────────────────────────────────
MapKit and CoreLocation deliver their results to completion handlers dispatched
on the MAIN queue. Nova's tools run on the ``nova-llm`` worker thread, whose run
loop never services that queue — verified experimentally: an MKLocalSearch
started from a worker thread simply never calls back and hangs until timeout.

So every MapKit call runs in a short-lived child process (this file, executed as
__main__) that owns its own main thread, prints one JSON line, and exits. That
mirrors how the rest of Nova shells out to `osascript`, gives us a hard timeout
for free, and keeps the voice loop from ever blocking on Apple's servers.

PRIVACY NOTE (see CLAUDE.md invariant 3)
────────────────────────────────────────────────────────────────────────────
This is the ONE part of Nova that is not purely on-device. Finding "the nearest
CVS" and computing a drive time are Apple network services, and doing so sends
an approximate location to Apple. Nothing is sent to any third party, no API key
exists, and nothing here touches the LLM or Nova's memory. Location is requested
only when the user asks a location question, and cached briefly so a follow-up
doesn't re-query.

PERMISSION
────────────────────────────────────────────────────────────────────────────
CoreLocation needs a real app identity: run headless, authorization stays
`notDetermined` forever and no prompt appears. Location therefore only works
when the backend runs under Nova.app, which must declare
NSLocationWhenInUseUsageDescription and the location entitlement. Every function
degrades gracefully when location is unavailable — navigation still works,
because opening Maps with a destination needs no location at all.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
import urllib.parse
from typing import Optional

log = logging.getLogger("nova.maps")

_HELPER_TIMEOUT = 25.0          # hard cap; Apple lookups are normally 1-3s
_LOCATION_TTL   = 300.0         # re-use a fix for 5 minutes
_location_cache: dict = {"at": 0.0, "coord": None, "denied": False}

TRANSPORT = {"driving": "automobile", "walking": "walking", "transit": "transit"}


# ══════════════════════════════════════════════════════════════════════════
# Parent-side API
# ══════════════════════════════════════════════════════════════════════════
def _call(payload: dict, timeout: float = _HELPER_TIMEOUT) -> dict:
    """Run one MapKit operation in the helper subprocess. Never raises."""
    try:
        r = subprocess.run(
            [sys.executable, __file__, json.dumps(payload)],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout"}
    except Exception as e:
        return {"ok": False, "error": str(e)}
    out = (r.stdout or "").strip().splitlines()
    for line in reversed(out):                    # last line is our JSON
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except Exception:
                continue
    return {"ok": False, "error": (r.stderr or "no output").strip()[:200]}


def set_location_from_app(payload: dict) -> None:
    """Accept a fix pushed by the Swift app (POST /api/location).

    This is the ONLY way Nova can obtain a location. Asking CoreLocation from
    here is futile: the helper runs as a bare python binary with no Info.plist,
    so `requestWhenInUseAuthorization()` is silently ignored, no prompt is ever
    shown, NovaOS never appears under Location Services, and the status stays
    notDetermined forever. Only the signed app bundle carries
    NSLocationWhenInUseUsageDescription — see LocationProvider.swift.
    """
    if not isinstance(payload, dict):
        raise ValueError("location payload must be an object")
    if not payload.get("available"):
        _location_cache.update(at=0.0, coord=None,
                               denied=(payload.get("reason") == "denied"))
        log.info(f"location unavailable from app: {payload.get('reason')}")
        return
    lat, lon = float(payload["lat"]), float(payload["lon"])
    _location_cache.update(at=time.time(), coord=(lat, lon), denied=False)
    # Never log the coordinate itself.
    log.info("location updated from app")


def location_was_denied() -> bool:
    """True when the app told us the user declined, so we can say so exactly."""
    return bool(_location_cache.get("denied"))


def current_location(force: bool = False) -> Optional[tuple]:
    """(lat, lon) or None. Supplied by the Swift app; cached for _LOCATION_TTL."""
    now = time.time()
    coord = _location_cache.get("coord")
    if coord and now - _location_cache["at"] < _LOCATION_TTL:
        return coord
    if coord:
        log.info("location fix is stale; waiting for a fresh one from the app")
    else:
        log.info("no location available (the app has not supplied a fix)")
    return None


def nearest(query: str, mode: str = "driving") -> dict:
    """Nearest match for `query` plus its travel time from where we are.

    Returns {ok, name, address, minutes, miles} or {ok: False, error: ...}
    where error is 'no_location' when CoreLocation isn't authorised.
    """
    coord = current_location()
    if coord is None:
        return {"ok": False, "error": "no_location"}
    return _call({"op": "nearest", "query": query,
                  "lat": coord[0], "lon": coord[1],
                  "mode": TRANSPORT.get(mode, "automobile")})


def eta_to(destination: str, mode: str = "driving") -> dict:
    """Travel time to a named place/address from the current location."""
    return nearest(destination, mode=mode)


def directions_url(destination: str, mode: str = "driving") -> str:
    flag = {"driving": "d", "walking": "w", "transit": "r"}.get(mode, "d")
    return (f"maps://?daddr={urllib.parse.quote(destination)}&dirflg={flag}")


def open_directions(destination: str, mode: str = "driving") -> bool:
    """Open Apple Maps with directions. Needs no location permission."""
    try:
        subprocess.run(["open", directions_url(destination, mode)],
                       capture_output=True, timeout=10)
        return True
    except Exception as e:
        log.warning(f"open directions failed: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════
# Child process — runs on its OWN main thread, prints one JSON line
# ══════════════════════════════════════════════════════════════════════════
def _child(payload: dict) -> dict:
    import CoreLocation
    import MapKit
    from Foundation import NSRunLoop, NSDate

    def pump(done, seconds=15.0, step=0.25):
        end = time.time() + seconds
        while time.time() < end:
            NSRunLoop.currentRunLoop().runUntilDate_(
                NSDate.dateWithTimeIntervalSinceNow_(step))
            if done():
                return True
        return False

    op = payload.get("op")

    if op == "location":
        if not CoreLocation.CLLocationManager.locationServicesEnabled():
            return {"ok": False, "error": "location services disabled"}
        mgr = CoreLocation.CLLocationManager.alloc().init()
        try:
            mgr.requestWhenInUseAuthorization()
        except Exception:
            pass
        mgr.startUpdatingLocation()
        if not pump(lambda: mgr.location() is not None, seconds=12.0):
            return {"ok": False,
                    "error": f"no fix (auth={mgr.authorizationStatus()})"}
        c = mgr.location().coordinate()
        return {"ok": True, "lat": c.latitude, "lon": c.longitude}

    if op == "nearest":
        lat, lon, q = payload["lat"], payload["lon"], payload["query"]
        here = CoreLocation.CLLocationCoordinate2DMake(lat, lon)
        req = MapKit.MKLocalSearchRequest.alloc().init()
        req.setNaturalLanguageQuery_(q)
        req.setRegion_(MapKit.MKCoordinateRegionMakeWithDistance(here, 16000, 16000))
        search = MapKit.MKLocalSearch.alloc().initWithRequest_(req)
        box = {}
        search.startWithCompletionHandler_(lambda r, e: box.update(r=r, e=e))
        if not pump(lambda: bool(box)):
            return {"ok": False, "error": "search timeout"}
        if not box.get("r"):
            return {"ok": False, "error": "no results"}
        items = list(box["r"].mapItems())
        if not items:
            return {"ok": False, "error": "no results"}

        # Rank by straight-line distance, then get a real ETA for the closest.
        origin = CoreLocation.CLLocation.alloc().initWithLatitude_longitude_(lat, lon)
        def straight(it):
            c = it.placemark().coordinate()
            return CoreLocation.CLLocation.alloc().initWithLatitude_longitude_(
                c.latitude, c.longitude).distanceFromLocation_(origin)
        items.sort(key=straight)
        best = items[0]

        mode = {"automobile": MapKit.MKDirectionsTransportTypeAutomobile,
                "walking": MapKit.MKDirectionsTransportTypeWalking,
                "transit": MapKit.MKDirectionsTransportTypeTransit,
                }.get(payload.get("mode", "automobile"),
                      MapKit.MKDirectionsTransportTypeAutomobile)
        src = MapKit.MKMapItem.alloc().initWithPlacemark_(
            MapKit.MKPlacemark.alloc().initWithCoordinate_(here))
        dreq = MapKit.MKDirectionsRequest.alloc().init()
        dreq.setSource_(src)
        dreq.setDestination_(best)
        dreq.setTransportType_(mode)
        box2 = {}
        MapKit.MKDirections.alloc().initWithRequest_(dreq) \
            .calculateETAWithCompletionHandler_(lambda r, e: box2.update(r=r, e=e))

        pm = best.placemark()
        parts = [pm.thoroughfare(), pm.locality()]
        address = ", ".join(p for p in parts if p)
        result = {"ok": True, "name": best.name() or payload["query"],
                  "address": address,
                  "miles": round(straight(best) / 1609.34, 1)}
        if pump(lambda: bool(box2), seconds=12.0) and box2.get("r"):
            eta = box2["r"]
            result["minutes"] = int(round(eta.expectedTravelTime() / 60.0))
            result["miles"] = round(eta.distance() / 1609.34, 1)
        return result

    return {"ok": False, "error": f"unknown op {op!r}"}


if __name__ == "__main__":
    try:
        print(json.dumps(_child(json.loads(sys.argv[1]))))
    except Exception as exc:                       # never let the child crash silently
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
