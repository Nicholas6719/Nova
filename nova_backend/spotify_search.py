"""
Name -> Spotify URI. The ONLY reason this exists.

Spotify's desktop app exposes exactly six AppleScript commands:

    next track · previous track · playpause · pause · play · play track

`play track` needs a URI ("spotify:track:4uLU..."), and there is no `search`
anywhere in its scripting dictionary. So the desktop app can play anything —
it just cannot tell anyone what a NAME refers to. This module does that one
lookup and hands the URI back; playback stays entirely local, through the app
Nicholas already has open.

WHAT THIS COSTS, PLAINLY (invariant 3, his call 2026-08-13):
  - one HTTPS request to Spotify carrying the words he asked for
  - a client ID and secret, which live in NOVA_DATA_DIR and NOT in the repo
  - no user login, no OAuth redirect, no Premium: this is client-credentials,
    which reads the PUBLIC catalog only

WHAT IT THEREFORE CANNOT DO: his own library, his saved songs, and personal or
algorithmic playlists (Discover Weekly, Release Radar) are invisible to
client-credentials search. Those need a user token. Nova says so rather than
pretending the name was not found.

Nothing here raises. Every failure returns a dict with `ok: False` and a reason
Nova can say out loud.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

log = logging.getLogger("nova.spotify")

TOKEN_URL  = "https://accounts.spotify.com/api/token"
SEARCH_URL = "https://api.spotify.com/v1/search"
_TIMEOUT   = 8.0

# The secret does NOT belong in config.json — that file is committed. This
# path is the same place the memory database lives: local, per-user, untracked.
CREDENTIALS_FILENAME = "spotify_credentials.json"

_token: dict = {"value": None, "expires_at": 0.0}


def _data_dir() -> Path:
    return Path(os.environ.get(
        "NOVA_DATA_DIR", Path.home() / "Library" / "Application Support" / "Nova"))


def credentials_path() -> Path:
    return _data_dir() / CREDENTIALS_FILENAME


def load_credentials() -> Optional[tuple]:
    """(client_id, client_secret) or None. Environment wins, for testing."""
    env_id = os.environ.get("SPOTIFY_CLIENT_ID", "").strip()
    env_secret = os.environ.get("SPOTIFY_CLIENT_SECRET", "").strip()
    if env_id and env_secret:
        return env_id, env_secret
    try:
        data = json.loads(credentials_path().read_text())
        cid = str(data.get("client_id", "")).strip()
        sec = str(data.get("client_secret", "")).strip()
        if cid and sec:
            return cid, sec
    except FileNotFoundError:
        return None
    except Exception as exc:
        log.warning(f"could not read Spotify credentials: {type(exc).__name__}")
    return None


def is_configured() -> bool:
    return load_credentials() is not None


def _get_token() -> Optional[str]:
    """Client-credentials token, cached until shortly before it expires."""
    if _token["value"] and time.time() < _token["expires_at"]:
        return _token["value"]

    creds = load_credentials()
    if not creds:
        return None
    cid, secret = creds
    auth = base64.b64encode(f"{cid}:{secret}".encode()).decode()
    try:
        req = urllib.request.Request(
            TOKEN_URL,
            data=urllib.parse.urlencode({"grant_type": "client_credentials"}).encode(),
            headers={"Authorization": f"Basic {auth}",
                     "Content-Type": "application/x-www-form-urlencoded"},
        )
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
            data = json.loads(r.read().decode())
    except Exception as exc:
        # Never log the secret or the response body.
        log.warning(f"Spotify token request failed: {type(exc).__name__}")
        return None

    tok = data.get("access_token")
    if not tok:
        return None
    _token["value"] = tok
    # Refresh a minute early rather than racing the expiry.
    _token["expires_at"] = time.time() + max(60, int(data.get("expires_in", 3600))) - 60
    return tok


# Order matters: a bare name is far more likely to be a song than a playlist,
# but an explicit "playlist" in the request flips that. See search().
_DEFAULT_TYPES = ("track", "album", "artist", "playlist")


def search(query: str, prefer: Optional[str] = None, market: str = "US") -> dict:
    """Best match for `query`. Returns {"ok", "uri", "label", "kind"}.

    `prefer` biases the result when he said what he wanted ("the album X",
    "the artist X", "a playlist of X").
    """
    if not query or not query.strip():
        return {"ok": False, "reason": "empty"}
    if not is_configured():
        return {"ok": False, "reason": "not_configured"}

    token = _get_token()
    if not token:
        return {"ok": False, "reason": "auth_failed"}

    types = list(_DEFAULT_TYPES)
    if prefer and prefer in types:
        types.remove(prefer)
        types.insert(0, prefer)

    params = urllib.parse.urlencode({
        "q": query.strip(),
        "type": ",".join(types),
        "limit": 5,
        "market": market,
    })
    try:
        req = urllib.request.Request(
            f"{SEARCH_URL}?{params}",
            headers={"Authorization": f"Bearer {token}"},
        )
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
            data = json.loads(r.read().decode())
    except Exception as exc:
        log.warning(f"Spotify search failed: {type(exc).__name__}")
        return {"ok": False, "reason": "unreachable"}

    for kind in types:
        items = ((data.get(f"{kind}s") or {}).get("items") or [])
        # Spotify returns null entries in playlist results often enough that
        # skipping them matters — a None here would crash the whole turn.
        items = [i for i in items if i]
        if not items:
            continue
        top = items[0]
        uri = top.get("uri")
        if not uri:
            continue
        return {"ok": True, "uri": uri, "kind": kind,
                "label": _label(kind, top)}

    return {"ok": False, "reason": "not_found"}


def _label(kind: str, item: dict) -> str:
    """How Nova will say what it found. Read back from the RESULT, never from
    what he asked for — otherwise Nova echoes the request and sounds like it
    succeeded when it actually matched something else."""
    name = item.get("name") or "that"
    if kind in ("track", "album"):
        artists = ", ".join(a.get("name", "") for a in (item.get("artists") or [])
                            if a.get("name"))
        return f"{name} by {artists}" if artists else name
    if kind == "playlist":
        owner = (item.get("owner") or {}).get("display_name")
        return f"the {name} playlist" + (f" by {owner}" if owner else "")
    if kind == "artist":
        return name
    return name
