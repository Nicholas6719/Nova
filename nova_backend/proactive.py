"""
Nova speaking first — and, mostly, deciding not to.

Every engine needed for this already existed. What did not exist was the
judgement about WHEN, and that is the entire problem: an assistant that
volunteers things is delightful about twice a day and unbearable at five times
an hour. So the interesting code here is not what gets announced, it is the
pile of reasons not to.

THE RULES, and each one is a way she could become annoying:

  * Never interrupt. She waits for the gap — never while she is speaking,
    never mid-conversation, never while he is being listened to. `_announce`
    already knows how to find a safe moment; this only decides there is
    something worth saying.
  * Never twice. A meeting is announced once, ever, keyed on its identity, not
    on its text. Restarting Nova does not re-announce the morning.
  * Never trivia. Only things that are TIME-BOUND and would be worse to learn
    late: an event about to start, a reminder already overdue. His calendar
    for next week is a thing he asks about, not a thing she volunteers.
  * Never a flood. One announcement at a time, and a floor between them, so a
    packed morning does not become a monologue.
  * Never when he cannot act. Muted, asleep, or the app in the puck while he
    is working — all silence.

Nothing here raises: a failed calendar read costs an announcement, never the
voice loop.
"""

from __future__ import annotations

import datetime
import logging
import threading
import time
from typing import Callable, Optional

log = logging.getLogger("nova.proactive")


class ProactiveMonitor:
    """Watches the day and occasionally says something about it."""

    def __init__(self, config: dict, assistant, announce: Callable[[str], None]) -> None:
        cfg = (config or {}).get("proactive", {})
        self.enabled = bool(cfg.get("enabled", True))
        self.lead_minutes = int(cfg.get("lead_minutes", 10))
        self.min_gap_s = float(cfg.get("min_gap_seconds", 600))
        self.poll_s = float(cfg.get("poll_seconds", 60))
        self._assistant = assistant
        self._announce = announce
        self._said: set[str] = set()
        self._last_spoke = 0.0
        self._thread: Optional[threading.Thread] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="nova-proactive",
                                        daemon=True)
        self._thread.start()
        log.info(f"proactive monitor running (lead {self.lead_minutes}m)")

    def _loop(self) -> None:
        while True:
            time.sleep(self.poll_s)
            try:
                line = self.next_announcement()
                if line:
                    self._last_spoke = time.time()
                    self._announce(line)
            except Exception as exc:
                log.warning(f"proactive check failed: {exc}")

    # ── The decision ──────────────────────────────────────────────────────────
    def quiet_reason(self, now: Optional[float] = None) -> Optional[str]:
        """Why Nova should say nothing right now, or None if she may speak.

        Returned as a REASON rather than a bool so the log says which rule held
        her back — with five overlapping rules, "it stayed quiet" is not a
        debuggable observation.
        """
        now = now if now is not None else time.time()
        a = self._assistant
        if getattr(a, "is_muted", False):
            return "muted"
        if not getattr(a, "is_awake", True):
            return "asleep"
        if getattr(a, "work_mode", False):
            return "working alongside him"
        # Mid-turn: she is listening, thinking or talking. The gap is what she
        # is waiting for, and `_announce` would block anyway — better to decide
        # here than to queue something that lands two minutes stale.
        tts = getattr(a, "tts", None)
        if tts is not None and getattr(tts, "is_speaking", lambda: False)():
            return "she is speaking"
        if now - self._last_spoke < self.min_gap_s:
            return "too soon after the last one"
        return None

    def next_announcement(self, now: Optional[datetime.datetime] = None) -> Optional[str]:
        """The one thing worth saying, or None. Marks it said."""
        reason = self.quiet_reason()
        if reason:
            log.debug(f"proactive: staying quiet ({reason})")
            return None
        now = now or datetime.datetime.now()
        for key, line in self._candidates(now):
            if key in self._said:
                continue
            self._said.add(key)
            return line
        return None

    def _candidates(self, now: datetime.datetime) -> list[tuple[str, str]]:
        """(identity, spoken line) for everything currently worth saying.

        Identity is the EVENT, not the sentence: the same meeting must not be
        announced again because its title was edited, and the wording is free
        to change without re-announcing the day.
        """
        out: list[tuple[str, str]] = []
        soon = now + datetime.timedelta(minutes=self.lead_minutes)
        try:
            import calendar_reminders as cal
        except Exception:
            return out

        # Events about to start.
        try:
            for e in cal.get_today_events():
                start = self._parse(e.get("start"))
                if start is None or not (now <= start <= soon):
                    continue
                mins = max(1, int((start - now).total_seconds() // 60))
                title = (e.get("title") or "an event").strip()
                out.append((f"event:{title}:{start.isoformat(timespec='minutes')}",
                            f"{title} starts in {mins} minute"
                            f"{'s' if mins != 1 else ''}."))
        except Exception as exc:
            log.debug(f"proactive: calendar unavailable ({exc})")

        # Reminders that have gone past due. Not ones merely due later today —
        # that is a thing he asks about, not a thing she interrupts for.
        try:
            for r in cal.get_all_reminders():
                iso = r.get("due_iso") or ""
                if not iso:
                    continue
                due = self._parse(iso)
                if due is None or due > now:
                    continue
                if (now - due).total_seconds() > 3600:
                    continue          # hours late is history, not news
                title = (r.get("title") or "a reminder").strip()
                out.append((f"reminder:{title}:{iso}",
                            f"Your reminder to {title.lower()} was due."))
        except Exception as exc:
            log.debug(f"proactive: reminders unavailable ({exc})")
        return out

    @staticmethod
    def _parse(value) -> Optional[datetime.datetime]:
        """Engine records carry either an ISO string or a long spoken date."""
        if not value:
            return None
        text = str(value)
        try:
            return datetime.datetime.fromisoformat(text)
        except ValueError:
            pass
        for fmt in ("%A, %B %d, %Y at %I:%M:%S %p", "%A, %B %d, %Y at %I:%M %p"):
            try:
                return datetime.datetime.strptime(text, fmt)
            except ValueError:
                continue
        return None
