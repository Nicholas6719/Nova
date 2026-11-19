"""
Night school — what Nova does while he sleeps (Stage 3 of the brain port).

Ported from JARVIS (MiniPCJarvis sidecar/brain/night_school.py). Three jobs,
in one quiet window, all SILENT: nothing is ever said about them unless he
asks ("what did you learn last night?") — JARVIS's resolved rule, and Nova's
own: nothing changes what she does unprompted out loud.

1. THE FACT AUDIT. Each stored world fact is re-checked against the Wikipedia
   page it came from. Better than JARVIS's: Wikipedia gives every page a
   revision id, so ONE request covers up to fifty facts, and a page that has
   not been edited since is confirmed without reading it and without the
   model. An edited page is re-read: if the stored answer is still grounded in
   the new text it is confirmed; otherwise the local model is asked, and may
   only make things worse — CHANGED demotes now, anything else strikes (two
   strikes demote). It is never allowed to CONFIRM: measured, the 3B said
   SAME for a fact against an unrelated sentence. A page that has gone is a
   strike. Distrust is the default.
2. OVERNIGHT CURIOSITY (his option C). General-knowledge questions he asked
   in the last week that the language model answered from memory are looked
   up properly; the normal intake (world_facts.consider) decides whether a
   verified, timeless fact is born. Anchored to what HE asked — never his
   browser, never open crawling — and each question is sent at most once,
   ever. Changeable questions are skipped outright: they could never be
   stored, so sending them would leak a question for nothing.
3. PARAPHRASE DISTILLATION (local, no network). Sentences the brain fell
   through on and a regex stage then handled are paraphrased by the local
   model, and the router is taught the paraphrases it would still miss —
   only through brain.learn's full bar, so a paraphrase its skill cannot
   execute, or that the brain itself would rank elsewhere, is not learned.

WHEN: inside `facts.night_hours` (default 01:00-06:00), after he has been
quiet for `facts.idle_minutes`, at most once every `facts.min_gap_hours`, and
only on mains power (the model runs; a laptop on battery overnight should not
spend it on this). Every model call goes to the nova-llm thread at BACKGROUND
priority as its own small job, and the run checks between items whether he is
back — if he is, it stops, so a turn never waits behind it for more than one
step.
"""
from __future__ import annotations

import datetime
import json
import logging
import re
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Optional

import world_facts as W

log = logging.getLogger("nova.night")

_PARA_SYSTEM = "You write short paraphrases of voice commands. One per line. No numbering, no quotes."
_PARA_PROMPT = 'Give {n} different natural ways a person might say this to a voice assistant: "{u}"'

_REPORT_RE = re.compile(
    r"^\s*(?:nova[,\s]+)?(?:what\s+did\s+you\s+(?:learn|look\s+up|do|study|find\s+out)\s+"
    r"(?:last\s+night|overnight|while\s+i\s+(?:was\s+)?(?:slept|sleeping|asleep))|"
    r"(?:how\s+did\s+)?night\s+school(?:\s+(?:report|go))?|what\s+did\s+you\s+look\s+up)"
    r"\s*(?:nova)?[\s.!?]*$", re.I)


def wants_report(text: str) -> bool:
    return bool(_REPORT_RE.search(text or ""))


class NightSchool:
    def __init__(self, db_path: Path, config: Optional[dict], facts: "W.WorldFacts",
                 generate: Callable[..., str],
                 here: Callable[[], bool],
                 learn: Optional[Callable[[str, str], bool]] = None,
                 acts: Optional[Callable[[str], bool]] = None,
                 skill_for: Optional[Callable[[str], Optional[str]]] = None,
                 on_power: Optional[Callable[[], Optional[bool]]] = None,
                 pages: Optional[Callable] = None,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        cfg = (config or {}).get("facts", {})
        self.enabled = bool(cfg.get("enabled", True)) and bool(cfg.get("night_school", True))
        self.hours = cfg.get("night_hours", [1, 6])
        self.idle_s = float(cfg.get("idle_minutes", 45)) * 60
        self.gap_s = float(cfg.get("min_gap_hours", 20)) * 3600
        self.fact_budget = int(cfg.get("fact_budget", 20))
        self.curiosity_budget = int(cfg.get("curiosity_budget", 5))
        self.distill_budget = int(cfg.get("distill_budget", 8))
        self.spacing_s = float(cfg.get("request_spacing_s", 15))
        self.recheck_s = float(cfg.get("recheck_days", 3)) * 86400
        self.db_path = Path(db_path)
        self.facts = facts
        self._generate = generate
        self._here = here
        self._learn = learn
        self._acts = acts
        self._skill_for = skill_for
        self._on_power = on_power
        self._pages = pages
        self._sleep = sleep
        self._thread: Optional[threading.Thread] = None
        self._running = threading.Lock()
        with self._db() as db:
            db.execute("CREATE TABLE IF NOT EXISTS night_meta (key TEXT PRIMARY KEY, value TEXT)")

    @contextmanager
    def _db(self):
        conn = sqlite3.connect(str(self.db_path), timeout=5.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _meta(self, key: str) -> Optional[str]:
        with self._db() as db:
            r = db.execute("SELECT value FROM night_meta WHERE key=?", (key,)).fetchone()
        return r[0] if r else None

    def _set_meta(self, key: str, value: str) -> None:
        with self._db() as db:
            db.execute("INSERT OR REPLACE INTO night_meta (key, value) VALUES (?,?)", (key, value))

    # ── when ─────────────────────────────────────────────────────────────────
    def in_window(self, now: Optional[float] = None) -> bool:
        h = datetime.datetime.fromtimestamp(now or time.time()).hour
        start, end = int(self.hours[0]), int(self.hours[1])
        return (start <= h or h < end) if start > end else (start <= h < end)

    def why_not(self, now: Optional[float] = None, idle_for: Optional[float] = None) -> Optional[str]:
        """The reason not to run now, or None if it is time."""
        now = now or time.time()
        if not self.enabled:
            return "disabled"
        if not self.in_window(now):
            return "outside the night window"
        if idle_for is not None and idle_for < self.idle_s:
            return "he was here recently"
        last = float(self._meta("last_run") or 0)
        if now - last < self.gap_s:
            return "already ran"
        if self._on_power is not None:
            try:
                if self._on_power() is False:
                    return "on battery"
            except Exception:
                pass
        return None

    def start(self, idle_for: Callable[[], float], poll_s: float = 600) -> None:
        if not self.enabled or self._thread is not None:
            return

        def _loop():
            while True:
                time.sleep(poll_s)
                try:
                    if self.why_not(idle_for=idle_for()) is None:
                        self.run()
                except Exception as exc:
                    log.warning(f"[night] run failed: {exc}")
        self._thread = threading.Thread(target=_loop, name="nova-night", daemon=True)
        self._thread.start()

    # ── the run ──────────────────────────────────────────────────────────────
    def run(self, force: bool = False) -> dict:
        """One full pass. Stops between items the moment he is back (unless
        forced). Returns the report, which is also kept for "what did you
        learn last night?"."""
        if not self._running.acquire(blocking=False):
            return {"skipped": "already running"}
        report = {"started": time.time(), "audited": 0, "confirmed": 0, "changed": 0,
                  "unclear": 0, "looked_up": [], "stored": [], "learned": [],
                  "stopped": None}
        try:
            def back() -> bool:
                return not force and self._here()

            self._audit(report, back)
            if not report["stopped"]:
                self._curiosity(report, back)
            if not report["stopped"]:
                self._distill(report, back)
        except Exception as exc:
            log.warning(f"[night] stopped by an error: {exc}")
            report["stopped"] = f"error: {type(exc).__name__}"
        finally:
            report["finished"] = time.time()
            self._set_meta("last_run", str(report["finished"]))
            self._set_meta("last_report", json.dumps(report))
            self._running.release()
        log.info(f"[night] {json.dumps(report)}")
        return report

    # job 1 ─────────────────────────────────────────────────────────────────
    def _audit(self, report: dict, back: Callable[[], bool]) -> None:
        due = self.facts.due_for_audit(self.fact_budget, older_than_s=self.recheck_s)
        if not due:
            return
        if self._pages is None:
            import wiki
            pages = wiki.pages
        else:
            pages = self._pages
        ids = sorted({f["sources"][0]["pageid"] for f in due if f["sources"]})
        meta = pages(ids, with_extract=False)
        self._log_audit(len(ids))
        if not meta.get("ok"):
            report["stopped"] = f"wikipedia unavailable ({meta.get('error')})"
            return
        now_pages = meta["pages"]
        edited: list[dict] = []
        for f in due:
            if back():
                report["stopped"] = "he came back"
                return
            src = f["sources"][0] if f["sources"] else None
            report["audited"] += 1
            page = now_pages.get(src["pageid"]) if src else None
            if page is None:
                self._unclear(f, report, "page gone")
            elif page["revid"] == src.get("revid"):
                self.facts.mark_verified(f["id"])
                report["confirmed"] += 1
            else:
                edited.append(f)
        if not edited:
            return
        self._sleep(self.spacing_s)
        fresh = pages(sorted({f["sources"][0]["pageid"] for f in edited}), with_extract=True)
        self._log_audit(len(edited))
        if not fresh.get("ok"):
            for f in edited:
                self._unclear(f, report, "could not re-read")
            return
        for f in edited:
            if back():
                report["stopped"] = "he came back"
                return
            src = f["sources"][0]
            page = fresh["pages"].get(src["pageid"])
            text = (page or {}).get("extract", "")
            if not text:
                self._unclear(f, report, "page empty")
                continue
            new_sources = [{**src, "revid": page["revid"]}] + f["sources"][1:]
            if W.grounded(f["answer"], text):
                # Still said in the new revision: confirmed without the model.
                self.facts.mark_verified(f["id"], new_sources)
                report["confirmed"] += 1
                continue
            # Not grounded in the new text any more. The model may say it
            # CHANGED; it may NOT confirm it — measured, the 3B answered SAME
            # for a stored fact against a sentence about cinema screens. So
            # SAME and UNCLEAR are both a strike, and two strikes demote.
            verdict = W.last_word(self._generate(
                W.COMPARE_SYSTEM, W.COMPARE_PROMPT.format(q=f["question"], a=f["answer"],
                                                          src=text[:2200]), 0.0, 8),
                ("SAME", "CHANGED", "UNCLEAR"))
            if verdict == "CHANGED":
                self.facts.demote(f["id"], "the source changed")
                report["changed"] += 1
            else:
                self._unclear(f, report, "unclear after an edit")

    def _unclear(self, f: dict, report: dict, why: str) -> None:
        if self.facts.strike(f["id"], why) >= 2:
            report["changed"] += 1
        else:
            report["unclear"] += 1

    def _log_audit(self, n: int) -> None:
        # What left the machine, for the record: page ids only.
        self.facts._log_lookup(f"(re-check of {n} stored page{'s' if n != 1 else ''})",
                               "audit", "sent")

    # job 2 ─────────────────────────────────────────────────────────────────
    def candidates(self, days: float = 7.0, limit: int = 200) -> list[str]:
        """His recent questions the model answered from memory, that are safe
        to send and worth sending. Pure: nothing leaves the machine here."""
        since = time.time() - days * 86400
        try:
            with self._db() as db:
                rows = db.execute(
                    "SELECT text FROM turn_stats WHERE ts >= ? AND path IN ('llm', 'conversation') "
                    "ORDER BY id DESC LIMIT ?", (since, limit)).fetchall()
        except sqlite3.OperationalError:
            return []
        names = self.facts._safe_names()
        out, seen = [], set()
        for (text,) in rows:
            q = W._clean(text)
            key = q.lower()
            if key in seen:
                continue
            seen.add(key)
            if not W.is_factual_question(q) or W.changeable(q):
                continue
            if W.private_reason(q, names):
                continue
            if self.facts.looked_up(q) or self.facts.lookup(q):
                continue
            out.append(q)
        return out

    def _curiosity(self, report: dict, back: Callable[[], bool]) -> None:
        import wiki
        if self.facts._search is None and wiki.backing_off():
            report["stopped"] = "wikipedia asked us to slow down"
            return
        sent = 0
        for q in self.candidates():
            if sent >= self.curiosity_budget:
                break
            if back():
                report["stopped"] = "he came back"
                return
            if sent:
                self._sleep(self.spacing_s)
            ref = self.facts.reference(q, origin="night")
            if ref.get("reason") == "rate limited":
                report["stopped"] = "wikipedia asked us to slow down"
                return
            sent += 1
            report["looked_up"].append(q)
            if not ref.get("ok"):
                continue
            if back():
                report["stopped"] = "he came back"
                return
            fid = self.facts.consider(ref, self._generate, origin="night")
            if fid:
                f = self.facts.get(fid)
                report["stored"].append({"question": f["question"], "answer": f["answer"]})

    # job 3 ─────────────────────────────────────────────────────────────────
    def _distill(self, report: dict, back: Callable[[], bool]) -> None:
        if not (self._learn and self._acts and self._skill_for):
            return
        since = time.time() - 7 * 86400
        try:
            with self._db() as db:
                rows = db.execute(
                    "SELECT DISTINCT text FROM turn_stats WHERE ts >= ? AND path LIKE 'regex:%' "
                    "ORDER BY id DESC LIMIT 30", (since,)).fetchall()
        except sqlite3.OperationalError:
            return
        for (text,) in rows:
            if len(report["learned"]) >= self.distill_budget:
                return
            if back():
                report["stopped"] = "he came back"
                return
            skill = self._skill_for(text)
            if not skill:
                continue
            raw = self._generate(_PARA_SYSTEM, _PARA_PROMPT.format(n=3, u=text.strip()), 0.7, 120)
            for line in (raw or "").splitlines():
                p = re.sub(r"^\s*(?:\d+[.)]|[-•*])\s*", "", line).strip().strip('"').strip()
                p = p.rstrip(".")
                if not (6 < len(p) < 90) or len(report["learned"]) >= self.distill_budget:
                    continue
                if self._acts(p):
                    continue                 # the router already gets this phrasing
                if self._learn(p, skill):
                    report["learned"].append({"text": p, "skill": skill})

    # ── what he can ask ──────────────────────────────────────────────────────
    def last_report(self) -> Optional[dict]:
        raw = self._meta("last_report")
        try:
            return json.loads(raw) if raw else None
        except ValueError:
            return None


def spoken(report: Optional[dict], label: Callable[[str], str] = lambda s: s) -> str:
    """The answer to "what did you learn last night?" — from the record."""
    if not report:
        return "I haven't had a night school yet. It runs in the small hours while you're away."
    when = datetime.datetime.fromtimestamp(report.get("finished") or report["started"])
    day = "Last night" if (datetime.date.today() - when.date()).days <= 1 else \
        f"On {when.strftime('%A')} night"
    parts = []
    a = report.get("audited", 0)
    if a:
        bits = [f"{report['confirmed']} still right"]
        if report.get("changed"):
            bits.append(f"{report['changed']} I stopped trusting")
        if report.get("unclear"):
            bits.append(f"{report['unclear']} I couldn't confirm yet")
        parts.append(f"I re-checked {a} fact{'s' if a != 1 else ''}: " + _join(bits))
    looked = report.get("looked_up") or []
    if looked:
        parts.append(f"I looked up {len(looked)} of your question{'s' if len(looked) != 1 else ''} "
                     f"on Wikipedia: {_join(looked)}")
    stored = report.get("stored") or []
    if stored:
        parts.append("I learned that " + _join([s["answer"].rstrip(".") for s in stored]))
    learned = report.get("learned") or []
    if learned:
        n = len(learned)
        parts.append(f"I picked up {n} new way{'s' if n != 1 else ''} of asking me to "
                     f"{label(learned[0]['skill'])}")
    if not parts:
        out = f"{day} I had nothing to study."
    else:
        out = f"{day}, " + ". ".join(parts) + "."
    if report.get("stopped") == "he came back":
        out += " I stopped early because you were back."
    return out.strip()


def _join(items: list[str]) -> str:
    items = [str(i) for i in items]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]
