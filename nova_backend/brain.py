"""
Nova's brain — a learned intent router between speech-to-text and the LLM.

Ported from JARVIS (MiniPCJarvis sidecar/brain/router.py, docs/BRAIN.md). The
design is JARVIS's; the details below are Nova's.

WHAT IT IS
A nearest-neighbour classifier over example sentences. Every request Nova can
handle by itself is a SKILL (skills.py): a name, example phrasings ("seeds"), a
slot function that pulls the arguments out of the sentence — or returns None,
meaning "this looks like mine but I cannot act on it" — and a runner that does
the work through the SAME handler the regex chain uses. Every example is
embedded once (bge-small, ~4ms each on this Mac); a new sentence is embedded
the same way and compared against all of them. The closest example names the
skill.

    confident AND the skill's slots extract  ->  a REFLEX: run it, no LLM
    the skill's guard refuses                ->  offer the turn to the runner-up
    nearly knew                              ->  kept in `self.unsure` (Stage 2 asks)
    no idea                                  ->  None: the old regex chain runs

THE CONFIDENCE RULE IS JARVIS'S, COPIED EXACTLY
    margin = top - rival            (rival = best example of ANOTHER skill)
    confidence = top if margin >= 0.06 else top - (0.06 - margin) * 3.0
A sentence that sits between two skills is never confident, however close it is
to both. Thresholds 0.82 (act) and 0.66 (nearly knew) were tuned on real speech
WITH bge-small — which is why Nova uses the same embedder rather than the MiniLM
model Chroma already ships: the numbers are a property of the model.

TWO QUESTIONS, BOTH MUST BE YES
confidence.py asks "was the TRANSCRIPT right?" The brain asks "which intent is
this?" They are independent. A skill carries a consequence tier, and the gate
uses the stricter of its own regex tier and the skill's: the brain can raise
the bar for acting, never lower it.

STORAGE (BRAIN.md, "Mistakes already made", #1 and #9)
Tables in nova_memory.db. Every write commits immediately on a short-lived
connection — an uncommitted write once held SQLite's only write lock for the
life of the JARVIS process and sent him ~2,600 messages in a night. Nothing is
ever held across a thread hop. Matrices built from np.frombuffer are copied,
because frombuffer returns a read-only view.

THREADS
Nova is threaded, not async. classify()/decide() are synchronous and guarded by
one lock (the main loop asks for the tier while the nova-llm thread routes).
Embedding is single-threaded on purpose: one short sentence is too little work
to fan out, and MLX needs the cores more.

The brain never touches the LLM, so a reflex cannot disturb the MLX prompt cache.
"""
from __future__ import annotations

import logging
import os
import re
import sqlite3
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

log = logging.getLogger("nova.brain")

# ── Defaults (config.json "brain" overrides) ───────────────────────────────────
THRESHOLD = 0.82          # at or above: act, if the slots extract
ASK_THRESHOLD = 0.66      # raw top similarity at or above this: "nearly knew"
# ...AND the margin-adjusted confidence at or above this. Nova's addition: with
# JARVIS's raw 0.66 alone, his REAL turns produced 6 "did you mean" questions
# and all 6 were wrong ("when do I go to the gym?" -> "read your calendar?").
# Measured 2026-11-19 over his turns + both paraphrase sets: every useful offer
# scored >= 0.727, every useless one <= 0.68. 0.70 kept 9 good, dropped all 12
# bad, lost 3 good. Tuned on that data, so re-measure as real turns accumulate.
ASK_MIN_CONFIDENCE = 0.70
MARGIN = 0.06             # how far ahead of the rival skill the top must be
MARGIN_PENALTY = 3.0      # how hard a near-tie is punished
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
MODEL_DIR = Path(__file__).resolve().parent / "models" / "fastembed"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS brain_examples (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        REAL NOT NULL,
    text      TEXT NOT NULL UNIQUE,     -- the CANONICAL form (_norm)
    skill     TEXT NOT NULL,
    source    TEXT NOT NULL DEFAULT 'seed',   -- seed | user | learned
    embedding BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS brain_commands (
    -- phrases he taught by voice: "when I say lights out, mute and lock"
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        REAL NOT NULL,
    phrase    TEXT NOT NULL UNIQUE,     -- _light() form of what he says
    steps     TEXT NOT NULL,            -- JSON [{"text": ..., "skill": ...}, ...]
    embedding BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS brain_rejections (
    -- "did you mean X?" ... "no". Never offer X for this sentence again.
    text  TEXT NOT NULL,
    skill TEXT NOT NULL,
    ts    REAL NOT NULL,
    PRIMARY KEY (text, skill)
);
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Skills
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class Skill:
    """One kind of request Nova can handle without the LLM.

    `slots(va, text)` returns the arguments, or None to refuse ("looks like
    mine, but I cannot act on it" — the turn then goes to the runner-up).
    `run(va, text, args)` does it, through the same stage handler the regex
    chain calls, and speaks for itself.

    Slots and run receive the ORIGINAL text, not the normalised one: Nova's
    extractors read capitals (a place name, a company) and digits.
    """
    name: str
    stage: str                       # the routing-corpus label it belongs to
    seeds: list[str]
    slots: Callable = lambda va, t: {}
    run: Optional[Callable] = None   # None = no tool (the `general` skill)
    tier: str = "low"                # confidence.py consequence: low|medium|high
    label: str = ""                  # "did you mean <label>?" (Stage 2)
    fixed_args: dict = field(default_factory=dict)
    # SITUATION, kept apart from the SENTENCE. Actuation only acts in work
    # mode; that is a fact about now, not about the words. Folded into slots,
    # a load-time "could this skill run this stored sentence?" check made
    # outside work mode would answer no and delete every learned actuation
    # phrasing on every start. decide() checks both; _executable() only slots.
    gate: Optional[Callable] = None


@dataclass
class Decision:
    skill: Skill
    args: dict
    confidence: float


# ═══════════════════════════════════════════════════════════════════════════════
# Normalisation
# ═══════════════════════════════════════════════════════════════════════════════
# Deliberately minimal (BRAIN.md: "start with only lowercasing, wake-word
# stripping and number/time placeholders; add a canon rule only when a real
# sentence needs it"). Each JARVIS canon rule exists because a real sentence of
# HIS went wrong; none of them is evidence about Nova.
# Only when something follows: "hey Nova" on its own is a GREETING, and
# stripping it whole left an empty sentence that matched nothing.
_WAKE = re.compile(r"^(?:(?:hey|hi|ok|okay|so)[,\s]+)?nova\b[,.!?\s]*(?=\S)")
_TRAILING_WAKE = re.compile(r"[,\s]+nova[.!?]*$")
_POLITE = re.compile(r"\b(?:please|for me|can you|could you|would you|will you)\b")


def _light(t: str) -> str:
    """Lowercase, strip the wake word and politeness. Keeps every content word."""
    t = (t or "").lower().strip()
    t = _WAKE.sub("", t)
    t = _TRAILING_WAKE.sub("", t)
    t = _POLITE.sub(" ", t)
    # Whisper's habits. "You Tube" as two words missed every "youtube" pattern
    # on JARVIS (BRAIN.md mistake #8); same recogniser family here.
    t = re.sub(r"\byou\s+tube\b", "youtube", t)
    t = re.sub(r"[^\w\s%':.-]", " ", t)
    t = re.sub(r"(?<!\d)[.:](?!\d)", " ", t)       # keep 5:30 and 2.5, drop sentence dots
    return re.sub(r"\s+", " ", t).strip()


_CANON: list[tuple[str, str]] = [
    # Numbers are objects, not intent: "set the volume to 40" and "to 75" are
    # the same request. Digit runs (and times like 5:30) become N.
    (r"\d+(?:[:.]\d+)?", "N"),
]


def _norm(t: str) -> str:
    """Canonical intent form used for embeddings (objects -> placeholders).

    Unlike JARVIS's first-match-only canon list, every rule applies: Nova's
    rules are object placeholders that compose, not whole-sentence rewrites.
    """
    t = _light(t)
    for pat, rep in _CANON:
        t = re.sub(pat, rep, t)
    return re.sub(r"\s+", " ", t).strip()


# What the online learner must never learn from the model's tool choice
# (Stage 2; defined here so load() can apply it to stored rows too).
_ANSWER_ONLY = re.compile(
    r"(?:yes|yeah|yep|yup|no|nope|nah|ok|okay|sure|please|thanks|thank you|"
    r"go ahead|do it|go for it|yes please|no thanks|fine|alright|right)[.!?]?")


# ═══════════════════════════════════════════════════════════════════════════════
# The brain
# ═══════════════════════════════════════════════════════════════════════════════
class Brain:
    def __init__(self, db_path: Path, skills: list[Skill],
                 config: Optional[dict] = None, va=None,
                 embed_cache: Optional[dict] = None) -> None:
        cfg = (config or {}).get("brain", {}) if config else {}
        self.cfg = cfg
        self.threshold = float(cfg.get("threshold", THRESHOLD))
        self.ask_threshold = float(cfg.get("ask_threshold", ASK_THRESHOLD))
        self.ask_min_confidence = float(cfg.get("ask_min_confidence", ASK_MIN_CONFIDENCE))
        self.margin = float(cfg.get("margin", MARGIN))
        self.margin_penalty = float(cfg.get("margin_penalty", MARGIN_PENALTY))
        self.embed_threads = int(cfg.get("embed_threads", 1))
        self.db_path = Path(db_path)
        self.va = va
        self.skills = list(skills)
        self.by_name = {s.name: s for s in self.skills}
        self._lock = threading.RLock()
        self._embedder = None
        self._vec_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._texts: list[str] = []
        self._skills: list[str] = []
        self._sources: list[str] = []
        self._matrix: Optional[np.ndarray] = None
        self._rejected: set = set()
        self._cmd_phrases: list[str] = []
        self._cmd_steps: list[list[dict]] = []
        self._cmd_matrix: Optional[np.ndarray] = None
        self.ready = False
        self.last_match: Optional[dict] = None
        # The near-miss from the last decide(): the right idea, scored too low
        # to act on alone, kept so the turn can ASK (Stage 2). JARVIS mistake
        # #5: the runner-up used to be computed, used, and thrown away.
        self.unsure: Optional[dict] = None
        self.stats = {"reflex": 0, "fallthrough": 0, "learned": 0}
        # CONFIDENTLY CONVERSATION. Set by decide() when the nearest examples
        # are the `general` skill's and the confidence clears the action bar:
        # the brain is as sure this is NOT a command as it would need to be to
        # run one. nova.py then skips the regex command stages — which is how
        # "the weather has been so weird this year" stops getting a forecast —
        # but never the unsupported-action refusal, which is a safety net.
        self.conversation = False
        # Shared across Brain instances (tests build several); text -> vector.
        self._shared = embed_cache

    # ── storage ───────────────────────────────────────────────────────────────
    @contextmanager
    def _db(self):
        """A short-lived connection that COMMITS before it is released.

        Never a long-lived one: the JARVIS outage was a single connection
        holding an open write transaction for the life of the process.
        """
        conn = sqlite3.connect(str(self.db_path), timeout=5.0)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ── embeddings ────────────────────────────────────────────────────────────
    def _get_embedder(self):
        if self._embedder is None:
            from fastembed import TextEmbedding
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            kw = dict(cache_dir=str(MODEL_DIR),
                      threads=self.embed_threads if self.embed_threads > 0 else None)
            try:
                # Offline first: the model is downloaded ONCE (~64MB, from
                # Hugging Face) and lives in nova_backend/models/. After that
                # the brain never touches the network.
                self._embedder = TextEmbedding(EMBED_MODEL, local_files_only=True, **kw)
            except Exception:
                log.info("brain: embedding model not on disk yet, downloading it once")
                self._embedder = TextEmbedding(EMBED_MODEL, **kw)
        return self._embedder

    def _embed(self, texts: list[str]) -> np.ndarray:
        if self._shared is not None:
            todo = [t for t in texts if t not in self._shared]
            if todo:
                self._shared.update(zip(todo, self._embed_raw(todo)))
            return np.stack([self._shared[t] for t in texts])
        return self._embed_raw(texts)

    def _embed_raw(self, texts: list[str]) -> np.ndarray:
        if len(texts) == 1:
            hit = self._vec_cache.get(texts[0])
            if hit is not None:
                self._vec_cache.move_to_end(texts[0])
                return hit
        vecs = np.array(list(self._get_embedder().embed(texts)), dtype=np.float32)
        vecs /= (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
        if len(texts) == 1:
            self._vec_cache[texts[0]] = vecs
            while len(self._vec_cache) > 256:
                self._vec_cache.popitem(last=False)
        return vecs

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def canonical_seeds(self) -> dict[str, str]:
        """canonical text -> skill. First skill listed wins a collision —
        silently, which is why tests/test_brain_seeds.py exists."""
        canon: dict[str, str] = {}
        for sk in self.skills:
            for u in sk.seeds:
                c = _norm(u)
                if c:
                    canon.setdefault(c, sk.name)
        return canon

    def load(self) -> None:
        """Seed on first run, re-validate what is stored, load into memory."""
        with self._lock:
            canon = self.canonical_seeds()
            with self._db() as db:
                db.executescript(_SCHEMA)
            dropped = 0
            with self._db() as db:
                # Seeds that changed or moved skill are dropped and re-embedded.
                for text, skill in db.execute(
                        "SELECT text, skill FROM brain_examples WHERE source='seed'").fetchall():
                    if canon.get(text) != skill:
                        db.execute("DELETE FROM brain_examples WHERE text=?", (text,))
                        dropped += 1
                # Seeds are ground truth: a learned row that contradicts one is
                # a mislabel; one its skill can no longer execute is useless.
                for text, skill in db.execute(
                        "SELECT text, skill FROM brain_examples WHERE source!='seed'").fetchall():
                    # The executability check needs the live engines; a brain
                    # loaded without them (a test, a bare tool) must not judge
                    # — and so must not delete — what it cannot evaluate.
                    if (text in canon and canon[text] != skill) \
                            or (self.va is not None
                                and not self._executable(text, skill)):
                        db.execute("DELETE FROM brain_examples WHERE text=?", (text,))
                        dropped += 1
                        log.info(f"brain: dropped stored example {text!r} ({skill})")
            # ^ committed HERE, as the block exits — before the embedding below,
            #   so no write transaction is ever open while we do slow work.
            if dropped:
                log.info(f"brain: dropped {dropped} stale example(s)")
            with self._db() as db:
                have = {r[0] for r in db.execute("SELECT text FROM brain_examples")}
            missing = [(sk, t) for t, sk in canon.items() if t not in have]
            if missing:
                t0 = time.time()
                vecs = self._embed([t for _, t in missing])
                with self._db() as db:
                    db.executemany(
                        "INSERT OR IGNORE INTO brain_examples "
                        "(ts, text, skill, source, embedding) VALUES (?,?,?,?,?)",
                        [(time.time(), t, sk, "seed", v.tobytes())
                         for (sk, t), v in zip(missing, vecs)])
                log.info(f"brain: embedded {len(missing)} new seed(s) in "
                         f"{time.time() - t0:.1f}s")
            with self._db() as db:
                rows = db.execute(
                    "SELECT text, skill, source, embedding FROM brain_examples").fetchall()
                self._rejected = {(r[0], r[1]) for r in db.execute(
                    "SELECT text, skill FROM brain_rejections")}
                crows = db.execute(
                    "SELECT phrase, steps, embedding FROM brain_commands").fetchall()
            rows = [r for r in rows if r[1] in self.by_name]   # a removed skill's rows stay inert
            self._texts = [r[0] for r in rows]
            self._skills = [r[1] for r in rows]
            self._sources = [r[2] for r in rows]
            # .copy(): np.frombuffer is a READ-ONLY view (JARVIS mistake #9).
            self._matrix = (np.frombuffer(b"".join(r[3] for r in rows), dtype=np.float32)
                            .reshape(len(rows), -1).copy()) if rows else None
            import json
            self._cmd_phrases = [r[0] for r in crows]
            self._cmd_steps = [json.loads(r[1]) for r in crows]
            self._cmd_matrix = (np.frombuffer(b"".join(r[2] for r in crows), dtype=np.float32)
                                .reshape(len(crows), -1).copy()) if crows else None
            self._get_embedder()
            self.ready = self._matrix is not None
            log.info(f"brain loaded: {len(rows)} examples across "
                     f"{len(set(self._skills))} skills")

    def load_background(self) -> threading.Thread:
        """Load off the voice loop. Until it is ready decide() returns None and
        the old regex chain answers everything, exactly as before."""
        def _run():
            try:
                self.load()
            except Exception as exc:
                log.warning(f"brain failed to load, regex routing only: {exc}")
        t = threading.Thread(target=_run, name="nova-brain-load", daemon=True)
        t.start()
        return t

    @property
    def example_count(self) -> int:
        return len(self._texts)

    # ── classification ────────────────────────────────────────────────────────
    def classify(self, text: str, k: int = 5,
                 exclude: Optional[set] = None) -> tuple[Optional[str], float]:
        """(skill, confidence). Top match names the skill; confidence is the top
        similarity, cut when the best example of a DIFFERENT skill is close."""
        if not self.ready or self._matrix is None or not (text or "").strip():
            return None, 0.0
        canon = _norm(text)
        if not canon:
            return None, 0.0
        with self._lock:
            q = self._embed([canon])[0]
            sims = self._matrix @ q
            order = np.argsort(-sims)
            if exclude:
                # decide() asks again without a skill whose guard refused, so a
                # guarded top match hands over to the runner-up instead of
                # dropping the whole turn.
                order = [i for i in order if self._skills[i] not in exclude]
                if not len(order):
                    return None, 0.0
            best = self._skills[order[0]]
            top = float(sims[order[0]])
            rival, rival_skill = 0.0, None
            for i in order[1:k * 4]:
                if self._skills[i] != best:
                    rival, rival_skill = float(sims[i]), self._skills[i]
                    break
            margin = top - rival
            confidence = top if margin >= self.margin \
                else top - (self.margin - margin) * self.margin_penalty
            confidence = round(max(0.0, confidence), 3)
            self.last_match = {
                "text": self._texts[order[0]], "skill": best,
                "source": self._sources[order[0]], "query": text,
                "canon": canon, "confidence": confidence,
                "rival": rival_skill, "margin": round(margin, 3),
                "top": round(top, 3),
            }
            if best == "general":
                return None, confidence
            return best, confidence

    def decide(self, text: str, va=None) -> Optional[Decision]:
        """A confident skill whose slots extract, or None.

        A skill whose extractor REFUSES steps aside and the next-best gets a
        turn — bounded, so a chain of refusals still ends at the fallback.
        """
        va = va if va is not None else self.va
        self.unsure = None
        self.conversation = False
        if not self.ready:
            return None
        # A skill he said NO to for this sentence ("did you mean X?" — "no",
        # or "no, I meant ..." after a reflex) is never chosen for it again.
        canon = _norm(text)
        refused: set = {sk for (c, sk) in self._rejected if c == canon}
        for attempt in range(3):
            name, conf = self.classify(text, exclude=refused or None)
            if attempt == 0 and (self.last_match or {}).get("skill") == "general" \
                    and conf >= self.threshold:
                self.conversation = True
            if not name or conf < self.threshold:
                # NEARLY KNEW IS NOT THE SAME AS NO IDEA. Kept for Stage 2's
                # "did you mean ...?" — never acted on here.
                m = self.last_match or {}
                top = float(m.get("top") or 0.0)
                if name and top >= self.ask_threshold and conf >= self.ask_min_confidence:
                    sk = self.by_name.get(name)
                    args = self._slots(sk, va, text) \
                        if sk is not None and self._gate(sk, va) else None
                    if sk is not None and args is not None \
                            and (canon, name) not in self._rejected:
                        self.unsure = {"skill": name, "rival": m.get("rival"),
                                       "confidence": conf, "top": top,
                                       "margin": m.get("margin"), "args": args}
                self.stats["fallthrough"] += 1
                return None
            sk = self.by_name[name]
            args = self._slots(sk, va, text) if self._gate(sk, va) else None
            if args is not None:
                self.stats["reflex"] += 1
                return Decision(sk, {**sk.fixed_args, **args}, conf)
            refused.add(name)
        self.stats["fallthrough"] += 1
        return None

    def tier_for(self, text: str, va=None) -> Optional[str]:
        """The consequence tier of the skill this sentence would run, or None.

        Asked by confidence.py's gate BEFORE the turn is routed. It must not
        disturb what the turn itself will see, so the per-turn state is put
        back afterwards (the embedding stays cached, so routing it again is
        nearly free)."""
        saved = (self.last_match, self.unsure, self.conversation, dict(self.stats))
        try:
            hit = self.match_command(text)
            if hit is not None:
                # A taught phrase is as consequential as its strictest step.
                rank = {"low": 0, "medium": 1, "high": 2}
                tiers = [self.by_name[st["skill"]].tier for st in hit[1]
                         if st.get("skill") in self.by_name]
                tiers += ["medium" for st in hit[1]
                          if str(st.get("skill", "")).startswith("regex:")]
                return max(tiers, key=lambda t: rank[t]) if tiers else None
            d = self.decide(text, va)
            return d.skill.tier if d is not None else None
        except Exception:
            return None
        finally:
            self.last_match, self.unsure, self.conversation, self.stats = saved

    @staticmethod
    def _gate(sk: Skill, va) -> bool:
        if sk.gate is None:
            return True
        try:
            return bool(sk.gate(va))
        except Exception:
            return False

    @staticmethod
    def _slots(sk: Skill, va, text: str) -> Optional[dict]:
        """A guard that throws must not cost him the turn: treat it as a refusal."""
        try:
            return sk.slots(va, text)
        except Exception as exc:
            log.warning(f"brain: slots for {sk.name} raised on {text!r}: {exc}")
            return None

    # ── learning support (Stage 2 builds on these) ────────────────────────────
    def _executable(self, text: str, skill: str) -> bool:
        """Could this skill act on this STORED sentence by itself?

        Stored rows are canonical (JARVIS mistake #2): `N` stands where a
        number was, and an extractor that needs a number refuses "N" — which
        on JARVIS deleted every confirmed example on the next boot. So the
        placeholders get stand-in values before the question is asked.
        """
        sk = self.by_name.get(skill)
        if sk is None or skill == "general" or sk.run is None:
            return False
        probe = re.sub(r"\bN\b", "10", text)
        return self._slots(sk, self.va, probe) is not None

    # ── Stage 2: learning ─────────────────────────────────────────────────────
    MAX_LEARN_WORDS = 14

    def learn(self, text: str, skill: str, source: str = "learned",
              va=None) -> bool:
        """Store a sentence as a new example of `skill`. True if stored.

        JARVIS's learner learned from the LLM's single tool call. Nova's LLM
        has no tools, so the teacher is the regex chain behind the brain: a
        sentence the brain fell through on and a regex stage then claimed.
        The bar is JARVIS's (BRAIN.md "How it learns"), every rule from a real
        mistake there:
          - at most 14 words;
          - never a bare answer ("yes" was once learned as "render again");
          - only what the skill's OWN slots can execute — checked on the raw
            sentence AND on the canonical form that will be stored, because
            load() re-validates the stored form (mistake #2);
          - never a sentence that contradicts a seed (seeds are ground truth);
          - not when the brain was already fairly sure it was conversation
            (general >= 0.7): the claim was the regex's whim.
        `source="user"` (a yes to "did you mean ...?") skips the last three:
        he said so himself.
        """
        va = va if va is not None else self.va
        t = _norm(text)
        sk = self.by_name.get(skill)
        if not t or sk is None or sk.run is None or skill == "general":
            return False
        if len(t.split()) > self.MAX_LEARN_WORDS or _ANSWER_ONLY.fullmatch(t):
            return False
        seed_owner = self.canonical_seeds().get(t)
        if seed_owner is not None:
            return False                      # already a seed (of this or another skill)
        if source != "user":
            if self._slots(sk, va, text) is None or not self._executable(t, skill):
                return False
            saved = (self.last_match, self.unsure, self.conversation)
            try:
                self.classify(text)
                m = self.last_match or {}
                if m.get("skill") == "general" and float(m.get("confidence") or 0) >= 0.7:
                    return False
                # TWO JUDGES MUST AGREE. The teacher here is the regex chain,
                # and the regex chain makes its own wrong calls ("my battery
                # died on the way to work" claims a battery reading). JARVIS
                # mistake #4 — the teacher's choice is not ground truth — so a
                # regex lesson is taken only when the brain's OWN best guess,
                # however unsure, already points at the same skill.
                if m.get("skill") != skill:
                    return False
            finally:
                self.last_match, self.unsure, self.conversation = saved
        with self._lock:
            if t in self._texts:
                i = self._texts.index(t)
                if self._skills[i] == skill:
                    return False
                if self._sources[i] == "user" and source != "user":
                    return False              # his word beats the regex's
                self._drop_index(i)           # replace a weaker, contradicting row
            v = self._embed([t])[0]
            with self._db() as db:
                db.execute("DELETE FROM brain_examples WHERE text=?", (t,))
                db.execute(
                    "INSERT INTO brain_examples (ts, text, skill, source, embedding) "
                    "VALUES (?,?,?,?,?)", (time.time(), t, skill, source, v.tobytes()))
            self._texts.append(t)
            self._skills.append(skill)
            self._sources.append(source)
            self._matrix = v[None, :] if self._matrix is None else np.vstack([self._matrix, v])
            self.stats["learned"] += 1
        log.info(f"brain learned ({source}): {t!r} -> {skill}")
        return True

    def _drop_index(self, i: int) -> None:
        with self._db() as db:
            db.execute("DELETE FROM brain_examples WHERE text=?", (self._texts[i],))
        self._texts.pop(i)
        self._skills.pop(i)
        self._sources.pop(i)
        self._matrix = np.delete(self._matrix, i, axis=0) if self._texts else None

    def reject(self, text: str, skill: str) -> None:
        """He said no to <skill> for this sentence: never choose or offer it
        for this sentence again. Persisted — being asked the same wrong
        question tomorrow is the same annoyance as twice in a row. Keyed on
        sentence AND skill, so the other readings stay available."""
        t = _norm(text)
        if not t or not skill:
            return
        try:
            with self._db() as db:
                db.execute("INSERT OR IGNORE INTO brain_rejections (text, skill, ts) "
                           "VALUES (?,?,?)", (t, skill, time.time()))
            self._rejected.add((t, skill))
            log.info(f"brain will not use {skill} for {t!r} again")
        except Exception as exc:
            # A rejection that cannot be stored must never cost him the turn.
            log.warning(f"brain: could not store a rejection: {exc}")

    def was_rejected(self, text: str, skill: str) -> bool:
        return (_norm(text), skill) in self._rejected

    def unlearn(self, match: Optional[dict]) -> Optional[str]:
        """A correction: the reflex behind `match` was wrong. Drops the stored
        example that caused it if it was LEARNED or taught by him; a seed is
        never dropped (seeds are ground truth — the rejection that goes with a
        correction is what keeps the sentence away from it). Returns the skill
        whose example was removed, or None."""
        if not match:
            return None
        if match.get("skill") == "command":
            return "command" if self.forget_command(match.get("text", "")) else None
        if match.get("source") not in ("learned", "user"):
            return None
        with self._lock:
            t = match.get("text")
            if t in self._texts:
                i = self._texts.index(t)
                skill = self._skills[i]
                self._drop_index(i)
                log.info(f"brain unlearned {t!r} ({skill})")
                return skill
        return None

    def learned(self, limit: int = 25) -> list[dict]:
        """Most recent non-seed examples, newest first (for the log/UI)."""
        with self._db() as db:
            rows = db.execute(
                "SELECT ts, text, skill, source FROM brain_examples WHERE source != 'seed' "
                "ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
        return [{"ts": r[0], "text": r[1], "skill": r[2], "source": r[3]} for r in rows]

    # ── Stage 2: phrases taught by voice ──────────────────────────────────────
    COMMAND_MATCH = 0.92      # near-exact only: a taught phrase runs SEVERAL actions

    def teach_command(self, phrase: str, steps: list[dict]) -> None:
        import json
        p = _light(phrase)
        if not p or not steps:
            raise ValueError("a taught phrase needs words and at least one step")
        with self._lock:
            v = self._embed([p])[0]
            with self._db() as db:
                db.execute("INSERT OR REPLACE INTO brain_commands (ts, phrase, steps, embedding) "
                           "VALUES (?,?,?,?)", (time.time(), p, json.dumps(steps), v.tobytes()))
            if p in self._cmd_phrases:
                i = self._cmd_phrases.index(p)
                self._cmd_steps[i] = steps
                self._cmd_matrix[i] = v        # writable: copied at load
            else:
                self._cmd_phrases.append(p)
                self._cmd_steps.append(steps)
                self._cmd_matrix = v[None, :] if self._cmd_matrix is None \
                    else np.vstack([self._cmd_matrix, v])
        log.info(f"brain taught command {p!r} -> {steps}")

    def match_command(self, text: str) -> Optional[tuple[str, list[dict], float]]:
        """(phrase, steps, similarity) for a taught phrase, near-exact only."""
        if self._cmd_matrix is None or not (text or "").strip():
            return None
        p = _light(text)
        if not p:
            return None
        with self._lock:
            if p in self._cmd_phrases:
                i = self._cmd_phrases.index(p)
                return self._cmd_phrases[i], self._cmd_steps[i], 1.0
            q = self._embed([p])[0]
            sims = self._cmd_matrix @ q
            i = int(np.argmax(sims))
            if float(sims[i]) >= self.COMMAND_MATCH:
                return self._cmd_phrases[i], self._cmd_steps[i], float(sims[i])
        return None

    def forget_command(self, phrase: str) -> bool:
        p = _light(phrase)
        with self._lock:
            if p not in self._cmd_phrases:
                # tolerate "forget lights out" when he taught "lights out please"
                hit = self.match_command(phrase)
                if not hit:
                    return False
                p = hit[0]
            i = self._cmd_phrases.index(p)
            with self._db() as db:
                db.execute("DELETE FROM brain_commands WHERE phrase=?", (p,))
            self._cmd_phrases.pop(i)
            self._cmd_steps.pop(i)
            self._cmd_matrix = np.delete(self._cmd_matrix, i, axis=0) \
                if self._cmd_phrases else None
        log.info(f"brain forgot command {p!r}")
        return True

    def commands(self) -> list[dict]:
        return [{"phrase": p, "steps": st} for p, st in zip(self._cmd_phrases, self._cmd_steps)]

