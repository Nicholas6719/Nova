"""
Nova File Intents — the natural-language layer over `file_manager.py`.

`detect_intent` is strict regex only: no file word, no intent, no interference
with ordinary conversation. `handle` returns one spoken string and may leave a
pending question behind — Nova asks before it touches anything.

Runs on the ``nova-llm`` worker thread (dispatched from `_handle_turn_impl`),
so calling `self.llm.generate` from here is thread-safe.

Extraction is DELIBERATELY deterministic. The calendar build taught this the
hard way: the 3B model mangled "from Downloads to Documents" into
destination=Downloads, and hand-written regex beat it every time. The LLM is
used for exactly one thing here — summarizing a file's contents out loud,
which is a judgment task rather than a parsing one.

Safety model, in order of how much it matters:
  * Nothing that changes the filesystem happens without an explicit spoken
    yes. Anything that is not a yes cancels.
  * Ambiguity is resolved BEFORE the confirmation, by asking which file —
    not by walking through "is this the one?" one file at a time.
  * Delete is refused outright and said so. Falling through to the LLM would
    let it claim it deleted something it did not.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Callable, Optional

import file_manager as fm
import panels as P

log = logging.getLogger("nova.files")


def _folder_panel(folder, label: str, info: dict) -> dict:
    """A folder listing as a panel: folders first, then files, all of them.

    Paths are deliberately absent. Invariant: a spoken answer never reads a
    filesystem path aloud, and there is no reason to put one on screen either
    when the folder is already the panel's subtitle.
    """
    rows = [{"title": name, "meta": "folder"} for name in info["folders"]]
    rows += [{"title": name} for name in info["files"]]
    n = info["n_files"] + info["n_folders"]
    return P.panel(
        title=label.capitalize(),
        subtitle=f"{n} item{'s' if n != 1 else ''}",
        blocks=[P.items(rows) if rows else P.note("This folder is empty.")],
    )


# ── Vocabulary ──────────────────────────────────────────────────────────────
# A noun that means "we are talking about a file". Generic on purpose; the
# distinctive-token guard in detect_intent is what stops it over-firing.
_FILE_NOUN_RE = re.compile(
    r"\b(file|files|document|documents|doc|docs|pdf|pdfs|"
    r"image|images|picture|pictures|photo|photos|screenshot|screenshots|"
    r"spreadsheet|spreadsheets|presentation|presentations|slides|"
    r"note|notes|resume|resumes|cv|essay|report|reports|invoice|invoices|"
    r"receipt|receipts|letter|budget|notebook)\b"
)
_FILE_EXT_RE = re.compile(
    r"\.(?:pdf|docx?|txt|md|rtf|pages|numbers|key|png|jpe?g|gif|heic|tiff|"
    r"bmp|webp|csv|xlsx?|pptx?|json|zip)\b"
)
# "What's in my Documents folder?" used to reach no handler at all and fall
# through to the LLM, which invented contents for a folder it never read. This
# fires BEFORE the searchable-token guard, because a folder question names a
# location, not a file, so it legitimately tokenizes to nothing.
_LIST_FOLDER_RE = re.compile(
    r"\b(?:"
    r"what(?:'?s| is)\s+(?:in|inside|on)\s+(?:my|the)\s+(?P<a>[\w ]+?)(?:\s+folder)?"
    r"|what\s+(?:files|else)\s+(?:are|is)\s+(?:in|on)\s+(?:my|the)\s+(?P<b>[\w ]+?)(?:\s+folder)?"
    r"|(?:list|show\s+me|what(?:'?s| is)\s+in)\s+(?:my|the)\s+(?P<c>[\w ]+?)\s+(?:folder|directory)"
    r"|what\s+do\s+i\s+have\s+(?:in|on)\s+(?:my|the)\s+(?P<d>[\w ]+?)(?:\s+folder)?"
    r")\s*[.?!]*$",
    re.I,
)

_PRONOUN_RE = re.compile(r"\b(it|that|this|that\s+one|the\s+same\s+one)\b")

# A calendar COMMAND, not merely a calendar word. "remind me to read the report"
# is a reminder even though it says "read" and "report"; "find the meeting notes
# file" is a file even though it says "meeting". So this matches the command
# phrasing only, never a bare noun. calendar_intents has the mirror guard.
_CALENDAR_COMMAND_RE = re.compile(
    r"\bremind\s+me\b"
    r"|\b(?:set|create|add|make|new)\s+(?:a\s+|an\s+|another\s+)?"
    r"(?:reminder|event|appointment|meeting)\b"
    r"|\bon\s+(?:my\s+)?calendar\b"
    r"|\bschedule\s+(?:a|an|my|the)\b"
)

_MOVE_VERB_RE = re.compile(
    r"\b(?:move[sd]?|moving|put(?:s|ting)?|transfer(?:s|red|ring)?|"
    r"send(?:s|ing)?|sent|drop(?:s|ped|ping)?|relocate[sd]?|relocating|"
    r"stick(?:s|ing)?|stuck)\b"
)
_COPY_VERB_RE = re.compile(
    r"\b(?:copy|copies|copied|copying|duplicate[sd]?|duplicating|"
    r"back\s*up|backed\s*up)\b"
)
_RENAME_VERB_RE = re.compile(r"\b(?:rename[sd]?|renaming)\b|\bchange\s+the\s+name\s+of\b")
_DELETE_VERB_RE = re.compile(
    r"\b(?:delete[sd]?|deleting|erase[sd]?|erasing|remove[sd]?|removing|"
    r"trash(?:es|ed|ing)?|get\s+rid\s+of|wipe[sd]?|shred(?:s|ded|ding)?)\b"
)
_DESCRIBE_RE = re.compile(
    r"\b(?:summari[sz]e[sd]?|summari[sz]ing|summary|describe[sd]?|describing|"
    r"read)\b|\bwhat(?:'?s|\s+is|\s+does)\s+(?:in|inside)\b|"
    r"\btell\s+me\s+what(?:'?s|\s+is)\s+(?:in|inside)\b"
)
_FIND_RE = re.compile(
    r"\b(?:find|finds|finding|locate[sd]?|locating|look\s+for|search\s+for|"
    r"where\s+is|where(?:'?s)?|do\s+i\s+have)\b"
)
_OPEN_RE = re.compile(r"\b(?:open|pull\s+up|bring\s+up|show\s+me|launch)\b")

# Destination: the folder named after to/into/onto. "from <somewhere>" spans are
# stripped first so "from my downloads to documents" can only match documents.
_DEST_RE = re.compile(
    r"\b(?:to|into|onto)\s+(?:my\s+|the\s+|a\s+)?"
    r"([A-Za-z][\w'-]*(?:\s+[A-Za-z][\w'-]*)?)"
    r"(?:\s+(?:folder|directory))?\b"
)
_FROM_RE = re.compile(r"\bfrom\s+(?:my\s+|the\s+)?[\w'-]+(?:\s+(?:folder|directory))?\b")
# Only these bare words are taken as a destination without the word "folder";
# anything else must be said as "... folder" so "send it to Mom" is not read
# as a filesystem move.
_KNOWN_DEST_WORDS = frozenset({
    "desktop", "downloads", "download", "documents", "document", "docs",
    "pictures", "picture", "photos", "music", "movies", "videos", "home",
})

_RENAME_SPLIT_RE = re.compile(
    r"\b(?:rename[sd]?|renaming|change\s+the\s+name\s+of)\b\s*(.*?)\s+"
    r"(?:to|as|into)\s+(.+)$"
)

_YES_RE = re.compile(
    r"^\s*(?:ye(?:s|ah|p|up)|sure|please|ok|okay|do\s+it|go\s+ahead|"
    r"confirm|correct|that'?s\s+(?:right|the\s+one)|affirmative)\b"
)
_NO_RE = re.compile(
    r"^\s*(?:no|nope|nah|don'?t|do\s+not|stop|cancel|never\s*mind|"
    r"forget\s+it|wrong\s+one|that'?s\s+not)\b"
)

# Picking a candidate, in PRIORITY ORDER. The tiers exist because "the second
# one" contains the word "one": matching number words at the same priority as
# ordinals made "the second one" resolve to candidate 1. Ordinals are checked
# first, then digits, then bare number words.
_CHOICE_TIERS: tuple[dict[str, int], ...] = (
    {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3,
     "fourth": 4, "4th": 4, "fifth": 5, "5th": 5},
    {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5},
    {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5},
)

# How long a "move it to Documents" still refers to the file we just touched.
_FOLLOWUP_WINDOW_S = 180.0


class NovaFiles:
    def __init__(self, config: dict, llm) -> None:
        self.config = config
        self.llm = llm
        self.name = config["user"]["address_as"]
        # (view_name, payload) for the panel; cleared at the top of every
        # handle() so a listing never lingers onto an unrelated answer.
        self.last_panel = None

        # A question Nova is waiting on: which file, or is this the one.
        # nova.py checks this BEFORE routing so the answer isn't misread as a
        # new command. An unrelated utterance clears it and routes normally.
        self._pending: Optional[dict] = None
        # The file the last successful operation touched, so "move it to
        # Documents" works as a follow-up. Expires — a stale pronoun should
        # never resolve to a file from an hour ago.
        self._last_file: Optional[dict] = None
        # Soft offer ("want me to show you where it is?"). nova.py reuses the
        # same one-shot slot as the calendar and maps offers.
        self.pending_offer: Optional[Callable[[], str]] = None

        self._file_system = (
            f"You are Nova, a sharp, composed AI assistant. You are describing a "
            f"file's contents out loud to {self.name}. Speak naturally, the way a "
            "person would say it. Never use markdown, bullet points, numbered "
            "lists, or dashes. "
            # The 3B will happily do arithmetic and get it wrong: asked to
            # summarize a budget listing groceries at 380, it announced that
            # 193 a month was left over for groceries. Report, don't reason.
            "Only state things the text actually says. Do not do arithmetic, "
            "do not add up numbers, and do not draw conclusions the text does "
            "not state. If you are unsure, leave it out."
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Pending question state
    # ═══════════════════════════════════════════════════════════════════════
    def has_pending(self) -> bool:
        return self._pending is not None

    def clear_pending(self) -> None:
        self._pending = None

    # ═══════════════════════════════════════════════════════════════════════
    # Intent detection
    # ═══════════════════════════════════════════════════════════════════════
    def detect_intent(self, text: str) -> Optional[str]:
        """One of file_find / file_describe / file_open / file_move /
        file_copy / file_rename / file_delete, or None.

        Returns None unless the utterance both reads as a file request AND
        names something searchable. That second half is what keeps "open my
        photos" pointed at the Photos app instead of a file search.
        """
        t = (text or "").lower().strip()
        if not t:
            return None

        # Calendar commands win outright. The pipeline already runs calendar
        # first, but declining here too means neither module depends on the
        # other's position to behave correctly.
        if _CALENDAR_COMMAND_RE.search(t):
            return None

        has_noun = bool(_FILE_NOUN_RE.search(t))
        has_ext = bool(_FILE_EXT_RE.search(t))
        m = _LIST_FOLDER_RE.search(t)
        if m:
            spoken = next((g for g in m.groupdict().values() if g), "")
            if fm.resolve_folder(spoken) is not None:
                return "list_folder"

        followup = self._recent_file() is not None and bool(_PRONOUN_RE.search(t))
        if not (has_noun or has_ext or followup):
            return None

        # Delete is answered honestly whether or not we could find the file,
        # so it skips the searchable-token guard below.
        if _DELETE_VERB_RE.search(t):
            return "file_delete"

        intent: Optional[str] = None
        if _RENAME_VERB_RE.search(t):
            intent = "file_rename"
        elif _COPY_VERB_RE.search(t) and re.search(r"\b(?:to|into|onto)\b", t):
            intent = "file_copy"
        elif _MOVE_VERB_RE.search(t) and re.search(r"\b(?:to|into|onto)\b", t):
            intent = "file_move"
        elif _DESCRIBE_RE.search(t):
            intent = "file_describe"
        elif _FIND_RE.search(t):
            intent = "file_find"
        elif _OPEN_RE.search(t):
            intent = "file_open"

        if intent is None:
            return None

        # The searchable-token guard. A follow-up pronoun supplies the file
        # instead, so it is exempt.
        if followup:
            return intent
        if not self._query_tokens(t, intent):
            log.info(f"declining {intent}: no distinctive file token in {t!r}")
            return None
        return intent

    # ═══════════════════════════════════════════════════════════════════════
    # Handling
    # ═══════════════════════════════════════════════════════════════════════
    def handle(self, intent: str, text: str) -> str:
        self.pending_offer = None
        self.last_panel = None
        try:
            return self._handle(intent, text)
        except Exception as exc:
            log.exception(f"file intent failed: {exc}")
            self._pending = None
            return "Something went wrong with that file request. Want to try again?"

    def _do_list_folder(self, text: str) -> str:
        """Speak the REAL contents of a folder. Fully deterministic — the LLM
        is never consulted, because a directory listing is a fact."""
        m = _LIST_FOLDER_RE.search(text.strip())
        spoken = next((g for g in m.groupdict().values() if g), "") if m else ""
        folder = fm.resolve_folder(spoken)
        if folder is None:
            return "I'm not sure which folder you mean."

        info = fm.list_folder(folder)
        label = folder.name or "home"
        if info["error"] == "permission":
            return (f"I don't have permission to read your {label} folder. You "
                    "can grant Nova access under Privacy and Security, Files and Folders.")
        if info["error"] == "missing":
            return f"I couldn't find a {label} folder."
        if info["error"]:
            return f"I wasn't able to read your {label} folder."

        nf, nd = info["n_files"], info["n_folders"]
        # Built before the early return: asking about an EMPTY folder must
        # replace the panel too, not leave the last one on screen while Nova
        # says it's empty.
        self.last_panel = ("files", _folder_panel(folder, label, info))
        if nf == 0 and nd == 0:
            return f"Your {label} folder is empty."

        parts = []
        if nd:
            parts.append(f"{nd} folder" + ("s" if nd != 1 else ""))
        if nf:
            parts.append(f"{nf} file" + ("s" if nf != 1 else ""))
        lead = f"You've got {_spoken_join(parts)} in {label}."

        # The panel got the WHOLE folder above; the voice gets four names.
        # Reading thirty filenames aloud is unusable, thirty rows on a screen
        # is just a list — the same trade as the weather week.
        names = [fm.spoken_name(str(folder / n)) for n in
                 (info["folders"] + info["files"])[:4]]
        if names:
            shown = _spoken_join(names)
            more = (nd + nf) - len(names)
            tail = f", and {more} more" if more > 0 else ""
            return f"{lead} There's {shown}{tail}."
        return lead

    def _handle(self, intent: str, text: str) -> str:
        t = (text or "").lower().strip()

        if intent == "list_folder":
            return self._do_list_folder(text)

        if intent == "file_delete":
            return ("I don't delete files. That's the one thing I won't do by "
                    "voice, since there's no undo if I get the wrong one. You "
                    "can drag it to the Trash yourself, or I can show you where "
                    "it is.")

        action = {
            "file_find": "find", "file_describe": "describe", "file_open": "open",
            "file_move": "move", "file_copy": "copy", "file_rename": "rename",
        }[intent]

        # What are we operating on?
        recent = self._recent_file()
        if recent and _PRONOUN_RE.search(t) and not _FILE_EXT_RE.search(t):
            candidates = [recent]
            log.info(f"follow-up pronoun resolves to {recent}")
        else:
            query = " ".join(self._query_tokens(t, intent))
            candidates = fm.search_file(query)
            if not candidates:
                return self._nothing_found()

        # Destination / new name, both deterministic.
        destination = None
        new_name = None
        if action in ("move", "copy"):
            spoken = self._extract_destination(t)
            if not spoken:
                return ("I didn't catch where you wanted it to go. Try again and "
                        "name the folder.")
            destination = fm.resolve_destination(spoken)
            if not destination:
                return (f"I couldn't find a folder called {spoken}. "
                        "Where should I put it?")
        elif action == "rename":
            new_name = self._extract_new_name(text)
            if not new_name:
                return "I didn't catch what you wanted to rename it to."

        if len(candidates) > 1:
            return self._ask_which(action, candidates, destination, new_name)
        return self._act_on(action, candidates[0], destination, new_name)

    # ── One candidate: do it, or ask to confirm first ───────────────────────
    def _act_on(self, action: str, path: str, destination: Optional[str],
                new_name: Optional[str]) -> str:
        if action == "find":
            return self._do_find(path)
        if action == "describe":
            return self._do_describe(path)
        if action == "open":
            return self._do_open(path)

        # move / copy / rename all change the filesystem: confirm first.
        if action in ("move", "copy") and destination:
            if Path(path).expanduser().parent.resolve() == Path(destination).resolve():
                where = Path(destination).name or "that folder"
                verb = "moving" if action == "move" else "copying"
                return (f"{fm.spoken_name(path)} is already in {where}, so "
                        f"{verb} it there wouldn't do anything. Did you mean "
                        "somewhere else?")

        self._pending = {
            "kind": "confirm", "action": action, "path": path,
            "destination": destination, "new_name": new_name, "rest": [],
        }
        return self._confirm_question(action, path, destination, new_name)

    def _confirm_question(self, action: str, path: str,
                          destination: Optional[str], new_name: Optional[str]) -> str:
        where = fm.folder_label(path)
        name = fm.spoken_name(path)
        if action == "move":
            return (f"I found {name}, {fm.spoken_kind(path)}, in {where}. "
                    f"Move it to {Path(destination or '').name}?")
        if action == "copy":
            return (f"I found {name}, {fm.spoken_kind(path)}, in {where}. "
                    f"Copy it to {Path(destination or '').name}?")
        return (f"I found {name}, {fm.spoken_kind(path)}, in {where}. "
                f"Rename it to {new_name}?")

    # ── Several candidates: ask which, before anything else ─────────────────
    def _ask_which(self, action: str, candidates: list[str],
                   destination: Optional[str], new_name: Optional[str]) -> str:
        shown = candidates[:4]
        self._pending = {
            "kind": "choose", "action": action, "candidates": shown,
            "destination": destination, "new_name": new_name,
        }
        parts = []
        for i, path in enumerate(shown, start=1):
            word = ("One", "Two", "Three", "Four")[i - 1]
            parts.append(f"{word}, {fm.spoken_name(path)} in {fm.folder_label(path)}.")
        # Say so when the list is truncated rather than implying it's complete.
        lead = (f"I found {self._count_word(len(candidates))} that could match, "
                f"here are the closest {self._count_word(len(shown))}. "
                if len(candidates) > len(shown) else
                f"I found {self._count_word(len(shown))} that could match. ")
        return lead + " ".join(parts) + " Which one?"

    @staticmethod
    def _count_word(n: int) -> str:
        return {2: "two files", 3: "three files", 4: "four files",
                5: "five files"}.get(n, f"{n} files")

    # ═══════════════════════════════════════════════════════════════════════
    # Answering Nova's pending question
    # ═══════════════════════════════════════════════════════════════════════
    def resolve_pending(self, text: str) -> Optional[str]:
        """Interpret a reply to the outstanding file question.

        Returns the spoken response, or None when the utterance is not an
        answer at all — in which case the question is dropped and the caller
        routes the utterance normally. Nothing is ever modified on a reply
        Nova did not clearly understand.
        """
        pending = self._pending
        if not pending:
            return None
        t = (text or "").lower().strip().rstrip(".?!,;")

        if _NO_RE.search(t):
            self._pending = None
            return "Okay, leaving it alone."

        if pending["kind"] == "confirm":
            if _YES_RE.search(t):
                self._pending = None
                return self._execute(pending["action"], pending["path"],
                                     pending.get("destination"),
                                     pending.get("new_name"))
            # Not a clear yes: drop the question rather than guess.
            self._pending = None
            return None

        # kind == "choose"
        chosen = self._match_choice(t, pending["candidates"])
        if chosen is None:
            self._pending = None
            return None
        self._pending = None
        return self._act_on(pending["action"], chosen,
                            pending.get("destination"), pending.get("new_name"))

    def _match_choice(self, t: str, candidates: list[str]) -> Optional[str]:
        """Resolve 'the second one' / 'two' / 'the budget one' to a candidate."""
        if re.search(r"\b(?:last|bottom)\s+one\b", t):
            return candidates[-1]
        for tier in _CHOICE_TIERS:
            for word, idx in tier.items():
                if re.search(rf"\b{re.escape(word)}\b", t) and idx <= len(candidates):
                    return candidates[idx - 1]
        # By name: the candidate whose filename contains a distinctive word
        # the user just said, and no other candidate does.
        said = set(fm.tokenize_query(t))
        if said:
            hits = [p for p in candidates
                    if said & set(fm.tokenize_query(Path(p).stem))]
            if len(hits) == 1:
                return hits[0]
        # By folder, for "the one in downloads".
        m = re.search(r"\bin\s+(?:my\s+|the\s+)?([\w'-]+)", t)
        if m:
            folder = m.group(1).lower()
            hits = [p for p in candidates if fm.folder_label(p).lower() == folder]
            if len(hits) == 1:
                return hits[0]
        return None

    # ═══════════════════════════════════════════════════════════════════════
    # The operations themselves
    # ═══════════════════════════════════════════════════════════════════════
    def _execute(self, action: str, path: str, destination: Optional[str],
                 new_name: Optional[str]) -> str:
        if action == "move":
            ok, msg = fm.move_file(path, destination or "")
            if ok:
                self._remember_file(msg)
                return f"Done. {fm.spoken_name(msg)} is in {fm.folder_label(msg)} now."
            return f"I couldn't move it, {msg}."

        if action == "copy":
            ok, msg = fm.copy_file(path, destination or "")
            if ok:
                self._remember_file(msg)
                return (f"Done. There's a copy in {fm.folder_label(msg)}, and the "
                        f"original is still in {fm.folder_label(path)}.")
            return f"I couldn't copy it, {msg}."

        if action == "rename":
            ok, msg = fm.rename_file(path, new_name or "")
            if ok:
                self._remember_file(msg)
                return f"Done. It's called {fm.spoken_name(msg)} now."
            return f"I couldn't rename it, {msg}."

        return "I'm not sure what to do with that."

    def _do_find(self, path: str) -> str:
        self._remember_file(path)
        self.pending_offer = lambda p=path: self._reveal(p)
        return (f"{fm.spoken_name(path)} is in {fm.folder_label(path)}. "
                f"It's {fm.spoken_kind(path)}, {self._size_phrase(path)}. "
                "Want me to show you?")

    def _reveal(self, path: str) -> str:
        if fm.reveal_in_finder(path):
            return "There it is."
        return "I couldn't open a Finder window for that."

    def _do_open(self, path: str) -> str:
        self._remember_file(path)
        if fm.open_file(path):
            return f"Opening {fm.spoken_name(path)}."
        return f"I couldn't open {fm.spoken_name(path)}."

    def _do_describe(self, path: str) -> str:
        self._remember_file(path)
        text, problem = fm.extract_text(path)

        if problem:
            # Nothing readable. Say what it IS rather than making something up.
            if fm.get_file_type(path) == "image":
                dims = fm.image_dimensions(path)
                size = f" It's {dims[0]} by {dims[1]} pixels." if dims else ""
                self.pending_offer = lambda p=path: self._open_offer(p)
                return (f"{problem}{size} It's in {fm.folder_label(path)}, "
                        f"{self._size_phrase(path)}. Want me to open it?")
            return f"{problem} It's in {fm.folder_label(path)}."

        prompt = (
            "Give a brief spoken summary of this file's contents, two to four "
            "sentences, conversational. Describe only what is written. If it's "
            "mostly data or code, say what it is and what it's for rather than "
            "reading it out.\n\n"
            f"FILE: {Path(path).name}\n\nCONTENT:\n{text}"
        )
        try:
            # temperature 0: this is a faithfulness task, not a creative one.
            summary = self.llm.generate(self._file_system, [], prompt,
                                        temperature=0.0, max_tokens=220).strip()
        except Exception as exc:
            log.warning(f"summary generation failed: {exc}")
            summary = ""

        if not summary:
            # Deterministic fallback — never silence, never invention.
            words = len(text.split())
            return (f"{fm.spoken_name(path)} is {fm.spoken_kind(path)} in "
                    f"{fm.folder_label(path)}, about {words} words. "
                    "I wasn't able to summarize it.")
        return _clean_spoken(summary)

    def _open_offer(self, path: str) -> str:
        return self._do_open(path)

    @staticmethod
    def _size_phrase(path: str) -> str:
        try:
            return fm.human_size(Path(path).stat().st_size)
        except OSError:
            return "unknown size"

    def _nothing_found(self) -> str:
        denied = fm.get_last_permission_errors()
        if denied:
            return ("I can't see your files right now. Nova needs access to "
                    "your folders in System Settings, under Privacy and "
                    "Security, then I'll be able to look again.")
        return "I couldn't find anything matching that. Can you be more specific?"

    # ═══════════════════════════════════════════════════════════════════════
    # Extraction helpers (deterministic)
    # ═══════════════════════════════════════════════════════════════════════
    def _query_tokens(self, t: str, intent: str) -> list[str]:
        """Distinctive search tokens from the utterance.

        For a rename the new name is cut off first, otherwise "rename the
        groceries file to shopping list" would search for a file matching
        groceries AND shopping AND list, and find nothing.
        """
        source = t
        if intent == "file_rename":
            m = _RENAME_SPLIT_RE.search(t)
            if m:
                source = m.group(1)
        elif intent in ("file_move", "file_copy"):
            # Cut the source and destination phrases off before tokenizing.
            # The stopword list covers the STANDARD folder names, but a folder
            # Nicholas invented does not appear in it — "move the budget file
            # to my projects folder" searched for a file matching budget AND
            # projects, and found nothing.
            source = _FROM_RE.sub(" ", source)
            dests = list(_DEST_RE.finditer(source))
            if dests:
                source = source[:dests[-1].start()]
        return fm.tokenize_query(source)

    def _extract_destination(self, t: str) -> Optional[str]:
        """The folder named after the FINAL to/into/onto, ignoring any
        'from <somewhere>' source phrase."""
        cleaned = _FROM_RE.sub(" ", t)
        matches = _DEST_RE.findall(cleaned)
        if not matches:
            return None
        for raw in reversed(matches):
            cand = raw.strip()
            # The capture is up to two words, so "my projects folder" arrives as
            # "projects folder" — peel the folder word off and remember we saw it.
            named_folder = re.match(r"^(.*?)\s+(?:folder|directory)$", cand)
            if named_folder:
                cand = named_folder.group(1).strip()
            if not cand:
                continue
            head = cand.split()[0]
            # A bare word is only a destination if it's a standard folder;
            # otherwise the user had to say "folder" for us to treat it as one,
            # so that "send this to Mom" is never read as a filesystem move.
            if head in _KNOWN_DEST_WORDS:
                return head
            if named_folder:
                return cand
            if re.search(rf"\b{re.escape(cand)}\s+(?:folder|directory)\b", cleaned):
                return cand
        return None

    def _extract_new_name(self, text: str) -> Optional[str]:
        m = _RENAME_SPLIT_RE.search((text or "").strip())
        if not m:
            return None
        new = m.group(2).strip().rstrip(".?!")
        new = re.sub(r"^(?:the\s+|a\s+)", "", new, flags=re.I)
        new = re.sub(r"\s+(?:please|instead|now)$", "", new, flags=re.I)
        # Spoken names come through lowercase; title-case them the way the
        # calendar layer title-cases reminder titles.
        if new.islower():
            new = " ".join(w.capitalize() if len(w) > 2 else w for w in new.split())
            new = new[:1].upper() + new[1:]
        return new or None

    # ═══════════════════════════════════════════════════════════════════════
    # Follow-up memory
    # ═══════════════════════════════════════════════════════════════════════
    def _remember_file(self, path: str) -> None:
        self._last_file = {"path": path, "ts": time.time()}

    def _recent_file(self) -> Optional[str]:
        lf = self._last_file
        if not lf:
            return None
        if time.time() - lf["ts"] >= _FOLLOWUP_WINDOW_S:
            self._last_file = None
            return None
        if not Path(lf["path"]).exists():
            self._last_file = None
            return None
        return lf["path"]


def _spoken_join(items: list) -> str:
    items = [str(i) for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _clean_spoken(text: str) -> str:
    """Strip anything the LLM added that Nova would read aloud as punctuation
    (CLAUDE.md invariant 10: no markdown, no lists, no em dashes)."""
    text = re.sub(r"[*_`#]+", "", text)
    text = re.sub(r"^\s*[-•]\s*", "", text, flags=re.M)
    text = re.sub(r"^\s*\d+[.)]\s*", "", text, flags=re.M)
    text = text.replace("—", ", ").replace("–", ", ")
    # A spaced hyphen is a dash the model reached for anyway ("that's it, really
    # - the report just..."). Same treatment.
    text = re.sub(r"\s+-\s+", ", ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
