#!/usr/bin/env python3
"""
RAG tests — Nova must only quote a document when it actually has one.

Chroma returns the nearest k chunks no matter how far away they are, so RAG
used to inject its three closest neighbours on EVERY conversational turn. With
~/Documents indexed wholesale that meant base64 from draco_encoder.js and glyph
tables from a font, handed to a 3B model under the heading "relevant content
from Nicholas's personal documents". It cost answer quality on every turn, not
just latency.

Fidelity: real rag.py, real on-disk index (the one Nova queries). The
exclusion rules are checked as pure functions and need no index at all.
"""
from __future__ import annotations

import os
import socket
import sys
from pathlib import Path as _Path

TESTS_DIR = _Path(__file__).resolve().parent
BACKEND = str(TESTS_DIR.parent)
sys.path.insert(0, BACKEND)
sys.path.insert(0, str(TESTS_DIR))
os.chdir(BACKEND)

PASS = FAIL = 0
FAILURES: list[str] = []


def check(cond, label, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {label}")
    else:
        FAIL += 1
        FAILURES.append(label)
        print(f"  FAIL  {label}")
    if detail:
        print(f"        {detail}")


def section(t):
    print("\n" + "=" * 70)
    print(t)
    print("=" * 70)


import rag as rag_mod
from rag import _is_excluded, DEFAULT_MAX_DISTANCE, EXCLUDED_DIRS

# ══════════════════════════════════════════════════════════════════════════
section("WHAT MAY BE INDEXED  (pure rules — no index needed)")
# ══════════════════════════════════════════════════════════════════════════
ROOT = _Path("/Users/nick/Documents")

MUST_SKIP = [
    "Coding_Projects/Jarvis/frontend/node_modules/three/build/three.webgpu.js",
    "Coding_Projects/Jarvis/.venv311/lib/python3.11/site-packages/hf_api.py",
    "Coding_Projects/Jarvis/frontend/dist/assets/index.js",
    "Coding_Projects/MyAgent/.git/config.json",
    "Projects/App/build/output.json",
    "Projects/App/DerivedData/Build/x.swift",
    "Projects/thing/__pycache__/mod.py",
    "Projects/ios/Pods/Alamofire/Source/Response.swift",
]
for rel in MUST_SKIP:
    check(_is_excluded(ROOT / rel, ROOT), f"skipped: {rel[:58]}")

MUST_KEEP = [
    "resume.pdf",
    "notes/Nova architecture.md",
    "Coding_Projects/Nova/nova_backend/memory.py",
    "School/ACC 101 Syllabus.pdf",
    "finance-degree-roadmap.html",
]
for rel in MUST_KEEP:
    check(not _is_excluded(ROOT / rel, ROOT), f"indexed: {rel[:58]}")

# A docs_dir that itself lives under a dotted or excluded path must not
# exclude every file inside it — only the parts BELOW the root count.
odd_root = _Path("/Users/nick/.config/mydocs")
check(not _is_excluded(odd_root / "notes.md", odd_root),
      "a docs_dir under a hidden path still indexes its own files")
check(_is_excluded(odd_root / "node_modules/x.js", odd_root),
      "…but still skips dependencies inside it")

check("node_modules" in EXCLUDED_DIRS and "site-packages" in EXCLUDED_DIRS,
      "the two directories that were 98% of the corpus are excluded")


# ══════════════════════════════════════════════════════════════════════════
section("RELEVANCE  (real index — what Nova will actually retrieve)")
# ══════════════════════════════════════════════════════════════════════════
# Chroma has no concurrent-writer support: opening the index while the app
# holds it corrupts it, and that has already cost a rebuild once.
def _port_busy(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.4)
        return s.connect_ex(("127.0.0.1", port)) == 0


if _port_busy(5001):
    print("\n  SKIPPED — Nova is running (port 5001 is in use).")
    print("  Chroma has no concurrent-writer support; opening the index while")
    print("  the app holds it corrupts it. Quit Nova and re-run.")
    print(f"\n  {PASS}/{PASS + FAIL} (rule checks only)")
    sys.exit(1 if FAIL else 0)

import json

config = json.loads((_Path(BACKEND) / "config.json").read_text())
DATA_DIR = _Path(os.environ.get(
    "NOVA_DATA_DIR", _Path.home() / "Library/Application Support/Nova")).expanduser()
index_dir = DATA_DIR / "rag_index"

if not index_dir.exists():
    print("\n  SKIPPED — no RAG index on this machine yet.")
    print(f"\n  {PASS}/{PASS + FAIL} (rule checks only)")
    sys.exit(1 if FAIL else 0)

from rag import NovaRAG

r = NovaRAG(index_dir, docs_dir=_Path(config["memory"]["rag_docs_dir"]).expanduser(),
            max_distance=config["memory"].get("rag_max_distance", DEFAULT_MAX_DISTANCE))
total = r._collection.count()
print(f"  index holds {total} chunks")

if total == 0:
    print("\n  SKIPPED — index is empty (still re-ingesting in the background).")
    print(f"\n  {PASS}/{PASS + FAIL} (rule checks only)")
    sys.exit(1 if FAIL else 0)

# Nothing from an excluded directory may survive in the stored index.
sample = r._collection.get(limit=3000, include=["metadatas"])
leaked = [m.get("source", "") for m in sample["metadatas"]
          if set(_Path(m.get("source", "")).parts) & EXCLUDED_DIRS]
check(not leaked, "no chunk in the index comes from an excluded directory",
      "" if not leaked else f"{len(leaked)} leaked, e.g. {leaked[0][:90]}")

# Ordinary conversation must retrieve NOTHING.
CHATTY = [
    "how are you doing", "good morning", "tell me a joke",
    "what do you think about the weather today", "tell me something interesting",
    "what's the meaning of life", "are you there", "what's new",
    "do you ever get bored", "I had a long day", "say something funny",
    "what's your favourite colour",
]
noisy = [q for q in CHATTY if r.query(q, n_results=3)]
check(not noisy, f"ordinary conversation retrieves nothing ({len(CHATTY)} phrases)",
      "" if not noisy else f"these injected document text: {noisy}")

# And a real document question must still work, or the feature is pointless.
DOCSEEK = [
    "what is in my notes about the Nova architecture",
    "what does my finance degree roadmap say",
    "how does my memory module store facts",
]
hits = [q for q in DOCSEEK if r.query(q, n_results=3)]
check(len(hits) >= 2,
      f"document questions still retrieve ({len(hits)}/{len(DOCSEEK)})",
      f"missed: {[q for q in DOCSEEK if q not in hits]}")

# The threshold must be doing the work — not an empty index faking a pass.
raw = r._collection.query(query_texts=["how are you doing"], n_results=3)
check(len(raw["documents"][0]) > 0,
      "the index DOES return neighbours for chatty text (threshold is what filters)",
      f"nearest distance {min(raw['distances'][0]):.3f} "
      f"> max_distance {r.max_distance}")


# ══════════════════════════════════════════════════════════════════════════
section("RESULT")
# ══════════════════════════════════════════════════════════════════════════
print(f"\n  {PASS}/{PASS + FAIL}")
for f in FAILURES:
    print(f"    ✗ {f}")
sys.exit(1 if FAIL else 0)
