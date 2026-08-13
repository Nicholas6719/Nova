"""
Nova RAG — Local document retrieval via ChromaDB.

100% on-device. Your documents never leave your machine.

Supports: .txt, .md, .pdf, .py, .swift, .js, .json
Watches: ~/Documents by default (configurable via config.json)

Ingestion runs in the background on startup and re-runs periodically.
query() returns relevant text chunks for LLM context enrichment.
"""

from __future__ import annotations

import logging
import shutil
import threading
import time
from pathlib import Path

log = logging.getLogger("nova.rag")

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".py", ".swift", ".js", ".json", ".html"}
CHUNK_SIZE     = 500    # characters per chunk
CHUNK_OVERLAP  = 100    # overlap between chunks for context continuity
REINGEST_INTERVAL_S = 3600   # re-scan for new files every hour

# Directories that are machine-generated dependencies or build output, not
# anything Nicholas wrote. Measured under ~/Documents: 14,699 of 15,059
# indexable files lived in these (Jarvis's node_modules plus a Python venv)
# against 360 real documents — so 98% of the index was third-party source.
# The retrieved "personal documents" were things like base64 blobs from
# draco_encoder.js and glyph tables from a font, injected into the prompt on
# every single conversational turn.
EXCLUDED_DIRS = {
    "node_modules", "site-packages", "dist-packages", "bower_components",
    ".venv", "venv", "env", ".env", "virtualenv",
    "build", "dist", "out", "target", ".build", "DerivedData", "Pods",
    "Carthage", ".next", ".nuxt", "vendor",
    ".git", ".hg", ".svn", "__pycache__", ".mypy_cache", ".pytest_cache",
    ".tox", ".cache", ".gradle", ".idea", ".terraform",
}

# Bump when SUPPORTED_EXTENSIONS or EXCLUDED_DIRS change: the stored index was
# built under the OLD rules, and excluding a directory going forward does not
# remove what is already in there. Changing the rules therefore has to rebuild.
INGEST_RULES_VERSION = 2

# Cosine distance above which a "match" is not worth putting in front of the
# model. Chroma always returns the nearest k chunks no matter how far away they
# are, so without this every chatty turn ("how are you doing") still retrieved
# its three nearest neighbours out of the whole corpus and presented them to a
# 3B model as relevant personal context.
#
# Swept over 40 conversational utterances and 8 real document questions. The
# two overlap — "what's the best way to fall asleep" (0.451) sits nearer than
# several genuine document questions — so there is no threshold that is right
# every time, only a trade:
#     0.45 -> 0 of 40 chatty turns inject noise, 6 of 8 document questions hit
#     0.50 -> 1 of 40 inject noise,              6 of 8 hit   (strictly worse)
#     0.60 -> 5 of 40 inject noise,              8 of 8 hit
# 0.45 is chosen because nearly every utterance is conversational, and noise
# there costs Nicholas answer quality on EVERY turn, while a miss costs one
# retrieval he can get by asking again more specifically. Tune via
# memory.rag_max_distance in config.json.
DEFAULT_MAX_DISTANCE = 0.45


def _is_excluded(path: Path, root: Path) -> bool:
    """True for dependency/build output, and for anything under a hidden
    directory. Only the part BELOW the docs root is examined — a docs_dir that
    itself sits under a dotted path must not exclude everything."""
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = path
    for part in rel.parts[:-1]:
        if part in EXCLUDED_DIRS or (part.startswith(".") and part not in (".", "..")):
            return True
    return False


class NovaRAG:
    def __init__(self, index_dir: Path, docs_dir: Path,
                 max_distance: float = DEFAULT_MAX_DISTANCE) -> None:
        self.index_dir    = index_dir
        self.docs_dir     = docs_dir
        self.max_distance = max_distance
        index_dir.mkdir(parents=True, exist_ok=True)

        self._open_collection()

        # Initial ingestion + periodic re-scan
        t = threading.Thread(target=self._ingest_loop, daemon=True)
        t.start()

    def _open_collection(self, _retry: bool = True) -> None:
        """Open the Chroma collection, rebuilding it once if it is corrupt.

        The index is DERIVED data — every chunk can be re-read from the user's
        documents — so a damaged index is a rebuild, not a loss. Chroma has no
        concurrent-writer support, and two Nova backends pointed at the same
        directory (an app instance plus a manually started one) leaves it
        raising `Failed to apply logs to the hnsw segment writer` forever. Left
        alone that silently disables RAG for good, since every later load hits
        the same corrupt files.
        """
        import chromadb

        try:
            self._client = chromadb.PersistentClient(path=str(self.index_dir))
            self._collection = self._client.get_or_create_collection(
                name="nova_docs",
                metadata={"hnsw:space": "cosine"},
            )
            log.info(f"RAG collection: {self._collection.count()} chunks indexed")
            self._rebuild_if_rules_changed()
            return
        except Exception as exc:
            if not _retry:
                raise
            log.warning(f"RAG index unreadable ({exc}); rebuilding from scratch.")

        # Move the damaged index aside rather than deleting outright, so a bad
        # diagnosis is recoverable, then start clean. The stale copy from a
        # previous rebuild is what we discard.
        try:
            self._client = None
            self._collection = None
            # Chroma caches one system per path, so simply constructing a new
            # PersistentClient for the same directory hands back the SAME broken
            # system — the retry then fails even against an empty directory.
            # Measured exactly that. Drop the cache before reopening.
            try:
                from chromadb.api.shared_system_client import SharedSystemClient
                SharedSystemClient.clear_system_cache()
            except Exception as exc:
                log.debug(f"could not clear chroma system cache: {exc}")
            broken = self.index_dir.with_name(self.index_dir.name + ".broken")
            if broken.exists():
                shutil.rmtree(broken, ignore_errors=True)
            self.index_dir.rename(broken)
            self.index_dir.mkdir(parents=True, exist_ok=True)
            log.warning(f"Damaged RAG index moved to {broken.name}; "
                        "it will re-ingest in the background.")
        except Exception as exc:
            log.error(f"Could not reset the RAG index: {exc}")
            raise

        self._open_collection(_retry=False)

    def _rules_file(self) -> Path:
        return self.index_dir / "ingest_rules.json"

    def _rebuild_if_rules_changed(self) -> None:
        """Drop the collection when the ingest rules have changed.

        Narrowing what gets indexed does nothing to what is already stored, so
        without this the excluded third-party source would keep being retrieved
        forever. The index is derived data — every chunk can be re-read from
        disk — so throwing it away costs only a background re-ingest.
        """
        import json

        current = {
            "version": INGEST_RULES_VERSION,
            "extensions": sorted(SUPPORTED_EXTENSIONS),
            "excluded": sorted(EXCLUDED_DIRS),
        }
        try:
            stored = json.loads(self._rules_file().read_text())
        except Exception:
            stored = None

        if stored == current:
            return

        n_before = 0
        try:
            n_before = self._collection.count()
        except Exception:
            pass

        if n_before:
            log.warning(
                f"RAG ingest rules changed; discarding {n_before} chunks indexed "
                "under the old rules and re-ingesting in the background."
            )
            try:
                self._client.delete_collection("nova_docs")
                self._collection = self._client.get_or_create_collection(
                    name="nova_docs", metadata={"hnsw:space": "cosine"},
                )
            except Exception as exc:
                log.error(f"Could not reset the RAG collection: {exc}")
                return

        try:
            self._rules_file().write_text(json.dumps(current, indent=2))
        except Exception as exc:
            log.debug(f"could not record ingest rules: {exc}")

    # ── Query ─────────────────────────────────────────────────────────────────────
    def query(self, text: str, n_results: int = 3) -> str:
        """
        Return relevant document chunks for a query, or "" when nothing indexed
        is actually relevant.

        The relevance test is the point. Chroma returns the nearest k chunks
        unconditionally, so this used to hand the model its three closest
        neighbours for every utterance — "how are you doing" retrieved font
        glyph tables and told the model they were relevant personal context.
        """
        try:
            total = self._collection.count()
        except Exception as exc:
            log.debug(f"RAG count failed: {exc}")
            return ""
        if total == 0:
            return ""

        n = min(n_results, total)
        try:
            results = self._collection.query(
                query_texts=[text],
                n_results=n,
            )
        except Exception as exc:
            log.debug(f"RAG query failed: {exc}")
            return ""

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = (results.get("distances") or [[]])[0]
        if not docs:
            return ""

        parts = []
        for i, doc in enumerate(docs):
            # No distance reported (older Chroma): keep the old behaviour rather
            # than silently dropping every result.
            if i < len(dists) and dists[i] > self.max_distance:
                continue
            meta = metas[i] if i < len(metas) else None
            source = Path(meta.get("source", "")).name if meta else ""
            parts.append(f"[{source}]\n{doc}" if source else doc)

        if not parts:
            log.debug(f"RAG: nothing relevant for {text!r} "
                      f"(nearest {min(dists):.3f})" if dists else "RAG: no match")
            return ""

        return "\n---\n".join(parts)

    # ── Ingestion ─────────────────────────────────────────────────────────────────
    def _ingest_loop(self) -> None:
        while True:
            self._ingest_docs()
            time.sleep(REINGEST_INTERVAL_S)

    def _ingest_docs(self) -> None:
        if not self.docs_dir.exists():
            log.warning(f"RAG docs dir not found: {self.docs_dir}")
            return

        new_count = 0
        for path in self.docs_dir.rglob("*"):
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            if _is_excluded(path, self.docs_dir):
                continue
            try:
                if path.stat().st_size > 5 * 1024 * 1024:   # skip files > 5MB
                    continue
            except OSError:
                continue
            try:
                self._ingest_file(path)
                new_count += 1
            except Exception as exc:
                log.debug(f"Skipped {path.name}: {exc}")

        if new_count:
            log.info(f"RAG: processed {new_count} files ({self._collection.count()} total chunks)")

    def _ingest_file(self, path: Path) -> None:
        """Read, chunk, and index a single file. Skips if already indexed."""
        doc_id = str(path.resolve())
        mtime  = str(path.stat().st_mtime)

        # Check if already indexed with same mtime
        existing = self._collection.get(where={"source": doc_id}, limit=1)
        if existing["ids"]:
            existing_mtime = existing["metadatas"][0].get("mtime", "")
            if existing_mtime == mtime:
                return   # unchanged — skip
            # File changed — delete old chunks
            self._collection.delete(where={"source": doc_id})

        text = self._read_file(path)
        if not text or len(text.strip()) < 50:
            return

        chunks = _chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        if not chunks:
            return

        ids   = [f"{doc_id}::{i}" for i in range(len(chunks))]
        metas = [{"source": doc_id, "filename": path.name, "mtime": mtime}
                 for _ in chunks]

        self._collection.add(
            documents=chunks,
            ids=ids,
            metadatas=metas,
        )

    def _read_file(self, path: Path) -> str:
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            try:
                import pypdf
                reader = pypdf.PdfReader(str(path))
                return " ".join(
                    (page.extract_text() or "") for page in reader.pages
                )
            except Exception:
                return ""

        if suffix == ".json":
            try:
                import json
                with open(path) as f:
                    data = json.load(f)
                return json.dumps(data, indent=2)
            except Exception:
                pass

        return path.read_text(errors="ignore")


# ── Text chunking ─────────────────────────────────────────────────────────────────
def _chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks for better retrieval coverage."""
    chunks = []
    start  = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end].strip())
        start += size - overlap
    return [c for c in chunks if len(c) >= 30]
