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
import threading
import time
from pathlib import Path

log = logging.getLogger("nova.rag")

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".py", ".swift", ".js", ".json", ".html"}
CHUNK_SIZE     = 500    # characters per chunk
CHUNK_OVERLAP  = 100    # overlap between chunks for context continuity
REINGEST_INTERVAL_S = 3600   # re-scan for new files every hour


class NovaRAG:
    def __init__(self, index_dir: Path, docs_dir: Path) -> None:
        self.index_dir = index_dir
        self.docs_dir  = docs_dir
        index_dir.mkdir(parents=True, exist_ok=True)

        import chromadb
        self._client = chromadb.PersistentClient(path=str(index_dir))
        self._collection = self._client.get_or_create_collection(
            name="nova_docs",
            metadata={"hnsw:space": "cosine"},
        )
        log.info(
            f"RAG collection: {self._collection.count()} chunks indexed"
        )

        # Initial ingestion + periodic re-scan
        t = threading.Thread(target=self._ingest_loop, daemon=True)
        t.start()

    # ── Query ─────────────────────────────────────────────────────────────────────
    def query(self, text: str, n_results: int = 3) -> str:
        """
        Return relevant document chunks for a query.
        Returns empty string if nothing indexed or no relevant results.
        """
        if self._collection.count() == 0:
            return ""

        n = min(n_results, self._collection.count())
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
        if not docs:
            return ""

        parts = []
        for doc, meta in zip(docs, metas):
            source = Path(meta.get("source", "")).name if meta else ""
            parts.append(f"[{source}]\n{doc}" if source else doc)

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
            if path.stat().st_size > 5 * 1024 * 1024:   # skip files > 5MB
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
