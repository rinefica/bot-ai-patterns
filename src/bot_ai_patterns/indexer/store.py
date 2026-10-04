"""FAISS index + JSON metadata store."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import faiss
import numpy as np

from .chunkers import Chunk


class IndexStore:
    """
    Stores vectors in a FAISS IndexFlatIP (inner product = cosine for normalized vectors).
    Metadata (all Chunk fields except text) is kept in a parallel JSON file.
    """

    def __init__(self, index_dir: str | Path):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._index: faiss.IndexFlatIP | None = None
        self._meta: list[dict] = []

    # ------------------------------------------------------------------
    # Building

    def build(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        """Create FAISS index from chunks and their vectors."""
        dim = vectors.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(vectors)
        self._meta = [
            {
                "chunk_id": c.chunk_id,
                "source": c.source,
                "title": c.title,
                "section": c.section,
                "strategy": c.strategy,
                "char_start": c.char_start,
                "char_len": c.char_len,
                "text": c.text,
                **c.metadata,
            }
            for c in chunks
        ]

    # ------------------------------------------------------------------
    # Persistence

    def save(self, name: str) -> None:
        """Save index and metadata under data/index/<name>.*"""
        faiss_path = self.index_dir / f"{name}.faiss"
        meta_path = self.index_dir / f"{name}.json"
        faiss.write_index(self._index, str(faiss_path))
        meta_path.write_text(json.dumps(self._meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved → {faiss_path} + {meta_path}")

    def load(self, name: str) -> None:
        faiss_path = self.index_dir / f"{name}.faiss"
        meta_path = self.index_dir / f"{name}.json"
        self._index = faiss.read_index(str(faiss_path))
        self._meta = json.loads(meta_path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # Search

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> list[dict]:
        """Return top-k metadata dicts sorted by cosine similarity."""
        scores, indices = self._index.search(query_vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            entry = dict(self._meta[idx])
            entry["score"] = float(score)
            results.append(entry)
        return results

    # ------------------------------------------------------------------
    # Stats

    def stats(self) -> dict:
        if not self._meta:
            return {}
        char_lens = [m["char_len"] for m in self._meta]
        return {
            "total_chunks": len(self._meta),
            "total_chars": sum(char_lens),
            "avg_chars": round(sum(char_lens) / len(char_lens), 1),
            "min_chars": min(char_lens),
            "max_chars": max(char_lens),
            "unique_sources": len({m["source"] for m in self._meta}),
        }
