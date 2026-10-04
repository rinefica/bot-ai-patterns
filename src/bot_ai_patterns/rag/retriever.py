"""Retriever: embed query → search FAISS → return ranked chunks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bot_ai_patterns.indexer.embedder import Embedder
from bot_ai_patterns.indexer.store import IndexStore


@dataclass
class SearchResult:
    chunk_id: str
    score: float
    title: str
    section: str
    text: str
    source: str


class Retriever:
    def __init__(self, index_dir: str | Path, strategy: str = "structure"):
        self._store = IndexStore(index_dir)
        self._store.load(strategy)
        self._embedder = Embedder()
        self.strategy = strategy

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        vec = self._embedder.embed_query(query)
        raw = self._store.search(vec, top_k=top_k)
        return [
            SearchResult(
                chunk_id=r["chunk_id"],
                score=r["score"],
                title=r["title"],
                section=r["section"],
                text=r["text"],
                source=r["source"],
            )
            for r in raw
        ]

    def format_context(self, results: list[SearchResult]) -> str:
        """Format retrieved chunks into a context block for the prompt."""
        parts = []
        for i, r in enumerate(results, 1):
            header = f"[{i}] {r.title}"
            if r.section:
                header += f" / {r.section}"
            parts.append(f"{header}\n{r.text.strip()}")
        return "\n\n---\n\n".join(parts)
