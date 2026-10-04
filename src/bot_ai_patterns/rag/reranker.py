"""Cross-encoder reranker + relevance filter."""

from __future__ import annotations

from dataclasses import dataclass

from sentence_transformers import CrossEncoder

from .retriever import SearchResult

_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class RankedResult:
    original: SearchResult
    bi_score: float      # original cosine score from FAISS
    cross_score: float   # cross-encoder score


class Reranker:
    """
    Two-stage reranker:
    1. Score every (query, chunk) pair with a cross-encoder.
    2. Filter out pairs below `threshold`.
    3. Return top-k by cross-encoder score.
    """

    def __init__(self, model_name: str = _MODEL, threshold: float = 0.0):
        self._model = CrossEncoder(model_name)
        self.threshold = threshold

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int = 5,
    ) -> list[RankedResult]:
        if not candidates:
            return []

        pairs = [(query, c.text) for c in candidates]
        scores = self._model.predict(pairs)

        ranked = sorted(
            [
                RankedResult(original=c, bi_score=c.score, cross_score=float(s))
                for c, s in zip(candidates, scores)
            ],
            key=lambda r: r.cross_score,
            reverse=True,
        )

        # Apply threshold filter
        filtered = [r for r in ranked if r.cross_score >= self.threshold]
        return filtered[:top_k]
