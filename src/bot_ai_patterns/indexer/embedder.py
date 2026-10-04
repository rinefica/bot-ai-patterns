"""Embedding generation using sentence-transformers (local, no API key needed)."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from .chunkers import Chunk

_MODEL_NAME = "all-MiniLM-L6-v2"  # 22 MB, 384-dim, fast & good quality


class Embedder:
    def __init__(self, model_name: str = _MODEL_NAME):
        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_embedding_dimension()

    def embed_chunks(self, chunks: list[Chunk], batch_size: int = 64) -> np.ndarray:
        """Return float32 matrix of shape (len(chunks), dim)."""
        texts = [c.text for c in chunks]
        vectors = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,   # cosine similarity via inner product
            convert_to_numpy=True,
        )
        return vectors.astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Return shape (1, dim) float32 array for a query string."""
        vec = self._model.encode([text], normalize_embeddings=True, convert_to_numpy=True)
        return vec.astype(np.float32)
