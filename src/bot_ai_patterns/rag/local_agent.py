"""
LocalRagAgent: RagAgent с локальной LLM вместо облака.

Retrieval — без изменений (FAISS + sentence-transformers, уже локально).
Generation — qwen2.5:7b через Ollama (LocalLLMClient).
"""

from __future__ import annotations

from pathlib import Path

from bot_ai_patterns.local_llm import LocalLLMClient
from .agent import RagAgent, RagResponse, _SYSTEM_BASE, _SYSTEM_RAG, _RAG_PROMPT
from .retriever import SearchResult


class LocalRagAgent(RagAgent):
    """RagAgent с локальной LLM (Ollama)."""

    def __init__(
        self,
        index_dir: str | Path,
        model: str = "qwen2.5:7b",
        retrieve_k: int = 10,
        final_k: int = 5,
        threshold: float = 0.0,
        strategy: str = "structure",
    ):
        # Инициализируем родительский класс (retriever, reranker, rewriter)
        super().__init__(
            index_dir=index_dir,
            retrieve_k=retrieve_k,
            final_k=final_k,
            threshold=threshold,
            strategy=strategy,
        )
        # Заменяем облачный клиент на локальный
        self._client = LocalLLMClient(model=model)
        self._model = model

    def _llm(self, system: str, user: str) -> tuple[str, int, int]:
        """Перекрываем метод родителя — используем LocalLLMClient."""
        from bot_ai_patterns.config import MAX_TOKENS
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=MAX_TOKENS,
        )
        msg = response.choices[0].message.content or ""
        return msg, response.usage.prompt_tokens, response.usage.completion_tokens
