"""RAG agent: plain / rag / rerank / full (rewrite+rerank) modes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from bot_ai_patterns.client import get_client
from bot_ai_patterns.config import MODEL_URI
from .retriever import Retriever, SearchResult
from .reranker import Reranker, RankedResult
from .query_rewriter import QueryRewriter

_SYSTEM_BASE = (
    "Ты — помощник по пошиву одежды. Отвечай точно и по делу, "
    "опираясь на инструкции по пошиву. Если информации недостаточно — скажи об этом."
)

_SYSTEM_RAG = (
    "Ты — помощник по пошиву одежды. Тебе предоставлены фрагменты инструкций по пошиву. "
    "Отвечай ТОЛЬКО на основе этих фрагментов. "
    "Если ответ есть в контексте — дай точный ответ со ссылкой на источник. "
    "Если информации нет — честно скажи об этом."
)

_RAG_PROMPT = """\
Контекст из инструкций по пошиву:

{context}

---

Вопрос: {question}

Ответь на вопрос, опираясь на контекст выше. Укажи, из каких инструкций взята информация."""


@dataclass
class RagResponse:
    answer: str
    mode: str
    sources: list[SearchResult] = field(default_factory=list)
    ranked: list[RankedResult] = field(default_factory=list)
    rewritten_query: str = ""
    candidates_before: int = 0   # chunks before reranker filter
    candidates_after: int = 0    # chunks after reranker filter
    prompt_tokens: int = 0
    completion_tokens: int = 0


class RagAgent:
    def __init__(
        self,
        index_dir: str | Path,
        retrieve_k: int = 10,   # candidates for reranker
        final_k: int = 5,       # chunks sent to LLM
        threshold: float = 0.0, # cross-encoder score cutoff
        strategy: str = "structure",
    ):
        self._retriever = Retriever(index_dir, strategy=strategy)
        self._reranker = Reranker(threshold=threshold)
        self._rewriter = QueryRewriter()
        self._client = get_client()
        self._retrieve_k = retrieve_k
        self._final_k = final_k

    # ------------------------------------------------------------------
    # Internal helpers

    def _llm(self, system: str, user: str) -> tuple[str, int, int]:
        response = self._client.chat.completions.create(
            model=MODEL_URI,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=1024,
        )
        msg = response.choices[0].message.content or ""
        return msg, response.usage.prompt_tokens, response.usage.completion_tokens

    def _build_context(self, sources: list[SearchResult]) -> str:
        return self._retriever.format_context(sources)

    # ------------------------------------------------------------------
    # Public API

    def ask_plain(self, question: str) -> RagResponse:
        """No retrieval — raw LLM answer."""
        answer, pt, ct = self._llm(_SYSTEM_BASE, question)
        return RagResponse(answer=answer, mode="plain", prompt_tokens=pt, completion_tokens=ct)

    def ask_rag(self, question: str) -> RagResponse:
        """Bi-encoder retrieval only (no reranker)."""
        sources = self._retriever.search(question, top_k=self._final_k)
        prompt = _RAG_PROMPT.format(context=self._build_context(sources), question=question)
        answer, pt, ct = self._llm(_SYSTEM_RAG, prompt)
        return RagResponse(
            answer=answer, mode="rag", sources=sources,
            candidates_before=len(sources), candidates_after=len(sources),
            prompt_tokens=pt, completion_tokens=ct,
        )

    def ask_rerank(self, question: str) -> RagResponse:
        """Bi-encoder retrieval → cross-encoder rerank → filter."""
        candidates = self._retriever.search(question, top_k=self._retrieve_k)
        ranked = self._reranker.rerank(question, candidates, top_k=self._final_k)
        sources = [r.original for r in ranked]
        prompt = _RAG_PROMPT.format(context=self._build_context(sources), question=question)
        answer, pt, ct = self._llm(_SYSTEM_RAG, prompt)
        return RagResponse(
            answer=answer, mode="rerank", sources=sources, ranked=ranked,
            candidates_before=len(candidates), candidates_after=len(ranked),
            prompt_tokens=pt, completion_tokens=ct,
        )

    def ask_full(self, question: str) -> RagResponse:
        """Query rewrite → bi-encoder → cross-encoder rerank → filter."""
        rewritten = self._rewriter.rewrite(question)
        candidates = self._retriever.search(rewritten, top_k=self._retrieve_k)
        ranked = self._reranker.rerank(rewritten, candidates, top_k=self._final_k)
        sources = [r.original for r in ranked]
        # Answer prompt uses original question but retrieved with rewritten
        prompt = _RAG_PROMPT.format(context=self._build_context(sources), question=question)
        answer, pt, ct = self._llm(_SYSTEM_RAG, prompt)
        return RagResponse(
            answer=answer, mode="full", sources=sources, ranked=ranked,
            rewritten_query=rewritten,
            candidates_before=len(candidates), candidates_after=len(ranked),
            prompt_tokens=pt, completion_tokens=ct,
        )
