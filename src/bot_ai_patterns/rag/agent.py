"""RAG agent: two modes — with retrieval and without."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from bot_ai_patterns.client import get_client
from bot_ai_patterns.config import MODEL_URI
from .retriever import Retriever, SearchResult

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
    mode: str          # "rag" | "plain"
    sources: list[SearchResult] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0


class RagAgent:
    def __init__(self, index_dir: str | Path, top_k: int = 5, strategy: str = "structure"):
        self._retriever = Retriever(index_dir, strategy=strategy)
        self._client = get_client()
        self._top_k = top_k

    # ------------------------------------------------------------------
    # Public API

    def ask_plain(self, question: str) -> RagResponse:
        """Ask LLM without any retrieval."""
        response = self._client.chat.completions.create(
            model=MODEL_URI,
            messages=[
                {"role": "system", "content": _SYSTEM_BASE},
                {"role": "user", "content": question},
            ],
            max_tokens=1024,
        )
        msg = response.choices[0].message.content or ""
        return RagResponse(
            answer=msg,
            mode="plain",
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
        )

    def ask_rag(self, question: str) -> RagResponse:
        """Retrieve relevant chunks, then ask LLM with augmented prompt."""
        sources = self._retriever.search(question, top_k=self._top_k)
        context = self._retriever.format_context(sources)
        prompt = _RAG_PROMPT.format(context=context, question=question)

        response = self._client.chat.completions.create(
            model=MODEL_URI,
            messages=[
                {"role": "system", "content": _SYSTEM_RAG},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1024,
        )
        msg = response.choices[0].message.content or ""
        return RagResponse(
            answer=msg,
            mode="rag",
            sources=sources,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
        )
