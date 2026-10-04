"""
CitedAgent — RAG с обязательными источниками, цитатами и режимом "не знаю".

Возвращает структурированный JSON-ответ:
  answer        — ответ на вопрос
  quotes        — список цитат из чанков (точный текст)
  sources       — список источников (файл / секция)
  confidence    — "high" | "low" | "none"
  no_answer_reason — причина, если confidence == "none"
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from bot_ai_patterns.client import get_client
from bot_ai_patterns.config import MODEL_URI
from .reranker import Reranker, RankedResult
from .retriever import Retriever, SearchResult
from .query_rewriter import QueryRewriter

# Порог cross-encoder: если лучший score ниже — говорим "не знаю"
DEFAULT_NO_ANSWER_THRESHOLD = 1.0

_SYSTEM = """\
Ты — точный ассистент по пошиву одежды. Работаешь ТОЛЬКО с предоставленными фрагментами.

Правила:
1. Если в контексте есть ответ — дай его, ОБЯЗАТЕЛЬНО процитировав точные фразы из контекста.
2. Цитаты должны быть дословными фрагментами из текста контекста (не перефразируй).
3. Если контекст не содержит ответа — установи confidence=none и объясни почему.
4. Никогда не придумывай факты вне контекста.

Отвечай СТРОГО в формате JSON (без markdown-обёртки):
{
  "answer": "<ответ на вопрос, опираясь на контекст>",
  "quotes": [
    {"text": "<дословная цитата из контекста>", "source": "<имя файла>", "section": "<секция>"}
  ],
  "sources": ["<файл / секция>"],
  "confidence": "<high|low|none>",
  "no_answer_reason": "<причина, только если confidence=none, иначе null>"
}"""

_USER_TEMPLATE = """\
Контекст из инструкций по пошиву:

{context}

---

Вопрос: {question}"""


@dataclass
class Quote:
    text: str
    source: str
    section: str


@dataclass
class CitedResponse:
    answer: str
    quotes: list[Quote]
    sources: list[str]
    confidence: str          # "high" | "low" | "none"
    no_answer_reason: str | None
    rewritten_query: str
    best_cross_score: float
    candidates_before: int
    candidates_after: int
    ranked: list[RankedResult] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0


class CitedAgent:
    def __init__(
        self,
        index_dir: str | Path,
        retrieve_k: int = 10,
        final_k: int = 5,
        rerank_threshold: float = 0.0,
        no_answer_threshold: float = DEFAULT_NO_ANSWER_THRESHOLD,
        strategy: str = "structure",
    ):
        self._retriever = Retriever(index_dir, strategy=strategy)
        self._reranker = Reranker(threshold=rerank_threshold)
        self._rewriter = QueryRewriter()
        self._client = get_client()
        self._retrieve_k = retrieve_k
        self._final_k = final_k
        self._no_answer_threshold = no_answer_threshold

    def ask(self, question: str) -> CitedResponse:
        # 1. Query rewrite
        rewritten = self._rewriter.rewrite(question)

        # 2. Bi-encoder retrieval
        candidates = self._retriever.search(rewritten, top_k=self._retrieve_k)

        # 3. Cross-encoder rerank + filter
        ranked = self._reranker.rerank(rewritten, candidates, top_k=self._final_k)

        best_score = ranked[0].cross_score if ranked else -999.0

        # 4. If below no-answer threshold — skip LLM, return "не знаю"
        if best_score < self._no_answer_threshold or not ranked:
            return CitedResponse(
                answer="",
                quotes=[],
                sources=[],
                confidence="none",
                no_answer_reason=(
                    f"Ни один из найденных фрагментов не достиг порога релевантности "
                    f"(лучший score: {best_score:.2f}, порог: {self._no_answer_threshold}). "
                    "Уточните вопрос или переформулируйте его."
                ),
                rewritten_query=rewritten,
                best_cross_score=best_score,
                candidates_before=len(candidates),
                candidates_after=len(ranked),
                ranked=ranked,
            )

        # 5. Build context with source labels
        sources_used = [r.original for r in ranked]
        context = self._retriever.format_context(sources_used)

        # 6. LLM call with JSON mode
        response = self._client.chat.completions.create(
            model=MODEL_URI,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": _USER_TEMPLATE.format(
                    context=context, question=question
                )},
            ],
            response_format={"type": "json_object"},
            max_tokens=2048,
        )
        raw = response.choices[0].message.content or "{}"

        # 7. Parse JSON
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {"answer": raw, "quotes": [], "sources": [], "confidence": "low",
                    "no_answer_reason": None}

        quotes = [
            Quote(
                text=q.get("text", ""),
                source=q.get("source", ""),
                section=q.get("section", ""),
            )
            for q in data.get("quotes", [])
            if q.get("text")
        ]

        return CitedResponse(
            answer=data.get("answer", ""),
            quotes=quotes,
            sources=data.get("sources", []),
            confidence=data.get("confidence", "low"),
            no_answer_reason=data.get("no_answer_reason"),
            rewritten_query=rewritten,
            best_cross_score=best_score,
            candidates_before=len(candidates),
            candidates_after=len(ranked),
            ranked=ranked,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
        )
