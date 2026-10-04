"""ChatAgent: RAG + dialog history + task state memory."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from bot_ai_patterns.client import get_client
from bot_ai_patterns.config import MODEL_URI
from .chat_session import ChatSession
from .task_tracker import TaskStateTracker
from .reranker import Reranker, RankedResult
from .retriever import Retriever, SearchResult
from .query_rewriter import QueryRewriter

_SYSTEM = """\
Ты — профессиональный ассистент по пошиву одежды. Помогаешь пользователю шаг за шагом.

Правила:
1. Отвечай ТОЛЬКО на основе предоставленного контекста из инструкций.
2. Учитывай всю историю диалога и зафиксированное состояние задачи.
3. Если контекст не содержит ответа — честно скажи об этом.
4. В конце ответа ВСЕГДА указывай источники в формате «📌 Источник: файл / раздел».
5. Если в ответе нет источников — явно напиши «⚠️ Ответ основан на общих знаниях, не на документах».
6. Не повторяй предыдущие ответы — продвигай диалог вперёд."""

_USER_TEMPLATE = """\
=== СОСТОЯНИЕ ЗАДАЧИ ===
{task_state}

=== ИСТОРИЯ ДИАЛОГА ===
{history}

=== КОНТЕКСТ ИЗ ИНСТРУКЦИЙ ===
{context}

=== ТЕКУЩИЙ ВОПРОС ===
{question}

Ответь, опираясь на контекст и историю. Укажи источники."""

_NO_CONTEXT_TEMPLATE = """\
=== СОСТОЯНИЕ ЗАДАЧИ ===
{task_state}

=== ИСТОРИЯ ДИАЛОГА ===
{history}

=== ТЕКУЩИЙ ВОПРОС ===
{question}

Релевантных фрагментов в базе не найдено (best score: {best_score:.2f}).
Ответь честно, что информации нет, и предложи уточнить вопрос."""


@dataclass
class ChatResponse:
    answer: str
    sources: list[str]
    ranked: list[RankedResult]
    rewritten_query: str
    best_score: float
    candidates_before: int
    candidates_after: int
    prompt_tokens: int = 0
    completion_tokens: int = 0


class ChatAgent:
    def __init__(
        self,
        index_dir: str | Path,
        retrieve_k: int = 10,
        final_k: int = 4,
        rerank_threshold: float = 0.0,
        no_context_threshold: float = 1.0,
        strategy: str = "structure",
    ):
        self._retriever = Retriever(index_dir, strategy=strategy)
        self._reranker = Reranker(threshold=rerank_threshold)
        self._rewriter = QueryRewriter()
        self._tracker = TaskStateTracker()
        self._client = get_client()
        self._retrieve_k = retrieve_k
        self._final_k = final_k
        self._no_context_threshold = no_context_threshold

    def chat(self, session: ChatSession, question: str) -> ChatResponse:
        # 1. Update task state from new message
        history_snippet = session.history_text(last_n=6)
        session.state = self._tracker.update(session.state, question, history_snippet)

        # 2. Query rewrite (use goal + question for richer query)
        query_with_context = question
        if session.state.goal:
            query_with_context = f"{session.state.goal}. {question}"
        rewritten = self._rewriter.rewrite(query_with_context)

        # 3. Retrieval + rerank
        candidates = self._retriever.search(rewritten, top_k=self._retrieve_k)
        ranked = self._reranker.rerank(rewritten, candidates, top_k=self._final_k)
        best_score = ranked[0].cross_score if ranked else -999.0
        sources_used = [r.original for r in ranked]

        # 4. Build prompt
        task_summary = session.state.summary()
        history_text = session.history_text(last_n=8)

        if best_score >= self._no_context_threshold and ranked:
            context = self._retriever.format_context(sources_used)
            prompt = _USER_TEMPLATE.format(
                task_state=task_summary,
                history=history_text,
                context=context,
                question=question,
            )
        else:
            prompt = _NO_CONTEXT_TEMPLATE.format(
                task_state=task_summary,
                history=history_text,
                question=question,
                best_score=best_score,
            )

        # 5. LLM call
        response = self._client.chat.completions.create(
            model=MODEL_URI,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1024,
        )
        answer = response.choices[0].message.content or ""

        # 6. Extract source list
        sources = list({
            f"{r.original.title} / {r.original.section or '—'}"
            for r in ranked
        })

        # 7. Update session history
        session.add("user", question)
        session.add("assistant", answer, sources=sources)

        return ChatResponse(
            answer=answer,
            sources=sources,
            ranked=ranked,
            rewritten_query=rewritten,
            best_score=best_score,
            candidates_before=len(candidates),
            candidates_after=len(ranked),
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
        )
