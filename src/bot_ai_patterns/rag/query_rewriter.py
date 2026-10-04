"""Query rewriter: expand/rephrase a user query for better retrieval."""

from __future__ import annotations

from bot_ai_patterns.client import get_client
from bot_ai_patterns.config import MODEL_URI

_SYSTEM = (
    "Ты — помощник по улучшению поисковых запросов для базы инструкций по пошиву одежды. "
    "Перефразируй запрос пользователя, чтобы он лучше совпадал с терминологией швейных инструкций. "
    "Добавь синонимы, раскрой аббревиатуры, уточни технические термины. "
    "Верни ТОЛЬКО переформулированный запрос, без объяснений, одной строкой."
)


class QueryRewriter:
    def __init__(self):
        self._client = get_client()

    def rewrite(self, query: str) -> str:
        """Return an expanded/rephrased version of the query."""
        response = self._client.chat.completions.create(
            model=MODEL_URI,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": query},
            ],
            max_tokens=128,
            temperature=0.3,
        )
        rewritten = (response.choices[0].message.content or "").strip()
        # Fallback to original if something goes wrong
        return rewritten if rewritten else query
