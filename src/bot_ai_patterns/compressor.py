"""Компрессор истории диалога: сжимает старые сообщения в текстовое резюме."""
from __future__ import annotations

import openai

from bot_ai_patterns.config import MODEL_URI

_SYSTEM = (
    "Ты — ассистент для сжатия диалогов. "
    "Создай краткое резюме переписки: о чём говорили, какие факты важны для продолжения. "
    "Если передано предыдущее резюме — объедини с новым фрагментом. "
    "3–7 предложений, на русском языке."
)


class Compressor:
    """Сжимает список сообщений в текстовое резюме через LLM."""

    def __init__(self, client: openai.OpenAI) -> None:
        self._client = client

    def summarize(
        self,
        messages: list[dict[str, str]],
        existing_summary: str = "",
    ) -> str:
        """Создать или обновить резюме.

        Args:
            messages: новые сообщения для сжатия (без системных).
            existing_summary: уже накопленное резюме (если есть).

        Returns:
            Обновлённое текстовое резюме.
        """
        parts: list[str] = []
        if existing_summary:
            parts.append(f"Предыдущее резюме:\n{existing_summary}")

        dialogue = "\n".join(
            f"{'Пользователь' if m['role'] == 'user' else 'Ассистент'}: {m['content']}"
            for m in messages
            if m["role"] in ("user", "assistant")
        )
        parts.append(f"Новый фрагмент диалога:\n{dialogue}")

        response = self._client.chat.completions.create(
            model=MODEL_URI,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": "\n\n".join(parts)},
            ],
            max_tokens=512,
        )
        return response.choices[0].message.content or ""
