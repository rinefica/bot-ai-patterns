"""Рабочая память — данные текущей задачи (K-V, авто-экстракция).

Область действия: текущая сессия. Сбрасывается вместе с историей диалога.
"""
from __future__ import annotations

import openai

from bot_ai_patterns.config import MODEL_URI

_EXTRACT_SYSTEM = (
    "Ты извлекаешь данные о текущей задаче из диалога.\n"
    "Выведи ТОЛЬКО строки формата «ключ: значение», по одной на строке.\n"
    "Не более 8 пар. Если важных данных о задаче нет — верни пустую строку."
)


class WorkingMemory:
    """Хранит ключевые параметры текущей задачи.

    Авто-обновляется после каждого обмена через LLM.
    Сбрасывается при reset() вместе с историей диалога.
    """

    def __init__(self, client: openai.OpenAI) -> None:
        self._client = client
        self._data: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def data(self) -> dict[str, str]:
        return dict(self._data)

    def set(self, key: str, value: str) -> None:
        self._data[key] = value

    def remove(self, key: str) -> bool:
        return self._data.pop(key, None) is not None

    def reset(self) -> None:
        self._data = {}

    def update_from_exchange(self, user_msg: str, assistant_msg: str) -> None:
        """Обновить данные задачи на основе нового обмена."""
        existing = (
            "\n".join(f"{k}: {v}" for k, v in self._data.items()) or "нет"
        )
        exchange = f"Пользователь: {user_msg}\nАссистент: {assistant_msg}"
        prompt = (
            f"Текущие данные задачи:\n{existing}\n\n"
            f"Новый обмен:\n{exchange}\n\n"
            "Обнови данные о задаче (тип продукта, параметры, стек, прогресс, ограничения). "
            "Удали неактуальное, добавь новое. Не более 8 пар."
        )
        try:
            resp = self._client.chat.completions.create(
                model=MODEL_URI,
                messages=[
                    {"role": "system", "content": _EXTRACT_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=200,
            )
            raw = resp.choices[0].message.content or ""
            new_data: dict[str, str] = {}
            for line in raw.strip().splitlines():
                line = line.strip().lstrip("•-").strip()
                if ":" in line:
                    key, _, val = line.partition(":")
                    key, val = key.strip(), val.strip()
                    if key and val:
                        new_data[key] = val
            if new_data:
                self._data = new_data
        except Exception:
            pass  # не ломаем диалог

    def format_block(self) -> str | None:
        if not self._data:
            return None
        lines = "\n".join(f"• {k}: {v}" for k, v in self._data.items())
        return f"[Рабочая память — текущая задача]\n{lines}"

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def get_state(self) -> dict:
        return {"data": dict(self._data)}

    def load_state(self, state: dict) -> None:
        self._data = state.get("data", {})
