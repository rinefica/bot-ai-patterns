"""Стратегия Sticky Facts — ключевые факты + последние N сообщений."""
from __future__ import annotations

import openai

from bot_ai_patterns.config import FACTS_WINDOW_SIZE, MODEL_URI
from bot_ai_patterns.context_strategies.base import ContextStrategy


class StickyFactsStrategy(ContextStrategy):
    """Извлекает ключевые факты из каждого обмена и хранит их отдельно.

    Контекст = [system] + [facts_block] + messages[-window_size:]

    Факты (цели, ограничения, предпочтения, договорённости) обновляются
    после каждого ответа ассистента через отдельный запрос к LLM.
    Позволяет помнить суть диалога при маленьком окне.
    """

    def __init__(self, client: openai.OpenAI, window_size: int = FACTS_WINDOW_SIZE) -> None:
        self._client = client
        self._window = window_size
        self._messages: list[dict[str, str]] = []
        self._facts: dict[str, str] = {}
        self._total: int = 0

    @property
    def name(self) -> str:
        return "sticky_facts"

    @property
    def display_name(self) -> str:
        return f"Sticky Facts (N={self._window})"

    def add_user(self, content: str) -> None:
        self._messages.append({"role": "user", "content": content})
        self._total += 1

    def add_assistant(self, content: str) -> None:
        self._messages.append({"role": "assistant", "content": content})
        self._extract_facts()

    def rollback_user(self) -> None:
        if self._messages and self._messages[-1]["role"] == "user":
            self._messages.pop()
            self._total = max(0, self._total - 1)

    def build_messages(self, system_prompt: str) -> list[dict[str, str]]:
        msgs: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        if self._facts:
            facts_text = "\n".join(f"• {k}: {v}" for k, v in self._facts.items())
            msgs.append({"role": "system", "content": f"[Ключевые факты диалога]\n{facts_text}"})
        msgs.extend(self._messages[-self._window:])
        return msgs

    def _extract_facts(self) -> None:
        """Обновить факты на основе последнего обмена user+assistant."""
        exchange = self._messages[-2:] if len(self._messages) >= 2 else []
        if not exchange:
            return

        dialogue = "\n".join(
            f"{'Пользователь' if m['role'] == 'user' else 'Ассистент'}: {m['content']}"
            for m in exchange
        )
        existing_str = (
            "\n".join(f"{k}: {v}" for k, v in self._facts.items())
            if self._facts else "нет"
        )
        prompt = (
            f"Текущие факты:\n{existing_str}\n\n"
            f"Новый обмен:\n{dialogue}\n\n"
            "Обнови список важных фактов (цели, ограничения, предпочтения, договорённости). "
            "Ответь ТОЛЬКО строками формата «ключ: значение», по одной на строке. "
            "Не более 10 фактов. Если новых фактов нет — верни текущие без изменений."
        )
        try:
            resp = self._client.chat.completions.create(
                model=MODEL_URI,
                messages=[
                    {"role": "system", "content": "Ты извлекаешь факты из диалогов."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=256,
            )
            raw = resp.choices[0].message.content or ""
            new_facts: dict[str, str] = {}
            for line in raw.strip().splitlines():
                line = line.strip().lstrip("•-").strip()
                if ":" in line:
                    key, _, val = line.partition(":")
                    key, val = key.strip(), val.strip()
                    if key and val:
                        new_facts[key] = val
            if new_facts:
                self._facts = new_facts
        except Exception:
            pass  # не ломаем диалог из-за ошибки извлечения

    def reset(self) -> None:
        self._messages = []
        self._facts = {}
        self._total = 0

    def init_from_messages(self, messages: list[dict[str, str]]) -> None:
        self._messages = list(messages[-self._window:])
        self._total = len(messages)

    def get_state(self) -> dict:
        return {
            "strategy": self.name,
            "messages": self._messages,
            "facts": self._facts,
            "total": self._total,
        }

    def load_state(self, state: dict) -> None:
        self._messages = state.get("messages", [])
        self._facts = state.get("facts", {})
        self._total = state.get("total", len(self._messages))

    @property
    def facts(self) -> dict[str, str]:
        return dict(self._facts)

    @property
    def message_count(self) -> int:
        return min(len(self._messages), self._window)

    @property
    def total_count(self) -> int:
        return self._total

    def stats_lines(self) -> list[str]:
        return [
            f"Стратегия: {self.display_name}",
            f"В контексте: {self.message_count} / {len(self._messages)} сообщ.",
            f"Фактов: {len(self._facts)}",
        ]
