"""Стратегия Sliding Window — скользящее окно последних N сообщений."""
from __future__ import annotations

from bot_ai_patterns.config import WINDOW_SIZE
from bot_ai_patterns.context_strategies.base import ContextStrategy


class SlidingWindowStrategy(ContextStrategy):
    """Хранит только последние N сообщений, остальное отбрасывает.

    Контекст = [system] + messages[-window_size:]

    Самая простая стратегия: дёшево, предсказуемо, но теряет ранний контекст.
    """

    def __init__(self, window_size: int = WINDOW_SIZE) -> None:
        self._window = window_size
        self._messages: list[dict[str, str]] = []
        self._total: int = 0

    @property
    def name(self) -> str:
        return "sliding_window"

    @property
    def display_name(self) -> str:
        return f"Sliding Window (N={self._window})"

    def add_user(self, content: str) -> None:
        self._messages.append({"role": "user", "content": content})
        self._total += 1

    def add_assistant(self, content: str) -> None:
        self._messages.append({"role": "assistant", "content": content})

    def rollback_user(self) -> None:
        if self._messages and self._messages[-1]["role"] == "user":
            self._messages.pop()
            self._total = max(0, self._total - 1)

    def build_messages(self, system_prompt: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": system_prompt},
            *self._messages[-self._window:],
        ]

    def reset(self) -> None:
        self._messages = []
        self._total = 0

    def init_from_messages(self, messages: list[dict[str, str]]) -> None:
        self._messages = list(messages[-self._window:])
        self._total = len(messages)

    def get_state(self) -> dict:
        return {"strategy": self.name, "messages": self._messages, "total": self._total}

    def load_state(self, state: dict) -> None:
        self._messages = state.get("messages", [])
        self._total = state.get("total", len(self._messages))

    def stats_lines(self) -> list[str]:
        in_ctx = min(len(self._messages), self._window)
        return [
            f"Стратегия: {self.display_name}",
            f"В контексте: {in_ctx} / {len(self._messages)} сообщ.",
        ]

    @property
    def message_count(self) -> int:
        return min(len(self._messages), self._window)

    @property
    def total_count(self) -> int:
        return self._total
