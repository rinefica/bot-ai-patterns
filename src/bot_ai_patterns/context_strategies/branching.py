"""Стратегия Branching — независимые ветки диалога от общего checkpoint."""
from __future__ import annotations

from bot_ai_patterns.context_strategies.base import ContextStrategy


class Branch:
    """Одна ветка диалога."""

    def __init__(self, name: str, messages: list[dict[str, str]] | None = None) -> None:
        self.name = name
        self.messages: list[dict[str, str]] = list(messages or [])


class BranchingStrategy(ContextStrategy):
    """Поддерживает независимые ветки диалога от общего checkpoint.

    Workflow:
    1. Ведём диалог в ветке 'main'
    2. set_checkpoint() — фиксируем точку ветвления
    3. create_branch(name) — создаём ветку от checkpoint
    4. switch_branch(name) — переключаемся между ветками
    5. Каждая ветка развивается независимо

    Контекст = [system] + messages текущей ветки (полностью).
    """

    def __init__(self) -> None:
        self._branches: dict[str, Branch] = {"main": Branch("main")}
        self._current: str = "main"
        self._checkpoint: list[dict[str, str]] = []
        self._checkpoint_set: bool = False

    @property
    def name(self) -> str:
        return "branching"

    @property
    def display_name(self) -> str:
        return f"Branching [{self._current}]"

    def add_user(self, content: str) -> None:
        self._current_branch.messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str) -> None:
        self._current_branch.messages.append({"role": "assistant", "content": content})

    def rollback_user(self) -> None:
        msgs = self._current_branch.messages
        if msgs and msgs[-1]["role"] == "user":
            msgs.pop()

    def build_messages(self, system_prompt: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": system_prompt},
            *self._current_branch.messages,
        ]

    # --- Методы управления ветками ---

    def set_checkpoint(self) -> None:
        """Зафиксировать текущее состояние как точку ветвления."""
        self._checkpoint = list(self._current_branch.messages)
        self._checkpoint_set = True

    def create_branch(self, name: str) -> bool:
        """Создать ветку от последнего checkpoint. False если невозможно."""
        if not self._checkpoint_set or name in self._branches:
            return False
        self._branches[name] = Branch(name, self._checkpoint)
        return True

    def switch_branch(self, name: str) -> bool:
        """Переключиться на ветку. False если не существует."""
        if name not in self._branches:
            return False
        self._current = name
        return True

    @property
    def branch_names(self) -> list[str]:
        return list(self._branches.keys())

    @property
    def current_branch_name(self) -> str:
        return self._current

    @property
    def checkpoint_set(self) -> bool:
        return self._checkpoint_set

    @property
    def _current_branch(self) -> Branch:
        return self._branches[self._current]

    def reset(self) -> None:
        self._branches = {"main": Branch("main")}
        self._current = "main"
        self._checkpoint = []
        self._checkpoint_set = False

    def init_from_messages(self, messages: list[dict[str, str]]) -> None:
        self._branches["main"].messages = list(messages)

    def get_state(self) -> dict:
        return {
            "strategy": self.name,
            "current": self._current,
            "checkpoint": self._checkpoint,
            "checkpoint_set": self._checkpoint_set,
            "branches": {n: b.messages for n, b in self._branches.items()},
        }

    def load_state(self, state: dict) -> None:
        self._current = state.get("current", "main")
        self._checkpoint = state.get("checkpoint", [])
        self._checkpoint_set = state.get("checkpoint_set", False)
        branches_data = state.get("branches", {"main": []})
        self._branches = {n: Branch(n, msgs) for n, msgs in branches_data.items()}
        if "main" not in self._branches:
            self._branches["main"] = Branch("main")

    @property
    def message_count(self) -> int:
        return len(self._current_branch.messages)

    @property
    def total_count(self) -> int:
        return len(self._current_branch.messages)

    def stats_lines(self) -> list[str]:
        lines = [
            f"Стратегия: {self.display_name}",
            f"Веток: {len(self._branches)} ({', '.join(self._branches.keys())})",
            f"Сообщений в [{self._current}]: {len(self._current_branch.messages)}",
        ]
        if self._checkpoint_set:
            lines.append(f"Checkpoint: {len(self._checkpoint)} сообщ.")
        return lines
