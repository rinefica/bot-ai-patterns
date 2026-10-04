"""ChatSession: dialog history + structured task state."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Message:
    role: str          # "user" | "assistant"
    content: str
    sources: list[str] = field(default_factory=list)
    ts: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))


@dataclass
class TaskState:
    goal: str = ""                    # что пользователь хочет сшить / узнать
    clarifications: list[str] = field(default_factory=list)   # что уже уточнено
    constraints: list[str] = field(default_factory=list)      # ограничения и условия
    terms: dict[str, str] = field(default_factory=dict)       # термины → определения
    open_questions: list[str] = field(default_factory=list)   # что ещё не прояснено

    def is_empty(self) -> bool:
        return not any([self.goal, self.clarifications, self.constraints,
                        self.terms, self.open_questions])

    def summary(self) -> str:
        parts = []
        if self.goal:
            parts.append(f"Цель: {self.goal}")
        if self.clarifications:
            parts.append("Уточнено: " + "; ".join(self.clarifications))
        if self.constraints:
            parts.append("Ограничения: " + "; ".join(self.constraints))
        if self.terms:
            t = ", ".join(f"{k}={v}" for k, v in list(self.terms.items())[:5])
            parts.append(f"Термины: {t}")
        if self.open_questions:
            parts.append("Открытые вопросы: " + "; ".join(self.open_questions))
        return "\n".join(parts) if parts else "(пусто)"

    def to_json(self) -> str:
        return json.dumps(
            {
                "goal": self.goal,
                "clarifications": self.clarifications,
                "constraints": self.constraints,
                "terms": self.terms,
                "open_questions": self.open_questions,
            },
            ensure_ascii=False,
            indent=2,
        )

    @classmethod
    def from_dict(cls, d: dict) -> "TaskState":
        return cls(
            goal=d.get("goal", ""),
            clarifications=d.get("clarifications", []),
            constraints=d.get("constraints", []),
            terms=d.get("terms", {}),
            open_questions=d.get("open_questions", []),
        )


class ChatSession:
    def __init__(self, max_history: int = 20):
        self.history: list[Message] = []
        self.state = TaskState()
        self._max_history = max_history

    def add(self, role: str, content: str, sources: list[str] | None = None) -> None:
        self.history.append(Message(role=role, content=content, sources=sources or []))
        if len(self.history) > self._max_history:
            # Keep the first message (sets context) + recent ones
            self.history = self.history[:1] + self.history[-(self._max_history - 1):]

    def history_text(self, last_n: int = 10) -> str:
        """Format last N messages for the prompt."""
        recent = self.history[-last_n:]
        lines = []
        for m in recent:
            prefix = "Пользователь" if m.role == "user" else "Ассистент"
            lines.append(f"[{prefix}]: {m.content}")
        return "\n".join(lines)

    def reset(self) -> None:
        self.history = []
        self.state = TaskState()
