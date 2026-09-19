"""Состояние задачи как конечный автомат.

Этапы: planning → execution → validation → done
Каждый этап — явное состояние с текущим шагом и ожидаемым действием.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

Stage = Literal["planning", "execution", "validation", "done"]

STAGE_ORDER: list[Stage] = ["planning", "execution", "validation", "done"]

STAGE_LABELS: dict[str, str] = {
    "planning":   "Планирование",
    "execution":  "Выполнение",
    "validation": "Проверка",
    "done":       "Завершено",
}

STAGE_DEFAULT_ACTIONS: dict[str, str] = {
    "planning":   "Уточните требования, стек и критерии готовности",
    "execution":  "Реализуйте текущий шаг согласно плану",
    "validation": "Проверьте результат на соответствие требованиям",
    "done":       "Задача завершена — подведи итоги",
}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class TaskState:
    """Состояние задачи: этап, шаг, ожидаемое действие, пауза."""

    title: str = ""
    stage: Stage = "planning"
    current_step: str = ""
    expected_action: str = field(
        default_factory=lambda: STAGE_DEFAULT_ACTIONS["planning"]
    )
    notes: list[str] = field(default_factory=list)
    paused: bool = False
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    # ------------------------------------------------------------------
    # Свойства
    # ------------------------------------------------------------------

    @property
    def stage_label(self) -> str:
        return STAGE_LABELS[self.stage]

    @property
    def is_done(self) -> bool:
        return self.stage == "done"

    @property
    def next_stage(self) -> Stage | None:
        idx = STAGE_ORDER.index(self.stage)
        if idx < len(STAGE_ORDER) - 1:
            return STAGE_ORDER[idx + 1]
        return None

    # ------------------------------------------------------------------
    # Переходы
    # ------------------------------------------------------------------

    def advance(self) -> bool:
        """Перейти на следующий этап. Возвращает False если уже done."""
        nxt = self.next_stage
        if nxt is None:
            return False
        self.stage = nxt
        self.current_step = ""
        self.expected_action = STAGE_DEFAULT_ACTIONS[nxt]
        self._touch()
        return True

    def pause(self) -> None:
        self.paused = True
        self._touch()

    def resume(self) -> None:
        self.paused = False
        self._touch()

    def set_step(self, step: str) -> None:
        self.current_step = step
        self._touch()

    def set_action(self, action: str) -> None:
        self.expected_action = action
        self._touch()

    def add_note(self, note: str) -> None:
        self.notes.append(note)
        self._touch()

    def _touch(self) -> None:
        self.updated_at = _now()

    # ------------------------------------------------------------------
    # Форматирование
    # ------------------------------------------------------------------

    def format_context_block(self, resuming: bool = False) -> str:
        """Блок для инжекции в системный контекст модели."""
        status = "на паузе" if self.paused else "активна"
        lines = [
            "[Состояние задачи]",
            f"Задача: {self.title or '(без названия)'}",
            f"Этап: {self.stage} — {self.stage_label}",
        ]
        if self.current_step:
            lines.append(f"Текущий шаг: {self.current_step}")
        if self.expected_action:
            lines.append(f"Ожидаемое действие: {self.expected_action}")
        lines.append(f"Статус: {status}")
        if resuming:
            lines.append(
                "Пользователь возобновил задачу — продолжай строго с текущего шага, "
                "не повторяй введение и уже сказанное."
            )
        return "\n".join(lines)

    def format_telegram(self) -> str:
        """Читаемое отображение для Telegram."""
        idx = STAGE_ORDER.index(self.stage)
        stages = " → ".join(
            f"<b>[{STAGE_LABELS[s]}]</b>" if s == self.stage else STAGE_LABELS[s]
            for s in STAGE_ORDER
        )
        lines = [
            f"<b>Задача:</b> {self.title or '(без названия)'}",
            f"<b>Прогресс:</b> {stages}",
        ]
        if self.current_step:
            lines.append(f"<b>Текущий шаг:</b> {self.current_step}")
        if self.expected_action:
            lines.append(f"<b>Ожидаемое действие:</b> {self.expected_action}")
        lines.append(f"<b>Статус:</b> {'⏸ на паузе' if self.paused else '▶ активна'}")
        if self.notes:
            lines.append(f"\n<b>Заметки ({len(self.notes)}):</b>")
            for note in self.notes[-3:]:
                lines.append(f"  • {note}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Сериализация
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "stage": self.stage,
            "current_step": self.current_step,
            "expected_action": self.expected_action,
            "notes": self.notes,
            "paused": self.paused,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TaskState:
        return cls(
            title=d.get("title", ""),
            stage=d.get("stage", "planning"),
            current_step=d.get("current_step", ""),
            expected_action=d.get("expected_action", STAGE_DEFAULT_ACTIONS["planning"]),
            notes=d.get("notes", []),
            paused=d.get("paused", False),
            created_at=d.get("created_at", _now()),
            updated_at=d.get("updated_at", _now()),
        )
