"""Состояние задачи как конечный автомат.

Этапы: requirements → implementation → architecture → result

Переходы явно описаны в ALLOWED_TRANSITIONS.
Любая попытка нелегального перехода вызывает StageTransitionError.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

Stage = Literal["requirements", "implementation", "architecture", "result"]

STAGE_ORDER: list[Stage] = ["requirements", "implementation", "architecture", "result"]

STAGE_LABELS: dict[str, str] = {
    "requirements":    "Опрос требований",
    "implementation":  "План реализации",
    "architecture":    "Архитектурный план",
    "result":          "Итоговый результат",
}

STAGE_DEFAULT_ACTIONS: dict[str, str] = {
    "requirements":   "Задавай уточняющие вопросы для сбора требований",
    "implementation": "Составь детальный план реализации",
    "architecture":   "Разработай архитектурный план с компонентами и схемами",
    "result":         "Предоставь итоговый документ архитектуры",
}

# Явная таблица допустимых переходов.
# Переход не в этом списке — запрещён, StageTransitionError.
ALLOWED_TRANSITIONS: dict[str, list[str]] = {
    "requirements":   ["implementation"],
    "implementation": ["architecture"],
    "architecture":   ["result"],
    "result":         [],  # финальный этап, выходов нет
}


class StageTransitionError(Exception):
    """Попытка нелегального перехода между этапами."""

    def __init__(self, from_stage: str, to_stage: str) -> None:
        self.from_stage = from_stage
        self.to_stage = to_stage
        allowed = ALLOWED_TRANSITIONS.get(from_stage, [])
        allowed_str = (
            " → ".join(STAGE_LABELS[s] for s in allowed)
            if allowed else "нет (финальный этап)"
        )
        from_label = STAGE_LABELS.get(from_stage, from_stage)
        to_label = STAGE_LABELS.get(to_stage, to_stage)
        super().__init__(
            f"Переход «{from_label}» → «{to_label}» запрещён.\n"
            f"Допустимые переходы из «{from_label}»: {allowed_str}.\n"
            f"Нельзя пропускать этапы — каждый должен быть пройден последовательно."
        )


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class TaskState:
    """Состояние задачи: этап, шаг, ожидаемое действие, пауза."""

    title: str = ""
    stage: Stage = "requirements"
    current_step: str = ""
    expected_action: str = field(
        default_factory=lambda: STAGE_DEFAULT_ACTIONS["requirements"]
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
        return self.stage == "result"

    @property
    def allowed_next(self) -> list[str]:
        """Список допустимых следующих этапов."""
        return ALLOWED_TRANSITIONS.get(self.stage, [])

    @property
    def next_stage(self) -> Stage | None:
        """Единственный следующий этап (None если финальный)."""
        allowed = self.allowed_next
        return allowed[0] if allowed else None

    def can_transition_to(self, stage: str) -> bool:
        return stage in self.allowed_next

    # ------------------------------------------------------------------
    # Переходы
    # ------------------------------------------------------------------

    def transition_to(self, stage: str) -> None:
        """Явный переход в указанный этап.

        Raises:
            StageTransitionError: если переход не разрешён.
        """
        if not self.can_transition_to(stage):
            raise StageTransitionError(self.stage, stage)
        self.stage = stage
        self.current_step = ""
        self.expected_action = STAGE_DEFAULT_ACTIONS[stage]
        self._touch()

    def advance(self) -> bool:
        """Перейти на единственный допустимый следующий этап.

        Возвращает False если уже в финальном этапе.
        """
        nxt = self.next_stage
        if nxt is None:
            return False
        self.transition_to(nxt)
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
            f"Текущий этап: {self.stage} — {self.stage_label}",
        ]
        if self.current_step:
            lines.append(f"Текущий шаг: {self.current_step}")
        if self.expected_action:
            lines.append(f"Ожидаемое действие: {self.expected_action}")

        # Явные правила переходов для модели
        if self.allowed_next:
            next_label = STAGE_LABELS[self.allowed_next[0]]
            lines.append(f"Следующий этап (после завершения текущего): {next_label}")
        blocked = [
            STAGE_LABELS[s] for s in STAGE_ORDER
            if s != self.stage and s not in self.allowed_next
        ]
        if blocked:
            lines.append(
                f"ЗАПРЕЩЕНО переходить к: {', '.join(blocked)} — этапы не пройдены."
            )

        lines.append(f"Статус: {status}")
        if resuming:
            lines.append(
                "Пользователь возобновил задачу — продолжай строго с текущего шага, "
                "не повторяй введение и уже сказанное."
            )
        return "\n".join(lines)

    def format_telegram(self) -> str:
        """Читаемое отображение для Telegram."""
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

        # Правила переходов
        if self.allowed_next:
            lines.append(
                f"\n<b>Следующий этап:</b> {STAGE_LABELS[self.allowed_next[0]]} "
                f"(команда /task advance)"
            )
        else:
            lines.append("\n<b>Финальный этап.</b> Задача завершена.")

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
            stage=d.get("stage", "requirements"),
            current_step=d.get("current_step", ""),
            expected_action=d.get("expected_action", STAGE_DEFAULT_ACTIONS["requirements"]),
            notes=d.get("notes", []),
            paused=d.get("paused", False),
            created_at=d.get("created_at", _now()),
            updated_at=d.get("updated_at", _now()),
        )
