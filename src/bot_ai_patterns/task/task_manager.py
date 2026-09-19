"""TaskManager — CRUD и персистентность FSM задачи."""
from __future__ import annotations

import json
from pathlib import Path

from bot_ai_patterns.task.task_state import STAGE_DEFAULT_ACTIONS, TaskState

TASKS_DIR = Path("data/tasks")


class TaskManager:
    """Управляет состоянием задачи: создание, переходы, пауза, восстановление.

    Персистируется в data/tasks/{user_id}_task.json.
    Переживает /reset (хранится независимо от истории диалога).
    """

    def __init__(self, user_id: int, tasks_dir: Path = TASKS_DIR) -> None:
        self._user_id = user_id
        self._dir = tasks_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._task: TaskState | None = None
        self._resuming: bool = False  # флаг первого сообщения после resume
        self._load()

    # ------------------------------------------------------------------
    # Свойства
    # ------------------------------------------------------------------

    @property
    def task(self) -> TaskState | None:
        return self._task

    @property
    def has_task(self) -> bool:
        return self._task is not None

    @property
    def resuming(self) -> bool:
        return self._resuming

    # ------------------------------------------------------------------
    # Управление задачей
    # ------------------------------------------------------------------

    def create(self, title: str) -> TaskState:
        """Создать новую задачу (заменяет предыдущую)."""
        self._task = TaskState(
            title=title,
            expected_action=STAGE_DEFAULT_ACTIONS["planning"],
        )
        self._resuming = False
        self._save()
        return self._task

    def advance(self) -> bool:
        """Перейти на следующий этап."""
        if self._task is None:
            return False
        result = self._task.advance()
        if result:
            self._save()
        return result

    def pause(self) -> bool:
        if self._task is None:
            return False
        self._task.pause()
        self._resuming = False
        self._save()
        return True

    def resume(self) -> bool:
        if self._task is None:
            return False
        self._task.resume()
        self._resuming = True  # следующее сообщение получит подсказку "продолжаем"
        self._save()
        return True

    def set_step(self, step: str) -> bool:
        if self._task is None:
            return False
        self._task.set_step(step)
        self._save()
        return True

    def set_action(self, action: str) -> bool:
        if self._task is None:
            return False
        self._task.set_action(action)
        self._save()
        return True

    def add_note(self, note: str) -> bool:
        if self._task is None:
            return False
        self._task.add_note(note)
        self._save()
        return True

    def clear(self) -> None:
        """Удалить текущую задачу."""
        self._task = None
        self._resuming = False
        path = self._path()
        if path.exists():
            path.unlink()

    # ------------------------------------------------------------------
    # Инжекция в контекст
    # ------------------------------------------------------------------

    def format_context_block(self) -> str | None:
        """Блок для инжекции в системный контекст.

        При первом сообщении после resume добавляет инструкцию
        "продолжай без повторений", затем сбрасывает флаг.
        """
        if self._task is None or self._task.is_done:
            return None
        block = self._task.format_context_block(resuming=self._resuming)
        if self._resuming:
            self._resuming = False  # однократная инструкция
        return block

    def format_telegram(self) -> str:
        if self._task is None:
            return (
                "Нет активной задачи.\n\n"
                "Создай командой: /task new &lt;название&gt;"
            )
        return self._task.format_telegram()

    # ------------------------------------------------------------------
    # Персистентность
    # ------------------------------------------------------------------

    def _path(self) -> Path:
        return self._dir / f"{self._user_id}_task.json"

    def _load(self) -> None:
        path = self._path()
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            self._task = TaskState.from_dict(data)

    def _save(self) -> None:
        if self._task is None:
            return
        self._path().write_text(
            json.dumps(self._task.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
