"""Тесты TaskManager."""
from __future__ import annotations

import tempfile
from pathlib import Path

from bot_ai_patterns.task.task_manager import TaskManager


def _tm(user_id: int = 1, tmp_dir: Path | None = None) -> TaskManager:
    return TaskManager(user_id, tmp_dir or Path(tempfile.mkdtemp()))


class TestTaskManager:
    def test_initial_no_task(self):
        tm = _tm()
        assert not tm.has_task
        assert tm.task is None

    def test_create_task(self):
        tm = _tm()
        task = tm.create("Разработка API")
        assert task.title == "Разработка API"
        assert task.stage == "planning"
        assert tm.has_task

    def test_advance_moves_stage(self):
        tm = _tm()
        tm.create("API")
        assert tm.advance() is True
        assert tm.task.stage == "execution"

    def test_advance_without_task_returns_false(self):
        tm = _tm()
        assert tm.advance() is False

    def test_pause_and_resume(self):
        tm = _tm()
        tm.create("API")
        assert tm.pause() is True
        assert tm.task.paused is True
        assert tm.resume() is True
        assert tm.task.paused is False

    def test_resume_sets_resuming_flag(self):
        tm = _tm()
        tm.create("API")
        tm.pause()
        tm.resume()
        assert tm.resuming is True

    def test_format_context_block_consumes_resuming_flag(self):
        tm = _tm()
        tm.create("API")
        tm.pause()
        tm.resume()
        # Первый вызов — с подсказкой
        block1 = tm.format_context_block()
        assert "без повторн" in block1 or "продолжай" in block1
        # Второй вызов — без подсказки
        block2 = tm.format_context_block()
        assert "возобновил" not in block2

    def test_format_context_block_none_when_no_task(self):
        tm = _tm()
        assert tm.format_context_block() is None

    def test_format_context_block_none_when_done(self):
        tm = _tm()
        tm.create("API")
        while tm.task and not tm.task.is_done:
            tm.advance()
        assert tm.format_context_block() is None

    def test_set_step(self):
        tm = _tm()
        tm.create("API")
        assert tm.set_step("POST /items") is True
        assert tm.task.current_step == "POST /items"

    def test_set_action(self):
        tm = _tm()
        tm.create("API")
        assert tm.set_action("Добавить валидацию") is True
        assert tm.task.expected_action == "Добавить валидацию"

    def test_add_note(self):
        tm = _tm()
        tm.create("API")
        assert tm.add_note("помни про кэш") is True
        assert len(tm.task.notes) == 1

    def test_clear_removes_task(self):
        tm = _tm()
        tm.create("API")
        tm.clear()
        assert not tm.has_task

    def test_persistence_across_instances(self):
        tmp = Path(tempfile.mkdtemp())
        tm1 = TaskManager(1, tmp)
        tm1.create("Маркетплейс")
        tm1.advance()
        tm1.set_step("Разработка каталога")

        tm2 = TaskManager(1, tmp)
        assert tm2.has_task
        assert tm2.task.title == "Маркетплейс"
        assert tm2.task.stage == "execution"
        assert tm2.task.current_step == "Разработка каталога"

    def test_different_users_isolated(self):
        tmp = Path(tempfile.mkdtemp())
        tm1 = TaskManager(1, tmp)
        tm1.create("Задача 1")

        tm2 = TaskManager(2, tmp)
        assert not tm2.has_task

    def test_format_telegram_no_task(self):
        tm = _tm()
        text = tm.format_telegram()
        assert "Нет активной задачи" in text

    def test_format_telegram_shows_stage(self):
        tm = _tm()
        tm.create("API")
        text = tm.format_telegram()
        assert "Планирование" in text
        assert "API" in text

    def test_format_telegram_shows_pause(self):
        tm = _tm()
        tm.create("API")
        tm.pause()
        text = tm.format_telegram()
        assert "паузе" in text
