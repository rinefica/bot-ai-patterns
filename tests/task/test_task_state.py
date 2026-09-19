"""Тесты TaskState FSM."""
from __future__ import annotations

from bot_ai_patterns.task.task_state import (
    STAGE_DEFAULT_ACTIONS,
    STAGE_ORDER,
    TaskState,
)


class TestTaskState:
    def test_initial_state(self):
        t = TaskState(title="API")
        assert t.stage == "planning"
        assert not t.paused
        assert not t.is_done

    def test_advance_through_all_stages(self):
        t = TaskState()
        for expected in ["execution", "validation", "done"]:
            assert t.advance() is True
            assert t.stage == expected
        assert t.is_done

    def test_advance_from_done_returns_false(self):
        t = TaskState(stage="done")
        assert t.advance() is False
        assert t.stage == "done"

    def test_advance_resets_step_and_sets_default_action(self):
        t = TaskState(stage="planning", current_step="шаг 1")
        t.advance()
        assert t.stage == "execution"
        assert t.current_step == ""
        assert t.expected_action == STAGE_DEFAULT_ACTIONS["execution"]

    def test_next_stage_property(self):
        t = TaskState(stage="planning")
        assert t.next_stage == "execution"
        t2 = TaskState(stage="done")
        assert t2.next_stage is None

    def test_pause_and_resume(self):
        t = TaskState()
        t.pause()
        assert t.paused is True
        t.resume()
        assert t.paused is False

    def test_set_step(self):
        t = TaskState()
        t.set_step("Разработка эндпоинта")
        assert t.current_step == "Разработка эндпоинта"

    def test_set_action(self):
        t = TaskState()
        t.set_action("Реализовать валидацию")
        assert t.expected_action == "Реализовать валидацию"

    def test_add_note(self):
        t = TaskState()
        t.add_note("помни про кэш")
        t.add_note("использовать Redis")
        assert len(t.notes) == 2
        assert "Redis" in t.notes[1]

    def test_format_context_block_contains_stage(self):
        t = TaskState(title="Маркетплейс", stage="execution")
        block = t.format_context_block()
        assert "execution" in block
        assert "Маркетплейс" in block

    def test_format_context_block_shows_step(self):
        t = TaskState()
        t.set_step("Настройка БД")
        block = t.format_context_block()
        assert "Настройка БД" in block

    def test_format_context_block_resuming_flag(self):
        t = TaskState()
        block = t.format_context_block(resuming=True)
        assert "возобновил" in block

    def test_format_context_block_no_resuming_hint_by_default(self):
        t = TaskState()
        block = t.format_context_block(resuming=False)
        assert "возобновил" not in block

    def test_stage_label_property(self):
        t = TaskState(stage="validation")
        assert t.stage_label == "Проверка"

    def test_serialization_roundtrip(self):
        t = TaskState(title="API", stage="execution")
        t.set_step("POST /items")
        t.add_note("заметка")
        t.pause()

        t2 = TaskState.from_dict(t.to_dict())
        assert t2.title == "API"
        assert t2.stage == "execution"
        assert t2.current_step == "POST /items"
        assert t2.paused is True
        assert len(t2.notes) == 1

    def test_all_stages_in_order(self):
        assert STAGE_ORDER == ["planning", "execution", "validation", "done"]
