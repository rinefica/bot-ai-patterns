"""Тесты TaskState FSM и явных переходов."""
from __future__ import annotations

import pytest

from bot_ai_patterns.task.task_state import (
    ALLOWED_TRANSITIONS,
    STAGE_DEFAULT_ACTIONS,
    STAGE_ORDER,
    StageTransitionError,
    TaskState,
)


class TestTaskState:
    def test_initial_state(self):
        t = TaskState(title="API")
        assert t.stage == "requirements"
        assert not t.paused
        assert not t.is_done

    def test_advance_through_all_stages(self):
        t = TaskState()
        for expected in ["implementation", "architecture", "result"]:
            assert t.advance() is True
            assert t.stage == expected
        assert t.is_done

    def test_advance_from_result_returns_false(self):
        t = TaskState(stage="result")
        assert t.advance() is False
        assert t.stage == "result"

    def test_advance_resets_step_and_sets_default_action(self):
        t = TaskState(stage="requirements", current_step="шаг 1")
        t.advance()
        assert t.stage == "implementation"
        assert t.current_step == ""
        assert t.expected_action == STAGE_DEFAULT_ACTIONS["implementation"]

    def test_next_stage_property(self):
        t = TaskState(stage="requirements")
        assert t.next_stage == "implementation"
        t2 = TaskState(stage="result")
        assert t2.next_stage is None

    def test_pause_and_resume(self):
        t = TaskState()
        t.pause()
        assert t.paused is True
        t.resume()
        assert t.paused is False

    def test_set_step(self):
        t = TaskState()
        t.set_step("Сбор нефункциональных требований")
        assert t.current_step == "Сбор нефункциональных требований"

    def test_set_action(self):
        t = TaskState()
        t.set_action("Уточнить масштаб")
        assert t.expected_action == "Уточнить масштаб"

    def test_add_note(self):
        t = TaskState()
        t.add_note("помни про кэш")
        t.add_note("использовать Redis")
        assert len(t.notes) == 2
        assert "Redis" in t.notes[1]

    def test_format_context_block_contains_stage(self):
        t = TaskState(title="Маркетплейс", stage="architecture")
        block = t.format_context_block()
        assert "architecture" in block
        assert "Маркетплейс" in block

    def test_format_context_block_shows_step(self):
        t = TaskState()
        t.set_step("Сбор требований к нагрузке")
        block = t.format_context_block()
        assert "Сбор требований к нагрузке" in block

    def test_format_context_block_resuming_flag(self):
        t = TaskState()
        block = t.format_context_block(resuming=True)
        assert "возобновил" in block

    def test_format_context_block_no_resuming_hint_by_default(self):
        t = TaskState()
        block = t.format_context_block(resuming=False)
        assert "возобновил" not in block

    def test_stage_label_property(self):
        t = TaskState(stage="architecture")
        assert t.stage_label == "Архитектурный план"

    def test_serialization_roundtrip(self):
        t = TaskState(title="API", stage="implementation")
        t.set_step("MVP задачи")
        t.add_note("заметка")
        t.pause()

        t2 = TaskState.from_dict(t.to_dict())
        assert t2.title == "API"
        assert t2.stage == "implementation"
        assert t2.current_step == "MVP задачи"
        assert t2.paused is True
        assert len(t2.notes) == 1

    def test_all_stages_in_order(self):
        assert STAGE_ORDER == ["requirements", "implementation", "architecture", "result"]

    def test_is_done_only_on_result(self):
        for stage in ["requirements", "implementation", "architecture"]:
            assert not TaskState(stage=stage).is_done
        assert TaskState(stage="result").is_done

    # ------------------------------------------------------------------
    # Явные переходы и ограничения
    # ------------------------------------------------------------------

    def test_allowed_transitions_table(self):
        assert ALLOWED_TRANSITIONS["requirements"] == ["implementation"]
        assert ALLOWED_TRANSITIONS["implementation"] == ["architecture"]
        assert ALLOWED_TRANSITIONS["architecture"] == ["result"]
        assert ALLOWED_TRANSITIONS["result"] == []

    def test_can_transition_to_next(self):
        t = TaskState(stage="requirements")
        assert t.can_transition_to("implementation") is True

    def test_cannot_skip_stage(self):
        t = TaskState(stage="requirements")
        assert t.can_transition_to("architecture") is False
        assert t.can_transition_to("result") is False

    def test_transition_to_valid(self):
        t = TaskState(stage="requirements")
        t.transition_to("implementation")
        assert t.stage == "implementation"

    def test_transition_to_invalid_raises(self):
        t = TaskState(stage="requirements")
        with pytest.raises(StageTransitionError) as exc_info:
            t.transition_to("architecture")
        assert "requirements" in str(exc_info.value) or "Опрос" in str(exc_info.value)
        assert "architecture" in str(exc_info.value) or "Архитектурный" in str(exc_info.value)

    def test_transition_to_skip_result_raises(self):
        t = TaskState(stage="requirements")
        with pytest.raises(StageTransitionError):
            t.transition_to("result")

    def test_transition_from_result_raises(self):
        t = TaskState(stage="result")
        with pytest.raises(StageTransitionError):
            t.transition_to("requirements")

    def test_advance_uses_transition_table(self):
        t = TaskState(stage="requirements")
        t.advance()
        assert t.stage == "implementation"
        t.advance()
        assert t.stage == "architecture"
        t.advance()
        assert t.stage == "result"
        assert t.advance() is False  # финальный

    def test_transition_error_message_mentions_allowed(self):
        t = TaskState(stage="implementation")
        with pytest.raises(StageTransitionError) as exc_info:
            t.transition_to("result")
        err = str(exc_info.value)
        assert "architecture" in err.lower() or "Архитектурный" in err

    def test_allowed_next_property(self):
        t = TaskState(stage="requirements")
        assert t.allowed_next == ["implementation"]
        t2 = TaskState(stage="result")
        assert t2.allowed_next == []

    def test_context_block_shows_forbidden_stages(self):
        t = TaskState(stage="requirements")
        block = t.format_context_block()
        assert "ЗАПРЕЩЕНО" in block
        assert "architecture" in block.lower() or "Архитектурный" in block

    def test_context_block_shows_next_allowed(self):
        t = TaskState(stage="requirements")
        block = t.format_context_block()
        assert "implementation" in block.lower() or "План" in block
