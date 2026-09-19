"""Тесты WorkingMemory."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from bot_ai_patterns.memory.working_memory import WorkingMemory


def _mock_client(reply: str = "цель: написать бота\nстек: Python") -> MagicMock:
    client = MagicMock()
    choice = SimpleNamespace(message=SimpleNamespace(content=reply))
    client.chat.completions.create.return_value = SimpleNamespace(choices=[choice])
    return client


class TestWorkingMemory:
    def test_initial_empty(self):
        wm = WorkingMemory(_mock_client())
        assert wm.data == {}

    def test_set_get(self):
        wm = WorkingMemory(_mock_client())
        wm.set("задача", "написать API")
        assert wm.data["задача"] == "написать API"

    def test_remove_existing(self):
        wm = WorkingMemory(_mock_client())
        wm.set("k", "v")
        assert wm.remove("k") is True
        assert "k" not in wm.data

    def test_remove_missing_returns_false(self):
        wm = WorkingMemory(_mock_client())
        assert wm.remove("nonexistent") is False

    def test_reset_clears_data(self):
        wm = WorkingMemory(_mock_client())
        wm.set("k", "v")
        wm.reset()
        assert wm.data == {}

    def test_update_from_exchange_populates_data(self):
        wm = WorkingMemory(_mock_client("задача: написать бота\nстек: Python"))
        wm.update_from_exchange("хочу сделать бота", "расскажи подробнее")
        assert len(wm.data) > 0

    def test_update_from_exchange_api_error_doesnt_break(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("API error")
        wm = WorkingMemory(client)
        wm.update_from_exchange("вопрос", "ответ")  # не должно бросить
        assert wm.data == {}

    def test_format_block_none_when_empty(self):
        wm = WorkingMemory(_mock_client())
        assert wm.format_block() is None

    def test_format_block_shows_data(self):
        wm = WorkingMemory(_mock_client())
        wm.set("тип", "маркетплейс")
        block = wm.format_block()
        assert block is not None
        assert "тип" in block
        assert "маркетплейс" in block

    def test_state_roundtrip(self):
        wm = WorkingMemory(_mock_client())
        wm.set("задача", "API")
        wm.set("стек", "FastAPI")
        state = wm.get_state()

        wm2 = WorkingMemory(_mock_client())
        wm2.load_state(state)
        assert wm2.data == {"задача": "API", "стек": "FastAPI"}
