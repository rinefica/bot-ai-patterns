"""Тесты LongTermMemory."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from bot_ai_patterns.memory.long_term_memory import LongTermMemory


def _mock_client(reply: str = "profile|язык: Python\ndecisions|стек: FastAPI") -> MagicMock:
    client = MagicMock()
    choice = SimpleNamespace(message=SimpleNamespace(content=reply))
    client.chat.completions.create.return_value = SimpleNamespace(choices=[choice])
    return client


def _ltm(user_id: int = 1, tmp_dir: Path | None = None) -> LongTermMemory:
    return LongTermMemory(user_id, tmp_dir or Path(tempfile.mkdtemp()))


class TestLongTermMemory:
    def test_initial_empty(self):
        ltm = _ltm()
        assert ltm.is_empty()

    def test_set_and_get(self):
        ltm = _ltm()
        ltm.set("profile", "язык", "Python")
        assert ltm.get_all()["profile"]["язык"] == "Python"

    def test_unknown_category_raises(self):
        ltm = _ltm()
        import pytest
        with pytest.raises(ValueError):
            ltm.set("unknown_cat", "k", "v")

    def test_remove_existing(self):
        ltm = _ltm()
        ltm.set("decisions", "стек", "FastAPI")
        assert ltm.remove("decisions", "стек") is True
        assert "стек" not in ltm.get_all()["decisions"]

    def test_remove_missing_returns_false(self):
        ltm = _ltm()
        assert ltm.remove("profile", "nonexistent") is False

    def test_persistence_across_instances(self):
        tmp = Path(tempfile.mkdtemp())
        ltm1 = LongTermMemory(42, tmp)
        ltm1.set("knowledge", "продукт", "маркетплейс")

        ltm2 = LongTermMemory(42, tmp)
        assert ltm2.get_all()["knowledge"]["продукт"] == "маркетплейс"

    def test_different_users_isolated(self):
        tmp = Path(tempfile.mkdtemp())
        ltm1 = LongTermMemory(1, tmp)
        ltm1.set("profile", "язык", "Python")

        ltm2 = LongTermMemory(2, tmp)
        assert ltm2.is_empty()

    def test_update_from_exchange_populates_data(self):
        tmp = Path(tempfile.mkdtemp())
        ltm = LongTermMemory(1, tmp)
        ltm.update_from_exchange(
            _mock_client("profile|язык: Python\ndecisions|стек: FastAPI"),
            "пишу на питоне",
            "отличный выбор",
        )
        all_data = ltm.get_all()
        assert all_data["profile"].get("язык") == "Python"
        assert all_data["decisions"].get("стек") == "FastAPI"

    def test_update_api_error_doesnt_break(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("API error")
        ltm = _ltm()
        ltm.update_from_exchange(client, "вопрос", "ответ")  # не должно бросить
        assert ltm.is_empty()

    def test_format_block_none_when_empty(self):
        ltm = _ltm()
        assert ltm.format_block() is None

    def test_format_block_shows_categories(self):
        ltm = _ltm()
        ltm.set("profile", "уровень", "senior")
        ltm.set("decisions", "стек", "FastAPI")
        block = ltm.format_block()
        assert block is not None
        assert "Профиль" in block
        assert "senior" in block
        assert "FastAPI" in block

    def test_format_telegram_empty(self):
        ltm = _ltm()
        text = ltm.format_telegram()
        assert "пуста" in text

    def test_format_telegram_shows_data(self):
        ltm = _ltm()
        ltm.set("knowledge", "продукт", "маркетплейс")
        text = ltm.format_telegram()
        assert "маркетплейс" in text
