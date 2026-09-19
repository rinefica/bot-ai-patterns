"""Тесты MemoryManager."""
from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from bot_ai_patterns.context_strategies.sliding_window import SlidingWindowStrategy
from bot_ai_patterns.memory.manager import MemoryManager


def _mock_client(reply: str = "") -> MagicMock:
    client = MagicMock()
    choice = SimpleNamespace(message=SimpleNamespace(content=reply))
    client.chat.completions.create.return_value = SimpleNamespace(choices=[choice])
    return client


def _manager(user_id: int = 1, client=None) -> MemoryManager:
    tmp = Path(tempfile.mkdtemp())
    mgr = MemoryManager(client or _mock_client(), user_id)
    # Подменяем директорию LTM на временную
    mgr.ltm._dir = tmp
    mgr.ltm._dir.mkdir(parents=True, exist_ok=True)
    return mgr


class TestMemoryManager:
    def test_build_messages_no_memory(self):
        mgr = _manager()
        strategy = SlidingWindowStrategy()
        strategy.add_user("привет")
        msgs = mgr.build_messages(strategy, "Ты ассистент.")
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "Ты ассистент."
        # Без памяти — только system + диалог
        roles = [m["role"] for m in msgs]
        assert roles.count("system") == 1

    def test_build_messages_injects_ltm_block(self):
        mgr = _manager()
        mgr.ltm.set("profile", "язык", "Python")
        strategy = SlidingWindowStrategy()
        strategy.add_user("вопрос")
        msgs = mgr.build_messages(strategy, "sys")
        system_contents = [m["content"] for m in msgs if m["role"] == "system"]
        assert any("Долговременная память" in c for c in system_contents)

    def test_build_messages_injects_wm_block(self):
        mgr = _manager()
        mgr.wm.set("задача", "написать бота")
        strategy = SlidingWindowStrategy()
        strategy.add_user("вопрос")
        msgs = mgr.build_messages(strategy, "sys")
        system_contents = [m["content"] for m in msgs if m["role"] == "system"]
        assert any("Рабочая память" in c for c in system_contents)

    def test_build_messages_order(self):
        mgr = _manager()
        mgr.ltm.set("decisions", "стек", "FastAPI")  # LTM без profile
        mgr.wm.set("задача", "API")
        strategy = SlidingWindowStrategy()
        strategy.add_user("q")
        msgs = mgr.build_messages(strategy, "sys")
        system_msgs = [m for m in msgs if m["role"] == "system"]
        assert len(system_msgs) == 3  # system_prompt + LTM + WM (профиль пуст)
        assert "Долговременная" in system_msgs[1]["content"]
        assert "Рабочая" in system_msgs[2]["content"]

    def test_reset_clears_wm_not_ltm(self):
        mgr = _manager()
        mgr.wm.set("k", "v")
        mgr.ltm.set("profile", "язык", "Python")
        mgr.reset()
        assert mgr.wm.data == {}
        assert not mgr.ltm.is_empty()

    def test_format_telegram_shows_both_layers(self):
        mgr = _manager()
        mgr.wm.set("задача", "маркетплейс")
        mgr.ltm.set("decisions", "стек", "FastAPI")
        mgr.profile.set("уровень", "senior")
        text = mgr.format_telegram()
        assert "WM" in text
        assert "LTM" in text
        assert "маркетплейс" in text
        assert "FastAPI" in text
        assert "senior" in text

    def test_wm_state_roundtrip(self):
        mgr = _manager()
        mgr.wm.set("k1", "v1")
        state = mgr.get_wm_state()

        mgr2 = _manager()
        mgr2.load_wm_state(state)
        assert mgr2.wm.data == {"k1": "v1"}

    def test_after_exchange_calls_extractors(self):
        client = _mock_client("задача: API")
        mgr = _manager(client=client)
        mgr.after_exchange("пишу API", "хорошо")
        # LLM был вызван для WM и LTM экстракции
        assert client.chat.completions.create.call_count == 2

    def test_profile_block_injected_before_ltm(self):
        mgr = _manager()
        mgr.profile.set("стиль", "кратко")
        mgr.ltm.set("decisions", "стек", "FastAPI")
        strategy = SlidingWindowStrategy()
        strategy.add_user("вопрос")
        msgs = mgr.build_messages(strategy, "sys")
        system_msgs = [m for m in msgs if m["role"] == "system"]
        # Порядок: system_prompt → profile → ltm
        assert "Профиль" in system_msgs[1]["content"]
        assert "Долговременная" in system_msgs[2]["content"]

    def test_profile_block_not_injected_when_empty(self):
        mgr = _manager()
        strategy = SlidingWindowStrategy()
        strategy.add_user("вопрос")
        msgs = mgr.build_messages(strategy, "sys")
        system_contents = [m["content"] for m in msgs if m["role"] == "system"]
        assert not any("Профиль" in c for c in system_contents)

    def test_format_telegram_shows_all_layers(self):
        mgr = _manager()
        mgr.wm.set("задача", "API")
        mgr.ltm.set("decisions", "стек", "FastAPI")
        mgr.profile.set("стиль", "кратко")
        text = mgr.format_telegram()
        assert "STM" in text
        assert "WM" in text
        assert "LTM" in text
        assert "Профиль" in text
