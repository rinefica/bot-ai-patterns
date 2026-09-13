"""Тесты StickyFactsStrategy."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from bot_ai_patterns.context_strategies.sticky_facts import StickyFactsStrategy


def _mock_client(facts_reply: str = "цель: разработать ТЗ\nязык: Python") -> MagicMock:
    client = MagicMock()
    choice = SimpleNamespace(message=SimpleNamespace(content=facts_reply))
    client.chat.completions.create.return_value = SimpleNamespace(choices=[choice])
    return client


class TestStickyFacts:
    def test_initial_no_facts(self):
        s = StickyFactsStrategy(_mock_client(), window_size=4)
        assert s.facts == {}
        assert s.message_count == 0

    def test_facts_extracted_after_assistant(self):
        s = StickyFactsStrategy(_mock_client("цель: разработать ТЗ"), window_size=4)
        s.add_user("хочу разработать ТЗ")
        s.add_assistant("хорошо")
        assert len(s.facts) > 0

    def test_facts_appear_in_context(self):
        s = StickyFactsStrategy(_mock_client("цель: написать бота"), window_size=4)
        s.add_user("пишу бота")
        s.add_assistant("понял")
        msgs = s.build_messages("sys")
        facts_block = next(
            (m for m in msgs if "Ключевые факты" in m.get("content", "")), None
        )
        assert facts_block is not None

    def test_no_facts_block_when_empty(self):
        s = StickyFactsStrategy(MagicMock(), window_size=4)
        msgs = s.build_messages("sys")
        assert all("Ключевые факты" not in m.get("content", "") for m in msgs)

    def test_window_limits_recent_messages(self):
        s = StickyFactsStrategy(_mock_client("к: в"), window_size=2)
        for i in range(6):
            s.add_user(f"q{i}")
            s.add_assistant(f"a{i}")
        msgs = s.build_messages("sys")
        recent = [m for m in msgs if m["role"] in ("user", "assistant")]
        assert len(recent) <= 2

    def test_rollback_removes_user_message(self):
        s = StickyFactsStrategy(MagicMock())
        s.add_user("вопрос")
        s.rollback_user()
        assert s.message_count == 0
        assert s.total_count == 0

    def test_extraction_failure_doesnt_break_chat(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("API error")
        s = StickyFactsStrategy(client)
        s.add_user("вопрос")
        s.add_assistant("ответ")  # не должно бросить исключение
        assert s.facts == {}

    def test_reset_clears_facts_and_messages(self):
        s = StickyFactsStrategy(_mock_client("к: в"))
        s.add_user("q")
        s.add_assistant("a")
        s.reset()
        assert s.facts == {}
        assert s.message_count == 0

    def test_state_roundtrip(self):
        s = StickyFactsStrategy(_mock_client("цель: тест"), window_size=4)
        s.add_user("вопрос")
        s.add_assistant("ответ")
        state = s.get_state()

        s2 = StickyFactsStrategy(MagicMock(), window_size=4)
        s2.load_state(state)
        assert s2.facts == s.facts

    def test_stats_lines_show_facts_count(self):
        s = StickyFactsStrategy(_mock_client("к: в"))
        s.add_user("q")
        s.add_assistant("a")
        lines = s.stats_lines()
        assert any("Фактов" in l for l in lines)
