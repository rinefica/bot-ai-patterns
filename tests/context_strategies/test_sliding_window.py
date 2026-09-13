"""Тесты SlidingWindowStrategy."""
from __future__ import annotations

import pytest

from bot_ai_patterns.context_strategies.sliding_window import SlidingWindowStrategy


class TestSlidingWindow:
    def test_initial_empty(self):
        s = SlidingWindowStrategy(window_size=4)
        assert s.message_count == 0
        assert s.total_count == 0

    def test_messages_added(self):
        s = SlidingWindowStrategy(window_size=4)
        s.add_user("привет")
        s.add_assistant("здравствуй")
        assert s.total_count == 1
        assert s.message_count == 2

    def test_build_messages_includes_system(self):
        s = SlidingWindowStrategy(window_size=4)
        s.add_user("вопрос")
        s.add_assistant("ответ")
        msgs = s.build_messages("Ты ассистент.")
        assert msgs[0] == {"role": "system", "content": "Ты ассистент."}

    def test_window_truncates_old_messages(self):
        s = SlidingWindowStrategy(window_size=4)
        for i in range(6):
            s.add_user(f"вопрос {i}")
            s.add_assistant(f"ответ {i}")
        msgs = s.build_messages("sys")
        assert len(msgs) == 5  # system + 4

    def test_window_keeps_last_n(self):
        s = SlidingWindowStrategy(window_size=2)
        s.add_user("первый")
        s.add_assistant("ответ 1")
        s.add_user("второй")
        s.add_assistant("ответ 2")
        s.add_user("третий")
        s.add_assistant("ответ 3")
        msgs = s.build_messages("sys")
        contents = [m["content"] for m in msgs[1:]]
        assert "первый" not in contents
        assert "третий" in contents

    def test_rollback_user(self):
        s = SlidingWindowStrategy()
        s.add_user("вопрос")
        s.rollback_user()
        assert s.total_count == 0
        assert s.message_count == 0

    def test_rollback_when_last_is_assistant_is_noop(self):
        s = SlidingWindowStrategy()
        s.add_assistant("ответ")
        s.rollback_user()
        assert s.message_count == 1

    def test_reset_clears_all(self):
        s = SlidingWindowStrategy()
        s.add_user("вопрос")
        s.add_assistant("ответ")
        s.reset()
        assert s.message_count == 0
        assert s.total_count == 0

    def test_init_from_messages_takes_last_n(self):
        s = SlidingWindowStrategy(window_size=2)
        history = [{"role": "user", "content": f"q{i}"} for i in range(6)]
        s.init_from_messages(history)
        msgs = s.build_messages("sys")
        assert len(msgs) == 3  # system + 2

    def test_state_roundtrip(self):
        s = SlidingWindowStrategy(window_size=4)
        s.add_user("раз")
        s.add_assistant("два")
        state = s.get_state()

        s2 = SlidingWindowStrategy(window_size=4)
        s2.load_state(state)
        assert s2.message_count == s.message_count

    def test_stats_lines_content(self):
        s = SlidingWindowStrategy(window_size=4)
        s.add_user("q")
        lines = s.stats_lines()
        assert any("Sliding Window" in l for l in lines)
        assert any("контексте" in l for l in lines)
