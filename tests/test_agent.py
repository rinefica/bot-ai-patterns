"""Тесты Agent: токены, стратегии, переполнение, сброс."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from bot_ai_patterns.agent import (
    Agent,
    ContextOverflowError,
    SessionStats,
    TokenUsage,
)
from bot_ai_patterns.config import CONTEXT_LIMIT, CONTEXT_WARN_THRESHOLD
from bot_ai_patterns.context_strategies import (
    BranchingStrategy,
    SlidingWindowStrategy,
    StickyFactsStrategy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_storage(saved: dict | None = None) -> MagicMock:
    storage = MagicMock()
    storage.load.return_value = saved or {}
    return storage


def _make_openai_response(
    reply: str = "ответ",
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
) -> SimpleNamespace:
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    choice = SimpleNamespace(message=SimpleNamespace(content=reply))
    return SimpleNamespace(choices=[choice], usage=usage)


def _make_agent(
    client=None,
    saved: dict | None = None,
    strategy=None,
) -> Agent:
    if client is None:
        client = MagicMock()
    storage = _make_storage(saved)
    return Agent(client, user_id=1, storage=storage, strategy=strategy)


# ---------------------------------------------------------------------------
# TokenUsage
# ---------------------------------------------------------------------------

class TestTokenUsage:
    def test_fields(self):
        u = TokenUsage(prompt_tokens=100, completion_tokens=40, total_tokens=140)
        assert u.prompt_tokens == 100
        assert u.completion_tokens == 40
        assert u.total_tokens == 140


# ---------------------------------------------------------------------------
# SessionStats
# ---------------------------------------------------------------------------

class TestSessionStats:
    def test_initial_state(self):
        s = SessionStats()
        assert s.turns == 0
        assert s.total_tokens == 0

    def test_add_accumulates(self):
        s = SessionStats()
        s.add(TokenUsage(100, 50, 150))
        s.add(TokenUsage(200, 60, 260))
        assert s.turns == 2
        assert s.total_tokens == 410

    def test_cost_rub(self):
        s = SessionStats()
        s.add(TokenUsage(500, 500, 1000))
        assert s.cost_rub == pytest.approx(6.0)

    def test_context_fill_pct(self):
        s = SessionStats()
        assert s.context_fill_pct(CONTEXT_LIMIT // 2) == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# Agent.chat — базовая работа
# ---------------------------------------------------------------------------

class TestAgentChat:
    def test_returns_reply_and_usage(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _make_openai_response(
            reply="привет", prompt_tokens=80, completion_tokens=20
        )
        agent = _make_agent(client)
        reply, usage = agent.chat("hello")

        assert reply == "привет"
        assert usage.prompt_tokens == 80
        assert usage.total_tokens == 100

    def test_last_usage_updated(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _make_openai_response(prompt_tokens=90)
        agent = _make_agent(client)
        agent.chat("вопрос")
        assert agent.last_usage.prompt_tokens == 90

    def test_stats_accumulate(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            _make_openai_response(prompt_tokens=100, completion_tokens=50),
            _make_openai_response(prompt_tokens=200, completion_tokens=60),
        ]
        agent = _make_agent(client)
        agent.chat("раз")
        agent.chat("два")

        assert agent.stats.turns == 2
        assert agent.stats.total_tokens == 410

    def test_history_len_grows_per_turn(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _make_openai_response()
        agent = _make_agent(client)

        assert agent.history_len == 0
        agent.chat("раз")
        assert agent.history_len == 2  # user + assistant
        agent.chat("два")
        assert agent.history_len == 4

    def test_strategy_receives_messages(self):
        """Стратегия должна получать add_user и add_assistant."""
        mock_strategy = MagicMock(spec=SlidingWindowStrategy)
        mock_strategy.build_messages.return_value = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "вопрос"},
        ]
        mock_strategy.name = "mock"

        client = MagicMock()
        client.chat.completions.create.return_value = _make_openai_response(reply="ответ")
        agent = _make_agent(client, strategy=mock_strategy)
        agent.chat("вопрос")

        mock_strategy.add_user.assert_called_once_with("вопрос")
        mock_strategy.add_assistant.assert_called_once_with("ответ")


# ---------------------------------------------------------------------------
# Agent.chat — переполнение контекста
# ---------------------------------------------------------------------------

class TestContextOverflow:
    def test_raises_on_overflow(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _make_openai_response(
            prompt_tokens=CONTEXT_LIMIT + 1
        )
        agent = _make_agent(client)

        with pytest.raises(ContextOverflowError) as exc_info:
            agent.chat("переполнение")

        assert exc_info.value.prompt_tokens == CONTEXT_LIMIT + 1

    def test_history_not_grows_on_overflow(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _make_openai_response(
            prompt_tokens=CONTEXT_LIMIT + 1
        )
        agent = _make_agent(client)

        with pytest.raises(ContextOverflowError):
            agent.chat("переполнение")

        assert agent.history_len == 0

    def test_strategy_rolled_back_on_overflow(self):
        mock_strategy = MagicMock(spec=SlidingWindowStrategy)
        mock_strategy.build_messages.return_value = [{"role": "user", "content": "q"}]
        mock_strategy.name = "mock"

        client = MagicMock()
        client.chat.completions.create.return_value = _make_openai_response(
            prompt_tokens=CONTEXT_LIMIT + 1
        )
        agent = _make_agent(client, strategy=mock_strategy)

        with pytest.raises(ContextOverflowError):
            agent.chat("переполнение")

        mock_strategy.rollback_user.assert_called_once()

    def test_stats_not_updated_on_overflow(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _make_openai_response(
            prompt_tokens=CONTEXT_LIMIT + 1
        )
        agent = _make_agent(client)

        with pytest.raises(ContextOverflowError):
            agent.chat("переполнение")

        assert agent.stats.turns == 0


# ---------------------------------------------------------------------------
# Agent.reset
# ---------------------------------------------------------------------------

class TestAgentReset:
    def test_reset_clears_stats_and_history(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _make_openai_response()
        agent = _make_agent(client)
        agent.chat("вопрос")

        agent.reset()

        assert agent.stats.turns == 0
        assert agent.history_len == 0
        assert agent.last_usage is None

    def test_reset_calls_strategy_reset(self):
        mock_strategy = MagicMock(spec=SlidingWindowStrategy)
        mock_strategy.build_messages.return_value = []
        mock_strategy.name = "mock"

        agent = _make_agent(strategy=mock_strategy)
        agent.reset()

        mock_strategy.reset.assert_called_once()


# ---------------------------------------------------------------------------
# Agent.switch_strategy
# ---------------------------------------------------------------------------

class TestSwitchStrategy:
    def test_switch_changes_strategy(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _make_openai_response()
        agent = _make_agent(client, strategy=SlidingWindowStrategy())

        agent.chat("вопрос")
        new_strategy = BranchingStrategy()
        agent.switch_strategy(new_strategy)

        assert agent.strategy is new_strategy

    def test_new_strategy_initialized_from_history(self):
        """switch_strategy передаёт аудит-лог в init_from_messages."""
        client = MagicMock()
        client.chat.completions.create.return_value = _make_openai_response()
        agent = _make_agent(client, strategy=SlidingWindowStrategy())
        agent.chat("вопрос")

        mock_strategy = MagicMock(spec=BranchingStrategy)
        mock_strategy.name = "branching"
        agent.switch_strategy(mock_strategy)

        mock_strategy.init_from_messages.assert_called_once()
        passed_messages = mock_strategy.init_from_messages.call_args[0][0]
        assert len(passed_messages) == 2  # user + assistant

    def test_default_strategy_is_sliding_window(self):
        agent = _make_agent()
        assert isinstance(agent.strategy, SlidingWindowStrategy)


# ---------------------------------------------------------------------------
# Предупреждение при приближении к лимиту
# ---------------------------------------------------------------------------

class TestWarningThreshold:
    def test_threshold_is_80_percent(self):
        assert CONTEXT_WARN_THRESHOLD == int(CONTEXT_LIMIT * 0.8)

    def test_threshold_below_limit(self):
        assert CONTEXT_WARN_THRESHOLD < CONTEXT_LIMIT

    def test_usage_below_threshold(self):
        u = TokenUsage(
            prompt_tokens=CONTEXT_WARN_THRESHOLD - 1,
            completion_tokens=50,
            total_tokens=CONTEXT_WARN_THRESHOLD + 49,
        )
        assert u.prompt_tokens < CONTEXT_WARN_THRESHOLD

    def test_usage_at_threshold(self):
        u = TokenUsage(
            prompt_tokens=CONTEXT_WARN_THRESHOLD,
            completion_tokens=50,
            total_tokens=CONTEXT_WARN_THRESHOLD + 50,
        )
        assert u.prompt_tokens >= CONTEXT_WARN_THRESHOLD


# ---------------------------------------------------------------------------
# Загрузка из хранилища
# ---------------------------------------------------------------------------

class TestStorageLoad:
    def test_loads_strategy_state_if_name_matches(self):
        saved = {
            "strategy_name": "sliding_window",
            "strategy_state": {
                "strategy": "sliding_window",
                "messages": [{"role": "user", "content": "привет"}],
                "total": 1,
            },
            "all_messages": [{"role": "user", "content": "привет"}],
        }
        agent = _make_agent(saved=saved, strategy=SlidingWindowStrategy())
        assert agent.history_len == 1

    def test_loads_all_messages_as_audit_log(self):
        saved = {
            "strategy_name": "sliding_window",
            "strategy_state": {},
            "all_messages": [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
            ],
        }
        agent = _make_agent(saved=saved)
        assert agent.history_len == 2
