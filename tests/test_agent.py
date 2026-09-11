"""Тесты подсчёта токенов в Agent."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from bot_ai_patterns.agent import (
    Agent,
    ContextOverflowError,
    SessionStats,
    TokenUsage,
)
from bot_ai_patterns.config import CONTEXT_LIMIT, CONTEXT_WARN_THRESHOLD


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_storage(history: list | None = None):
    """Заглушка JSONStorage."""
    storage = MagicMock()
    storage.load.return_value = history or []
    return storage


def _make_openai_response(
    reply: str = "ответ",
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
) -> MagicMock:
    """Возвращает объект, имитирующий ответ openai.ChatCompletion."""
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    choice = SimpleNamespace(message=SimpleNamespace(content=reply))
    return SimpleNamespace(choices=[choice], usage=usage)


def _make_agent(
    client=None,
    history=None,
) -> Agent:
    if client is None:
        client = MagicMock()
    storage = _make_storage(history)
    return Agent(client, user_id=1, storage=storage)


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
        assert s.history == []

    def test_add_accumulates(self):
        s = SessionStats()
        s.add(TokenUsage(100, 50, 150))
        s.add(TokenUsage(200, 60, 260))

        assert s.turns == 2
        assert s.total_prompt == 300
        assert s.total_completion == 110
        assert s.total_tokens == 410

    def test_cost_rub(self):
        s = SessionStats()
        s.add(TokenUsage(500, 500, 1000))
        # 1000 токенов * 6 руб / 1000 = 6 руб
        assert s.cost_rub == pytest.approx(6.0)

    def test_context_fill_pct(self):
        s = SessionStats()
        pct = s.context_fill_pct(CONTEXT_LIMIT // 2)
        assert pct == pytest.approx(50.0)

    def test_context_fill_pct_full(self):
        s = SessionStats()
        assert s.context_fill_pct(CONTEXT_LIMIT) == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Agent.chat — нормальная работа
# ---------------------------------------------------------------------------

class TestAgentChatTokens:
    def test_chat_returns_reply_and_usage(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _make_openai_response(
            reply="привет", prompt_tokens=80, completion_tokens=20
        )
        agent = _make_agent(client)

        reply, usage = agent.chat("hello")

        assert reply == "привет"
        assert usage.prompt_tokens == 80
        assert usage.completion_tokens == 20
        assert usage.total_tokens == 100

    def test_last_usage_updated_after_chat(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _make_openai_response(
            prompt_tokens=90, completion_tokens=30
        )
        agent = _make_agent(client)
        agent.chat("вопрос")

        assert agent.last_usage is not None
        assert agent.last_usage.prompt_tokens == 90

    def test_stats_accumulate_across_turns(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            _make_openai_response(prompt_tokens=100, completion_tokens=50),
            _make_openai_response(prompt_tokens=200, completion_tokens=60),
            _make_openai_response(prompt_tokens=350, completion_tokens=70),
        ]
        agent = _make_agent(client)

        agent.chat("раз")
        agent.chat("два")
        agent.chat("три")

        stats = agent.stats
        assert stats.turns == 3
        assert stats.total_prompt == 650
        assert stats.total_completion == 180
        assert stats.total_tokens == 830

    def test_prompt_tokens_grow_with_history(self):
        """Prompt_tokens должен расти с каждым ходом (история накапливается)."""
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            _make_openai_response(prompt_tokens=50,  completion_tokens=20),
            _make_openai_response(prompt_tokens=120, completion_tokens=25),
            _make_openai_response(prompt_tokens=200, completion_tokens=30),
        ]
        agent = _make_agent(client)

        agent.chat("первый")
        p1 = agent.last_usage.prompt_tokens

        agent.chat("второй")
        p2 = agent.last_usage.prompt_tokens

        agent.chat("третий")
        p3 = agent.last_usage.prompt_tokens

        assert p1 < p2 < p3, "Prompt_tokens должен расти с каждым ходом"

    def test_history_len_grows_per_turn(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _make_openai_response()
        agent = _make_agent(client)

        assert agent.history_len == 0
        agent.chat("раз")
        assert agent.history_len == 2   # user + assistant
        agent.chat("два")
        assert agent.history_len == 4


# ---------------------------------------------------------------------------
# Agent.chat — переполнение контекста
# ---------------------------------------------------------------------------

class TestContextOverflow:
    def test_raises_on_overflow(self):
        """Если API вернуло prompt_tokens > CONTEXT_LIMIT — бросаем ошибку."""
        client = MagicMock()
        client.chat.completions.create.return_value = _make_openai_response(
            prompt_tokens=CONTEXT_LIMIT + 1, completion_tokens=100
        )
        agent = _make_agent(client)

        with pytest.raises(ContextOverflowError) as exc_info:
            agent.chat("переполненный запрос")

        assert exc_info.value.prompt_tokens == CONTEXT_LIMIT + 1

    def test_history_not_modified_on_overflow(self):
        """После ContextOverflowError история не должна меняться."""
        client = MagicMock()
        client.chat.completions.create.return_value = _make_openai_response(
            prompt_tokens=CONTEXT_LIMIT + 1, completion_tokens=100
        )
        agent = _make_agent(client)
        initial_len = agent.history_len

        with pytest.raises(ContextOverflowError):
            agent.chat("переполненный запрос")

        # Сообщение пользователя откатывается, история не растёт
        assert agent.history_len == initial_len

    def test_stats_not_updated_on_overflow(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _make_openai_response(
            prompt_tokens=CONTEXT_LIMIT + 1, completion_tokens=100
        )
        agent = _make_agent(client)

        with pytest.raises(ContextOverflowError):
            agent.chat("переполнение")

        assert agent.stats.turns == 0

    def test_error_message_contains_limit(self):
        with pytest.raises(ContextOverflowError) as exc_info:
            raise ContextOverflowError(CONTEXT_LIMIT + 500)

        assert str(CONTEXT_LIMIT) in str(exc_info.value)


# ---------------------------------------------------------------------------
# Agent.reset
# ---------------------------------------------------------------------------

class TestAgentReset:
    def test_reset_clears_stats(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _make_openai_response(
            prompt_tokens=100, completion_tokens=50
        )
        agent = _make_agent(client)
        agent.chat("вопрос")
        assert agent.stats.turns == 1

        agent.reset()

        assert agent.stats.turns == 0
        assert agent.stats.total_tokens == 0
        assert agent.last_usage is None

    def test_reset_clears_history(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _make_openai_response()
        agent = _make_agent(client)
        agent.chat("вопрос")
        assert agent.history_len > 0

        agent.reset()

        assert agent.history_len == 0


# ---------------------------------------------------------------------------
# Предупреждение при приближении к лимиту
# ---------------------------------------------------------------------------

class TestWarningThreshold:
    def test_warn_threshold_is_below_limit(self):
        assert CONTEXT_WARN_THRESHOLD < CONTEXT_LIMIT

    def test_warn_threshold_is_80_percent(self):
        assert CONTEXT_WARN_THRESHOLD == int(CONTEXT_LIMIT * 0.8)

    def test_no_warning_below_threshold(self):
        """Ниже порога: токены не вызывают предупреждения."""
        usage = TokenUsage(
            prompt_tokens=CONTEXT_WARN_THRESHOLD - 1,
            completion_tokens=50,
            total_tokens=CONTEXT_WARN_THRESHOLD + 49,
        )
        assert usage.prompt_tokens < CONTEXT_WARN_THRESHOLD

    def test_warning_at_threshold(self):
        """На пороге: нужно показывать предупреждение."""
        usage = TokenUsage(
            prompt_tokens=CONTEXT_WARN_THRESHOLD,
            completion_tokens=50,
            total_tokens=CONTEXT_WARN_THRESHOLD + 50,
        )
        assert usage.prompt_tokens >= CONTEXT_WARN_THRESHOLD
