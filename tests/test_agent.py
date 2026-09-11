"""Тесты подсчёта токенов и компрессии истории в Agent."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from bot_ai_patterns.agent import (
    Agent,
    CompressionStats,
    ContextOverflowError,
    SessionStats,
    TokenUsage,
)
from bot_ai_patterns.config import (
    COMPRESS_AFTER_N,
    CONTEXT_LIMIT,
    CONTEXT_WARN_THRESHOLD,
    RECENT_KEEP,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_storage(saved: dict | None = None):
    """Заглушка JSONStorage: load возвращает dict (новый формат)."""
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


def _make_agent(client=None, saved: dict | None = None, use_compression: bool = False) -> Agent:
    if client is None:
        client = MagicMock()
    storage = _make_storage(saved)
    return Agent(client, user_id=1, storage=storage, use_compression=use_compression)


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
        assert s.cost_rub == pytest.approx(6.0)

    def test_context_fill_pct(self):
        s = SessionStats()
        assert s.context_fill_pct(CONTEXT_LIMIT // 2) == pytest.approx(50.0)

    def test_context_fill_pct_full(self):
        s = SessionStats()
        assert s.context_fill_pct(CONTEXT_LIMIT) == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# CompressionStats
# ---------------------------------------------------------------------------

class TestCompressionStats:
    def test_initial_state(self):
        cs = CompressionStats()
        assert cs.compressions_done == 0
        assert cs.messages_compressed == 0
        assert cs.tokens_saved == 0

    def test_tokens_saved(self):
        cs = CompressionStats(tokens_before=1000, tokens_after=300)
        assert cs.tokens_saved == 700

    def test_tokens_saved_no_negative(self):
        cs = CompressionStats(tokens_before=100, tokens_after=200)
        assert cs.tokens_saved == 0


# ---------------------------------------------------------------------------
# Agent.chat — нормальная работа (без компрессии)
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
        """Prompt_tokens растёт с каждым ходом — история накапливается."""
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

        assert p1 < p2 < p3

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
        client = MagicMock()
        client.chat.completions.create.return_value = _make_openai_response(
            prompt_tokens=CONTEXT_LIMIT + 1, completion_tokens=100
        )
        agent = _make_agent(client)

        with pytest.raises(ContextOverflowError) as exc_info:
            agent.chat("переполненный запрос")

        assert exc_info.value.prompt_tokens == CONTEXT_LIMIT + 1

    def test_history_not_modified_on_overflow(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _make_openai_response(
            prompt_tokens=CONTEXT_LIMIT + 1, completion_tokens=100
        )
        agent = _make_agent(client)
        initial_len = agent.history_len

        with pytest.raises(ContextOverflowError):
            agent.chat("переполненный запрос")

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

        agent.reset()

        assert agent.stats.turns == 0
        assert agent.stats.total_tokens == 0
        assert agent.last_usage is None

    def test_reset_clears_history(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _make_openai_response()
        agent = _make_agent(client)
        agent.chat("вопрос")

        agent.reset()

        assert agent.history_len == 0

    def test_reset_clears_summary_and_recent(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _make_openai_response()
        saved = {"history": [], "summary": "старое резюме", "recent": [{"role": "user", "content": "x"}]}
        agent = _make_agent(client, saved=saved, use_compression=True)

        agent.reset()

        assert agent.summary == ""
        assert agent.recent_len == 0
        assert agent.compress_stats.compressions_done == 0


# ---------------------------------------------------------------------------
# Компрессия: toggle_compression
# ---------------------------------------------------------------------------

class TestToggleCompression:
    def test_default_compression_off(self):
        agent = _make_agent()
        assert not agent.compression_enabled

    def test_toggle_turns_on(self):
        agent = _make_agent()
        result = agent.toggle_compression()
        assert result is True
        assert agent.compression_enabled

    def test_toggle_turns_off(self):
        agent = _make_agent(use_compression=True)
        result = agent.toggle_compression()
        assert result is False
        assert not agent.compression_enabled

    def test_double_toggle_restores(self):
        agent = _make_agent()
        agent.toggle_compression()
        agent.toggle_compression()
        assert not agent.compression_enabled


# ---------------------------------------------------------------------------
# Компрессия: recent накапливается и сжимается
# ---------------------------------------------------------------------------

class TestCompression:
    def _make_compressed_agent(self, n_responses: int = COMPRESS_AFTER_N + 2):
        client = MagicMock()
        client.chat.completions.create.return_value = _make_openai_response(
            prompt_tokens=100, completion_tokens=50
        )
        # summarize тоже вызывает client, настраиваем отдельно через Compressor mock
        agent = _make_agent(client, use_compression=True)
        # Мокаем compressor напрямую чтобы не делать реальных запросов
        agent._compressor = MagicMock()
        agent._compressor.summarize.return_value = "краткое резюме"
        return agent, client

    def test_recent_grows_with_messages(self):
        agent, client = self._make_compressed_agent()
        client.chat.completions.create.return_value = _make_openai_response()

        agent.chat("раз")
        agent.chat("два")

        assert agent.recent_len == 4  # 2 пары user+assistant

    def test_compression_triggered_at_threshold(self):
        agent, _ = self._make_compressed_agent()

        for i in range(COMPRESS_AFTER_N // 2):  # COMPRESS_AFTER_N сообщений = N/2 ходов
            agent.chat(f"вопрос {i}")

        assert agent.compress_stats.compressions_done == 1

    def test_recent_trimmed_after_compression(self):
        agent, _ = self._make_compressed_agent()

        for i in range(COMPRESS_AFTER_N // 2):
            agent.chat(f"вопрос {i}")

        assert agent.recent_len <= RECENT_KEEP

    def test_summary_set_after_compression(self):
        agent, _ = self._make_compressed_agent()

        for i in range(COMPRESS_AFTER_N // 2):
            agent.chat(f"вопрос {i}")

        assert agent.summary == "краткое резюме"

    def test_full_history_always_maintained(self):
        """_history растёт независимо от компрессии."""
        agent, _ = self._make_compressed_agent()

        for i in range(COMPRESS_AFTER_N // 2):
            agent.chat(f"вопрос {i}")

        # history_len = N/2 ходов * 2 сообщения
        assert agent.history_len == COMPRESS_AFTER_N

    def test_compressor_receives_old_messages(self):
        """Компрессор получает сообщения старше RECENT_KEEP, не весь recent."""
        agent, _ = self._make_compressed_agent()

        for i in range(COMPRESS_AFTER_N // 2):
            agent.chat(f"вопрос {i}")

        call_args = agent._compressor.summarize.call_args
        compressed_msgs = call_args[0][0]
        assert len(compressed_msgs) == COMPRESS_AFTER_N - RECENT_KEEP

    def test_messages_compressed_counter(self):
        agent, _ = self._make_compressed_agent()

        for i in range(COMPRESS_AFTER_N // 2):
            agent.chat(f"вопрос {i}")

        assert agent.compress_stats.messages_compressed == COMPRESS_AFTER_N - RECENT_KEEP

    def test_overflow_rolls_back_recent(self):
        """При ContextOverflowError recent не должен расти."""
        client = MagicMock()
        client.chat.completions.create.return_value = _make_openai_response(
            prompt_tokens=CONTEXT_LIMIT + 1, completion_tokens=100
        )
        agent = _make_agent(client, use_compression=True)
        agent._compressor = MagicMock()
        initial_recent = agent.recent_len

        with pytest.raises(ContextOverflowError):
            agent.chat("переполнение")

        assert agent.recent_len == initial_recent


# ---------------------------------------------------------------------------
# Storage backward compatibility
# ---------------------------------------------------------------------------

class TestStorageBackwardCompat:
    def test_loads_old_list_format(self):
        """Старый формат (список) загружается без ошибок."""
        old_history = [
            {"role": "system", "content": "Ты ассистент."},
            {"role": "user", "content": "привет"},
        ]
        saved = {"history": old_history, "summary": "", "recent": []}
        agent = _make_agent(saved=saved)

        assert agent.history_len == 1  # 1 сообщение без системного

    def test_loads_summary_and_recent(self):
        saved = {
            "history": [{"role": "system", "content": "Ты ассистент."}],
            "summary": "было обсуждение питона",
            "recent": [{"role": "user", "content": "продолжим?"}],
        }
        agent = _make_agent(saved=saved, use_compression=True)

        assert agent.summary == "было обсуждение питона"
        assert agent.recent_len == 1


# ---------------------------------------------------------------------------
# Предупреждение при приближении к лимиту
# ---------------------------------------------------------------------------

class TestWarningThreshold:
    def test_warn_threshold_is_below_limit(self):
        assert CONTEXT_WARN_THRESHOLD < CONTEXT_LIMIT

    def test_warn_threshold_is_80_percent(self):
        assert CONTEXT_WARN_THRESHOLD == int(CONTEXT_LIMIT * 0.8)

    def test_no_warning_below_threshold(self):
        usage = TokenUsage(
            prompt_tokens=CONTEXT_WARN_THRESHOLD - 1,
            completion_tokens=50,
            total_tokens=CONTEXT_WARN_THRESHOLD + 49,
        )
        assert usage.prompt_tokens < CONTEXT_WARN_THRESHOLD

    def test_warning_at_threshold(self):
        usage = TokenUsage(
            prompt_tokens=CONTEXT_WARN_THRESHOLD,
            completion_tokens=50,
            total_tokens=CONTEXT_WARN_THRESHOLD + 50,
        )
        assert usage.prompt_tokens >= CONTEXT_WARN_THRESHOLD
