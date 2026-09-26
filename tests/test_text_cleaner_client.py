"""Tests for text_cleaner_client.clean_and_summarize — MCP session is mocked."""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot_ai_patterns.text_cleaner_client import clean_and_summarize


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _AsyncCtx:
    def __init__(self, val):
        self._val = val

    async def __aenter__(self):
        return self._val

    async def __aexit__(self, *_):
        pass


def _tool_result(data: dict) -> MagicMock:
    item = MagicMock()
    item.text = json.dumps(data)
    result = MagicMock()
    result.is_error = False
    result.content = [item]
    return result


def _error_result() -> MagicMock:
    result = MagicMock()
    result.is_error = True
    result.content = []
    return result


def _patched(session: AsyncMock):
    stdio_mock = MagicMock()
    stdio_mock.return_value = _AsyncCtx((MagicMock(), MagicMock()))
    session_cls = MagicMock()
    session_cls.return_value = _AsyncCtx(session)
    return stdio_mock, session_cls


_SUMMARY_DATA = {
    "cleaned_text": "Очищенный текст",
    "item_type": "платье",
    "summary": "Летнее платье A-силуэт.",
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCleanAndSummarizeClient:
    @pytest.mark.asyncio
    async def test_returns_dict_with_summary(self):
        session = AsyncMock()
        session.initialize = AsyncMock()
        session.call_tool = AsyncMock(return_value=_tool_result(_SUMMARY_DATA))

        stdio_mock, session_cls = _patched(session)
        with (
            patch("bot_ai_patterns.text_cleaner_client.stdio_client", stdio_mock),
            patch("bot_ai_patterns.text_cleaner_client.ClientSession", session_cls),
        ):
            result = await clean_and_summarize("сырой текст")

        assert result["summary"] == _SUMMARY_DATA["summary"]
        assert result["item_type"] == "платье"

    @pytest.mark.asyncio
    async def test_calls_correct_tool(self):
        session = AsyncMock()
        session.initialize = AsyncMock()
        session.call_tool = AsyncMock(return_value=_tool_result(_SUMMARY_DATA))

        stdio_mock, session_cls = _patched(session)
        with (
            patch("bot_ai_patterns.text_cleaner_client.stdio_client", stdio_mock),
            patch("bot_ai_patterns.text_cleaner_client.ClientSession", session_cls),
        ):
            await clean_and_summarize("текст")

        session.call_tool.assert_called_once_with("clean_and_summarize", {"text": "текст"})

    @pytest.mark.asyncio
    async def test_raises_on_error_result(self):
        session = AsyncMock()
        session.initialize = AsyncMock()
        session.call_tool = AsyncMock(return_value=_error_result())

        stdio_mock, session_cls = _patched(session)
        with (
            patch("bot_ai_patterns.text_cleaner_client.stdio_client", stdio_mock),
            patch("bot_ai_patterns.text_cleaner_client.ClientSession", session_cls),
        ):
            with pytest.raises(RuntimeError, match="clean_and_summarize error"):
                await clean_and_summarize("текст")

    @pytest.mark.asyncio
    async def test_returns_empty_dict_on_no_content(self):
        session = AsyncMock()
        session.initialize = AsyncMock()
        result_mock = MagicMock()
        result_mock.is_error = False
        result_mock.content = []
        session.call_tool = AsyncMock(return_value=result_mock)

        stdio_mock, session_cls = _patched(session)
        with (
            patch("bot_ai_patterns.text_cleaner_client.stdio_client", stdio_mock),
            patch("bot_ai_patterns.text_cleaner_client.ClientSession", session_cls),
        ):
            result = await clean_and_summarize("текст")

        assert result == {}

    @pytest.mark.asyncio
    async def test_session_initialized(self):
        session = AsyncMock()
        session.initialize = AsyncMock()
        session.call_tool = AsyncMock(return_value=_tool_result(_SUMMARY_DATA))

        stdio_mock, session_cls = _patched(session)
        with (
            patch("bot_ai_patterns.text_cleaner_client.stdio_client", stdio_mock),
            patch("bot_ai_patterns.text_cleaner_client.ClientSession", session_cls),
        ):
            await clean_and_summarize("текст")

        session.initialize.assert_called_once()
