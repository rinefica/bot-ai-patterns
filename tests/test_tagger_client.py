"""Tests for tagger_client.extract_tags — MCP session is mocked."""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot_ai_patterns.tagger_client import extract_tags


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


_TAGS_DATA = {
    "clothing_type": "платье",
    "fabric": ["хлопок", "лён"],
    "silhouette": "A-силуэт",
    "season": ["лето"],
    "length": "миди",
    "style": "casual",
    "closure": "молния",
    "additional_tags": ["без рукавов"],
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestExtractTagsClient:
    @pytest.mark.asyncio
    async def test_returns_dict_with_tags(self):
        session = AsyncMock()
        session.initialize = AsyncMock()
        session.call_tool = AsyncMock(return_value=_tool_result(_TAGS_DATA))

        stdio_mock, session_cls = _patched(session)
        with (
            patch("bot_ai_patterns.tagger_client.stdio_client", stdio_mock),
            patch("bot_ai_patterns.tagger_client.ClientSession", session_cls),
        ):
            result = await extract_tags("описание изделия")

        assert result["clothing_type"] == "платье"
        assert "хлопок" in result["fabric"]

    @pytest.mark.asyncio
    async def test_calls_correct_tool(self):
        session = AsyncMock()
        session.initialize = AsyncMock()
        session.call_tool = AsyncMock(return_value=_tool_result(_TAGS_DATA))

        stdio_mock, session_cls = _patched(session)
        with (
            patch("bot_ai_patterns.tagger_client.stdio_client", stdio_mock),
            patch("bot_ai_patterns.tagger_client.ClientSession", session_cls),
        ):
            await extract_tags("описание")

        session.call_tool.assert_called_once_with("extract_tags", {"summary": "описание"})

    @pytest.mark.asyncio
    async def test_raises_on_error_result(self):
        session = AsyncMock()
        session.initialize = AsyncMock()
        session.call_tool = AsyncMock(return_value=_error_result())

        stdio_mock, session_cls = _patched(session)
        with (
            patch("bot_ai_patterns.tagger_client.stdio_client", stdio_mock),
            patch("bot_ai_patterns.tagger_client.ClientSession", session_cls),
        ):
            with pytest.raises(RuntimeError, match="extract_tags error"):
                await extract_tags("описание")

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
            patch("bot_ai_patterns.tagger_client.stdio_client", stdio_mock),
            patch("bot_ai_patterns.tagger_client.ClientSession", session_cls),
        ):
            result = await extract_tags("описание")

        assert result == {}

    @pytest.mark.asyncio
    async def test_all_tag_fields_present(self):
        session = AsyncMock()
        session.initialize = AsyncMock()
        session.call_tool = AsyncMock(return_value=_tool_result(_TAGS_DATA))

        stdio_mock, session_cls = _patched(session)
        with (
            patch("bot_ai_patterns.tagger_client.stdio_client", stdio_mock),
            patch("bot_ai_patterns.tagger_client.ClientSession", session_cls),
        ):
            result = await extract_tags("описание")

        for field in ("clothing_type", "fabric", "silhouette", "season", "length", "style", "closure", "additional_tags"):
            assert field in result

    @pytest.mark.asyncio
    async def test_session_initialized(self):
        session = AsyncMock()
        session.initialize = AsyncMock()
        session.call_tool = AsyncMock(return_value=_tool_result(_TAGS_DATA))

        stdio_mock, session_cls = _patched(session)
        with (
            patch("bot_ai_patterns.tagger_client.stdio_client", stdio_mock),
            patch("bot_ai_patterns.tagger_client.ClientSession", session_cls),
        ):
            await extract_tags("описание")

        session.initialize.assert_called_once()
