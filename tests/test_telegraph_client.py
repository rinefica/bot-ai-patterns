"""Tests for telegraph_client — MCP session is mocked."""
import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot_ai_patterns.telegraph_client import create_page, upload_photo


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


_SAMPLE_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50


# ---------------------------------------------------------------------------
# Tests: upload_photo
# ---------------------------------------------------------------------------

class TestUploadPhotoClient:
    @pytest.mark.asyncio
    async def test_returns_url_string(self):
        session = AsyncMock()
        session.initialize = AsyncMock()
        session.call_tool = AsyncMock(return_value=_tool_result({"url": "https://telegra.ph/file/abc.png"}))
        stdio_mock, session_cls = _patched(session)
        with (
            patch("bot_ai_patterns.telegraph_client.stdio_client", stdio_mock),
            patch("bot_ai_patterns.telegraph_client.ClientSession", session_cls),
        ):
            result = await upload_photo(_SAMPLE_PNG)
        assert result == "https://telegra.ph/file/abc.png"

    @pytest.mark.asyncio
    async def test_calls_upload_photo_tool(self):
        session = AsyncMock()
        session.initialize = AsyncMock()
        session.call_tool = AsyncMock(return_value=_tool_result({"url": "https://telegra.ph/x"}))
        stdio_mock, session_cls = _patched(session)
        with (
            patch("bot_ai_patterns.telegraph_client.stdio_client", stdio_mock),
            patch("bot_ai_patterns.telegraph_client.ClientSession", session_cls),
        ):
            await upload_photo(_SAMPLE_PNG)
        session.call_tool.assert_called_once()
        call_args = session.call_tool.call_args
        assert call_args[0][0] == "upload_photo"

    @pytest.mark.asyncio
    async def test_passes_base64_encoded_bytes(self):
        session = AsyncMock()
        session.initialize = AsyncMock()
        session.call_tool = AsyncMock(return_value=_tool_result({"url": "https://telegra.ph/x"}))
        stdio_mock, session_cls = _patched(session)
        with (
            patch("bot_ai_patterns.telegraph_client.stdio_client", stdio_mock),
            patch("bot_ai_patterns.telegraph_client.ClientSession", session_cls),
        ):
            await upload_photo(_SAMPLE_PNG)
        args = session.call_tool.call_args[0][1]
        assert base64.b64decode(args["image_b64"]) == _SAMPLE_PNG

    @pytest.mark.asyncio
    async def test_raises_on_error(self):
        session = AsyncMock()
        session.initialize = AsyncMock()
        session.call_tool = AsyncMock(return_value=_error_result())
        stdio_mock, session_cls = _patched(session)
        with (
            patch("bot_ai_patterns.telegraph_client.stdio_client", stdio_mock),
            patch("bot_ai_patterns.telegraph_client.ClientSession", session_cls),
        ):
            with pytest.raises(RuntimeError, match="upload_photo error"):
                await upload_photo(_SAMPLE_PNG)

    @pytest.mark.asyncio
    async def test_returns_empty_string_on_no_content(self):
        session = AsyncMock()
        session.initialize = AsyncMock()
        mock_result = MagicMock()
        mock_result.is_error = False
        mock_result.content = []
        session.call_tool = AsyncMock(return_value=mock_result)
        stdio_mock, session_cls = _patched(session)
        with (
            patch("bot_ai_patterns.telegraph_client.stdio_client", stdio_mock),
            patch("bot_ai_patterns.telegraph_client.ClientSession", session_cls),
        ):
            result = await upload_photo(_SAMPLE_PNG)
        assert result == ""


# ---------------------------------------------------------------------------
# Tests: create_page
# ---------------------------------------------------------------------------

_ITEMS = [
    {
        "title": "Платье",
        "item_type": "платье",
        "summary": "Летнее платье.",
        "tags_json": "{}",
        "cover_url": "https://telegra.ph/file/x.png",
        "file_name": "dress.pdf",
    }
]


class TestCreatePageClient:
    @pytest.mark.asyncio
    async def test_returns_url_string(self):
        session = AsyncMock()
        session.initialize = AsyncMock()
        session.call_tool = AsyncMock(
            return_value=_tool_result({"url": "https://telegra.ph/Test-01-01", "path": "Test-01-01"})
        )
        stdio_mock, session_cls = _patched(session)
        with (
            patch("bot_ai_patterns.telegraph_client.stdio_client", stdio_mock),
            patch("bot_ai_patterns.telegraph_client.ClientSession", session_cls),
        ):
            result = await create_page("Test", _ITEMS)
        assert result == "https://telegra.ph/Test-01-01"

    @pytest.mark.asyncio
    async def test_calls_create_page_tool(self):
        session = AsyncMock()
        session.initialize = AsyncMock()
        session.call_tool = AsyncMock(
            return_value=_tool_result({"url": "https://telegra.ph/x", "path": "x"})
        )
        stdio_mock, session_cls = _patched(session)
        with (
            patch("bot_ai_patterns.telegraph_client.stdio_client", stdio_mock),
            patch("bot_ai_patterns.telegraph_client.ClientSession", session_cls),
        ):
            await create_page("Test", _ITEMS)
        session.call_tool.assert_called_once()
        assert session.call_tool.call_args[0][0] == "create_page"

    @pytest.mark.asyncio
    async def test_passes_title_and_items_json(self):
        session = AsyncMock()
        session.initialize = AsyncMock()
        session.call_tool = AsyncMock(
            return_value=_tool_result({"url": "https://telegra.ph/x", "path": "x"})
        )
        stdio_mock, session_cls = _patched(session)
        with (
            patch("bot_ai_patterns.telegraph_client.stdio_client", stdio_mock),
            patch("bot_ai_patterns.telegraph_client.ClientSession", session_cls),
        ):
            await create_page("Мой заголовок", _ITEMS)
        args = session.call_tool.call_args[0][1]
        assert args["title"] == "Мой заголовок"
        parsed = json.loads(args["items_json"])
        assert parsed[0]["title"] == "Платье"

    @pytest.mark.asyncio
    async def test_raises_on_error(self):
        session = AsyncMock()
        session.initialize = AsyncMock()
        session.call_tool = AsyncMock(return_value=_error_result())
        stdio_mock, session_cls = _patched(session)
        with (
            patch("bot_ai_patterns.telegraph_client.stdio_client", stdio_mock),
            patch("bot_ai_patterns.telegraph_client.ClientSession", session_cls),
        ):
            with pytest.raises(RuntimeError, match="create_page error"):
                await create_page("T", _ITEMS)

    @pytest.mark.asyncio
    async def test_session_initialized(self):
        session = AsyncMock()
        session.initialize = AsyncMock()
        session.call_tool = AsyncMock(
            return_value=_tool_result({"url": "https://telegra.ph/x", "path": "x"})
        )
        stdio_mock, session_cls = _patched(session)
        with (
            patch("bot_ai_patterns.telegraph_client.stdio_client", stdio_mock),
            patch("bot_ai_patterns.telegraph_client.ClientSession", session_cls),
        ):
            await create_page("T", _ITEMS)
        session.initialize.assert_called_once()
