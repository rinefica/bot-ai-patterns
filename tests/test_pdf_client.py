"""Tests for pdf_client.process_pdf — MCP session is mocked."""
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pymupdf
import pytest
import pytest_asyncio

from bot_ai_patterns.pdf_client import process_pdf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _AsyncCtx:
    """Wraps a value into an async context manager."""
    def __init__(self, val):
        self._val = val

    async def __aenter__(self):
        return self._val

    async def __aexit__(self, *_):
        pass


def _text_result(page_num: int, text: str) -> MagicMock:
    item = MagicMock()
    item.text = json.dumps({
        "pdf_path": "/tmp/input.pdf",
        "total_pages": 3,
        "page": page_num,
        "text": text,
    })
    result = MagicMock()
    result.is_error = False
    result.content = [item]
    return result


def _error_result() -> MagicMock:
    result = MagicMock()
    result.is_error = True
    result.content = []
    return result


def _png_result() -> MagicMock:
    result = MagicMock()
    result.is_error = False
    result.content = []
    return result


@pytest.fixture
def real_pdf_bytes(tmp_path) -> bytes:
    """Minimal 3-page PDF as bytes."""
    doc = pymupdf.open()
    for i in range(1, 4):
        page = doc.new_page()
        page.insert_text((72, 100), f"Page {i} text")
    path = tmp_path / "test.pdf"
    doc.save(str(path))
    return path.read_bytes()


@pytest.fixture
def real_png_bytes(tmp_path) -> bytes:
    """Minimal valid PNG bytes via pymupdf."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 100), "test")
    pix = page.get_pixmap(matrix=pymupdf.Matrix(1, 1))
    path = tmp_path / "test.png"
    pix.save(str(path))
    return path.read_bytes()


# ---------------------------------------------------------------------------
# Fixtures: patched MCP stack
# ---------------------------------------------------------------------------

def _make_patches(session: AsyncMock):
    """Return (stdio_patch, session_patch) with session wired in."""
    stdio_mock = MagicMock()
    stdio_mock.return_value = _AsyncCtx((MagicMock(), MagicMock()))

    session_cls_mock = MagicMock()
    session_cls_mock.return_value = _AsyncCtx(session)

    return stdio_mock, session_cls_mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestProcessPdfReturnsTextAndPng:
    @pytest.mark.asyncio
    async def test_returns_png_bytes_and_text(self, real_pdf_bytes, real_png_bytes, tmp_path):
        session = AsyncMock()
        session.initialize = AsyncMock()

        png_file: Path | None = None

        async def call_tool(name, args):
            nonlocal png_file
            if name == "pdf_read_text":
                page = args["page"]
                return _text_result(page, f"текст страницы {page}")
            if name == "pdf_extract_first_page":
                png_file = Path(args["output_path"])
                png_file.write_bytes(real_png_bytes)
                return _png_result()

        session.call_tool = call_tool
        stdio_mock, session_cls_mock = _make_patches(session)

        with (
            patch("bot_ai_patterns.pdf_client.stdio_client", stdio_mock),
            patch("bot_ai_patterns.pdf_client.ClientSession", session_cls_mock),
        ):
            png, text = await process_pdf(real_pdf_bytes)

        assert png == real_png_bytes
        assert "текст страницы 1" in text

    @pytest.mark.asyncio
    async def test_text_contains_up_to_5_pages(self, real_pdf_bytes, real_png_bytes):
        session = AsyncMock()
        session.initialize = AsyncMock()
        calls: list[int] = []

        async def call_tool(name, args):
            if name == "pdf_read_text":
                page = args["page"]
                calls.append(page)
                return _text_result(page, f"p{page}")
            png_path = Path(args["output_path"])
            png_path.write_bytes(real_png_bytes)
            return _png_result()

        session.call_tool = call_tool
        stdio_mock, session_cls_mock = _make_patches(session)

        with (
            patch("bot_ai_patterns.pdf_client.stdio_client", stdio_mock),
            patch("bot_ai_patterns.pdf_client.ClientSession", session_cls_mock),
        ):
            await process_pdf(real_pdf_bytes)

        assert calls == [1, 2, 3, 4, 5]

    @pytest.mark.asyncio
    async def test_stops_iteration_on_error_result(self, real_pdf_bytes, real_png_bytes):
        session = AsyncMock()
        session.initialize = AsyncMock()
        calls: list[int] = []

        async def call_tool(name, args):
            if name == "pdf_read_text":
                page = args["page"]
                calls.append(page)
                if page >= 3:
                    return _error_result()
                return _text_result(page, f"p{page}")
            png_path = Path(args["output_path"])
            png_path.write_bytes(real_png_bytes)
            return _png_result()

        session.call_tool = call_tool
        stdio_mock, session_cls_mock = _make_patches(session)

        with (
            patch("bot_ai_patterns.pdf_client.stdio_client", stdio_mock),
            patch("bot_ai_patterns.pdf_client.ClientSession", session_cls_mock),
        ):
            _, text = await process_pdf(real_pdf_bytes)

        assert calls == [1, 2, 3]
        assert "p1" in text
        assert "p2" in text
        assert "p3" not in text

    @pytest.mark.asyncio
    async def test_returns_empty_png_when_file_not_written(self, real_pdf_bytes):
        session = AsyncMock()
        session.initialize = AsyncMock()

        async def call_tool(name, args):
            if name == "pdf_read_text":
                return _text_result(args["page"], "text")
            # intentionally do NOT write the PNG file
            return _png_result()

        session.call_tool = call_tool
        stdio_mock, session_cls_mock = _make_patches(session)

        with (
            patch("bot_ai_patterns.pdf_client.stdio_client", stdio_mock),
            patch("bot_ai_patterns.pdf_client.ClientSession", session_cls_mock),
        ):
            png, _ = await process_pdf(real_pdf_bytes)

        assert png == b""

    @pytest.mark.asyncio
    async def test_text_pages_separated_by_header(self, real_pdf_bytes, real_png_bytes):
        session = AsyncMock()
        session.initialize = AsyncMock()

        async def call_tool(name, args):
            if name == "pdf_read_text":
                page = args["page"]
                return _text_result(page, f"content {page}")
            png_path = Path(args["output_path"])
            png_path.write_bytes(real_png_bytes)
            return _png_result()

        session.call_tool = call_tool
        stdio_mock, session_cls_mock = _make_patches(session)

        with (
            patch("bot_ai_patterns.pdf_client.stdio_client", stdio_mock),
            patch("bot_ai_patterns.pdf_client.ClientSession", session_cls_mock),
        ):
            _, text = await process_pdf(real_pdf_bytes)

        assert "--- Страница 1 ---" in text
        assert "--- Страница 2 ---" in text

    @pytest.mark.asyncio
    async def test_skips_pages_with_empty_text(self, real_pdf_bytes, real_png_bytes):
        session = AsyncMock()
        session.initialize = AsyncMock()

        async def call_tool(name, args):
            if name == "pdf_read_text":
                page = args["page"]
                text = "real content" if page == 1 else ""
                return _text_result(page, text)
            png_path = Path(args["output_path"])
            png_path.write_bytes(real_png_bytes)
            return _png_result()

        session.call_tool = call_tool
        stdio_mock, session_cls_mock = _make_patches(session)

        with (
            patch("bot_ai_patterns.pdf_client.stdio_client", stdio_mock),
            patch("bot_ai_patterns.pdf_client.ClientSession", session_cls_mock),
        ):
            _, text = await process_pdf(real_pdf_bytes)

        assert "real content" in text
        assert "--- Страница 2 ---" not in text
