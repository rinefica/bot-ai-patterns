"""Tests for mcp_orchestrator.process_collection — all _call() invocations mocked."""
import json
from unittest.mock import AsyncMock, patch

import pytest

from bot_ai_patterns.mcp_orchestrator import process_collection

# ---------------------------------------------------------------------------
# Preset _call responses for a valid sewing pattern file
# ---------------------------------------------------------------------------

_SCAN = {"text": "Платье летнее A-силуэт. Хлопок.", "total_pages": 5, "pages_scanned": 2}
_DETECT = {"is_pattern": True, "confidence": 0.95, "reason": "Описание выкройки"}
_CLEANED = {"cleaned_text": "Платье A-силуэт.", "item_type": "платье", "summary": "Летнее платье."}
_TAGS = {"clothing_type": "платье", "fabric": ["хлопок"], "silhouette": "A-силуэт",
         "season": ["лето"], "length": "миди", "style": "casual",
         "closure": "молния", "additional_tags": []}
_IMG = {"output_path": "/tmp/cover.png", "width_px": 400, "height_px": 566, "dpi": 150}
_UPLOAD = {"url": "https://telegra.ph/file/abc.png"}
_PAGE = {"url": "https://telegra.ph/Test-Page-01-01", "path": "Test-Page-01-01"}

_SAMPLE_PDF = b"%PDF-1.4 sample"
_SAMPLE_FILE = (_SAMPLE_PDF, "dress.pdf", "https://api.telegram.org/file/botTOKEN/dress.pdf")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _call_side_effect(responses: list):
    """Return _call mock that yields responses in sequence."""
    it = iter(responses)

    async def _call_mock(server_path, tool, args, env=None):
        return next(it)

    return _call_mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestProcessCollection:
    @pytest.mark.asyncio
    async def test_returns_telegraph_url(self):
        responses = [_SCAN, _DETECT, _CLEANED, _TAGS, _IMG, _UPLOAD, _PAGE]
        with (
            patch("bot_ai_patterns.mcp_orchestrator._call", side_effect=_call_side_effect(responses)),
            patch("pathlib.Path.write_bytes"),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.read_bytes", return_value=b"\x89PNG\r\n\x1a\n"),
        ):
            result = await process_collection([_SAMPLE_FILE])
        assert result == "https://telegra.ph/Test-Page-01-01"

    @pytest.mark.asyncio
    async def test_interrupts_chain_when_no_text(self):
        empty_scan = {"text": "", "total_pages": 1, "pages_scanned": 0}
        with (
            patch("bot_ai_patterns.mcp_orchestrator._call", side_effect=_call_side_effect([empty_scan])),
            patch("pathlib.Path.write_bytes"),
            patch("pathlib.Path.exists", return_value=False),
        ):
            with pytest.raises(RuntimeError, match="не найдено описание"):
                await process_collection([_SAMPLE_FILE])

    @pytest.mark.asyncio
    async def test_interrupts_chain_when_not_sewing(self):
        not_pattern = {"is_pattern": False, "confidence": 0.9, "reason": "Не выкройка"}
        with (
            patch("bot_ai_patterns.mcp_orchestrator._call",
                  side_effect=_call_side_effect([_SCAN, not_pattern])),
            patch("pathlib.Path.write_bytes"),
        ):
            with pytest.raises(RuntimeError, match="не найдено описание"):
                await process_collection([_SAMPLE_FILE])

    @pytest.mark.asyncio
    async def test_raises_when_no_valid_files(self):
        with (
            patch("bot_ai_patterns.mcp_orchestrator._call",
                  side_effect=_call_side_effect([_SCAN, {"is_pattern": False, "confidence": 0.8, "reason": "нет"}])),
            patch("pathlib.Path.write_bytes"),
        ):
            with pytest.raises(RuntimeError):
                await process_collection([_SAMPLE_FILE])

    @pytest.mark.asyncio
    async def test_calls_scan_first(self):
        calls = []

        async def track_call(server_path, tool, args, env=None):
            calls.append(tool)
            if tool == "scan_for_description":
                return _SCAN
            if tool == "is_sewing_pattern":
                return _DETECT
            if tool == "clean_and_summarize":
                return _CLEANED
            if tool == "extract_tags":
                return _TAGS
            if tool == "pdf_extract_first_page":
                return _IMG
            if tool == "upload_photo":
                return _UPLOAD
            if tool == "create_page":
                return _PAGE
            return {}

        with (
            patch("bot_ai_patterns.mcp_orchestrator._call", side_effect=track_call),
            patch("pathlib.Path.write_bytes"),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.read_bytes", return_value=b"\x89PNG\r\n\x1a\n"),
        ):
            await process_collection([_SAMPLE_FILE])

        assert calls[0] == "scan_for_description"

    @pytest.mark.asyncio
    async def test_tool_call_order(self):
        calls = []

        async def track_call(server_path, tool, args, env=None):
            calls.append(tool)
            mapping = {
                "scan_for_description": _SCAN,
                "is_sewing_pattern": _DETECT,
                "clean_and_summarize": _CLEANED,
                "extract_tags": _TAGS,
                "pdf_extract_first_page": _IMG,
                "upload_photo": _UPLOAD,
                "create_page": _PAGE,
            }
            return mapping.get(tool, {})

        with (
            patch("bot_ai_patterns.mcp_orchestrator._call", side_effect=track_call),
            patch("pathlib.Path.write_bytes"),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.read_bytes", return_value=b"\x89PNG\r\n\x1a\n"),
        ):
            await process_collection([_SAMPLE_FILE])

        expected_order = [
            "scan_for_description",
            "is_sewing_pattern",
            "clean_and_summarize",
            "extract_tags",
            "pdf_extract_first_page",
            "upload_photo",
            "create_page",
        ]
        assert calls == expected_order

    @pytest.mark.asyncio
    async def test_second_file_attached_not_processed(self):
        """After the first sewing pattern is found, subsequent files are attached, not processed."""
        tools_called = []

        async def track_call(server_path, tool, args, env=None):
            tools_called.append(tool)
            mapping = {
                "scan_for_description": _SCAN,
                "is_sewing_pattern": _DETECT,
                "clean_and_summarize": _CLEANED,
                "extract_tags": _TAGS,
                "pdf_extract_first_page": _IMG,
                "upload_photo": _UPLOAD,
                "create_page": _PAGE,
            }
            return mapping.get(tool, {})

        file2 = (b"%PDF second", "shirt.pdf", "https://api.telegram.org/file/botTOKEN/shirt.pdf")
        with (
            patch("bot_ai_patterns.mcp_orchestrator._call", side_effect=track_call),
            patch("pathlib.Path.write_bytes"),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.read_bytes", return_value=b"\x89PNG\r\n\x1a\n"),
        ):
            await process_collection([_SAMPLE_FILE, file2])

        # Second file should NOT trigger scan/detect/clean/tags
        assert tools_called.count("scan_for_description") == 1
        assert tools_called.count("is_sewing_pattern") == 1
        assert "create_page" in tools_called

    @pytest.mark.asyncio
    async def test_progress_callback_called(self):
        messages = []

        async def on_prog(msg: str) -> None:
            messages.append(msg)

        async def track_call(server_path, tool, args, env=None):
            mapping = {
                "scan_for_description": _SCAN,
                "is_sewing_pattern": _DETECT,
                "clean_and_summarize": _CLEANED,
                "extract_tags": _TAGS,
                "pdf_extract_first_page": _IMG,
                "upload_photo": _UPLOAD,
                "create_page": _PAGE,
            }
            return mapping.get(tool, {})

        with (
            patch("bot_ai_patterns.mcp_orchestrator._call", side_effect=track_call),
            patch("pathlib.Path.write_bytes"),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.read_bytes", return_value=b"\x89PNG\r\n\x1a\n"),
        ):
            await process_collection([_SAMPLE_FILE], on_progress=on_prog)

        assert len(messages) > 0

    @pytest.mark.asyncio
    async def test_on_cover_ready_called_with_png_bytes(self):
        cover_calls = []

        async def on_cover(png: bytes) -> str:
            cover_calls.append(png)
            return "https://telegram.org/cover.png"

        async def track_call(server_path, tool, args, env=None):
            mapping = {
                "scan_for_description": _SCAN,
                "is_sewing_pattern": _DETECT,
                "clean_and_summarize": _CLEANED,
                "extract_tags": _TAGS,
                "pdf_extract_first_page": _IMG,
                "create_page": _PAGE,
            }
            return mapping.get(tool, {})

        with (
            patch("bot_ai_patterns.mcp_orchestrator._call", side_effect=track_call),
            patch("pathlib.Path.write_bytes"),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.read_bytes", return_value=b"\x89PNG\r\n\x1a\n"),
        ):
            await process_collection([_SAMPLE_FILE], on_cover_ready=on_cover)

        assert len(cover_calls) == 1
        assert cover_calls[0] == b"\x89PNG\r\n\x1a\n"

    @pytest.mark.asyncio
    async def test_file_url_passed_to_create_page(self):
        captured = {}

        async def track_call(server_path, tool, args, env=None):
            if tool == "create_page":
                captured.update(args)
                return _PAGE
            mapping = {
                "scan_for_description": _SCAN,
                "is_sewing_pattern": _DETECT,
                "clean_and_summarize": _CLEANED,
                "extract_tags": _TAGS,
                "pdf_extract_first_page": _IMG,
                "upload_photo": _UPLOAD,
            }
            return mapping.get(tool, {})

        with (
            patch("bot_ai_patterns.mcp_orchestrator._call", side_effect=track_call),
            patch("pathlib.Path.write_bytes"),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.read_bytes", return_value=b"\x89PNG\r\n\x1a\n"),
        ):
            await process_collection([_SAMPLE_FILE])

        items = json.loads(captured.get("items_json", "[]"))
        assert items[0]["file_url"] == "https://api.telegram.org/file/botTOKEN/dress.pdf"

    @pytest.mark.asyncio
    async def test_create_page_receives_items(self):
        captured_args = {}

        async def track_call(server_path, tool, args, env=None):
            if tool == "create_page":
                captured_args.update(args)
                return _PAGE
            mapping = {
                "scan_for_description": _SCAN,
                "is_sewing_pattern": _DETECT,
                "clean_and_summarize": _CLEANED,
                "extract_tags": _TAGS,
                "pdf_extract_first_page": _IMG,
                "upload_photo": _UPLOAD,
            }
            return mapping.get(tool, {})

        with (
            patch("bot_ai_patterns.mcp_orchestrator._call", side_effect=track_call),
            patch("pathlib.Path.write_bytes"),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.read_bytes", return_value=b"\x89PNG\r\n\x1a\n"),
        ):
            await process_collection([_SAMPLE_FILE], page_title="Моя коллекция")

        assert captured_args["title"] == "Моя коллекция"
        items = json.loads(captured_args["items_json"])
        assert len(items) == 1
        assert items[0]["file_name"] == "dress.pdf"
