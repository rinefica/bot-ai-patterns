"""Unit tests for telegraph-mcp/server.py (HTTP calls are mocked)."""
import base64
import importlib.util
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Import server module directly
# ---------------------------------------------------------------------------
_SERVER_FILE = Path(__file__).parent.parent.parent / "telegraph-mcp" / "server.py"
_spec = importlib.util.spec_from_file_location("telegraph_server", _SERVER_FILE)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

upload_photo = _mod.upload_photo
create_page = _mod.create_page
UploadResult = _mod.UploadResult
PageResult = _mod.PageResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_get(token: str = "test-token") -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = {"ok": True, "result": {"access_token": token}}
    resp.raise_for_status = MagicMock()
    return resp


def _mock_upload_response(src: str = "/file/abc123.png") -> MagicMock:
    resp = MagicMock()
    resp.ok = True
    resp.json.return_value = [{"src": src}]
    resp.raise_for_status = MagicMock()
    return resp


def _mock_create_response(url: str = "https://telegra.ph/Test-01-01", path: str = "Test-01-01") -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = {"ok": True, "result": {"url": url, "path": path}}
    resp.raise_for_status = MagicMock()
    return resp


_SAMPLE_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
_SAMPLE_B64 = base64.b64encode(_SAMPLE_PNG).decode()

_SAMPLE_ITEM = {
    "title": "Летнее платье",
    "item_type": "платье",
    "summary": "Лёгкое летнее платье A-силуэта.",
    "tags_json": json.dumps({"clothing_type": "платье", "fabric": ["хлопок"], "style": "casual"}),
    "cover_url": "https://telegra.ph/file/abc.png",
    "file_name": "dress.pdf",
}


# ---------------------------------------------------------------------------
# Tests: upload_photo
# ---------------------------------------------------------------------------

class TestUploadPhoto:
    def _call(self, b64: str = _SAMPLE_B64) -> UploadResult:
        with patch.object(_mod.requests, "post", return_value=_mock_upload_response()) as mock_post:
            result = upload_photo(b64)
            return result

    def test_returns_upload_result(self):
        assert isinstance(self._call(), UploadResult)

    def test_url_starts_with_telegra_ph(self):
        assert self._call().url.startswith("https://telegra.ph")

    def test_url_contains_src_path(self):
        assert "/file/abc123.png" in self._call().url

    def test_posts_as_multipart(self):
        with patch.object(_mod.requests, "post", return_value=_mock_upload_response()) as mock_post:
            upload_photo(_SAMPLE_B64)
            assert mock_post.called
            _, kwargs = mock_post.call_args
            assert "files" in kwargs

    def test_decodes_base64_correctly(self):
        received_bytes = None
        def capture_post(url, files, **kwargs):
            nonlocal received_bytes
            received_bytes = files["file"][1]
            return _mock_upload_response()
        with patch.object(_mod.requests, "post", side_effect=capture_post):
            upload_photo(_SAMPLE_B64)
        assert received_bytes == _SAMPLE_PNG

    def test_returns_empty_url_on_unexpected_response(self):
        bad_resp = MagicMock()
        bad_resp.ok = False
        bad_resp.json.return_value = {"error": "oops"}
        bad_resp.raise_for_status = MagicMock()
        with patch.object(_mod.requests, "post", return_value=bad_resp):
            result = upload_photo(_SAMPLE_B64)
        assert result.url == ""

    def test_returns_empty_url_on_network_error(self):
        with patch.object(_mod.requests, "post", side_effect=Exception("timeout")):
            result = upload_photo(_SAMPLE_B64)
        assert result.url == ""


# ---------------------------------------------------------------------------
# Tests: create_page
# ---------------------------------------------------------------------------

class TestCreatePage:
    def _call(self, title: str = "Test Page", items: list[dict] | None = None) -> PageResult:
        items = items if items is not None else [_SAMPLE_ITEM]
        with (
            patch.object(_mod.requests, "get", return_value=_mock_get()),
            patch.object(_mod.requests, "post", return_value=_mock_create_response()),
        ):
            return create_page(title, json.dumps(items, ensure_ascii=False))

    def test_returns_page_result(self):
        assert isinstance(self._call(), PageResult)

    def test_url_in_result(self):
        result = self._call()
        assert result.url.startswith("https://telegra.ph")

    def test_path_in_result(self):
        assert self._call().path == "Test-01-01"

    def test_calls_create_account(self):
        with (
            patch.object(_mod.requests, "get", return_value=_mock_get()) as mock_get,
            patch.object(_mod.requests, "post", return_value=_mock_create_response()),
        ):
            create_page("T", json.dumps([_SAMPLE_ITEM]))
            mock_get.assert_called_once()
            assert "createAccount" in mock_get.call_args[0][0]

    def test_uses_env_token_when_set(self):
        with (
            patch.dict(_mod.os.environ, {"TELEGRAPH_ACCESS_TOKEN": "env-token"}),
            patch.object(_mod.requests, "get", return_value=_mock_get()) as mock_get,
            patch.object(_mod.requests, "post", return_value=_mock_create_response()),
        ):
            create_page("T", json.dumps([_SAMPLE_ITEM]))
            mock_get.assert_not_called()

    def test_raises_on_api_error(self):
        bad_resp = MagicMock()
        bad_resp.json.return_value = {"ok": False, "error": "TITLE_INVALID"}
        bad_resp.raise_for_status = MagicMock()
        with (
            patch.object(_mod.requests, "get", return_value=_mock_get()),
            patch.object(_mod.requests, "post", return_value=bad_resp),
        ):
            with pytest.raises(RuntimeError, match="TITLE_INVALID"):
                create_page("T", json.dumps([_SAMPLE_ITEM]))

    def test_empty_items_creates_page(self):
        result = self._call(items=[])
        assert isinstance(result, PageResult)

    def test_item_with_missing_fields(self):
        result = self._call(items=[{"title": "x"}])
        assert isinstance(result, PageResult)

    def test_tags_json_parsed_and_included(self):
        captured_payload = {}
        def capture(url, json=None, **kwargs):
            captured_payload.update(json or {})
            return _mock_create_response()
        with (
            patch.object(_mod.requests, "get", return_value=_mock_get()),
            patch.object(_mod.requests, "post", side_effect=capture),
        ):
            create_page("Tags Test", json.dumps([_SAMPLE_ITEM]))
        content = captured_payload.get("content", "[]")
        nodes = json.loads(content) if isinstance(content, str) else content
        all_text = json.dumps(nodes, ensure_ascii=False)
        assert "платье" in all_text
