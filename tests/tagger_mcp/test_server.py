"""Unit tests for tagger-mcp/server.py (DeepSeek is mocked)."""
import importlib.util
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Import server module directly
# ---------------------------------------------------------------------------
_SERVER_FILE = Path(__file__).parent.parent.parent / "tagger-mcp" / "server.py"
_spec = importlib.util.spec_from_file_location("tagger_server", _SERVER_FILE)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

extract_tags = _mod.extract_tags
TagResult = _mod.TagResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_completion(data: dict) -> MagicMock:
    msg = MagicMock()
    msg.content = json.dumps(data)
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


_VALID_RESPONSE = {
    "clothing_type": "платье",
    "fabric": ["хлопок", "лён"],
    "silhouette": "A-силуэт",
    "season": ["лето", "весна"],
    "length": "миди",
    "style": "casual",
    "closure": "молния",
    "additional_tags": ["без рукавов", "V-образный вырез"],
}

_SAMPLE_SUMMARY = "Летнее платье с A-силуэтом из хлопка. Приталенный верх, расклешённая юбка до колена."


# ---------------------------------------------------------------------------
# Tests: extract_tags
# ---------------------------------------------------------------------------

class TestExtractTags:
    def _call(self, summary: str = _SAMPLE_SUMMARY, response: dict = None) -> TagResult:
        resp_data = response if response is not None else _VALID_RESPONSE
        with patch.object(_mod, "_get_client") as mock_get_client:
            client = MagicMock()
            client.chat.completions.create.return_value = _mock_completion(resp_data)
            mock_get_client.return_value = client
            return extract_tags(summary)

    def test_returns_tag_result(self):
        result = self._call()
        assert isinstance(result, TagResult)

    def test_clothing_type(self):
        assert self._call().clothing_type == "платье"

    def test_fabric_is_list(self):
        result = self._call()
        assert isinstance(result.fabric, list)
        assert "хлопок" in result.fabric

    def test_silhouette(self):
        assert self._call().silhouette == "A-силуэт"

    def test_season_is_list(self):
        result = self._call()
        assert "лето" in result.season

    def test_length(self):
        assert self._call().length == "миди"

    def test_style(self):
        assert self._call().style == "casual"

    def test_closure(self):
        assert self._call().closure == "молния"

    def test_additional_tags_is_list(self):
        result = self._call()
        assert isinstance(result.additional_tags, list)
        assert "без рукавов" in result.additional_tags

    def test_raises_on_empty_summary(self):
        with pytest.raises(ValueError, match="empty"):
            extract_tags("   ")

    def test_raises_on_invalid_json(self):
        with patch.object(_mod, "_get_client") as mock_get_client:
            client = MagicMock()
            msg = MagicMock()
            msg.content = "not json {"
            client.chat.completions.create.return_value.choices = [MagicMock(message=msg)]
            mock_get_client.return_value = client
            with pytest.raises(RuntimeError, match="invalid JSON"):
                extract_tags(_SAMPLE_SUMMARY)

    def test_raises_when_api_key_missing(self):
        with patch.dict(_mod.os.environ, {}, clear=True):
            with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
                extract_tags(_SAMPLE_SUMMARY)

    def test_missing_fields_use_fallbacks(self):
        result = self._call(response={})
        assert result.clothing_type == "не определено"
        assert result.fabric == []
        assert result.additional_tags == []

    def test_deepseek_called_with_json_mode(self):
        with patch.object(_mod, "_get_client") as mock_get_client:
            client = MagicMock()
            client.chat.completions.create.return_value = _mock_completion(_VALID_RESPONSE)
            mock_get_client.return_value = client
            extract_tags(_SAMPLE_SUMMARY)
            call_kwargs = client.chat.completions.create.call_args.kwargs
            assert call_kwargs["response_format"] == {"type": "json_object"}

    def test_summary_passed_to_deepseek(self):
        with patch.object(_mod, "_get_client") as mock_get_client:
            client = MagicMock()
            client.chat.completions.create.return_value = _mock_completion(_VALID_RESPONSE)
            mock_get_client.return_value = client
            extract_tags(_SAMPLE_SUMMARY)
            messages = client.chat.completions.create.call_args.kwargs["messages"]
            assert any(_SAMPLE_SUMMARY in m["content"] for m in messages)
