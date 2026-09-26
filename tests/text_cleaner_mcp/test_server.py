"""Unit tests for text-cleaner-mcp/server.py (DeepSeek is mocked)."""
import importlib.util
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Import server module directly
# ---------------------------------------------------------------------------
_SERVER_FILE = Path(__file__).parent.parent.parent / "text-cleaner-mcp" / "server.py"
_spec = importlib.util.spec_from_file_location("text_cleaner_server", _SERVER_FILE)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

clean_and_summarize = _mod.clean_and_summarize
SummaryResult = _mod.SummaryResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_completion(data: dict) -> MagicMock:
    """Build a fake openai ChatCompletion response."""
    msg = MagicMock()
    msg.content = json.dumps(data)
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


_VALID_RESPONSE = {
    "cleaned_text": "Платье с A-силуэтом, приталенный верх, расклешённая юбка.",
    "item_type": "платье",
    "summary": "Летнее платье с A-силуэтом. Приталенный верх и расклешённая юбка. Подходит для casual-образов.",
}

_SAMPLE_TEXT = "Page 1\nПлатье летнее A-силуэт\nPage 2\nМатериал: хлопок 100%"


# ---------------------------------------------------------------------------
# Tests: clean_and_summarize
# ---------------------------------------------------------------------------

class TestCleanAndSummarize:
    def _call(self, text: str = _SAMPLE_TEXT, response: dict = None) -> SummaryResult:
        resp_data = response or _VALID_RESPONSE
        with (
            patch.object(_mod, "_get_client") as mock_get_client,
        ):
            client = MagicMock()
            client.chat.completions.create.return_value = _mock_completion(resp_data)
            mock_get_client.return_value = client
            return clean_and_summarize(text)

    def test_returns_summary_result(self):
        result = self._call()
        assert isinstance(result, SummaryResult)

    def test_summary_populated(self):
        result = self._call()
        assert result.summary == _VALID_RESPONSE["summary"]

    def test_item_type_populated(self):
        result = self._call()
        assert result.item_type == "платье"

    def test_cleaned_text_populated(self):
        result = self._call()
        assert result.cleaned_text == _VALID_RESPONSE["cleaned_text"]

    def test_raises_on_empty_text(self):
        with pytest.raises(ValueError, match="empty"):
            clean_and_summarize("   ")

    def test_raises_on_invalid_json_from_deepseek(self):
        with patch.object(_mod, "_get_client") as mock_get_client:
            client = MagicMock()
            msg = MagicMock()
            msg.content = "not valid json {"
            client.chat.completions.create.return_value.choices = [MagicMock(message=msg)]
            mock_get_client.return_value = client
            with pytest.raises(RuntimeError, match="invalid JSON"):
                clean_and_summarize(_SAMPLE_TEXT)

    def test_raises_when_api_key_missing(self):
        with patch.dict(_mod.os.environ, {}, clear=True):
            with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
                clean_and_summarize(_SAMPLE_TEXT)

    def test_missing_fields_use_fallbacks(self):
        result = self._call(response={"cleaned_text": "x"})
        assert result.item_type == "неизвестно"
        assert result.summary == ""

    def test_text_truncated_to_8000_chars(self):
        long_text = "а" * 10_000
        with patch.object(_mod, "_get_client") as mock_get_client:
            client = MagicMock()
            client.chat.completions.create.return_value = _mock_completion(_VALID_RESPONSE)
            mock_get_client.return_value = client
            clean_and_summarize(long_text)
            call_args = client.chat.completions.create.call_args
            user_msg = call_args.kwargs["messages"][-1]["content"]
            assert len(user_msg) < 8100  # prompt header + 8000 chars max

    def test_deepseek_called_with_json_mode(self):
        with patch.object(_mod, "_get_client") as mock_get_client:
            client = MagicMock()
            client.chat.completions.create.return_value = _mock_completion(_VALID_RESPONSE)
            mock_get_client.return_value = client
            clean_and_summarize(_SAMPLE_TEXT)
            call_kwargs = client.chat.completions.create.call_args.kwargs
            assert call_kwargs["response_format"] == {"type": "json_object"}
