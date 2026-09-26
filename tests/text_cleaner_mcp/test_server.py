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
is_sewing_pattern = _mod.is_sewing_pattern
SummaryResult = _mod.SummaryResult
PatternDetectResult = _mod.PatternDetectResult


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


# ---------------------------------------------------------------------------
# Tests: is_sewing_pattern
# ---------------------------------------------------------------------------

_PATTERN_RESPONSE = {"is_pattern": True, "confidence": 0.95, "reason": "Описание выкройки"}
_NOT_PATTERN_RESPONSE = {"is_pattern": False, "confidence": 0.9, "reason": "Не выкройка"}
_SEWING_TEXT = "Платье летнее A-силуэт. Материал: хлопок 100%. Выкройка размера 42-44."


class TestIsSewingPattern:
    def _call(self, text: str = _SEWING_TEXT, response: dict = None) -> PatternDetectResult:
        resp_data = response if response is not None else _PATTERN_RESPONSE
        with patch.object(_mod, "_get_client") as mock_get_client:
            client = MagicMock()
            client.chat.completions.create.return_value = _mock_completion(resp_data)
            mock_get_client.return_value = client
            return is_sewing_pattern(text)

    def test_returns_pattern_detect_result(self):
        assert isinstance(self._call(), PatternDetectResult)

    def test_is_pattern_true_for_sewing_text(self):
        assert self._call().is_pattern is True

    def test_confidence_is_float(self):
        result = self._call()
        assert isinstance(result.confidence, float)
        assert 0.0 <= result.confidence <= 1.0

    def test_reason_is_string(self):
        assert isinstance(self._call().reason, str)

    def test_is_pattern_false_for_non_sewing(self):
        result = self._call(response=_NOT_PATTERN_RESPONSE)
        assert result.is_pattern is False

    def test_empty_text_returns_false_without_api_call(self):
        result = is_sewing_pattern("   ")
        assert result.is_pattern is False
        assert result.confidence == 1.0

    def test_raises_on_invalid_json(self):
        with patch.object(_mod, "_get_client") as mock_get_client:
            client = MagicMock()
            msg = MagicMock()
            msg.content = "not json {"
            client.chat.completions.create.return_value.choices = [MagicMock(message=msg)]
            mock_get_client.return_value = client
            with pytest.raises(RuntimeError, match="invalid JSON"):
                is_sewing_pattern(_SEWING_TEXT)

    def test_raises_when_api_key_missing(self):
        with patch.dict(_mod.os.environ, {}, clear=True):
            with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
                is_sewing_pattern(_SEWING_TEXT)

    def test_called_with_json_mode(self):
        with patch.object(_mod, "_get_client") as mock_get_client:
            client = MagicMock()
            client.chat.completions.create.return_value = _mock_completion(_PATTERN_RESPONSE)
            mock_get_client.return_value = client
            is_sewing_pattern(_SEWING_TEXT)
            call_kwargs = client.chat.completions.create.call_args.kwargs
            assert call_kwargs["response_format"] == {"type": "json_object"}

    def test_text_truncated_to_3000_chars(self):
        long_text = "а" * 5000
        with patch.object(_mod, "_get_client") as mock_get_client:
            client = MagicMock()
            client.chat.completions.create.return_value = _mock_completion(_PATTERN_RESPONSE)
            mock_get_client.return_value = client
            is_sewing_pattern(long_text)
            messages = client.chat.completions.create.call_args.kwargs["messages"]
            user_content = next(m["content"] for m in messages if m["role"] == "user")
            assert len(user_content) <= 3100
