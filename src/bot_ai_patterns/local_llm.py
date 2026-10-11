"""
LocalLLMClient: OpenAI-совместимый адаптер для Ollama HTTP API.

Реализует минимальный интерфейс openai.OpenAI.chat.completions.create,
чтобы RagAgent мог переключаться между облаком и локальной моделью.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import httpx

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:7b"


@dataclass
class _Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class _Message:
    content: str
    role: str = "assistant"


@dataclass
class _Choice:
    message: _Message


@dataclass
class _Response:
    choices: list[_Choice]
    usage: _Usage = field(default_factory=_Usage)


class _Completions:
    def __init__(self, model: str, base_url: str, timeout: int):
        self._model = model
        self._base_url = base_url
        self._timeout = timeout

    def create(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int = 1024,
        **_kwargs,
    ) -> _Response:
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(
                f"{self._base_url}/api/chat",
                json={
                    "model": model or self._model,
                    "messages": messages,
                    "stream": False,
                    "options": {"num_predict": max_tokens},
                },
            )
            resp.raise_for_status()
            data = resp.json()

        content = data["message"]["content"]
        # Ollama возвращает статистику токенов
        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)

        return _Response(
            choices=[_Choice(message=_Message(content=content))],
            usage=_Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )


class _Chat:
    def __init__(self, model: str, base_url: str, timeout: int):
        self.completions = _Completions(model, base_url, timeout)


class LocalLLMClient:
    """Замена openai.OpenAI для локального Ollama."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = OLLAMA_URL,
        timeout: int = 120,
    ):
        self.model = model
        self.chat = _Chat(model, base_url, timeout)

    @staticmethod
    def is_available(base_url: str = OLLAMA_URL) -> bool:
        try:
            with httpx.Client(timeout=5) as client:
                return client.get(f"{base_url}/api/tags").status_code == 200
        except httpx.ConnectError:
            return False
