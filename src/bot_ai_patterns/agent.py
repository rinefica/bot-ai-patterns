from __future__ import annotations

import openai

from bot_ai_patterns.config import MODEL_URI, MAX_TOKENS
from bot_ai_patterns.storage import JSONStorage

_DEFAULT_SYSTEM = "Ты полезный ассистент."


class Agent:
    """Агент, инкапсулирующий логику общения с LLM.

    Хранит историю диалога в памяти и персистентно в JSON.
    При создании загружает сохранённую историю — диалог продолжается
    после перезапуска приложения.
    """

    def __init__(
        self,
        client: openai.OpenAI,
        user_id: int,
        storage: JSONStorage,
        system_prompt: str = _DEFAULT_SYSTEM,
    ) -> None:
        self._client = client
        self._user_id = user_id
        self._storage = storage
        self._system_prompt = system_prompt

        saved = storage.load(user_id)
        self._history: list[dict[str, str]] = (
            saved if saved else [{"role": "system", "content": system_prompt}]
        )

    def chat(self, user_input: str) -> str:
        self._history.append({"role": "user", "content": user_input})

        response = self._client.chat.completions.create(
            model=MODEL_URI,
            messages=self._history,
            max_tokens=MAX_TOKENS,
        )

        reply = response.choices[0].message.content or ""
        self._history.append({"role": "assistant", "content": reply})
        self._storage.save(self._user_id, self._history)
        return reply

    def reset(self) -> None:
        self._history = [{"role": "system", "content": self._system_prompt}]
        self._storage.delete(self._user_id)
