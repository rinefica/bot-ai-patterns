from __future__ import annotations

from dataclasses import dataclass, field

import openai

from bot_ai_patterns.config import (
    CONTEXT_LIMIT,
    CONTEXT_WARN_THRESHOLD,
    MAIN_MODEL_PRICE_PER_1K,
    MAX_TOKENS,
    MODEL_URI,
)
from bot_ai_patterns.storage import JSONStorage

_DEFAULT_SYSTEM = "Ты полезный ассистент."


@dataclass
class TokenUsage:
    """Статистика токенов одного запроса."""

    prompt_tokens: int       # токены входа (история + новый запрос)
    completion_tokens: int   # токены ответа модели
    total_tokens: int        # сумма


@dataclass
class SessionStats:
    """Накопленная статистика токенов за весь диалог."""

    turns: int = 0
    total_prompt: int = 0
    total_completion: int = 0
    total_tokens: int = 0
    history: list[TokenUsage] = field(default_factory=list)

    def add(self, usage: TokenUsage) -> None:
        self.turns += 1
        self.total_prompt += usage.prompt_tokens
        self.total_completion += usage.completion_tokens
        self.total_tokens += usage.total_tokens
        self.history.append(usage)

    @property
    def cost_rub(self) -> float:
        return self.total_tokens * MAIN_MODEL_PRICE_PER_1K / 1000

    def context_fill_pct(self, last_prompt_tokens: int) -> float:
        """Процент заполненности контекстного окна текущим промптом."""
        return last_prompt_tokens / CONTEXT_LIMIT * 100


class ContextOverflowError(Exception):
    """Вызывается когда история превышает лимит контекста."""

    def __init__(self, prompt_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        super().__init__(
            f"История ({prompt_tokens} токенов) превышает лимит модели ({CONTEXT_LIMIT}). "
            "Сбросьте историю командой /reset."
        )


class Agent:
    """Агент, инкапсулирующий логику общения с LLM.

    Хранит историю диалога в памяти и персистентно в JSON.
    При создании загружает сохранённую историю — диалог продолжается
    после перезапуска приложения.

    Считает токены по каждому запросу: входные (вся история), выходные (ответ),
    суммарные. Предупреждает при приближении к лимиту контекста.
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

        self._stats = SessionStats()
        self._last_usage: TokenUsage | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, user_input: str) -> tuple[str, TokenUsage]:
        """Отправить сообщение и получить ответ вместе со статистикой токенов.

        Returns:
            (reply, TokenUsage) — текст ответа и статистика текущего запроса.

        Raises:
            ContextOverflowError: когда история превышает CONTEXT_LIMIT.
        """
        self._history.append({"role": "user", "content": user_input})

        try:
            response = self._client.chat.completions.create(
                model=MODEL_URI,
                messages=self._history,
                max_tokens=MAX_TOKENS,
            )
        except openai.BadRequestError as exc:
            # Убираем только что добавленное сообщение — состояние не меняем
            self._history.pop()
            raise exc

        usage = response.usage
        token_usage = TokenUsage(
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
        )

        # Проверяем, не переполнен ли контекст (по факту ответа)
        if token_usage.prompt_tokens > CONTEXT_LIMIT:
            self._history.pop()
            raise ContextOverflowError(token_usage.prompt_tokens)

        reply = response.choices[0].message.content or ""
        self._history.append({"role": "assistant", "content": reply})
        self._storage.save(self._user_id, self._history)

        self._stats.add(token_usage)
        self._last_usage = token_usage

        return reply, token_usage

    def reset(self) -> None:
        self._history = [{"role": "system", "content": self._system_prompt}]
        self._storage.delete(self._user_id)
        self._stats = SessionStats()
        self._last_usage = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def stats(self) -> SessionStats:
        return self._stats

    @property
    def last_usage(self) -> TokenUsage | None:
        return self._last_usage

    @property
    def history_len(self) -> int:
        """Число сообщений в истории (без системного)."""
        return max(0, len(self._history) - 1)
