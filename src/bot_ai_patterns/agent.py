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
from bot_ai_patterns.context_strategies import ContextStrategy, SlidingWindowStrategy
from bot_ai_patterns.storage import JSONStorage

_DEFAULT_SYSTEM = "Ты полезный ассистент."


@dataclass
class TokenUsage:
    """Статистика токенов одного запроса."""

    prompt_tokens: int       # токены входа (контекст + запрос)
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
        return last_prompt_tokens / CONTEXT_LIMIT * 100


class ContextOverflowError(Exception):
    """Вызывается когда контекст превышает лимит модели."""

    def __init__(self, prompt_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        super().__init__(
            f"Контекст ({prompt_tokens} токенов) превышает лимит модели ({CONTEXT_LIMIT}). "
            "Сбросьте историю командой /reset или смените стратегию."
        )


class Agent:
    """Агент с подключаемыми стратегиями управления контекстом.

    Стратегия определяет какие сообщения отправлять в API.
    Полный аудит-лог хранится в _all_messages независимо от стратегии —
    это позволяет корректно инициализировать новую стратегию при переключении.
    """

    def __init__(
        self,
        client: openai.OpenAI,
        user_id: int,
        storage: JSONStorage,
        system_prompt: str = _DEFAULT_SYSTEM,
        strategy: ContextStrategy | None = None,
    ) -> None:
        self._client = client
        self._user_id = user_id
        self._storage = storage
        self._system_prompt = system_prompt
        self._strategy: ContextStrategy = strategy or SlidingWindowStrategy()

        # Аудит-лог: все сообщения за сессию (без системного)
        self._all_messages: list[dict[str, str]] = []

        saved = storage.load(user_id)
        if saved:
            self._all_messages = saved.get("all_messages", [])
            strategy_state = saved.get("strategy_state")
            if strategy_state and strategy_state.get("strategy") == self._strategy.name:
                self._strategy.load_state(strategy_state)
            elif self._all_messages:
                self._strategy.init_from_messages(self._all_messages)

        self._stats = SessionStats()
        self._last_usage: TokenUsage | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, user_input: str) -> tuple[str, TokenUsage]:
        """Отправить сообщение, получить ответ + статистику токенов.

        Raises:
            ContextOverflowError: если контекст превышает CONTEXT_LIMIT.
        """
        self._strategy.add_user(user_input)
        self._all_messages.append({"role": "user", "content": user_input})

        messages = self._strategy.build_messages(self._system_prompt)

        try:
            response = self._client.chat.completions.create(
                model=MODEL_URI,
                messages=messages,
                max_tokens=MAX_TOKENS,
            )
        except openai.BadRequestError as exc:
            self._strategy.rollback_user()
            self._all_messages.pop()
            raise exc

        usage = response.usage
        token_usage = TokenUsage(
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
        )

        if token_usage.prompt_tokens > CONTEXT_LIMIT:
            self._strategy.rollback_user()
            self._all_messages.pop()
            raise ContextOverflowError(token_usage.prompt_tokens)

        reply = response.choices[0].message.content or ""
        self._strategy.add_assistant(reply)
        self._all_messages.append({"role": "assistant", "content": reply})

        self._storage.save(
            self._user_id,
            strategy_name=self._strategy.name,
            strategy_state=self._strategy.get_state(),
            all_messages=self._all_messages,
        )
        self._stats.add(token_usage)
        self._last_usage = token_usage

        return reply, token_usage

    def switch_strategy(self, strategy: ContextStrategy) -> None:
        """Переключить стратегию, инициализируя её из аудит-лога."""
        strategy.init_from_messages(self._all_messages)
        self._strategy = strategy

    def reset(self) -> None:
        self._strategy.reset()
        self._all_messages = []
        self._storage.delete(self._user_id)
        self._stats = SessionStats()
        self._last_usage = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def strategy(self) -> ContextStrategy:
        return self._strategy

    @property
    def stats(self) -> SessionStats:
        return self._stats

    @property
    def last_usage(self) -> TokenUsage | None:
        return self._last_usage

    @property
    def history_len(self) -> int:
        """Полное число сообщений за сессию (аудит-лог, без системного)."""
        return len(self._all_messages)
