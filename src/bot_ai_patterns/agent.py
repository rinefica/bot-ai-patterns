from __future__ import annotations

from dataclasses import dataclass, field

import openai

from bot_ai_patterns.compressor import Compressor
from bot_ai_patterns.config import (
    COMPRESS_AFTER_N,
    CONTEXT_LIMIT,
    CONTEXT_WARN_THRESHOLD,
    MAIN_MODEL_PRICE_PER_1K,
    MAX_TOKENS,
    MODEL_URI,
    RECENT_KEEP,
)
from bot_ai_patterns.storage import JSONStorage

_DEFAULT_SYSTEM = "Ты полезный ассистент."


@dataclass
class TokenUsage:
    """Статистика токенов одного запроса."""

    prompt_tokens: int       # токены входа (контекст + новый запрос)
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


@dataclass
class CompressionStats:
    """Статистика работы компрессора."""

    compressions_done: int = 0       # сколько раз сжимали
    messages_compressed: int = 0    # сколько всего сообщений сжато
    tokens_before: int = 0          # prompt_tokens до последнего сжатия
    tokens_after: int = 0           # prompt_tokens после последнего сжатия

    @property
    def tokens_saved(self) -> int:
        """Токены сэкономленные последним сжатием."""
        return max(0, self.tokens_before - self.tokens_after)


class ContextOverflowError(Exception):
    """Вызывается когда история превышает лимит контекста."""

    def __init__(self, prompt_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        super().__init__(
            f"История ({prompt_tokens} токенов) превышает лимит модели ({CONTEXT_LIMIT}). "
            "Сбросьте историю командой /reset."
        )


class Agent:
    """Агент с управлением контекстом через компрессию.

    Два режима:
    - Обычный (use_compression=False): отправляет полную историю на каждый запрос.
    - Компрессионный (use_compression=True): старые сообщения заменяются summary,
      в контекст подставляются только последние RECENT_KEEP сообщений + резюме.

    Полная история (_history) хранится всегда — для аудита и переключения режимов.
    """

    def __init__(
        self,
        client: openai.OpenAI,
        user_id: int,
        storage: JSONStorage,
        system_prompt: str = _DEFAULT_SYSTEM,
        use_compression: bool = False,
    ) -> None:
        self._client = client
        self._user_id = user_id
        self._storage = storage
        self._system_prompt = system_prompt
        self._use_compression = use_compression
        self._compressor = Compressor(client)

        saved = storage.load(user_id)
        system_msg = {"role": "system", "content": system_prompt}

        self._history: list[dict[str, str]] = (
            saved["history"] if saved.get("history") else [system_msg]
        )
        self._summary: str = saved.get("summary", "")
        self._recent: list[dict[str, str]] = saved.get("recent", [])

        self._stats = SessionStats()
        self._compress_stats = CompressionStats()
        self._last_usage: TokenUsage | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, user_input: str) -> tuple[str, TokenUsage]:
        """Отправить сообщение, получить ответ + статистику токенов.

        Raises:
            ContextOverflowError: когда контекст превышает CONTEXT_LIMIT.
        """
        user_msg: dict[str, str] = {"role": "user", "content": user_input}
        self._history.append(user_msg)
        if self._use_compression:
            self._recent.append(user_msg)

        messages = self._build_messages()

        try:
            response = self._client.chat.completions.create(
                model=MODEL_URI,
                messages=messages,
                max_tokens=MAX_TOKENS,
            )
        except openai.BadRequestError as exc:
            self._history.pop()
            if self._use_compression:
                self._recent.pop()
            raise exc

        usage = response.usage
        token_usage = TokenUsage(
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
        )

        if token_usage.prompt_tokens > CONTEXT_LIMIT:
            self._history.pop()
            if self._use_compression:
                self._recent.pop()
            raise ContextOverflowError(token_usage.prompt_tokens)

        reply = response.choices[0].message.content or ""
        assistant_msg: dict[str, str] = {"role": "assistant", "content": reply}
        self._history.append(assistant_msg)

        if self._use_compression:
            self._recent.append(assistant_msg)
            self._maybe_compress(token_usage.prompt_tokens)

        self._storage.save(
            self._user_id,
            self._history,
            summary=self._summary,
            recent=self._recent,
        )
        self._stats.add(token_usage)
        self._last_usage = token_usage

        return reply, token_usage

    def toggle_compression(self) -> bool:
        """Переключить режим компрессии. Возвращает новое значение флага."""
        self._use_compression = not self._use_compression
        return self._use_compression

    def reset(self) -> None:
        self._history = [{"role": "system", "content": self._system_prompt}]
        self._summary = ""
        self._recent = []
        self._storage.delete(self._user_id)
        self._stats = SessionStats()
        self._compress_stats = CompressionStats()
        self._last_usage = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def compression_enabled(self) -> bool:
        return self._use_compression

    @property
    def summary(self) -> str:
        return self._summary

    @property
    def stats(self) -> SessionStats:
        return self._stats

    @property
    def compress_stats(self) -> CompressionStats:
        return self._compress_stats

    @property
    def last_usage(self) -> TokenUsage | None:
        return self._last_usage

    @property
    def history_len(self) -> int:
        """Число сообщений в полной истории (без системного)."""
        return max(0, len(self._history) - 1)

    @property
    def recent_len(self) -> int:
        """Число сообщений в текущем 'окне' компрессионного режима."""
        return len(self._recent)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _build_messages(self) -> list[dict[str, str]]:
        """Собрать контекст для отправки в API."""
        if not self._use_compression or not self._summary:
            return self._history

        msgs: list[dict[str, str]] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "system", "content": f"[Резюме предыдущего диалога]\n{self._summary}"},
        ]
        msgs.extend(self._recent)
        return msgs

    def _maybe_compress(self, prompt_tokens_before: int) -> None:
        """Сжать историю если recent достиг порога COMPRESS_AFTER_N."""
        if len(self._recent) < COMPRESS_AFTER_N:
            return

        to_compress = self._recent[:-RECENT_KEEP]
        self._summary = self._compressor.summarize(to_compress, self._summary)
        self._recent = self._recent[-RECENT_KEEP:]

        self._compress_stats.compressions_done += 1
        self._compress_stats.messages_compressed += len(to_compress)
        self._compress_stats.tokens_before = prompt_tokens_before
        # tokens_after будет обновлён после следующего запроса
