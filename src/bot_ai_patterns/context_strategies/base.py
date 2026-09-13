"""Базовый класс стратегии управления контекстом."""
from __future__ import annotations

from abc import ABC, abstractmethod


class ContextStrategy(ABC):
    """Определяет как формируется контекст запроса к LLM."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Slug-идентификатор стратегии."""

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Название для отображения пользователю."""

    def add_user(self, content: str) -> None:
        """Добавить сообщение пользователя."""

    def add_assistant(self, content: str) -> None:
        """Добавить ответ ассистента."""

    def rollback_user(self) -> None:
        """Откатить последнее сообщение пользователя (при ошибке API)."""

    @abstractmethod
    def build_messages(self, system_prompt: str) -> list[dict[str, str]]:
        """Собрать список сообщений для запроса к API."""

    def reset(self) -> None:
        """Сбросить состояние стратегии."""

    def init_from_messages(self, messages: list[dict[str, str]]) -> None:
        """Инициализировать из существующих сообщений (без системного)."""

    def get_state(self) -> dict:
        """Сериализуемое состояние для хранения."""
        return {"strategy": self.name}

    def load_state(self, state: dict) -> None:
        """Восстановить состояние из хранилища."""

    def stats_lines(self) -> list[str]:
        """Строки статистики для блока токенов в Telegram."""
        return [f"Стратегия: {self.display_name}"]

    @property
    @abstractmethod
    def message_count(self) -> int:
        """Число сообщений в текущем контексте (без системного)."""

    @property
    @abstractmethod
    def total_count(self) -> int:
        """Общее число сообщений, добавленных за сессию."""
