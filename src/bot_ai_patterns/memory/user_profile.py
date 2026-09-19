"""Профиль пользователя — структурированные предпочтения по стилю и формату.

Хранится в LTM категории 'profile'.
Инжектируется в каждый запрос как явная инструкция для модели.
"""
from __future__ import annotations

from bot_ai_patterns.memory.long_term_memory import LongTermMemory

# Стандартные поля профиля и их описания для отображения
STANDARD_FIELDS: dict[str, str] = {
    "имя": "Как обращаться",
    "стиль": "Стиль ответов",
    "формат": "Формат вывода",
    "уровень": "Уровень опыта",
    "стек": "Основные технологии",
    "ограничения": "Ограничения",
    "язык": "Язык общения",
}

# Пресеты для быстрой настройки
PRESETS: dict[str, dict[str, str]] = {
    "junior": {
        "стиль": "подробно, с объяснениями и примерами",
        "уровень": "начинающий разработчик",
        "формат": "шаг за шагом",
        "ограничения": "избегай сложных терминов без объяснений",
    },
    "senior": {
        "стиль": "кратко и по делу, без воды",
        "уровень": "опытный разработчик",
        "формат": "код и ключевые факты",
        "ограничения": "без длинных вступлений, без очевидных объяснений",
    },
    "manager": {
        "стиль": "структурировано, с выводами",
        "уровень": "технический менеджер",
        "формат": "bullet points, выводы в конце",
        "ограничения": "без лишнего кода, фокус на решениях и рисках",
    },
}


class UserProfile:
    """Профиль пользователя с предпочтениями по стилю общения.

    Обёртка над LongTermMemory[profile].
    Генерирует системный блок-инструкцию для модели.
    """

    def __init__(self, ltm: LongTermMemory) -> None:
        self._ltm = ltm

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def data(self) -> dict[str, str]:
        return self._ltm.get_all()["profile"]

    def is_empty(self) -> bool:
        return not self.data

    def set(self, field: str, value: str) -> None:
        """Установить поле профиля."""
        self._ltm.set("profile", field.strip().lower(), value.strip())

    def remove(self, field: str) -> bool:
        """Удалить поле профиля."""
        return self._ltm.remove("profile", field.strip().lower())

    def apply_preset(self, preset_name: str) -> bool:
        """Применить пресет настроек. Возвращает False если пресет не найден."""
        preset = PRESETS.get(preset_name.lower())
        if preset is None:
            return False
        for field, value in preset.items():
            self._ltm.set("profile", field, value)
        return True

    def clear(self) -> None:
        """Очистить весь профиль."""
        for field in list(self.data.keys()):
            self._ltm.remove("profile", field)

    # ------------------------------------------------------------------
    # Форматирование
    # ------------------------------------------------------------------

    def format_system_block(self) -> str | None:
        """Блок-инструкция для инжекции в системный контекст."""
        profile = self.data
        if not profile:
            return None

        lines = [
            "[Профиль пользователя — строго следуй этим предпочтениям в каждом ответе]"
        ]

        # Сначала стандартные поля в фиксированном порядке
        for key in STANDARD_FIELDS:
            if key in profile:
                label = STANDARD_FIELDS[key]
                lines.append(f"• {label}: {profile[key]}")

        # Затем нестандартные поля
        for key, val in profile.items():
            if key not in STANDARD_FIELDS:
                lines.append(f"• {key}: {val}")

        return "\n".join(lines)

    def format_telegram(self) -> str:
        """Отображение профиля в Telegram."""
        profile = self.data
        if not profile:
            return (
                "<b>Профиль пуст.</b>\n\n"
                "Установи предпочтения:\n"
                "  <code>/profile set стиль кратко и по делу</code>\n"
                "  <code>/profile set уровень senior</code>\n"
                "  <code>/profile set ограничения без эмодзи</code>\n\n"
                f"Или выбери пресет: /profile preset [{' | '.join(PRESETS)}]"
            )

        lines = ["<b>Профиль пользователя</b>\n"]
        for key in STANDARD_FIELDS:
            if key in profile:
                lines.append(f"• <b>{STANDARD_FIELDS[key]}</b>: {profile[key]}")
        for key, val in profile.items():
            if key not in STANDARD_FIELDS:
                lines.append(f"• <b>{key}</b>: {val}")

        lines.append(
            f"\n<i>Поля: {', '.join(f'<code>{k}</code>' for k in STANDARD_FIELDS)}</i>"
        )
        return "\n".join(lines)
