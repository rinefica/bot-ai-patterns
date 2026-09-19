"""MemoryManager — объединяет три слоя памяти.

Слои:
  STM (Short-Term Memory)  — текущий диалог, управляется ContextStrategy
  WM  (Working Memory)     — данные текущей задачи, авто-экстракция, сбрасывается с /reset
  LTM (Long-Term Memory)   — профиль/решения/знания, персистентная между сессиями
"""
from __future__ import annotations

import openai

from bot_ai_patterns.context_strategies.base import ContextStrategy
from bot_ai_patterns.memory.long_term_memory import LongTermMemory
from bot_ai_patterns.memory.user_profile import UserProfile
from bot_ai_patterns.memory.working_memory import WorkingMemory


class MemoryManager:
    """Управляет WM и LTM, инжектирует их в контекст поверх STM.

    Финальный порядок сообщений в запросе к API:
      [system_prompt]
      [LTM блок]      ← долговременная память, если непуста
      [WM блок]       ← рабочая память, если непуста
      [...STM msgs]   ← последние N сообщений диалога (стратегия)
    """

    def __init__(self, client: openai.OpenAI, user_id: int) -> None:
        self._client = client
        self.wm = WorkingMemory(client)
        self.ltm = LongTermMemory(user_id)
        self.profile = UserProfile(self.ltm)

    def build_messages(
        self, strategy: ContextStrategy, system_prompt: str
    ) -> list[dict[str, str]]:
        """Собрать финальный список сообщений с инжекцией памяти."""
        base = strategy.build_messages(system_prompt)

        # base[0] — system prompt; вставляем LTM и WM сразу после него
        if not base:
            return base

        result = [base[0]]

        # 1. Профиль — первым: явные инструкции по стилю для каждого ответа
        profile_block = self.profile.format_system_block()
        if profile_block:
            result.append({"role": "system", "content": profile_block})

        # 2. LTM — решения и знания из прошлых сессий
        ltm_block = self.ltm.format_block()
        if ltm_block:
            result.append({"role": "system", "content": ltm_block})

        # 3. WM — данные текущей задачи
        wm_block = self.wm.format_block()
        if wm_block:
            result.append({"role": "system", "content": wm_block})

        result.extend(base[1:])
        return result

    def after_exchange(self, user_msg: str, assistant_msg: str) -> None:
        """Обновить WM и LTM после успешного обмена.

        Вызывается асинхронно — не блокирует основной диалог.
        """
        self.wm.update_from_exchange(user_msg, assistant_msg)
        self.ltm.update_from_exchange(self._client, user_msg, assistant_msg)

    def reset(self) -> None:
        """Сбросить WM (LTM не сбрасывается)."""
        self.wm.reset()

    def format_telegram(self) -> str:
        """Отчёт о всех слоях памяти для Telegram."""
        sections = []

        # STM — краткая сводка
        sections.append("<b>STM (краткосрочная):</b> текущий диалог, управляется стратегией")

        # WM
        if self.wm.data:
            wm_lines = ["<b>WM (рабочая память — текущая задача):</b>"]
            for k, v in self.wm.data.items():
                wm_lines.append(f"  • <code>{k}</code>: {v}")
            sections.append("\n".join(wm_lines))
        else:
            sections.append("<b>WM (рабочая память):</b> пуста")

        # LTM (без profile — он отдельно)
        ltm_data = self.ltm.get_all()
        ltm_no_profile = {
            cat: d for cat, d in ltm_data.items() if cat != "profile"
        }
        if any(ltm_no_profile.values()):
            ltm_lines = ["<b>LTM (долговременная память):</b>"]
            labels = {"decisions": "Решения", "knowledge": "Знания"}
            for cat, items in ltm_no_profile.items():
                if items:
                    ltm_lines.append(f"  <i>{labels.get(cat, cat)}:</i>")
                    for k, v in items.items():
                        ltm_lines.append(f"    • <code>{k}</code>: {v}")
            sections.append("\n".join(ltm_lines))
        else:
            sections.append("<b>LTM (долговременная память):</b> пуста")

        # Профиль отдельным блоком
        sections.append(self.profile.format_telegram())

        return "\n\n".join(sections)

    # ------------------------------------------------------------------
    # Persistence helpers (WM only — LTM persists itself)
    # ------------------------------------------------------------------

    def get_wm_state(self) -> dict:
        return self.wm.get_state()

    def load_wm_state(self, state: dict) -> None:
        self.wm.load_state(state)
