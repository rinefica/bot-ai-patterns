"""TaskStateTracker: extracts and updates structured task state from dialog."""

from __future__ import annotations

import json

from bot_ai_patterns.client import get_client
from bot_ai_patterns.config import MODEL_URI
from .chat_session import TaskState

_SYSTEM = """\
Ты — анализатор диалога. По последнему сообщению пользователя и текущему состоянию задачи
обнови структуру состояния. Верни ТОЛЬКО JSON без markdown-обёртки.

Поля:
- goal: главная цель пользователя (что хочет сшить / узнать), строка или ""
- clarifications: список строк — что уже уточнено/подтверждено в диалоге
- constraints: список строк — ограничения (размер, ткань, навык, инструменты)
- terms: словарь термин→значение (швейные термины, которые упомянул пользователь)
- open_questions: список строк — что ещё не выяснено, но важно для цели

Правила:
- Не удаляй существующие данные, только дополняй или исправляй
- Если ничего нового — верни текущее состояние без изменений
- Не добавляй лишних полей"""


class TaskStateTracker:
    def __init__(self):
        self._client = get_client()

    def update(self, current_state: TaskState, user_message: str, history_snippet: str) -> TaskState:
        prompt = f"""\
Текущее состояние:
{current_state.to_json()}

Последние сообщения диалога:
{history_snippet}

Новое сообщение пользователя: {user_message}

Обнови состояние задачи."""

        try:
            response = self._client.chat.completions.create(
                model=MODEL_URI,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                max_tokens=512,
                temperature=0.1,
            )
            raw = response.choices[0].message.content or "{}"
            data = json.loads(raw)
            return TaskState.from_dict(data)
        except Exception:
            return current_state  # fallback — keep old state on error
