"""Долговременная память — профиль, решения, знания (персистентная).

Область действия: между сессиями. НЕ сбрасывается при /reset.
"""
from __future__ import annotations

import json
from pathlib import Path

import openai

from bot_ai_patterns.config import MODEL_URI

MEMORY_DIR = Path("data/memory")
CATEGORIES = ("profile", "decisions", "knowledge")

_CATEGORY_LABELS = {
    "profile": "Профиль пользователя",
    "decisions": "Принятые решения",
    "knowledge": "Накопленные знания",
}

_EXTRACT_SYSTEM = (
    "Ты извлекаешь долгосрочную информацию о пользователе из диалога.\n"
    "Выведи строки формата «категория|ключ: значение», по одной на строке.\n"
    "Категории: profile (предпочтения, язык, уровень опыта), "
    "decisions (принятые технические решения), "
    "knowledge (важные факты о проекте/предметной области).\n"
    "Только реально присутствующее в тексте. Если ничего — верни пустую строку."
)


class LongTermMemory:
    """Хранит факты о пользователе между сессиями.

    Три категории:
    - profile: язык, уровень опыта, предпочтения
    - decisions: принятые технические и продуктовые решения
    - knowledge: накопленные факты о предметной области/проекте

    Персистируется в data/memory/{user_id}_ltm.json.
    НЕ сбрасывается при /reset.
    """

    def __init__(self, user_id: int, memory_dir: Path = MEMORY_DIR) -> None:
        self._user_id = user_id
        self._dir = memory_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, dict[str, str]] = {c: {} for c in CATEGORIES}
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set(self, category: str, key: str, value: str) -> None:
        if category not in CATEGORIES:
            raise ValueError(f"Неизвестная категория: {category}. Допустимые: {CATEGORIES}")
        self._data[category][key] = value
        self._save()

    def remove(self, category: str, key: str) -> bool:
        if category not in self._data:
            return False
        was_there = self._data[category].pop(key, None) is not None
        if was_there:
            self._save()
        return was_there

    def get_all(self) -> dict[str, dict[str, str]]:
        return {cat: dict(d) for cat, d in self._data.items()}

    def is_empty(self) -> bool:
        return all(not d for d in self._data.values())

    def update_from_exchange(
        self, client: openai.OpenAI, user_msg: str, assistant_msg: str
    ) -> None:
        """Авто-обновить память на основе нового обмена."""
        exchange = f"Пользователь: {user_msg}\nАссистент: {assistant_msg}"
        try:
            resp = client.chat.completions.create(
                model=MODEL_URI,
                messages=[
                    {"role": "system", "content": _EXTRACT_SYSTEM},
                    {"role": "user", "content": exchange},
                ],
                max_tokens=200,
            )
            raw = resp.choices[0].message.content or ""
            for line in raw.strip().splitlines():
                line = line.strip().lstrip("•-").strip()
                if "|" in line and ":" in line:
                    cat_part, _, rest = line.partition("|")
                    cat = cat_part.strip().lower()
                    if cat not in CATEGORIES:
                        continue
                    key, _, val = rest.partition(":")
                    key, val = key.strip(), val.strip()
                    if key and val:
                        self._data[cat][key] = val
            self._save()
        except Exception:
            pass  # не ломаем диалог

    def format_block(self) -> str | None:
        if self.is_empty():
            return None
        lines = ["[Долговременная память]"]
        for cat in CATEGORIES:
            if self._data[cat]:
                lines.append(f"  {_CATEGORY_LABELS[cat]}:")
                for k, v in self._data[cat].items():
                    lines.append(f"    • {k}: {v}")
        return "\n".join(lines)

    def format_telegram(self) -> str:
        if self.is_empty():
            return "Долговременная память пуста."
        lines = ["<b>Долговременная память</b>"]
        for cat in CATEGORIES:
            if self._data[cat]:
                lines.append(f"\n<b>{_CATEGORY_LABELS[cat]}:</b>")
                for k, v in self._data[cat].items():
                    lines.append(f"  • <code>{k}</code>: {v}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _path(self) -> Path:
        return self._dir / f"{self._user_id}_ltm.json"

    def _load(self) -> None:
        path = self._path()
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            for cat in CATEGORIES:
                self._data[cat] = raw.get(cat, {})

    def _save(self) -> None:
        self._path().write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
