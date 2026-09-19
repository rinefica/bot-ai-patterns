"""Инварианты — ограничения, которые ассистент не имеет права нарушать.

Хранятся отдельно от диалога и памяти.
Инжектируются первым системным блоком с явным запретом нарушений.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

INVARIANTS_DIR = Path("data/invariants")

CATEGORIES: dict[str, str] = {
    "architecture": "Архитектура",
    "technical":    "Технические решения",
    "stack":        "Стек",
    "business":     "Бизнес-правила",
}

# Алиасы для удобного ввода
CATEGORY_ALIASES: dict[str, str] = {
    "arch":         "architecture",
    "architecture": "architecture",
    "tech":         "technical",
    "technical":    "technical",
    "stack":        "stack",
    "biz":          "business",
    "business":     "business",
}

# Системная инструкция — формулировка критична для поведения модели
_SYSTEM_PREAMBLE = (
    "ИНВАРИАНТЫ — НАРУШАТЬ НЕЛЬЗЯ.\n"
    "Следующие решения зафиксированы командой и не подлежат пересмотру.\n"
    "Правила:\n"
    "1. Если запрос пользователя противоречит инварианту — ОТКАЖИСЬ его выполнять.\n"
    "2. Явно назови, какой инвариант нарушен и почему.\n"
    "3. Предложи альтернативу, которая инвариант соблюдает.\n"
    "4. Не допускай частичного нарушения: нельзя «немного» отступить от инварианта."
)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class Invariant:
    id: int
    category: str
    description: str
    created_at: str = field(default_factory=_now)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "category": self.category,
            "description": self.description,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Invariant:
        return cls(
            id=d["id"],
            category=d["category"],
            description=d["description"],
            created_at=d.get("created_at", _now()),
        )


class InvariantsStore:
    """Хранилище инвариантов пользователя.

    Персистируется в data/invariants/{user_id}_invariants.json.
    Независимо от истории диалога, LTM и FSM задачи.
    """

    def __init__(self, user_id: int, inv_dir: Path = INVARIANTS_DIR) -> None:
        self._user_id = user_id
        self._dir = inv_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._items: list[Invariant] = []
        self._next_id: int = 1
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, category: str, description: str) -> Invariant:
        """Добавить инвариант. category — строка или алиас."""
        resolved = CATEGORY_ALIASES.get(category.lower())
        if resolved is None:
            raise ValueError(
                f"Неизвестная категория: '{category}'.\n"
                f"Допустимые: {', '.join(CATEGORIES)}\n"
                f"Алиасы: arch, tech, stack, biz"
            )
        inv = Invariant(id=self._next_id, category=resolved, description=description)
        self._items.append(inv)
        self._next_id += 1
        self._save()
        return inv

    def remove(self, inv_id: int) -> bool:
        before = len(self._items)
        self._items = [i for i in self._items if i.id != inv_id]
        if len(self._items) < before:
            self._save()
            return True
        return False

    def clear(self) -> None:
        self._items = []
        self._next_id = 1
        self._save()

    def get_all(self) -> list[Invariant]:
        return list(self._items)

    def by_category(self) -> dict[str, list[Invariant]]:
        result: dict[str, list[Invariant]] = {cat: [] for cat in CATEGORIES}
        for inv in self._items:
            result.setdefault(inv.category, []).append(inv)
        return result

    def is_empty(self) -> bool:
        return not self._items

    # ------------------------------------------------------------------
    # Форматирование
    # ------------------------------------------------------------------

    def format_context_block(self) -> str | None:
        """Системный блок для инжекции в контекст модели."""
        if self.is_empty():
            return None
        lines = [_SYSTEM_PREAMBLE, ""]
        by_cat = self.by_category()
        for cat, label in CATEGORIES.items():
            items = by_cat.get(cat, [])
            if items:
                lines.append(f"{label}:")
                for inv in items:
                    lines.append(f"  [{inv.id}] {inv.description}")
        return "\n".join(lines)

    def format_telegram(self) -> str:
        """Читаемое отображение для Telegram."""
        if self.is_empty():
            return (
                "<b>Инварианты не заданы.</b>\n\n"
                "Добавь командой:\n"
                "  <code>/invariant add arch монолит на FastAPI — не предлагать микросервисы</code>\n"
                "  <code>/invariant add stack только Python — без Go и Java</code>\n\n"
                f"Категории: {', '.join(f'<code>{a}</code>' for a in ['arch', 'tech', 'stack', 'biz'])}"
            )
        lines = ["<b>Инварианты (нельзя нарушать):</b>"]
        by_cat = self.by_category()
        for cat, label in CATEGORIES.items():
            items = by_cat.get(cat, [])
            if items:
                lines.append(f"\n<b>{label}:</b>")
                for inv in items:
                    lines.append(f"  <code>[{inv.id}]</code> {inv.description}")
        lines.append(
            "\n<i>Удалить: /invariant remove &lt;id&gt; | Очистить: /invariant clear</i>"
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Персистентность
    # ------------------------------------------------------------------

    def _path(self) -> Path:
        return self._dir / f"{self._user_id}_invariants.json"

    def _load(self) -> None:
        path = self._path()
        if not path.exists():
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        self._items = [Invariant.from_dict(d) for d in data.get("items", [])]
        self._next_id = data.get("next_id", len(self._items) + 1)

    def _save(self) -> None:
        data = {
            "next_id": self._next_id,
            "items": [i.to_dict() for i in self._items],
        }
        self._path().write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
