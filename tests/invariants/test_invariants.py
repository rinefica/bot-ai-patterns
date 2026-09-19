"""Тесты InvariantsStore."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from bot_ai_patterns.invariants.invariants import CATEGORIES, InvariantsStore


def _store(user_id: int = 1, tmp_dir: Path | None = None) -> InvariantsStore:
    return InvariantsStore(user_id, tmp_dir or Path(tempfile.mkdtemp()))


class TestInvariantsStore:
    def test_initial_empty(self):
        s = _store()
        assert s.is_empty()
        assert s.get_all() == []

    def test_add_by_full_name(self):
        s = _store()
        inv = s.add("architecture", "монолит на FastAPI")
        assert inv.category == "architecture"
        assert inv.description == "монолит на FastAPI"
        assert inv.id == 1

    def test_add_by_alias(self):
        s = _store()
        inv = s.add("arch", "монолит")
        assert inv.category == "architecture"
        inv2 = s.add("tech", "PostgreSQL обязателен")
        assert inv2.category == "technical"
        inv3 = s.add("biz", "данные только в EU")
        assert inv3.category == "business"

    def test_add_unknown_category_raises(self):
        s = _store()
        with pytest.raises(ValueError, match="Неизвестная категория"):
            s.add("unknown", "что-то")

    def test_ids_autoincrement(self):
        s = _store()
        i1 = s.add("stack", "Python only")
        i2 = s.add("arch", "монолит")
        assert i1.id == 1
        assert i2.id == 2

    def test_remove_existing(self):
        s = _store()
        inv = s.add("stack", "Python only")
        assert s.remove(inv.id) is True
        assert s.is_empty()

    def test_remove_missing_returns_false(self):
        s = _store()
        assert s.remove(999) is False

    def test_clear_removes_all(self):
        s = _store()
        s.add("arch", "монолит")
        s.add("stack", "Python")
        s.clear()
        assert s.is_empty()

    def test_by_category(self):
        s = _store()
        s.add("arch", "монолит")
        s.add("arch", "REST API")
        s.add("stack", "Python only")
        by_cat = s.by_category()
        assert len(by_cat["architecture"]) == 2
        assert len(by_cat["stack"]) == 1
        assert len(by_cat["business"]) == 0

    def test_format_context_block_none_when_empty(self):
        s = _store()
        assert s.format_context_block() is None

    def test_format_context_block_contains_preamble(self):
        s = _store()
        s.add("arch", "монолит")
        block = s.format_context_block()
        assert block is not None
        assert "НАРУШАТЬ НЕЛЬЗЯ" in block
        assert "ОТКАЖИСЬ" in block

    def test_format_context_block_contains_all_categories(self):
        s = _store()
        s.add("arch", "монолит")
        s.add("stack", "Python only")
        s.add("biz", "GDPR compliance")
        block = s.format_context_block()
        assert "Архитектура" in block
        assert "Стек" in block
        assert "Бизнес" in block
        assert "монолит" in block
        assert "Python only" in block
        assert "GDPR" in block

    def test_format_context_block_shows_ids(self):
        s = _store()
        inv = s.add("arch", "монолит")
        block = s.format_context_block()
        assert f"[{inv.id}]" in block

    def test_format_telegram_empty_shows_hint(self):
        s = _store()
        text = s.format_telegram()
        assert "не заданы" in text.lower() or "Инварианты не заданы" in text

    def test_format_telegram_shows_invariants(self):
        s = _store()
        s.add("arch", "монолит на FastAPI")
        s.add("stack", "только Python")
        text = s.format_telegram()
        assert "монолит на FastAPI" in text
        assert "только Python" in text

    def test_persistence_across_instances(self):
        tmp = Path(tempfile.mkdtemp())
        s1 = InvariantsStore(1, tmp)
        s1.add("arch", "монолит")
        s1.add("stack", "Python only")

        s2 = InvariantsStore(1, tmp)
        assert len(s2.get_all()) == 2
        assert s2.get_all()[0].description == "монолит"
        # ID продолжает с правильного значения
        inv = s2.add("biz", "EU only")
        assert inv.id == 3

    def test_different_users_isolated(self):
        tmp = Path(tempfile.mkdtemp())
        s1 = InvariantsStore(1, tmp)
        s1.add("arch", "монолит")

        s2 = InvariantsStore(2, tmp)
        assert s2.is_empty()

    def test_remove_and_add_keeps_unique_ids(self):
        s = _store()
        i1 = s.add("arch", "монолит")
        s.add("stack", "Python")
        s.remove(i1.id)
        i3 = s.add("biz", "GDPR")
        assert i3.id == 3  # не переиспользует удалённый ID

    def test_all_categories_valid(self):
        s = _store()
        for cat in CATEGORIES:
            inv = s.add(cat, f"тест {cat}")
            assert inv.category == cat
