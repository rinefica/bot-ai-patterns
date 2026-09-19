"""Тесты UserProfile."""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from bot_ai_patterns.memory.long_term_memory import LongTermMemory
from bot_ai_patterns.memory.user_profile import UserProfile, PRESETS


def _profile(tmp_dir: Path | None = None) -> UserProfile:
    tmp = tmp_dir or Path(tempfile.mkdtemp())
    ltm = LongTermMemory(1, tmp)
    return UserProfile(ltm)


class TestUserProfile:
    def test_initial_empty(self):
        p = _profile()
        assert p.is_empty()
        assert p.data == {}

    def test_set_and_get(self):
        p = _profile()
        p.set("стиль", "кратко")
        assert p.data["стиль"] == "кратко"

    def test_set_lowercases_field(self):
        p = _profile()
        p.set("Стиль", "кратко")
        assert "стиль" in p.data

    def test_remove_existing(self):
        p = _profile()
        p.set("уровень", "senior")
        assert p.remove("уровень") is True
        assert "уровень" not in p.data

    def test_remove_missing_returns_false(self):
        p = _profile()
        assert p.remove("nonexistent") is False

    def test_clear_removes_all_fields(self):
        p = _profile()
        p.set("стиль", "кратко")
        p.set("уровень", "senior")
        p.clear()
        assert p.is_empty()

    def test_apply_preset_junior(self):
        p = _profile()
        assert p.apply_preset("junior") is True
        data = p.data
        assert "уровень" in data
        assert "стиль" in data

    def test_apply_preset_senior(self):
        p = _profile()
        assert p.apply_preset("senior") is True
        assert "кратко" in p.data.get("стиль", "")

    def test_apply_preset_unknown_returns_false(self):
        p = _profile()
        assert p.apply_preset("unknown_preset") is False

    def test_all_presets_valid(self):
        for name in PRESETS:
            p = _profile()
            assert p.apply_preset(name) is True
            assert not p.is_empty()

    def test_format_system_block_none_when_empty(self):
        p = _profile()
        assert p.format_system_block() is None

    def test_format_system_block_contains_instruction(self):
        p = _profile()
        p.set("стиль", "кратко")
        block = p.format_system_block()
        assert block is not None
        assert "Профиль пользователя" in block
        assert "кратко" in block

    def test_format_system_block_standard_fields_ordered(self):
        p = _profile()
        p.set("ограничения", "без эмодзи")
        p.set("имя", "Саша")
        p.set("стиль", "подробно")
        block = p.format_system_block()
        # имя должно идти раньше стиля, стиль раньше ограничений
        assert block.index("Как обращаться") < block.index("Стиль ответов")
        assert block.index("Стиль ответов") < block.index("Ограничения")

    def test_format_telegram_empty_shows_hint(self):
        p = _profile()
        text = p.format_telegram()
        assert "Профиль пуст" in text
        assert "preset" in text

    def test_format_telegram_shows_fields(self):
        p = _profile()
        p.set("уровень", "senior")
        p.set("стек", "Python, FastAPI")
        text = p.format_telegram()
        assert "senior" in text
        assert "Python" in text

    def test_profile_persists_via_ltm(self):
        tmp = Path(tempfile.mkdtemp())
        p1 = _profile(tmp)
        p1.set("имя", "Александр")
        p1.set("стиль", "кратко")

        ltm2 = LongTermMemory(1, tmp)
        p2 = UserProfile(ltm2)
        assert p2.data.get("имя") == "Александр"
        assert p2.data.get("стиль") == "кратко"
