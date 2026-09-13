"""Тесты BranchingStrategy."""
from __future__ import annotations

from bot_ai_patterns.context_strategies.branching import BranchingStrategy


class TestBranching:
    def test_initial_has_main_branch(self):
        s = BranchingStrategy()
        assert "main" in s.branch_names
        assert s.current_branch_name == "main"

    def test_messages_go_to_current_branch(self):
        s = BranchingStrategy()
        s.add_user("привет")
        s.add_assistant("здравствуй")
        assert s.message_count == 2

    def test_build_messages_includes_system(self):
        s = BranchingStrategy()
        s.add_user("вопрос")
        msgs = s.build_messages("Ты ассистент.")
        assert msgs[0]["role"] == "system"
        assert msgs[1]["content"] == "вопрос"

    def test_set_checkpoint(self):
        s = BranchingStrategy()
        s.add_user("q1")
        s.add_assistant("a1")
        s.set_checkpoint()
        assert s.checkpoint_set
        assert len(s._checkpoint) == 2

    def test_create_branch_requires_checkpoint(self):
        s = BranchingStrategy()
        assert s.create_branch("feature") is False

    def test_create_branch_from_checkpoint(self):
        s = BranchingStrategy()
        s.add_user("q1")
        s.add_assistant("a1")
        s.set_checkpoint()
        assert s.create_branch("feature") is True
        assert "feature" in s.branch_names

    def test_branch_starts_from_checkpoint(self):
        s = BranchingStrategy()
        s.add_user("общее")
        s.add_assistant("основа")
        s.set_checkpoint()
        s.create_branch("b2")

        s.add_user("main exclusive")
        s.add_assistant("main answer")

        s.switch_branch("b2")
        contents = [m["content"] for m in s.build_messages("sys")]
        assert "общее" in contents
        assert "main exclusive" not in contents

    def test_branches_are_independent(self):
        s = BranchingStrategy()
        s.add_user("общий")
        s.add_assistant("ответ")
        s.set_checkpoint()
        s.create_branch("b2")

        s.add_user("main вопрос")
        s.add_assistant("main ответ")

        s.switch_branch("b2")
        s.add_user("b2 вопрос")
        s.add_assistant("b2 ответ")

        s.switch_branch("main")
        main_contents = [m["content"] for m in s.build_messages("sys")]
        assert "main вопрос" in main_contents
        assert "b2 вопрос" not in main_contents

        s.switch_branch("b2")
        b2_contents = [m["content"] for m in s.build_messages("sys")]
        assert "b2 вопрос" in b2_contents
        assert "main вопрос" not in b2_contents

    def test_switch_nonexistent_branch(self):
        s = BranchingStrategy()
        assert s.switch_branch("nonexistent") is False

    def test_create_duplicate_branch_fails(self):
        s = BranchingStrategy()
        s.set_checkpoint()
        s.create_branch("b2")
        assert s.create_branch("b2") is False

    def test_rollback_user(self):
        s = BranchingStrategy()
        s.add_user("вопрос")
        s.rollback_user()
        assert s.message_count == 0

    def test_reset_clears_all_branches(self):
        s = BranchingStrategy()
        s.add_user("q")
        s.set_checkpoint()
        s.create_branch("b2")
        s.reset()
        assert s.branch_names == ["main"]
        assert s.current_branch_name == "main"
        assert not s.checkpoint_set

    def test_state_roundtrip(self):
        s = BranchingStrategy()
        s.add_user("q1")
        s.add_assistant("a1")
        s.set_checkpoint()
        s.create_branch("b2")
        s.switch_branch("b2")
        s.add_user("q2")
        state = s.get_state()

        s2 = BranchingStrategy()
        s2.load_state(state)
        assert s2.current_branch_name == "b2"
        assert "b2" in s2.branch_names
        assert s2.message_count == s.message_count

    def test_init_from_messages_populates_main(self):
        s = BranchingStrategy()
        s.init_from_messages([
            {"role": "user", "content": "вопрос"},
            {"role": "assistant", "content": "ответ"},
        ])
        contents = [m["content"] for m in s.build_messages("sys")]
        assert "вопрос" in contents

    def test_stats_lines_show_branches(self):
        s = BranchingStrategy()
        s.set_checkpoint()
        s.create_branch("b2")
        lines = s.stats_lines()
        assert any("Веток" in l for l in lines)
        assert any("main" in l for l in lines)
