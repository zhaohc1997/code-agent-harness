from __future__ import annotations

from code_agent_harness.policies.code_assistant import build_code_assistant_policy


def test_policy_blocks_disabled_tool() -> None:
    policy = build_code_assistant_policy(disabled_tools={"shell"})

    decision = policy.evaluate("shell", {"command": "ls"})

    assert decision.outcome == "block"
    assert decision.reason == "disabled_tool"


def test_policy_requires_confirmation_for_destructive_patch() -> None:
    policy = build_code_assistant_policy(disabled_tools={"shell"})

    decision = policy.evaluate(
        "apply_patch",
        {
            "path": "calc.py",
            "replacements": [{"old_text": "return a + b", "new_text": ""}],
        },
    )

    assert decision.outcome == "require_confirmation"
    assert decision.reason == "destructive_patch"


def test_policy_reminds_on_broad_test_run() -> None:
    policy = build_code_assistant_policy(disabled_tools={"shell"})

    decision = policy.evaluate("run_tests", {})

    assert decision.outcome == "remind"
    assert decision.reason == "broad_test_run"


def test_policy_reminds_on_option_only_test_run() -> None:
    policy = build_code_assistant_policy(disabled_tools={"shell"})

    decision = policy.evaluate("run_tests", {"args": ["-q"]})

    assert decision.outcome == "remind"
    assert decision.reason == "broad_test_run"
    assert "targeted" in decision.message.lower()


def test_policy_reminds_on_large_patch_attempt() -> None:
    policy = build_code_assistant_policy(disabled_tools={"shell"})

    decision = policy.evaluate(
        "apply_patch",
        {
            "path": "calc.py",
            "replacements": [
                {"old_text": "a", "new_text": "b"},
                {"old_text": "c", "new_text": "d"},
                {"old_text": "e", "new_text": "f"},
                {"old_text": "g", "new_text": "h"},
            ],
        },
    )

    assert decision.outcome == "remind"
    assert decision.reason == "large_patch"
