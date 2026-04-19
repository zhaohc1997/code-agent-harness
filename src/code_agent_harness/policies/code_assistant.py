from __future__ import annotations

from code_agent_harness.policies.engine import PolicyDecision


class CodeAssistantPolicy:
    def __init__(self, *, disabled_tools: set[str]) -> None:
        self._disabled_tools = frozenset(disabled_tools)

    def evaluate(self, tool_name: str, arguments: dict[str, object]) -> PolicyDecision:
        if tool_name in self._disabled_tools:
            return PolicyDecision(
                outcome="block",
                reason="disabled_tool",
                message=f"Tool {tool_name} is disabled in code_assistant.",
            )
        if tool_name == "run_tests" and _looks_broad_test_run(arguments):
            return PolicyDecision(
                outcome="remind",
                reason="broad_test_run",
                message=(
                    "Prefer targeted tests before the full suite; pass args like "
                    "['-q', 'tests/test_file.py']."
                ),
            )
        if tool_name == "apply_patch" and _looks_destructive_patch(arguments):
            return PolicyDecision(
                outcome="require_confirmation",
                reason="destructive_patch",
                message="Deleting code requires user confirmation in practical mode.",
            )
        if tool_name == "apply_patch" and _looks_large_patch(arguments):
            return PolicyDecision(
                outcome="remind",
                reason="large_patch",
                message="Large patches should be split after inspecting the relevant file and tests.",
            )
        return PolicyDecision(outcome="execute", reason="allowed", message="Allowed.")


def build_code_assistant_policy(*, disabled_tools: set[str]) -> CodeAssistantPolicy:
    return CodeAssistantPolicy(disabled_tools=disabled_tools)


def _looks_broad_test_run(arguments: dict[str, object]) -> bool:
    args = arguments.get("args")
    if not isinstance(args, list) or not args:
        return True
    return not any(isinstance(arg, str) and not arg.startswith("-") for arg in args)


def _looks_destructive_patch(arguments: dict[str, object]) -> bool:
    replacements = arguments.get("replacements")
    if not isinstance(replacements, list):
        return False
    return any(
        isinstance(replacement, dict) and replacement.get("new_text") == ""
        for replacement in replacements
    )


def _looks_large_patch(arguments: dict[str, object]) -> bool:
    replacements = arguments.get("replacements")
    return isinstance(replacements, list) and len(replacements) > 3
