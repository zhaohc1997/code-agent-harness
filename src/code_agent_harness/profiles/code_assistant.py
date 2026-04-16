from dataclasses import dataclass
from pathlib import Path

from code_agent_harness.prompts.layers import PromptLayers


@dataclass(frozen=True)
class CodeAssistantProfile:
    name: str
    workspace_root: Path
    active_tool_names: tuple[str, ...]
    disabled_tools: tuple[str, ...]
    prompt_layers: PromptLayers
    provider_extra: dict[str, object]


def build_code_assistant_profile(
    *,
    workspace_root: Path,
    ablations: set[str] | None = None,
) -> CodeAssistantProfile:
    disabled_tools = {"shell"}
    active_tool_names = (
        "read_file",
        "search_text",
        "list_files",
        "apply_patch",
        "run_tests",
        "git_status",
        "ask_confirmation",
    )
    prompt_layers = PromptLayers(
        system="You are a code assistant for one local repository.",
        scenario="Use only the active tools. Prefer small, targeted investigation and tests.",
        execution="Reply in the user's language and report concrete code and test outcomes.",
    )
    provider_extra: dict[str, object] = {"thinking": {"enabled": True}}
    active_ablations = set() if ablations is None else set(ablations)

    if "tool_narrowing" in active_ablations:
        active_tool_names = active_tool_names + ("shell",)
        disabled_tools.clear()
    if "prompt_layers" in active_ablations:
        prompt_layers = PromptLayers(
            system=prompt_layers.system,
            scenario="",
            execution=prompt_layers.execution,
        )
    if "reasoning_mode" in active_ablations:
        provider_extra = {}

    return CodeAssistantProfile(
        name="code_assistant",
        workspace_root=workspace_root,
        active_tool_names=active_tool_names,
        disabled_tools=tuple(sorted(disabled_tools)),
        prompt_layers=prompt_layers,
        provider_extra=provider_extra,
    )
