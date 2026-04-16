from pathlib import Path

from code_agent_harness.profiles.code_assistant import build_code_assistant_profile


def test_code_assistant_profile_narrows_tools_and_enables_reasoning(tmp_path: Path) -> None:
    profile = build_code_assistant_profile(workspace_root=tmp_path)

    assert profile.name == "code_assistant"
    assert profile.workspace_root == tmp_path
    assert profile.active_tool_names == (
        "read_file",
        "search_text",
        "list_files",
        "apply_patch",
        "run_tests",
        "git_status",
        "ask_confirmation",
    )
    assert profile.disabled_tools == ("shell",)
    assert "code assistant" in profile.prompt_layers.system.lower()
    assert profile.provider_extra["thinking"]["enabled"] is True


def test_code_assistant_profile_supports_phase2_ablations(tmp_path: Path) -> None:
    profile = build_code_assistant_profile(
        workspace_root=tmp_path,
        ablations={"tool_narrowing", "prompt_layers", "reasoning_mode"},
    )

    assert "shell" in profile.active_tool_names
    assert profile.disabled_tools == ()
    assert profile.prompt_layers.scenario == ""
    assert profile.provider_extra == {}
