from pathlib import Path

from code_agent_harness.profiles.code_assistant import build_code_assistant_profile


def test_code_assistant_profile_narrows_tools_and_enables_reasoning(tmp_path: Path) -> None:
    profile = build_code_assistant_profile(workspace_root=tmp_path)

    assert profile.name == "code_assistant"
    assert profile.workspace_root == tmp_path
    assert "read_file" in profile.active_tool_names
    assert "run_tests" in profile.active_tool_names
    assert "search_text" in profile.active_tool_names
    assert "list_files" in profile.active_tool_names
    assert "apply_patch" in profile.active_tool_names
    assert "git_status" in profile.active_tool_names
    assert "ask_confirmation" in profile.active_tool_names
    assert "shell" in profile.disabled_tools
    assert "code assistant" in profile.prompt_layers.system.lower()
    assert profile.provider_extra["thinking"]["enabled"] is True
