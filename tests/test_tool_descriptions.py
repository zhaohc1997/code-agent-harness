from pathlib import Path

from code_agent_harness.tools.builtins import load_builtin_tools


def _tool_descriptions(tmp_path: Path) -> dict[str, str]:
    return {
        tool.definition.name: tool.definition.description or ""
        for tool in load_builtin_tools(tmp_path)
    }


def test_phase3_tool_descriptions_are_field_level_contracts(tmp_path: Path) -> None:
    descriptions = _tool_descriptions(tmp_path)

    assert "workspace-relative" in descriptions["read_file"]
    assert "path: string" in descriptions["read_file"]
    assert "start_line: integer" in descriptions["read_file"]
    assert "end_line: integer" in descriptions["read_file"]
    assert "exact text fragment" in descriptions["search_text"]
    assert "not a regex" in descriptions["search_text"]
    assert "replacements: array" in descriptions["apply_patch"]
    assert "old_text" in descriptions["apply_patch"]
    assert "new_text" in descriptions["apply_patch"]
    assert "args: array[string]" in descriptions["run_tests"]
    assert "Shell command strings are invalid" in descriptions["run_tests"]
    assert "message: string" in descriptions["ask_confirmation"]
