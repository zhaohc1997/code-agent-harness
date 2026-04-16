from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from code_agent_harness.tools.apply_patch_tool import create_apply_patch_handler
from code_agent_harness.tools.builtins import load_builtin_tools
from code_agent_harness.tools.read_file import create_read_file_handler
from code_agent_harness.tools.run_tests import create_run_tests_handler


def test_read_file_supports_line_ranges(tmp_path: Path) -> None:
    file_path = tmp_path / "example.txt"
    file_path.write_text("line1\nline2\nline3\nline4\n", encoding="utf-8")

    handler = create_read_file_handler(tmp_path)

    result = handler({"path": "example.txt", "start_line": 2, "end_line": 3})

    assert result == "line2\nline3\n"


def test_apply_patch_replaces_expected_text(tmp_path: Path) -> None:
    file_path = tmp_path / "app.py"
    file_path.write_text("print('before')\n", encoding="utf-8")

    handler = create_apply_patch_handler(tmp_path)

    result = handler(
        {
            "path": "app.py",
            "replacements": [
                {
                    "old_text": "before",
                    "new_text": "after",
                }
            ],
        }
    )

    assert "applied 1 replacement" in result
    assert file_path.read_text(encoding="utf-8") == "print('after')\n"


def test_run_tests_uses_argument_arrays(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append({"args": args, "kwargs": kwargs})
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout="passed\n",
            stderr="",
        )

    monkeypatch.setattr("code_agent_harness.tools.run_tests.subprocess.run", fake_run)

    handler = create_run_tests_handler(tmp_path)
    result = handler({"args": ["tests/test_code_assistant_tools.py", "-q"]})

    assert "passed" in result
    assert calls
    assert calls[0]["args"] == (["pytest", "tests/test_code_assistant_tools.py", "-q"],)
    assert calls[0]["kwargs"]["shell"] is False
    assert calls[0]["kwargs"]["cwd"] == tmp_path


def test_strengthened_tool_schema_lists_required_fields(tmp_path: Path) -> None:
    tools = {tool.definition.name: tool.definition for tool in load_builtin_tools(tmp_path)}

    assert tools["read_file"].input_schema["required"] == ["path"]
    assert tools["apply_patch"].input_schema["required"] == ["path", "replacements"]
    assert tools["run_tests"].input_schema["required"] == ["args"]
    assert tools["ask_confirmation"].input_schema["required"] == ["message"]
