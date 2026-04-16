from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from code_agent_harness.tools.apply_patch_tool import create_apply_patch_handler
from code_agent_harness.tools.builtins import load_builtin_tools
from code_agent_harness.tools.git_status import create_git_status_handler
from code_agent_harness.tools.read_file import create_read_file_handler
from code_agent_harness.tools.run_tests import create_run_tests_handler


def test_read_file_supports_line_ranges(tmp_path: Path) -> None:
    file_path = tmp_path / "example.txt"
    file_path.write_text("line1\nline2\nline3\nline4\n", encoding="utf-8")

    handler = create_read_file_handler(tmp_path)

    result = handler({"path": "example.txt", "start_line": 2, "end_line": 3})

    assert result == "line2\nline3\n"


def test_read_file_rejects_path_escape(tmp_path: Path) -> None:
    outside_path = tmp_path.parent / "outside.txt"
    outside_path.write_text("secret\n", encoding="utf-8")

    handler = create_read_file_handler(tmp_path)

    with pytest.raises(ValueError, match="path escapes workspace root"):
        handler({"path": "../outside.txt"})


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


def test_apply_patch_rejects_ambiguous_single_replacement(tmp_path: Path) -> None:
    file_path = tmp_path / "app.py"
    original = "print('before')\nprint('before')\n"
    file_path.write_text(original, encoding="utf-8")

    handler = create_apply_patch_handler(tmp_path)

    with pytest.raises(ValueError, match="ambiguous single replacement"):
        handler(
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

    assert file_path.read_text(encoding="utf-8") == original


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
    assert calls[0]["args"] == (
        [sys.executable, "-m", "pytest", "tests/test_code_assistant_tools.py", "-q"],
    )
    assert calls[0]["kwargs"]["shell"] is False
    assert calls[0]["kwargs"]["cwd"] == tmp_path


def test_git_status_reports_failure_when_git_returns_no_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=args[0], returncode=128, stdout="", stderr="")

    monkeypatch.setattr("code_agent_harness.tools.git_status.subprocess.run", fake_run)

    handler = create_git_status_handler(tmp_path)
    result = handler({})

    assert "git status failed" in result
    assert "128" in result


def test_strengthened_tool_schema_lists_required_fields(tmp_path: Path) -> None:
    tools = {tool.definition.name: tool.definition for tool in load_builtin_tools(tmp_path)}

    assert tools["read_file"].input_schema["required"] == ["path"]
    assert tools["apply_patch"].input_schema["required"] == ["path", "replacements"]
    assert tools["run_tests"].input_schema["required"] == ["args"]
    assert tools["ask_confirmation"].input_schema["required"] == ["message"]
