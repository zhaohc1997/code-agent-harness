from pathlib import Path

import code_agent_harness.llm as llm

from code_agent_harness.evals.runner import run_eval_task
from code_agent_harness.evals.tasks import EvalTask


def test_eval_runner_copies_fixture_into_isolated_workspace(tmp_path: Path) -> None:
    task = EvalTask(
        task_id="bugfix-basic",
        task_class="bugfix",
        fixture_name="bugfix_repo",
        user_input="Fix the add() function and run the smallest relevant test.",
        expected_tool_names=("read_file", "apply_patch", "run_tests"),
        required_response_substrings=("fixed", "passed"),
        repo_assertions=(("calc.py", "return a + b"),),
        live_eligible=True,
    )

    provider = llm.FakeProvider(
        script=[
            {
                "stop_reason": "tool_use",
                "content": [
                    {
                        "type": "tool_call",
                        "id": "tool-1",
                        "name": "read_file",
                        "arguments": {"path": "calc.py"},
                    }
                ],
            },
            {
                "stop_reason": "tool_use",
                "content": [
                    {
                        "type": "tool_call",
                        "id": "tool-2",
                        "name": "apply_patch",
                        "arguments": {
                            "path": "calc.py",
                            "replacements": [
                                {"old_text": "return a - b", "new_text": "return a + b"}
                            ],
                        },
                    }
                ],
            },
            {
                "stop_reason": "tool_use",
                "content": [
                    {
                        "type": "tool_call",
                        "id": "tool-3",
                        "name": "run_tests",
                        "arguments": {"args": ["-q", "tests/test_calc.py"]},
                    }
                ],
            },
            {
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "Fixed the bug and the targeted test passed."}],
            },
        ]
    )

    result = run_eval_task(
        task,
        provider=provider,
        fixtures_root=Path("tests/evals/fixtures"),
        tmp_root=tmp_path,
    )

    assert result.score.passed is True
    assert result.tool_names == ("read_file", "apply_patch", "run_tests")
    assert result.workspace_root != Path("tests/evals/fixtures/bugfix_repo")
    assert "return a - b" in (Path("tests/evals/fixtures/bugfix_repo") / "calc.py").read_text(
        encoding="utf-8"
    )
