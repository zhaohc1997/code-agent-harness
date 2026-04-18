from pathlib import Path

import code_agent_harness.llm as llm

from code_agent_harness.evals.runner import (
    EvalRunResult,
    compare_suite_results,
    run_eval_suite,
    run_eval_task,
)
from code_agent_harness.evals.scoring import EvalScore
from code_agent_harness.evals.tasks import load_default_tasks
from code_agent_harness.evals.trace import EvalTrace, TraceToolCall
from code_agent_harness.types.state import SessionState


def test_eval_runner_copies_fixture_and_exposes_trace(tmp_path: Path) -> None:
    task = next(task for task in load_default_tasks() if task.task_id == "bugfix-basic")
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
                                {"old_text": "return a - b", "new_text": "return a + b"},
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

    assert result.task_id == "bugfix-basic"
    assert result.trace.tool_calls[-1].tool_name == "run_tests"
    assert result.trace.tool_calls[-1].result_status == "ok"
    assert result.score.passed is True
    assert result.workspace_root != Path("tests/evals/fixtures/bugfix_repo")


def test_run_eval_suite_aggregates_dimension_averages(tmp_path: Path) -> None:
    tasks = tuple(
        task for task in load_default_tasks() if task.task_id in {"bugfix-basic", "analysis-timeout"}
    )

    def provider_factory(task):
        scripts = {
            "bugfix-basic": [
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
            ],
            "analysis-timeout": [
                {
                    "stop_reason": "tool_use",
                    "content": [
                        {
                            "type": "tool_call",
                            "id": "tool-1",
                            "name": "read_file",
                            "arguments": {"path": "service.py"},
                        }
                    ],
                },
                {
                    "stop_reason": "end_turn",
                    "content": [
                        {
                            "type": "text",
                            "text": "Default timeout is 45. Enabled features are search, history, and export.",
                        }
                    ],
                },
            ],
        }
        return llm.FakeProvider(script=scripts[task.task_id])

    suite = run_eval_suite(
        suite_name="default",
        tasks=tasks,
        provider_factory=provider_factory,
        fixtures_root=Path("tests/evals/fixtures"),
        tmp_root=tmp_path,
    )

    assert suite.passed_tasks == 2
    assert suite.total_tasks == 2
    assert suite.dimension_averages["workflow"] == 1.0


def test_compare_suite_results_reports_changed_tasks(tmp_path: Path) -> None:
    trace = EvalTrace(
        tool_calls=(
            TraceToolCall(
                index=0,
                tool_name="read_file",
                arguments={"path": "calc.py"},
                tool_use_id="tool-1",
                has_result=True,
                result_status="ok",
            ),
        ),
        final_output="done",
        final_state=SessionState.COMPLETED,
        workspace_root=tmp_path,
    )
    baseline = (
        EvalRunResult(
            task_id="bugfix-basic",
            workspace_root=tmp_path,
            trace=trace,
            score=EvalScore(
                passed=True,
                dimensions={
                    "tool_choice": 1.0,
                    "tool_arguments": 1.0,
                    "repository_state": 1.0,
                    "tests": 1.0,
                    "response_content": 1.0,
                    "workflow": 1.0,
                },
                evidence={
                    name: ""
                    for name in (
                        "tool_choice",
                        "tool_arguments",
                        "repository_state",
                        "tests",
                        "response_content",
                        "workflow",
                    )
                },
            ),
        ),
    )
    ablation = (
        EvalRunResult(
            task_id="bugfix-basic",
            workspace_root=tmp_path,
            trace=trace,
            score=EvalScore(
                passed=False,
                dimensions={
                    "tool_choice": 1.0,
                    "tool_arguments": 0.0,
                    "repository_state": 1.0,
                    "tests": 1.0,
                    "response_content": 1.0,
                    "workflow": 0.0,
                },
                evidence={
                    "tool_choice": "",
                    "tool_arguments": "bad args",
                    "repository_state": "",
                    "tests": "",
                    "response_content": "",
                    "workflow": "bad workflow",
                },
            ),
        ),
    )

    comparison = compare_suite_results("default", "policy_engine", baseline, ablation)

    assert comparison.ablation_name == "policy_engine"
    assert comparison.changed_tasks == ("bugfix-basic",)
    assert comparison.delta_by_dimension["tool_arguments"] == -1.0
