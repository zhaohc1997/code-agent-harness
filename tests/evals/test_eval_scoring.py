from pathlib import Path

from code_agent_harness.evals.scoring import score_eval_run
from code_agent_harness.evals.scoring import score_eval_task
from code_agent_harness.evals.tasks import (
    ArgumentExpectation,
    EvalTask,
    OutcomeExpectation,
    ToolExpectation,
    WorkflowExpectation,
)
from code_agent_harness.evals.trace import EvalTrace, TraceToolCall
from code_agent_harness.types.state import SessionState


def test_score_eval_run_keeps_dimension_failures_visible() -> None:
    score = score_eval_run(
        tool_choice_ok=True,
        tool_arguments_ok=True,
        repository_state_ok=True,
        tests_ok=True,
        response_content_ok=False,
        workflow_ok=True,
    )

    assert score.passed is False
    assert score.dimensions["repository_state"] == 1.0
    assert score.dimensions["response_content"] == 0.0


def test_score_eval_task_exposes_dimension_evidence(tmp_path: Path) -> None:
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    task = EvalTask(
        task_id="bugfix-basic",
        task_class="bugfix",
        fixture_name="bugfix_repo",
        user_input="Fix it",
        tool_expectations=(
            ToolExpectation(
                name="read_file",
                argument_expectations=(
                    ArgumentExpectation(field_path="path", match_mode="exact", expected="calc.py"),
                ),
            ),
            ToolExpectation(
                name="apply_patch",
                must_appear_after=("read_file",),
                argument_expectations=(
                    ArgumentExpectation(field_path="path", match_mode="exact", expected="calc.py"),
                ),
            ),
            ToolExpectation(
                name="run_tests",
                must_appear_after=("apply_patch",),
                argument_expectations=(
                    ArgumentExpectation(
                        field_path="args",
                        match_mode="contains",
                        expected="tests/test_calc.py",
                    ),
                ),
            ),
        ),
        workflow_expectations=WorkflowExpectation(
            must_read_before_patch=True,
            must_run_tests_before_finish=True,
        ),
        outcome_expectations=OutcomeExpectation(
            repo_assertions=(("calc.py", "return a + b"),),
            required_test_args_fragments=("tests/test_calc.py",),
            required_response_substrings=("fixed", "passed"),
        ),
        live_eligible=True,
    )
    trace = EvalTrace(
        tool_calls=(
            TraceToolCall(
                index=0,
                tool_name="apply_patch",
                arguments={"path": "calc.py"},
                tool_use_id="tool-1",
                has_result=True,
                result_status="ok",
            ),
            TraceToolCall(
                index=1,
                tool_name="run_tests",
                arguments={"args": ["-q", "tests/test_other.py"]},
                tool_use_id="tool-2",
                has_result=True,
                result_status="ok",
            ),
        ),
        final_output="Fixed the bug.",
        final_state=SessionState.COMPLETED,
        workspace_root=tmp_path,
    )

    score = score_eval_task(task, trace)

    assert score.passed is False
    assert score.dimensions["tool_choice"] == 0.0
    assert score.dimensions["tool_arguments"] == 0.0
    assert score.dimensions["workflow"] == 0.0
    assert score.evidence["tool_choice"] == "missing required tool read_file"
    assert "tests/test_calc.py" in score.evidence["tool_arguments"]
    assert "before patching" in score.evidence["workflow"]
