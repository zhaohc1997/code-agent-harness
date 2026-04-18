from pathlib import Path

from code_agent_harness.evals.diagnostics import (
    attribute_failures,
    compare_cost_metrics,
    compute_cost_metrics,
    recommend_mechanism,
)
from code_agent_harness.evals.scoring import EvalScore
from code_agent_harness.evals.trace import EvalTrace, TraceToolCall
from code_agent_harness.types.state import SessionState


def test_compute_cost_metrics_counts_trace_events(tmp_path: Path) -> None:
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
            TraceToolCall(
                index=1,
                tool_name="run_tests",
                arguments={"args": ["-q"]},
                tool_use_id="tool-2",
                has_result=True,
                result_status="remind",
            ),
            TraceToolCall(
                index=2,
                tool_name="run_tests",
                arguments={"args": ["-q", "tests/test_calc.py"]},
                tool_use_id="tool-3",
                has_result=True,
                result_status="ok",
            ),
            TraceToolCall(
                index=3,
                tool_name="apply_patch",
                arguments={"path": "calc.py"},
                tool_use_id="tool-4",
                has_result=True,
                result_status="require_confirmation",
            ),
        ),
        final_output="done",
        final_state=SessionState.COMPLETED,
        workspace_root=tmp_path,
        assistant_turn_count=4,
    )

    metrics = compute_cost_metrics(trace)

    assert metrics.values["tool_call_count"] == 4.0
    assert metrics.values["successful_tool_call_count"] == 2.0
    assert metrics.values["assistant_turn_count"] == 4.0
    assert metrics.values["test_invocation_count"] == 2.0
    assert metrics.values["patch_invocation_count"] == 1.0
    assert metrics.values["confirmation_count"] == 1.0
    assert metrics.values["reminder_count"] == 1.0
    assert metrics.values["targeted_test_ratio"] == 0.5


def test_attribute_failures_maps_failed_dimensions() -> None:
    score = EvalScore(
        passed=False,
        dimensions={
            "tool_choice": 0.0,
            "tool_arguments": 1.0,
            "repository_state": 0.0,
            "tests": 1.0,
            "response_content": 0.0,
            "workflow": 0.0,
        },
    )

    assert attribute_failures(score) == (
        "tool_selection_failure",
        "repo_state_failure",
        "response_failure",
        "workflow_failure",
    )


def test_compare_cost_metrics_and_recommendation() -> None:
    baseline = {"tool_call_count": 4.0, "assistant_turn_count": 4.0}
    cheaper_ablation = {"tool_call_count": 2.0, "assistant_turn_count": 3.0}

    assert compare_cost_metrics(baseline, cheaper_ablation) == {
        "assistant_turn_count": -1.0,
        "tool_call_count": -2.0,
    }
    assert (
        recommend_mechanism(
            baseline_pass_rate=1.0,
            ablation_pass_rate=1.0,
            dimension_deltas={"workflow": 0.0},
            cost_deltas={"tool_call_count": -2.0},
        )
        == "disable_or_rework"
    )
