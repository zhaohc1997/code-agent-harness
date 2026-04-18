from __future__ import annotations

from dataclasses import dataclass

from code_agent_harness.evals.scoring import EvalScore
from code_agent_harness.evals.trace import EvalTrace, TraceToolCall


FAILURE_BUCKETS = {
    "tool_choice": "tool_selection_failure",
    "tool_arguments": "tool_argument_failure",
    "repository_state": "repo_state_failure",
    "tests": "test_failure",
    "response_content": "response_failure",
    "workflow": "workflow_failure",
}


@dataclass(frozen=True)
class EvalCostMetrics:
    values: dict[str, float]


def compute_cost_metrics(trace: EvalTrace) -> EvalCostMetrics:
    test_calls = tuple(call for call in trace.tool_calls if call.tool_name == "run_tests")
    targeted_test_calls = tuple(call for call in test_calls if _is_targeted_test_call(call))
    test_invocation_count = float(len(test_calls))
    targeted_test_ratio = (
        len(targeted_test_calls) / test_invocation_count if test_invocation_count else 0.0
    )
    return EvalCostMetrics(
        values={
            "tool_call_count": float(len(trace.tool_calls)),
            "successful_tool_call_count": float(
                sum(
                    1
                    for call in trace.tool_calls
                    if call.has_result and call.result_status == "ok"
                )
            ),
            "assistant_turn_count": float(trace.assistant_turn_count),
            "test_invocation_count": test_invocation_count,
            "patch_invocation_count": float(
                sum(1 for call in trace.tool_calls if call.tool_name == "apply_patch")
            ),
            "confirmation_count": float(
                sum(
                    1
                    for call in trace.tool_calls
                    if call.tool_name == "ask_confirmation"
                    or call.result_status == "require_confirmation"
                )
            ),
            "blocked_call_count": float(
                sum(1 for call in trace.tool_calls if call.result_status in {"block", "blocked"})
            ),
            "reminder_count": float(
                sum(1 for call in trace.tool_calls if call.result_status == "remind")
            ),
            "targeted_test_ratio": targeted_test_ratio,
        }
    )


def attribute_failures(score: EvalScore) -> tuple[str, ...]:
    return tuple(
        bucket
        for dimension, bucket in FAILURE_BUCKETS.items()
        if score.dimensions.get(dimension, 1.0) != 1.0
    )


def average_cost_metrics(metrics: tuple[EvalCostMetrics, ...]) -> dict[str, float]:
    if not metrics:
        return {}
    names = sorted({name for metric in metrics for name in metric.values})
    return {
        name: sum(metric.values.get(name, 0.0) for metric in metrics) / len(metrics)
        for name in names
    }


def compare_cost_metrics(
    baseline: dict[str, float],
    ablation: dict[str, float],
) -> dict[str, float]:
    names = sorted(set(baseline) | set(ablation))
    return {name: ablation.get(name, 0.0) - baseline.get(name, 0.0) for name in names}


def recommend_mechanism(
    *,
    baseline_pass_rate: float,
    ablation_pass_rate: float,
    dimension_deltas: dict[str, float],
    cost_deltas: dict[str, float],
) -> str:
    if ablation_pass_rate < baseline_pass_rate:
        return "keep"
    if ablation_pass_rate > baseline_pass_rate:
        return "disable_or_rework"
    if any(value < 0.0 for value in dimension_deltas.values()):
        return "keep"
    if any(value > 0.0 for value in dimension_deltas.values()):
        return "disable_or_rework"
    if _total_cost_delta(cost_deltas) < 0.0:
        return "disable_or_rework"
    return "neutral"


def _is_targeted_test_call(call: TraceToolCall) -> bool:
    args = call.arguments.get("args")
    if not isinstance(args, list) or not args:
        return False
    return any(isinstance(arg, str) and not arg.startswith("-") for arg in args)


def _total_cost_delta(cost_deltas: dict[str, float]) -> float:
    return sum(
        value
        for name, value in cost_deltas.items()
        if name not in {"targeted_test_ratio"}
    )
