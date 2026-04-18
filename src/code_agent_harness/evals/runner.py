from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
import shutil

from code_agent_harness.engine.cancellation import CancellationToken
from code_agent_harness.engine.loop import AgentRuntime
from code_agent_harness.engine.state_machine import EngineStateMachine
from code_agent_harness.evals.diagnostics import (
    attribute_failures,
    average_cost_metrics,
    compare_cost_metrics,
    compute_cost_metrics,
    recommend_mechanism,
)
from code_agent_harness.evals.scoring import EvalScore, score_eval_task
from code_agent_harness.evals.tasks import EvalTask
from code_agent_harness.evals.trace import EvalTrace, extract_eval_trace
from code_agent_harness.policies.code_assistant import build_code_assistant_policy
from code_agent_harness.profiles.code_assistant import build_code_assistant_profile
from code_agent_harness.prompts.builders import build_system_prompt
from code_agent_harness.storage.checkpoints import CheckpointStore
from code_agent_harness.storage.sessions import SessionStore
from code_agent_harness.tools.builtins import load_builtin_tools
from code_agent_harness.tools.executor import ToolExecutor
from code_agent_harness.tools.registry import ToolRegistry
from code_agent_harness.types.state import SessionState


@dataclass(frozen=True)
class EvalRunResult:
    task_id: str
    workspace_root: Path
    trace: EvalTrace
    score: EvalScore
    cost_metrics: dict[str, float] = field(default_factory=dict)
    failure_attributions: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvalSuiteResult:
    suite_name: str
    results: tuple[EvalRunResult, ...]
    passed_tasks: int
    total_tasks: int
    dimension_averages: dict[str, float]
    cost_averages: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalComparisonResult:
    suite_name: str
    ablation_name: str
    baseline: tuple[EvalRunResult, ...]
    ablation: tuple[EvalRunResult, ...]
    delta_by_dimension: dict[str, float]
    changed_tasks: tuple[str, ...]
    delta_by_cost: dict[str, float] = field(default_factory=dict)
    recommendation: str = "neutral"


def _pass_fail_outcome(result: EvalRunResult | None) -> bool | None:
    if result is None:
        return None
    return result.score.passed


def _build_runtime(
    workspace_root: Path,
    runtime_root: Path,
    *,
    provider: object,
    ablations: set[str] | None,
) -> AgentRuntime:
    profile = build_code_assistant_profile(workspace_root=workspace_root, ablations=ablations)
    registry = ToolRegistry(
        lambda: [
            tool
            for tool in load_builtin_tools(profile.workspace_root)
            if tool.definition.name in profile.active_tool_names
        ]
    )
    return AgentRuntime(
        provider=provider,
        system_prompt=build_system_prompt(profile.prompt_layers),
        sessions=SessionStore(runtime_root / "sessions"),
        checkpoints=CheckpointStore(runtime_root / "checkpoints"),
        registry=registry,
        executor=ToolExecutor(registry=registry, blob_store_root=runtime_root),
        cancellation=CancellationToken(signal_root=runtime_root / "cancellations"),
        state_machine=EngineStateMachine(SessionState.IDLE),
        context_window_tokens=128_000,
        policy_engine=(
            None
            if ablations is not None and "policy_engine" in ablations
            else build_code_assistant_policy(disabled_tools=set(profile.disabled_tools))
        ),
        provider_extra=profile.provider_extra,
    )


def run_eval_task(
    task: EvalTask,
    *,
    provider: object,
    fixtures_root: Path,
    tmp_root: Path,
    ablations: set[str] | None = None,
) -> EvalRunResult:
    source_root = fixtures_root / task.fixture_name
    workspace_root = tmp_root / task.task_id
    runtime_root = tmp_root / ".agenth" / task.task_id

    if workspace_root.exists():
        shutil.rmtree(workspace_root)
    if runtime_root.exists():
        shutil.rmtree(runtime_root)
    shutil.copytree(source_root, workspace_root)

    runtime = _build_runtime(
        workspace_root,
        runtime_root,
        provider=provider,
        ablations=ablations,
    )
    runtime_result = runtime.run(session_id=task.task_id, user_input=task.user_input)
    trace = extract_eval_trace(runtime_result, workspace_root=workspace_root)
    score = score_eval_task(task, trace)
    cost_metrics = compute_cost_metrics(trace)
    failure_attributions = attribute_failures(score)
    return EvalRunResult(
        task_id=task.task_id,
        workspace_root=workspace_root,
        trace=trace,
        score=score,
        cost_metrics=cost_metrics.values,
        failure_attributions=failure_attributions,
    )


def run_eval_suite(
    *,
    suite_name: str,
    tasks: tuple[EvalTask, ...],
    provider_factory: Callable[[EvalTask], object],
    fixtures_root: Path,
    tmp_root: Path,
    ablations: set[str] | None = None,
) -> EvalSuiteResult:
    results = tuple(
        run_eval_task(
            task,
            provider=provider_factory(task),
            fixtures_root=fixtures_root,
            tmp_root=tmp_root,
            ablations=ablations,
        )
        for task in tasks
    )
    total_tasks = len(results)
    passed_tasks = sum(1 for result in results if result.score.passed)
    dimension_totals: dict[str, float] = defaultdict(float)
    for result in results:
        for name, value in result.score.dimensions.items():
            dimension_totals[name] += value
    dimension_averages = {
        name: total / total_tasks for name, total in sorted(dimension_totals.items())
    } if total_tasks else {}
    cost_averages = average_cost_metrics(
        tuple(compute_cost_metrics(result.trace) for result in results)
    )
    return EvalSuiteResult(
        suite_name=suite_name,
        results=results,
        passed_tasks=passed_tasks,
        total_tasks=total_tasks,
        dimension_averages=dimension_averages,
        cost_averages=cost_averages,
    )


def compare_suite_results(
    suite_name: str,
    ablation_name: str,
    baseline: tuple[EvalRunResult, ...],
    ablation: tuple[EvalRunResult, ...],
) -> EvalComparisonResult:
    baseline_by_task = {result.task_id: result for result in baseline}
    ablation_by_task = {result.task_id: result for result in ablation}
    task_ids = tuple(sorted(set(baseline_by_task) | set(ablation_by_task)))

    changed_tasks = tuple(
        task_id
        for task_id in task_ids
        if _pass_fail_outcome(baseline_by_task.get(task_id))
        != _pass_fail_outcome(ablation_by_task.get(task_id))
    )

    all_dimensions = sorted(
        {
            dimension
            for result in baseline + ablation
            for dimension in result.score.dimensions
        }
    )
    delta_by_dimension = {
        dimension: (
            (
                sum(
                    result.score.dimensions.get(dimension, 0.0)
                    for result in ablation
                ) / len(ablation)
            )
            - (
                sum(
                    result.score.dimensions.get(dimension, 0.0)
                    for result in baseline
                ) / len(baseline)
            )
        )
        for dimension in all_dimensions
        if baseline and ablation
    }
    baseline_cost_averages = _average_run_costs(baseline)
    ablation_cost_averages = _average_run_costs(ablation)
    delta_by_cost = compare_cost_metrics(baseline_cost_averages, ablation_cost_averages)
    baseline_pass_rate = (
        sum(1 for result in baseline if result.score.passed) / len(baseline)
        if baseline
        else 0.0
    )
    ablation_pass_rate = (
        sum(1 for result in ablation if result.score.passed) / len(ablation)
        if ablation
        else 0.0
    )

    return EvalComparisonResult(
        suite_name=suite_name,
        ablation_name=ablation_name,
        baseline=baseline,
        ablation=ablation,
        delta_by_dimension=delta_by_dimension,
        changed_tasks=changed_tasks,
        delta_by_cost=delta_by_cost,
        recommendation=recommend_mechanism(
            baseline_pass_rate=baseline_pass_rate,
            ablation_pass_rate=ablation_pass_rate,
            dimension_deltas=delta_by_dimension,
            cost_deltas=delta_by_cost,
        ),
    )


def _average_run_costs(results: tuple[EvalRunResult, ...]) -> dict[str, float]:
    if not results:
        return {}
    names = sorted({name for result in results for name in result.cost_metrics})
    return {
        name: sum(result.cost_metrics.get(name, 0.0) for result in results) / len(results)
        for name in names
    }
