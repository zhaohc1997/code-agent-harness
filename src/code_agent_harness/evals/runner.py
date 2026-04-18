from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import shutil

from code_agent_harness.engine.cancellation import CancellationToken
from code_agent_harness.engine.loop import AgentRuntime
from code_agent_harness.engine.state_machine import EngineStateMachine
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


@dataclass(frozen=True)
class EvalSuiteResult:
    suite_name: str
    results: tuple[EvalRunResult, ...]
    passed_tasks: int
    total_tasks: int
    dimension_averages: dict[str, float]


@dataclass(frozen=True)
class EvalComparisonResult:
    suite_name: str
    ablation_name: str
    baseline: tuple[EvalRunResult, ...]
    ablation: tuple[EvalRunResult, ...]
    delta_by_dimension: dict[str, float]
    changed_tasks: tuple[str, ...]


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
    return EvalRunResult(
        task_id=task.task_id,
        workspace_root=workspace_root,
        trace=trace,
        score=score,
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
    return EvalSuiteResult(
        suite_name=suite_name,
        results=results,
        passed_tasks=passed_tasks,
        total_tasks=total_tasks,
        dimension_averages=dimension_averages,
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
        if baseline_by_task.get(task_id) != ablation_by_task.get(task_id)
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

    return EvalComparisonResult(
        suite_name=suite_name,
        ablation_name=ablation_name,
        baseline=baseline,
        ablation=ablation,
        delta_by_dimension=delta_by_dimension,
        changed_tasks=changed_tasks,
    )
