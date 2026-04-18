# Code Assistant Phase 3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deepen the existing `code_assistant` scenario with a cost-directed optimization loop: expanded eval coverage, trace-derived cost metrics, failure attribution, stronger tool descriptions, richer policy reminders, and CLI comparison recommendations.

**Architecture:** Keep phase 3 logic outside the generic engine loop. Expand `evals` to compute diagnostics from existing runtime traces, strengthen `tools`, `profiles`, and `policies` as low-risk online improvements, and extend CLI output so phase 3 optimization decisions can be made from terminal reports.

**Tech Stack:** Python 3.11+, `pytest`, `dataclasses`, `pathlib`, existing `FakeProvider`, existing `AgentRuntime`, existing code-assistant profile/policy/tool/eval layers

---

## File Structure

- Modify: `README.md`
- Modify: `src/code_agent_harness/cli.py`
- Modify: `src/code_agent_harness/evals/__init__.py`
- Create: `src/code_agent_harness/evals/diagnostics.py`
- Modify: `src/code_agent_harness/evals/runner.py`
- Modify: `src/code_agent_harness/evals/tasks.py`
- Modify: `src/code_agent_harness/evals/trace.py`
- Modify: `src/code_agent_harness/policies/code_assistant.py`
- Modify: `src/code_agent_harness/profiles/code_assistant.py`
- Modify: `src/code_agent_harness/tools/builtins.py`
- Modify: `tests/evals/test_eval_ablations.py`
- Create: `tests/evals/test_eval_diagnostics.py`
- Modify: `tests/evals/test_eval_runner.py`
- Modify: `tests/evals/test_eval_tasks.py`
- Modify: `tests/test_cli_phase2.py`
- Modify: `tests/test_code_assistant_profile.py`
- Modify: `tests/test_policy_engine.py`
- Modify: `tests/test_tool_descriptions.py`

### Task 1: Expand The Default Code-Assistant Eval Suite

**Files:**
- Modify: `src/code_agent_harness/evals/tasks.py`
- Modify: `src/code_agent_harness/cli.py`
- Modify: `tests/evals/test_eval_tasks.py`

- [ ] **Step 1: Add failing tests for the phase 3 task set**

Append these tests to `tests/evals/test_eval_tasks.py`:

```python
def test_load_default_tasks_includes_phase3_boundary_cases() -> None:
    tasks = {task.task_id: task for task in load_default_tasks()}

    assert len(tasks) >= 5
    assert {"bugfix-basic", "feature-title-case", "analysis-timeout"}.issubset(tasks)
    assert "bugfix-targeted-report" in tasks
    assert "analysis-readme-no-write" in tasks

    task_classes = [task.task_class for task in tasks.values()]
    assert task_classes.count("bugfix") >= 2
    assert "feature" in task_classes
    assert "analysis" in task_classes

    targeted = tasks["bugfix-targeted-report"]
    assert targeted.outcome_expectations.required_test_args_fragments == ("tests/test_calc.py",)
    assert "tests/test_calc.py" in targeted.outcome_expectations.required_response_substrings

    readonly = tasks["analysis-readme-no-write"]
    assert readonly.workflow_expectations.forbid_patch is True
    assert readonly.workflow_expectations.forbid_test_runs is True
    assert readonly.outcome_expectations.required_response_substrings == (
        "Analysis Repo",
        "without modifying files",
    )
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
pytest tests/evals/test_eval_tasks.py::test_load_default_tasks_includes_phase3_boundary_cases -q
```

Expected: FAIL because `bugfix-targeted-report` and `analysis-readme-no-write` do not exist yet.

- [ ] **Step 3: Add the two default phase 3 tasks**

Modify `load_default_tasks()` in `src/code_agent_harness/evals/tasks.py` so the returned tuple includes these two tasks after the existing three tasks:

```python
        EvalTask(
            task_id="bugfix-targeted-report",
            task_class="bugfix",
            fixture_name="bugfix_repo",
            user_input=(
                "Fix add(), run the targeted calc test, and explicitly report the test file you ran."
            ),
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
                required_response_substrings=("fixed", "passed", "tests/test_calc.py"),
            ),
            live_eligible=True,
        ),
        EvalTask(
            task_id="analysis-readme-no-write",
            task_class="analysis",
            fixture_name="analysis_repo",
            user_input=(
                "Summarize the repository purpose from README.md. Do not modify files or run tests."
            ),
            tool_expectations=(
                ToolExpectation(
                    name="read_file",
                    argument_expectations=(
                        ArgumentExpectation(field_path="path", match_mode="exact", expected="README.md"),
                    ),
                ),
            ),
            workflow_expectations=WorkflowExpectation(
                forbid_patch=True,
                forbid_test_runs=True,
            ),
            outcome_expectations=OutcomeExpectation(
                required_response_substrings=("Analysis Repo", "without modifying files"),
            ),
            live_eligible=True,
        ),
```

- [ ] **Step 4: Add scripted providers for the new tasks**

In `src/code_agent_harness/cli.py`, extend `_build_scripted_eval_provider()` with two entries:

```python
        "bugfix-targeted-report": [
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
                "content": [
                    {
                        "type": "text",
                        "text": "Fixed add() and passed the targeted test tests/test_calc.py.",
                    }
                ],
            },
        ],
        "analysis-readme-no-write": [
            {
                "stop_reason": "tool_use",
                "content": [
                    {
                        "type": "tool_call",
                        "id": "tool-1",
                        "name": "read_file",
                        "arguments": {"path": "README.md"},
                    }
                ],
            },
            {
                "stop_reason": "end_turn",
                "content": [
                    {
                        "type": "text",
                        "text": "Analysis Repo answers repository questions without modifying files.",
                    }
                ],
            },
        ],
```

- [ ] **Step 5: Run task and suite tests**

Run:

```bash
pytest tests/evals/test_eval_tasks.py tests/evals/test_eval_runner.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add src/code_agent_harness/evals/tasks.py src/code_agent_harness/cli.py tests/evals/test_eval_tasks.py
git commit -m "feat: expand code assistant eval tasks"
```

### Task 2: Add Trace-Derived Diagnostics

**Files:**
- Modify: `src/code_agent_harness/evals/trace.py`
- Create: `src/code_agent_harness/evals/diagnostics.py`
- Modify: `src/code_agent_harness/evals/__init__.py`
- Create: `tests/evals/test_eval_diagnostics.py`

- [ ] **Step 1: Write failing diagnostics tests**

Create `tests/evals/test_eval_diagnostics.py`:

```python
from pathlib import Path

from code_agent_harness.evals.diagnostics import (
    attribute_failures,
    compute_cost_metrics,
    compare_cost_metrics,
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
    assert recommend_mechanism(
        baseline_pass_rate=1.0,
        ablation_pass_rate=1.0,
        dimension_deltas={"workflow": 0.0},
        cost_deltas={"tool_call_count": -2.0},
    ) == "disable_or_rework"
```

- [ ] **Step 2: Run the new diagnostics tests to verify they fail**

Run:

```bash
pytest tests/evals/test_eval_diagnostics.py -q
```

Expected: FAIL with `ModuleNotFoundError` for `code_agent_harness.evals.diagnostics`.

- [ ] **Step 3: Add assistant turn count to `EvalTrace`**

Modify `src/code_agent_harness/evals/trace.py`:

```python
@dataclass(frozen=True)
class EvalTrace:
    tool_calls: tuple[TraceToolCall, ...]
    final_output: str
    final_state: SessionState
    workspace_root: Path
    assistant_turn_count: int = 0
```

In `extract_eval_trace()`, add a counter:

```python
    assistant_turn_count = 0
```

Increment it inside the assistant-message loop:

```python
        assistant_turn_count += 1
```

Set it in the return value:

```python
        assistant_turn_count=assistant_turn_count,
```

- [ ] **Step 4: Implement diagnostics**

Create `src/code_agent_harness/evals/diagnostics.py`:

```python
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
                sum(1 for call in trace.tool_calls if call.has_result and call.result_status == "ok")
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
                    if call.tool_name == "ask_confirmation" or call.result_status == "require_confirmation"
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
```

- [ ] **Step 5: Export diagnostics types**

Modify `src/code_agent_harness/evals/__init__.py` to import and export:

```python
from code_agent_harness.evals.diagnostics import (
    EvalCostMetrics,
    attribute_failures,
    average_cost_metrics,
    compare_cost_metrics,
    compute_cost_metrics,
    recommend_mechanism,
)
```

Add these names to `__all__`.

- [ ] **Step 6: Run diagnostics tests**

Run:

```bash
pytest tests/evals/test_eval_diagnostics.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

Run:

```bash
git add src/code_agent_harness/evals/trace.py src/code_agent_harness/evals/diagnostics.py src/code_agent_harness/evals/__init__.py tests/evals/test_eval_diagnostics.py
git commit -m "feat: add eval cost diagnostics"
```

### Task 3: Wire Diagnostics Through Runner And CLI

**Files:**
- Modify: `src/code_agent_harness/evals/runner.py`
- Modify: `src/code_agent_harness/cli.py`
- Modify: `tests/evals/test_eval_runner.py`
- Modify: `tests/evals/test_eval_ablations.py`
- Modify: `tests/test_cli_phase2.py`

- [ ] **Step 1: Add failing runner tests for cost averages and recommendations**

Append this test to `tests/evals/test_eval_runner.py`:

```python
def test_run_eval_suite_exposes_cost_averages(tmp_path: Path) -> None:
    tasks = tuple(task for task in load_default_tasks() if task.task_id == "analysis-timeout")

    def provider_factory(task):
        del task
        return llm.FakeProvider(
            script=[
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
            ]
        )

    suite = run_eval_suite(
        suite_name="default",
        tasks=tasks,
        provider_factory=provider_factory,
        fixtures_root=Path("tests/evals/fixtures"),
        tmp_root=tmp_path,
    )

    assert suite.cost_averages["tool_call_count"] == 1.0
    assert suite.cost_averages["assistant_turn_count"] == 2.0
    assert suite.results[0].failure_attributions == ()
```

Append this test to the same file:

```python
def test_compare_suite_results_reports_cost_delta_and_recommendation(tmp_path: Path) -> None:
    trace = EvalTrace(
        tool_calls=(),
        final_output="done",
        final_state=SessionState.COMPLETED,
        workspace_root=tmp_path,
        assistant_turn_count=1,
    )
    baseline = (
        EvalRunResult(
            task_id="bugfix-basic",
            workspace_root=tmp_path,
            trace=trace,
            score=EvalScore(passed=True, dimensions={"workflow": 1.0}, evidence={}),
            cost_metrics={"tool_call_count": 3.0, "assistant_turn_count": 3.0},
        ),
    )
    ablation = (
        EvalRunResult(
            task_id="bugfix-basic",
            workspace_root=tmp_path,
            trace=trace,
            score=EvalScore(passed=True, dimensions={"workflow": 1.0}, evidence={}),
            cost_metrics={"tool_call_count": 1.0, "assistant_turn_count": 2.0},
        ),
    )

    comparison = compare_suite_results("default", "policy_engine", baseline, ablation)

    assert comparison.delta_by_cost["tool_call_count"] == -2.0
    assert comparison.delta_by_cost["assistant_turn_count"] == -1.0
    assert comparison.recommendation == "disable_or_rework"
```

- [ ] **Step 2: Run runner tests to verify they fail**

Run:

```bash
pytest tests/evals/test_eval_runner.py::test_run_eval_suite_exposes_cost_averages tests/evals/test_eval_runner.py::test_compare_suite_results_reports_cost_delta_and_recommendation -q
```

Expected: FAIL because `cost_averages`, `cost_metrics`, `failure_attributions`, `delta_by_cost`, and `recommendation` do not exist yet.

- [ ] **Step 3: Extend runner dataclasses and calculations**

Modify `src/code_agent_harness/evals/runner.py` imports:

```python
from dataclasses import dataclass, field

from code_agent_harness.evals.diagnostics import (
    attribute_failures,
    average_cost_metrics,
    compare_cost_metrics,
    compute_cost_metrics,
    recommend_mechanism,
)
```

Modify dataclasses:

```python
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
```

In `run_eval_task()`, compute diagnostics:

```python
    cost_metrics = compute_cost_metrics(trace)
    failure_attributions = attribute_failures(score)
```

Return:

```python
        cost_metrics=cost_metrics.values,
        failure_attributions=failure_attributions,
```

In `run_eval_suite()`, compute:

```python
    cost_averages = average_cost_metrics(
        tuple(compute_cost_metrics(result.trace) for result in results)
    )
```

Return `cost_averages=cost_averages`.

In `compare_suite_results()`, compute pass rates and cost deltas:

```python
    baseline_cost_averages = _average_run_costs(baseline)
    ablation_cost_averages = _average_run_costs(ablation)
    delta_by_cost = compare_cost_metrics(baseline_cost_averages, ablation_cost_averages)
    baseline_pass_rate = (sum(1 for result in baseline if result.score.passed) / len(baseline)) if baseline else 0.0
    ablation_pass_rate = (sum(1 for result in ablation if result.score.passed) / len(ablation)) if ablation else 0.0
```

Add helper:

```python
def _average_run_costs(results: tuple[EvalRunResult, ...]) -> dict[str, float]:
    if not results:
        return {}
    names = sorted({name for result in results for name in result.cost_metrics})
    return {
        name: sum(result.cost_metrics.get(name, 0.0) for result in results) / len(results)
        for name in names
    }
```

Return:

```python
        delta_by_cost=delta_by_cost,
        recommendation=recommend_mechanism(
            baseline_pass_rate=baseline_pass_rate,
            ablation_pass_rate=ablation_pass_rate,
            dimension_deltas=delta_by_dimension,
            cost_deltas=delta_by_cost,
        ),
```

- [ ] **Step 4: Add failing CLI output tests**

In `tests/evals/test_eval_ablations.py`, update `_make_run_result()` so returned `EvalRunResult` includes:

```python
        cost_metrics={
            "tool_call_count": 2.0 if passed else 4.0,
            "assistant_turn_count": 2.0 if passed else 4.0,
        },
        failure_attributions=() if passed else ("workflow_failure",),
```

Update the comparison fixture to include:

```python
        delta_by_cost={"tool_call_count": 2.0, "assistant_turn_count": 2.0},
        recommendation="keep",
```

Add assertions to `test_eval_cli_reports_single_task_evidence()`:

```python
    assert "cost_tool_call_count=4.0" in text
    assert "failure_attribution=workflow_failure" in text
```

Add assertions to `test_eval_cli_reports_suite_comparison()`:

```python
    assert "delta_cost_tool_call_count=2.0" in text
    assert "recommendation=keep" in text
```

In `tests/test_cli_phase2.py`, add a suite output test:

```python
def test_eval_suite_output_reports_cost_averages(monkeypatch, tmp_path: Path) -> None:
    from code_agent_harness import cli
    from code_agent_harness.evals.runner import EvalRunResult, EvalSuiteResult
    from code_agent_harness.evals.scoring import EvalScore
    from code_agent_harness.evals.trace import EvalTrace

    result = EvalRunResult(
        task_id="analysis-timeout",
        workspace_root=tmp_path,
        trace=EvalTrace(
            tool_calls=(),
            final_output="",
            final_state=SessionState.COMPLETED,
            workspace_root=tmp_path,
        ),
        score=EvalScore(passed=True, dimensions={"workflow": 1.0}, evidence={}),
        cost_metrics={"tool_call_count": 1.0},
    )
    suite = EvalSuiteResult(
        suite_name="default",
        results=(result,),
        passed_tasks=1,
        total_tasks=1,
        dimension_averages={"workflow": 1.0},
        cost_averages={"tool_call_count": 1.0},
    )

    monkeypatch.setattr(cli, "run_eval_suite", lambda *args, **kwargs: suite)
    monkeypatch.setattr(cli, "_build_scripted_eval_provider", lambda task_id: object())

    stdout = io.StringIO()
    stderr = io.StringIO()

    exit_code = main(
        ["eval", "--profile", "code_assistant", "--suite", "default"],
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 0
    assert "avg_cost_tool_call_count=1.0" in stdout.getvalue()
    assert stderr.getvalue() == ""
```

- [ ] **Step 5: Run CLI tests to verify they fail**

Run:

```bash
pytest tests/evals/test_eval_ablations.py tests/test_cli_phase2.py -q
```

Expected: FAIL because CLI output does not include cost metrics, failure attribution, cost deltas, or recommendations.

- [ ] **Step 6: Extend CLI output**

Modify `_write_task_result()` in `src/code_agent_harness/cli.py`:

```python
    for name, value in result.cost_metrics.items():
        stdout.write(f"cost_{name}={value}\n")
    for attribution in result.failure_attributions:
        stdout.write(f"failure_attribution={attribution}\n")
```

Change the signature from:

```python
def _write_task_result(task_id: str, score: object, stdout: TextIO) -> None:
```

to:

```python
def _write_task_result(result: object, stdout: TextIO) -> None:
```

Update callers and field references:

```python
    stdout.write(f"task={result.task_id}\n")
    stdout.write(f"passed={1 if result.score.passed else 0}\n")
```

Modify `_write_suite_result()`:

```python
    for name, value in result.cost_averages.items():
        stdout.write(f"avg_cost_{name}={value}\n")
```

Modify `_write_comparison_result()`:

```python
    for name, value in result.delta_by_cost.items():
        stdout.write(f"delta_cost_{name}={value}\n")
    stdout.write(f"recommendation={result.recommendation}\n")
```

- [ ] **Step 7: Run focused runner and CLI tests**

Run:

```bash
pytest tests/evals/test_eval_runner.py tests/evals/test_eval_ablations.py tests/test_cli_phase2.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

Run:

```bash
git add src/code_agent_harness/evals/runner.py src/code_agent_harness/cli.py tests/evals/test_eval_runner.py tests/evals/test_eval_ablations.py tests/test_cli_phase2.py
git commit -m "feat: surface eval cost diagnostics"
```

### Task 4: Strengthen Tool Descriptions And Prompt Layers

**Files:**
- Modify: `src/code_agent_harness/tools/builtins.py`
- Modify: `src/code_agent_harness/profiles/code_assistant.py`
- Modify: `tests/test_tool_descriptions.py`
- Modify: `tests/test_code_assistant_profile.py`

- [ ] **Step 1: Add failing tool-description tests**

Append to `tests/test_tool_descriptions.py`:

```python
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
```

Append to `tests/test_code_assistant_profile.py`:

```python
def test_code_assistant_profile_has_phase3_prompt_contract(tmp_path: Path) -> None:
    profile = build_code_assistant_profile(workspace_root=tmp_path)

    assert "inspect before editing" in profile.prompt_layers.scenario.lower()
    assert "targeted tests" in profile.prompt_layers.scenario.lower()
    assert "what changed" in profile.prompt_layers.execution.lower()
    assert "what tests ran" in profile.prompt_layers.execution.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_tool_descriptions.py::test_phase3_tool_descriptions_are_field_level_contracts tests/test_code_assistant_profile.py::test_code_assistant_profile_has_phase3_prompt_contract -q
```

Expected: FAIL because descriptions and prompt layers do not yet contain the required phase 3 contract text.

- [ ] **Step 3: Upgrade tool descriptions**

Modify descriptions in `src/code_agent_harness/tools/builtins.py`:

```python
description=(
    "Read a UTF-8 file before editing. Fields: path: string workspace-relative file path; "
    "start_line: integer optional inclusive first line; end_line: integer optional inclusive last line."
),
```

```python
description=(
    "Search workspace files for an exact text fragment, not a regex. Fields: "
    "pattern: string exact text fragment; path: string optional workspace-relative file or directory scope."
),
```

```python
description=(
    "List files under a workspace-relative directory. Fields: path: string optional directory; "
    "omit path to list the workspace root."
),
```

```python
description=(
    "Apply structured text replacements. Fields: path: string workspace-relative file path; "
    "replacements: array of objects with old_text: string, new_text: string, "
    "replace_all: boolean optional. Destructive edits may require confirmation."
),
```

```python
description=(
    "Run pytest from the workspace root. Fields: args: array[string] required pytest arguments, "
    "prefer targeted tests such as ['-q', 'tests/test_file.py']. Shell command strings are invalid."
),
```

```python
description=(
    "Record a required human confirmation step. Fields: message: string required prompt; "
    "default: boolean optional default answer. This does not execute the risky action."
),
```

- [ ] **Step 4: Upgrade prompt layer text**

Modify `src/code_agent_harness/profiles/code_assistant.py` prompt layer construction:

```python
    prompt_layers = PromptLayers(
        system="You are a code assistant for one local repository.",
        scenario=(
            "Use only the active tools. Inspect before editing. Prefer small, targeted investigation "
            "and targeted tests before broader validation. For read-only analysis, do not patch files "
            "or claim test execution unless a test tool actually ran."
        ),
        execution=(
            "Reply in the user's language. Report what changed, what tests ran, concrete outcomes, "
            "and any remaining risk or blocker."
        ),
    )
```

- [ ] **Step 5: Run focused tests**

Run:

```bash
pytest tests/test_tool_descriptions.py tests/test_code_assistant_profile.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add src/code_agent_harness/tools/builtins.py src/code_agent_harness/profiles/code_assistant.py tests/test_tool_descriptions.py tests/test_code_assistant_profile.py
git commit -m "feat: strengthen code assistant contracts"
```

### Task 5: Extend Code-Assistant Policy Reminders

**Files:**
- Modify: `src/code_agent_harness/policies/code_assistant.py`
- Modify: `tests/test_policy_engine.py`

- [ ] **Step 1: Add failing policy tests**

Append to `tests/test_policy_engine.py`:

```python
def test_policy_reminds_on_option_only_test_run() -> None:
    policy = build_code_assistant_policy(disabled_tools={"shell"})

    decision = policy.evaluate("run_tests", {"args": ["-q"]})

    assert decision.outcome == "remind"
    assert decision.reason == "broad_test_run"
    assert "targeted" in decision.message.lower()


def test_policy_reminds_on_large_patch_attempt() -> None:
    policy = build_code_assistant_policy(disabled_tools={"shell"})

    decision = policy.evaluate(
        "apply_patch",
        {
            "path": "calc.py",
            "replacements": [
                {"old_text": "a", "new_text": "b"},
                {"old_text": "c", "new_text": "d"},
                {"old_text": "e", "new_text": "f"},
                {"old_text": "g", "new_text": "h"},
            ],
        },
    )

    assert decision.outcome == "remind"
    assert decision.reason == "large_patch"
```

- [ ] **Step 2: Run policy tests to verify they fail**

Run:

```bash
pytest tests/test_policy_engine.py::test_policy_reminds_on_option_only_test_run tests/test_policy_engine.py::test_policy_reminds_on_large_patch_attempt -q
```

Expected: FAIL because option-only test runs and large patches are not yet detected.

- [ ] **Step 3: Implement policy helpers**

Modify `CodeAssistantPolicy.evaluate()` in `src/code_agent_harness/policies/code_assistant.py`:

```python
        if tool_name == "run_tests" and _looks_broad_test_run(arguments):
            return PolicyDecision(
                outcome="remind",
                reason="broad_test_run",
                message="Prefer targeted tests before the full suite; pass args like ['-q', 'tests/test_file.py'].",
            )
```

Add large patch reminder after destructive patch handling:

```python
        if tool_name == "apply_patch" and _looks_large_patch(arguments):
            return PolicyDecision(
                outcome="remind",
                reason="large_patch",
                message="Large patches should be split after inspecting the relevant file and tests.",
            )
```

Add helpers:

```python
def _looks_broad_test_run(arguments: dict[str, object]) -> bool:
    args = arguments.get("args")
    if not isinstance(args, list) or not args:
        return True
    return not any(isinstance(arg, str) and not arg.startswith("-") for arg in args)


def _looks_large_patch(arguments: dict[str, object]) -> bool:
    replacements = arguments.get("replacements")
    return isinstance(replacements, list) and len(replacements) > 3
```

- [ ] **Step 4: Run policy tests**

Run:

```bash
pytest tests/test_policy_engine.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/code_agent_harness/policies/code_assistant.py tests/test_policy_engine.py
git commit -m "feat: add phase3 policy reminders"
```

### Task 6: Document Phase 3 And Run Final Verification

**Files:**
- Modify: `README.md`
- Modify: `tests/evals/test_eval_ablations.py`
- Modify: `tests/evals/test_eval_diagnostics.py`
- Modify: `tests/evals/test_eval_runner.py`
- Modify: `tests/evals/test_eval_tasks.py`
- Modify: `tests/test_cli_phase2.py`

- [ ] **Step 1: Add README section**

Append this text to `README.md`:

    ## Phase 3 Cost-Directed Code Assistant

    Phase 3 keeps the project focused on `code_assistant` and optimizes by cost structure:

    1. Expand the eval set first.
    2. Strengthen tool descriptions before changing models.
    3. Add pre-tool policy reminders for rules the model may forget.
    4. Treat reasoning mode as an ablation, not a universal answer.
    5. Put model upgrades last.

    Run the default suite with cost diagnostics:

    ```bash
    code-agent-harness eval --profile code_assistant --suite default
    ```

    Compare a mechanism ablation:

    ```bash
    code-agent-harness eval --profile code_assistant --suite default --compare-ablation policy_engine
    ```

    The reported `cost_*` values are trace-derived proxy metrics, not provider billing data.

- [ ] **Step 2: Run focused phase 3 regression tests**

Run:

```bash
pytest tests/evals/test_eval_tasks.py tests/evals/test_eval_diagnostics.py tests/evals/test_eval_runner.py tests/evals/test_eval_ablations.py tests/test_cli_phase2.py tests/test_code_assistant_profile.py tests/test_policy_engine.py tests/test_tool_descriptions.py -q
```

Expected: PASS.

- [ ] **Step 3: Run CLI smoke commands**

Run:

```bash
PYTHONPATH=src python -m code_agent_harness.cli eval --profile code_assistant --suite default
```

Expected output includes:

```text
suite=default
passed_tasks=5/5
avg_cost_tool_call_count=
```

Run:

```bash
PYTHONPATH=src python -m code_agent_harness.cli eval --profile code_assistant --suite default --compare-ablation policy_engine
```

Expected output includes:

```text
compare_ablation=policy_engine
delta_cost_tool_call_count=
recommendation=
```

- [ ] **Step 4: Run the full repository test suite**

Run:

```bash
pytest -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add README.md tests/evals/test_eval_ablations.py tests/evals/test_eval_diagnostics.py tests/evals/test_eval_runner.py tests/evals/test_eval_tasks.py tests/test_cli_phase2.py tests/test_code_assistant_profile.py tests/test_policy_engine.py tests/test_tool_descriptions.py
git commit -m "docs: document phase3 cost optimization"
```

## Self-Review

### Spec Coverage

- Expanded default eval task set: Task 1
- Trace-derived cost metrics: Task 2 and Task 3
- Failure attribution: Task 2 and Task 3
- Cost averages and cost deltas: Task 3
- Mechanism recommendation: Task 2 and Task 3
- Stronger tool descriptions: Task 4
- Stronger prompt layers: Task 4
- Policy reminders: Task 5
- CLI output: Task 3
- README documentation and full verification: Task 6

### Placeholder Scan

The plan contains no placeholder markers or unspecified follow-up work. Each task includes exact files, concrete test snippets, implementation snippets, commands, and expected outcomes.

### Type Consistency

- `EvalCostMetrics.values` is used consistently as `dict[str, float]`.
- `EvalRunResult.cost_metrics` stores the raw metrics dictionary for CLI and comparison.
- `EvalSuiteResult.cost_averages` stores averaged numeric metrics.
- `EvalComparisonResult.delta_by_cost` stores ablation-minus-baseline deltas.
- `EvalComparisonResult.recommendation` stores one of `keep`, `neutral`, or `disable_or_rework`.
