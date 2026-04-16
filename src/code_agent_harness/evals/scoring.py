from __future__ import annotations

from dataclasses import dataclass, field

from code_agent_harness.evals.matching import match_argument_expectations
from code_agent_harness.evals.tasks import EvalTask
from code_agent_harness.evals.trace import EvalTrace


@dataclass(frozen=True)
class EvalScore:
    passed: bool
    dimensions: dict[str, float]
    evidence: dict[str, str] = field(default_factory=dict)


def _first_index(trace: EvalTrace, tool_name: str) -> int | None:
    for call in trace.tool_calls:
        if call.tool_name == tool_name:
            return call.index
    return None


def score_eval_task(task: EvalTask, trace: EvalTrace) -> EvalScore:
    dimensions: dict[str, float] = {}
    evidence: dict[str, str] = {}

    tool_choice_issues: list[str] = []
    for expected in task.tool_expectations:
        call_index = _first_index(trace, expected.name)
        if expected.required and call_index is None:
            tool_choice_issues.append(f"missing required tool {expected.name}")
            continue
        if call_index is None:
            continue
        for predecessor in expected.must_appear_after:
            predecessor_index = _first_index(trace, predecessor)
            if predecessor_index is None:
                tool_choice_issues.append(f"{expected.name} requires prior tool {predecessor}")
            elif predecessor_index >= call_index:
                tool_choice_issues.append(f"{expected.name} must appear after {predecessor}")
        for successor in expected.must_appear_before:
            successor_index = _first_index(trace, successor)
            if successor_index is None:
                tool_choice_issues.append(f"{expected.name} requires later tool {successor}")
            elif successor_index <= call_index:
                tool_choice_issues.append(f"{expected.name} must appear before {successor}")
    dimensions["tool_choice"] = 0.0 if tool_choice_issues else 1.0
    if tool_choice_issues:
        evidence["tool_choice"] = tool_choice_issues[0]

    tool_argument_issues: list[str] = []
    for expected in task.tool_expectations:
        if not expected.argument_expectations:
            continue
        matching_calls = [call for call in trace.tool_calls if call.tool_name == expected.name]
        if not matching_calls:
            tool_argument_issues.append(f"missing tool call for {expected.name}")
            continue
        call_failures: list[str] = []
        for call in matching_calls:
            matched, call_evidence = match_argument_expectations(call.arguments, expected.argument_expectations)
            if matched:
                call_failures = []
                break
            call_failures.append(", ".join(call_evidence))
        if call_failures:
            tool_argument_issues.append(f"{expected.name}: {' | '.join(call_failures)}")
    dimensions["tool_arguments"] = 0.0 if tool_argument_issues else 1.0
    if tool_argument_issues:
        evidence["tool_arguments"] = "; ".join(tool_argument_issues)

    repository_issues: list[str] = []
    for relative_path, required_text in task.outcome_expectations.repo_assertions:
        target = trace.workspace_root / relative_path
        if not target.exists():
            repository_issues.append(f"missing file {relative_path}")
            continue
        if required_text not in target.read_text(encoding="utf-8"):
            repository_issues.append(f"{relative_path} missing expected text {required_text!r}")
    dimensions["repository_state"] = 0.0 if repository_issues else 1.0
    if repository_issues:
        evidence["repository_state"] = "; ".join(repository_issues)

    tests_issues: list[str] = []
    run_test_args = [
        str(argument)
        for call in trace.tool_calls
        if call.tool_name == "run_tests"
        for argument in (
            call.arguments.get("args", [])
            if isinstance(call.arguments.get("args", []), list)
            else [call.arguments.get("args", "")]
        )
    ]
    for fragment in task.outcome_expectations.required_test_args_fragments:
        if not any(fragment in arg for arg in run_test_args):
            tests_issues.append(f"missing test target fragment {fragment}")
    dimensions["tests"] = 0.0 if tests_issues else 1.0
    if tests_issues:
        evidence["tests"] = "; ".join(tests_issues)

    response_issues = [
        f"missing response fragment {fragment}"
        for fragment in task.outcome_expectations.required_response_substrings
        if fragment.lower() not in trace.final_output.lower()
    ]
    dimensions["response_content"] = 0.0 if response_issues else 1.0
    if response_issues:
        evidence["response_content"] = "; ".join(response_issues)

    workflow_issues: list[str] = []
    read_index = _first_index(trace, "read_file")
    patch_index = _first_index(trace, "apply_patch")
    test_index = _first_index(trace, "run_tests")
    workflow = task.workflow_expectations
    if workflow.must_read_before_patch and patch_index is not None:
        if read_index is None or read_index >= patch_index:
            workflow_issues.append("must read files before patching")
    if workflow.must_run_tests_before_finish:
        if test_index is None:
            workflow_issues.append("missing required tests before completion")
        elif patch_index is not None and test_index <= patch_index:
            workflow_issues.append("must run tests before finishing")
    if workflow.forbid_patch and patch_index is not None:
        workflow_issues.append("patching is forbidden")
    if workflow.forbid_test_runs and test_index is not None:
        workflow_issues.append("test runs are forbidden")
    dimensions["workflow"] = 0.0 if workflow_issues else 1.0
    if workflow_issues:
        evidence["workflow"] = "; ".join(workflow_issues)

    return EvalScore(
        passed=all(value == 1.0 for value in dimensions.values()),
        dimensions=dimensions,
        evidence=evidence,
    )


def score_eval_run(
    *,
    tool_choice_ok: bool,
    tool_arguments_ok: bool,
    repository_state_ok: bool,
    tests_ok: bool,
    response_content_ok: bool,
    workflow_ok: bool,
) -> EvalScore:
    dimensions = {
        "tool_choice": 1.0 if tool_choice_ok else 0.0,
        "tool_arguments": 1.0 if tool_arguments_ok else 0.0,
        "repository_state": 1.0 if repository_state_ok else 0.0,
        "tests": 1.0 if tests_ok else 0.0,
        "response_content": 1.0 if response_content_ok else 0.0,
        "workflow": 1.0 if workflow_ok else 0.0,
    }
    return EvalScore(
        passed=all(value == 1.0 for value in dimensions.values()),
        dimensions=dimensions,
    )
