from dataclasses import dataclass


@dataclass(frozen=True)
class EvalScore:
    passed: bool
    dimensions: dict[str, float]


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
