from code_agent_harness.evals.tasks import (
    ArgumentExpectation,
    EvalTask,
    OutcomeExpectation,
    ToolExpectation,
    WorkflowExpectation,
    load_default_tasks,
)


def test_argument_expectation_rejects_unknown_match_mode() -> None:
    try:
        ArgumentExpectation(field_path="path", match_mode="unknown", expected="calc.py")
    except ValueError as exc:
        assert "match_mode" in str(exc)
    else:
        raise AssertionError("expected match mode validation error")


def test_eval_task_requires_known_task_class() -> None:
    try:
        EvalTask(
            task_id="bad",
            task_class="other",
            fixture_name="bugfix_repo",
            user_input="fix it",
            tool_expectations=(ToolExpectation(name="read_file"),),
            workflow_expectations=WorkflowExpectation(),
            outcome_expectations=OutcomeExpectation(),
            live_eligible=False,
        )
    except ValueError as exc:
        assert "task_class" in str(exc)
    else:
        raise AssertionError("expected validation error")


def test_eval_task_rejects_mixed_tool_expectation_inputs() -> None:
    try:
        EvalTask(
            task_id="bad",
            task_class="bugfix",
            fixture_name="bugfix_repo",
            user_input="fix it",
            tool_expectations=(ToolExpectation(name="read_file"),),
            workflow_expectations=WorkflowExpectation(),
            outcome_expectations=OutcomeExpectation(),
            live_eligible=False,
            expected_tool_names=("read_file",),
        )
    except ValueError as exc:
        assert "tool_expectations" in str(exc)
    else:
        raise AssertionError("expected conflict validation error")


def test_eval_task_rejects_mixed_outcome_inputs() -> None:
    try:
        EvalTask(
            task_id="bad",
            task_class="bugfix",
            fixture_name="bugfix_repo",
            user_input="fix it",
            tool_expectations=(ToolExpectation(name="read_file"),),
            workflow_expectations=WorkflowExpectation(),
            outcome_expectations=OutcomeExpectation(),
            live_eligible=False,
            repo_assertions=(("calc.py", "return a + b"),),
        )
    except ValueError as exc:
        assert "outcome_expectations" in str(exc)
    else:
        raise AssertionError("expected conflict validation error")


def test_load_default_tasks_uses_structured_expectations() -> None:
    tasks = {task.task_id: task for task in load_default_tasks()}

    assert set(tasks) == {"bugfix-basic", "feature-title-case", "analysis-timeout"}
    bugfix = tasks["bugfix-basic"]
    analysis = tasks["analysis-timeout"]

    assert bugfix.tool_expectations[0].name == "read_file"
    assert bugfix.tool_expectations[1].must_appear_after == ("read_file",)
    assert bugfix.tool_expectations[2].argument_expectations[0].field_path == "args"
    assert bugfix.workflow_expectations.must_read_before_patch is True
    assert bugfix.workflow_expectations.must_run_tests_before_finish is True
    assert bugfix.outcome_expectations.repo_assertions == (("calc.py", "return a + b"),)
    assert analysis.workflow_expectations.forbid_patch is True
    assert analysis.outcome_expectations.required_response_substrings == ("45", "search", "history", "export")
