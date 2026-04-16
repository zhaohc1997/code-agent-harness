from code_agent_harness.evals.tasks import EvalTask
from code_agent_harness.evals.tasks import load_default_tasks


def test_eval_task_requires_known_task_class() -> None:
    try:
        EvalTask(
            task_id="bad",
            task_class="other",
            fixture_name="bugfix_repo",
            user_input="fix it",
            expected_tool_names=("read_file",),
            required_response_substrings=("fixed",),
            repo_assertions=(),
            live_eligible=False,
        )
    except ValueError as exc:
        assert "task_class" in str(exc)
    else:
        raise AssertionError("expected validation error")


def test_load_default_tasks_covers_bugfix_feature_and_analysis() -> None:
    tasks = load_default_tasks()

    assert {task.task_class for task in tasks} == {"bugfix", "feature", "analysis"}
    assert len(tasks) >= 3
