from code_agent_harness.evals.scoring import score_eval_run


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
