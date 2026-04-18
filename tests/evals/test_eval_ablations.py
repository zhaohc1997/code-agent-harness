from __future__ import annotations

import io
from pathlib import Path

from code_agent_harness import cli
from code_agent_harness.evals.runner import EvalComparisonResult, EvalRunResult, EvalSuiteResult
from code_agent_harness.evals.scoring import EvalScore
from code_agent_harness.evals.trace import EvalTrace
from code_agent_harness.types.state import SessionState


def test_cli_parser_accepts_multiple_ablations() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "eval",
            "--profile",
            "code_assistant",
            "--task",
            "bugfix-basic",
            "--ablate",
            "policy_engine",
            "--ablate",
            "prompt_layers",
        ]
    )

    assert args.ablate == ["policy_engine", "prompt_layers"]


def _make_run_result(task_id: str, passed: bool, tmp_path: Path) -> EvalRunResult:
    value = 1.0 if passed else 0.0
    return EvalRunResult(
        task_id=task_id,
        workspace_root=tmp_path,
        trace=EvalTrace(
            tool_calls=(),
            final_output="",
            final_state=SessionState.COMPLETED,
            workspace_root=tmp_path,
        ),
        score=EvalScore(
            passed=passed,
            dimensions={
                "tool_choice": value,
                "tool_arguments": value,
                "repository_state": value,
                "tests": value,
                "response_content": value,
                "workflow": value,
            },
            evidence={
                "tool_choice": "" if passed else "missing required tool read_file",
                "tool_arguments": "" if passed else "bad args",
                "repository_state": "",
                "tests": "",
                "response_content": "",
                "workflow": "" if passed else "repository file must be read before patching",
            },
        ),
    )


def test_eval_cli_reports_single_task_evidence(monkeypatch, tmp_path: Path) -> None:
    def fake_run_eval_task(*args, **kwargs):
        del args, kwargs
        return _make_run_result("bugfix-basic", False, tmp_path)

    monkeypatch.setattr(cli, "run_eval_task", fake_run_eval_task)
    monkeypatch.setattr(cli, "_build_scripted_eval_provider", lambda task_id: object())

    stdout = io.StringIO()
    stderr = io.StringIO()

    exit_code = cli.main(
        ["eval", "--profile", "code_assistant", "--task", "bugfix-basic"],
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 0
    text = stdout.getvalue()
    assert "task=bugfix-basic" in text
    assert "passed=0" in text
    assert "tool_choice=0.0" in text
    assert "evidence_tool_choice=missing required tool read_file" in text
    assert stderr.getvalue() == ""


def test_eval_cli_reports_suite_comparison(monkeypatch, tmp_path: Path) -> None:
    baseline_result = _make_run_result("bugfix-basic", True, tmp_path)
    ablation_result = _make_run_result("bugfix-basic", False, tmp_path)
    baseline_suite = EvalSuiteResult(
        suite_name="default",
        results=(baseline_result,),
        passed_tasks=1,
        total_tasks=1,
        dimension_averages={name: 1.0 for name in baseline_result.score.dimensions},
    )
    ablation_suite = EvalSuiteResult(
        suite_name="default",
        results=(ablation_result,),
        passed_tasks=0,
        total_tasks=1,
        dimension_averages={name: 0.0 for name in baseline_result.score.dimensions},
    )
    comparison = EvalComparisonResult(
        suite_name="default",
        ablation_name="policy_engine",
        baseline=baseline_suite.results,
        ablation=ablation_suite.results,
        delta_by_dimension={name: -1.0 for name in baseline_result.score.dimensions},
        changed_tasks=("bugfix-basic",),
    )

    monkeypatch.setattr(cli, "compare_suite_results", lambda *args, **kwargs: comparison)
    monkeypatch.setattr(
        cli,
        "run_eval_suite",
        lambda *args, **kwargs: baseline_suite if kwargs.get("ablations") is None else ablation_suite,
    )
    monkeypatch.setattr(cli, "_build_scripted_eval_provider", lambda task_id: object())

    stdout = io.StringIO()
    stderr = io.StringIO()

    exit_code = cli.main(
        ["eval", "--profile", "code_assistant", "--suite", "default", "--compare-ablation", "policy_engine"],
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 0
    text = stdout.getvalue()
    assert "suite=default" in text
    assert "compare_ablation=policy_engine" in text
    assert "baseline_passed=1/1" in text
    assert "ablation_passed=0/1" in text
    assert "delta_tool_arguments=-1.0" in text
    assert "changed_tasks=bugfix-basic" in text
    assert stderr.getvalue() == ""
