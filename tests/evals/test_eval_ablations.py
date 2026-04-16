import io

from code_agent_harness.cli import build_parser
from code_agent_harness.cli import main


def test_cli_parser_accepts_multiple_ablations() -> None:
    parser = build_parser()
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


def test_eval_cli_reports_dimensioned_score() -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()

    exit_code = main(
        ["eval", "--profile", "code_assistant", "--task", "bugfix-basic", "--ablate", "policy_engine"],
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 0
    assert "tool_choice=" in stdout.getvalue()
    assert "repository_state=" in stdout.getvalue()
    assert stderr.getvalue() == ""
