from __future__ import annotations

import io
from pathlib import Path

import pytest

from code_agent_harness.cli import build_parser, main
from code_agent_harness.config import RuntimeConfig
from code_agent_harness.types.engine import RuntimeResult
from code_agent_harness.types.state import SessionState


def test_runtime_config_reads_deepseek_env(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CODE_AGENT_HARNESS_LIVE", "1")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("DEEPSEEK_MODEL", "DeepSeek-V3.2")
    monkeypatch.setenv("DEEPSEEK_REASONING", "0")

    config = RuntimeConfig.from_env(tmp_path / ".agenth", Path("/workspace"), "code_assistant")

    assert config.root == tmp_path / ".agenth"
    assert config.workspace_root == Path("/workspace")
    assert config.profile == "code_assistant"
    assert config.live is True
    assert config.live_provider.api_key == "test-key"
    assert config.live_provider.base_url == "https://api.deepseek.com"
    assert config.live_provider.model == "DeepSeek-V3.2"
    assert config.live_provider.reasoning_enabled is False
    assert config.paths.root == tmp_path / ".agenth"


def test_cli_parser_accepts_phase2_profile_and_eval_command() -> None:
    parser = build_parser()

    run_args = parser.parse_args(["run", "--session", "s1", "--input", "hello", "--profile", "code_assistant"])
    cancel_args = parser.parse_args(["cancel", "--session", "s1"])
    eval_args = parser.parse_args(
        [
            "eval",
            "--profile",
            "code_assistant",
            "--suite",
            "default",
            "--compare-ablation",
            "policy_engine",
        ]
    )

    assert run_args.command == "run"
    assert run_args.profile == "code_assistant"
    assert cancel_args.command == "cancel"
    assert eval_args.command == "eval"
    assert eval_args.profile == "code_assistant"
    assert eval_args.suite == "default"
    assert eval_args.compare_ablation == "policy_engine"


def test_cli_parser_requires_task_or_suite() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["eval", "--profile", "code_assistant"])


def test_run_command_passes_selected_profile_to_runtime_factory() -> None:
    received_profiles: list[str] = []

    class FakeRuntime:
        def run(self, session_id: str, user_input: str) -> RuntimeResult:
            assert session_id == "s1"
            assert user_input == "hello"
            return RuntimeResult(
                state=SessionState.COMPLETED,
                output_text="done",
                messages=[{"role": "assistant", "content": [{"type": "text", "text": "done"}]}],
            )

    def runtime_factory(profile: str) -> FakeRuntime:
        received_profiles.append(profile)
        return FakeRuntime()

    stdout = io.StringIO()
    stderr = io.StringIO()

    exit_code = main(
        ["run", "--session", "s1", "--input", "hello", "--profile", "reviewer"],
        runtime_factory=runtime_factory,
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 0
    assert received_profiles == ["reviewer"]
    assert stderr.getvalue() == ""
    assert "state=completed" in stdout.getvalue()
    assert "done" in stdout.getvalue()


def test_eval_command_returns_nonzero_for_unknown_task() -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()

    exit_code = main(
        ["eval", "--profile", "code_assistant", "--task", "missing-task"],
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code != 0
    assert stdout.getvalue() == ""
    assert "unknown task" in stderr.getvalue()


def test_eval_command_reports_live_config_errors(monkeypatch) -> None:
    monkeypatch.setenv("CODE_AGENT_HARNESS_LIVE", "1")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    stdout = io.StringIO()
    stderr = io.StringIO()

    exit_code = main(
        ["eval", "--profile", "code_assistant", "--task", "bugfix-basic", "--live"],
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 1
    assert stdout.getvalue() == ""
    assert "DEEPSEEK_API_KEY is required" in stderr.getvalue()
