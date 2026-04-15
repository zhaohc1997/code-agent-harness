from __future__ import annotations

from pathlib import Path

from code_agent_harness.cli import build_parser
from code_agent_harness.config import RuntimeConfig


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
        ["eval", "--profile", "code_assistant", "--task", "Fix a bug", "--ablate", "tool_calls", "--ablate", "memory", "--live"]
    )

    assert run_args.command == "run"
    assert run_args.profile == "code_assistant"
    assert cancel_args.command == "cancel"
    assert eval_args.command == "eval"
    assert eval_args.profile == "code_assistant"
    assert eval_args.task == "Fix a bug"
    assert eval_args.ablate == ["tool_calls", "memory"]
    assert eval_args.live is True
