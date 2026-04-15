from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, TextIO

from code_agent_harness.config import RuntimeConfig
from code_agent_harness.engine.cancellation import CancellationToken
from code_agent_harness.engine.loop import AgentRuntime
from code_agent_harness.engine.observability import Observability
from code_agent_harness.engine.state_machine import EngineStateMachine
from code_agent_harness.llm.openai_compatible import OpenAICompatibleProvider
from code_agent_harness.storage.checkpoints import CheckpointStore
from code_agent_harness.storage.logs import StructuredLogger
from code_agent_harness.storage.sessions import SessionStore
from code_agent_harness.tools.builtins import load_builtin_tools
from code_agent_harness.tools.executor import ToolExecutor
from code_agent_harness.tools.registry import ToolRegistry
from code_agent_harness.types.engine import RuntimeResult
from code_agent_harness.types.state import SessionState


class RuntimeRunner(Protocol):
    def run(self, session_id: str, user_input: str) -> RuntimeResult:
        ...


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="code-agent-harness")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--session", required=True)
    run_parser.add_argument("--input", required=True)

    cancel_parser = subparsers.add_parser("cancel")
    cancel_parser.add_argument("--session", required=True)

    return parser


def build_default_runtime() -> AgentRuntime:
    config = RuntimeConfig(root=Path.cwd() / ".agenth")
    paths = config.paths
    registry = ToolRegistry(load_builtin_tools)
    return AgentRuntime(
        provider=OpenAICompatibleProvider(client=object()),
        system_prompt=config.system_prompt,
        sessions=SessionStore(paths.sessions),
        checkpoints=CheckpointStore(paths.checkpoints),
        registry=registry,
        executor=ToolExecutor(registry=registry, blob_store_root=paths.root),
        cancellation=CancellationToken(signal_root=paths.cancellations),
        state_machine=EngineStateMachine(SessionState.IDLE),
        observability=Observability(StructuredLogger(paths.logs)),
        context_window_tokens=config.context_window_tokens,
        auto_summary_trigger_ratio=config.auto_summary_trigger_ratio,
        auto_summary_keep_recent=config.auto_summary_keep_recent,
    )


def _run_command(
    args: argparse.Namespace,
    runtime_factory: Callable[[], RuntimeRunner],
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    try:
        result = runtime_factory().run(session_id=args.session, user_input=args.input)
    except Exception as exc:
        stderr.write(f"error: {exc}\n")
        return 1

    stdout.write(f"state={result.state.value}\n")
    if result.output_text:
        stdout.write(f"{result.output_text}\n")
    return 0


def _cancel_command(args: argparse.Namespace, stdout: TextIO, stderr: TextIO) -> int:
    del stderr
    config = RuntimeConfig(root=Path.cwd() / ".agenth")
    token = CancellationToken(signal_root=config.paths.cancellations)
    token.cancel(args.session)
    stdout.write(f"cancelled={args.session}\n")
    return 0


def main(
    argv: list[str] | None = None,
    *,
    runtime_factory: Callable[[], RuntimeRunner] = build_default_runtime,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    if args.command == "run":
        return _run_command(args, runtime_factory, stdout, stderr)
    return _cancel_command(args, stdout, stderr)


if __name__ == "__main__":
    raise SystemExit(main())
