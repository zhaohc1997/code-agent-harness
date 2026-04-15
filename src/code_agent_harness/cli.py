from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, TextIO

from code_agent_harness.config import RuntimePaths
from code_agent_harness.engine.cancellation import CancellationToken
from code_agent_harness.engine.loop import AgentRuntime
from code_agent_harness.engine.state_machine import EngineStateMachine
from code_agent_harness.llm.openai_compatible import OpenAICompatibleProvider
from code_agent_harness.storage.checkpoints import CheckpointStore
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
    paths = RuntimePaths(Path.cwd() / ".agenth")
    registry = ToolRegistry(load_builtin_tools)
    return AgentRuntime(
        provider=OpenAICompatibleProvider(client=object()),
        system_prompt="You are code-agent-harness.",
        sessions=SessionStore(paths.sessions),
        checkpoints=CheckpointStore(paths.checkpoints),
        registry=registry,
        executor=ToolExecutor(registry=registry, blob_store_root=paths.root),
        cancellation=CancellationToken(),
        state_machine=EngineStateMachine(SessionState.IDLE),
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


def _cancel_command(stdout: TextIO, stderr: TextIO) -> int:
    del stdout
    stderr.write("error: cancel command is not wired to a runtime signal path yet\n")
    return 1


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
    return _cancel_command(stdout, stderr)


if __name__ == "__main__":
    raise SystemExit(main())
