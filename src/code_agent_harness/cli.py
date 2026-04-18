from __future__ import annotations

import argparse
import inspect
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, TextIO

from code_agent_harness import llm
from code_agent_harness.config import RuntimeConfig
from code_agent_harness.engine.cancellation import CancellationToken
from code_agent_harness.engine.loop import AgentRuntime
from code_agent_harness.engine.observability import Observability
from code_agent_harness.engine.state_machine import EngineStateMachine
from code_agent_harness.evals.runner import compare_suite_results, run_eval_suite, run_eval_task
from code_agent_harness.evals.tasks import EvalTask, load_default_tasks
from code_agent_harness.llm.openai_compatible import JsonHttpClient
from code_agent_harness.llm.openai_compatible import OpenAICompatibleProvider
from code_agent_harness.policies.code_assistant import build_code_assistant_policy
from code_agent_harness.profiles.code_assistant import build_code_assistant_profile
from code_agent_harness.prompts.builders import build_system_prompt
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
    run_parser.add_argument("--profile", default="code_assistant")

    cancel_parser = subparsers.add_parser("cancel")
    cancel_parser.add_argument("--session", required=True)

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--profile", default="code_assistant")
    target_group = eval_parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument("--task")
    target_group.add_argument("--suite")
    eval_parser.add_argument("--ablate", action="append", default=[])
    eval_parser.add_argument("--compare-ablation")
    eval_parser.add_argument("--live", action="store_true")

    return parser


def _load_runtime_config(profile: str) -> RuntimeConfig:
    return RuntimeConfig.from_env(Path.cwd() / ".agenth", Path.cwd(), profile)


def _build_live_provider(config: RuntimeConfig) -> OpenAICompatibleProvider:
    if config.live_provider is None:
        raise RuntimeError("Live provider disabled. Set CODE_AGENT_HARNESS_LIVE=1 and DeepSeek env vars.")

    return OpenAICompatibleProvider(
        client=JsonHttpClient(
            base_url=config.live_provider.base_url,
            api_key=config.live_provider.api_key,
        ),
        model=config.live_provider.model,
        base_url=config.live_provider.base_url,
        api_key=config.live_provider.api_key,
    )


def _build_scripted_eval_provider(task_id: str) -> llm.FakeProvider:
    scripts: dict[str, list[dict[str, object]]] = {
        "bugfix-basic": [
            {
                "stop_reason": "tool_use",
                "content": [
                    {
                        "type": "tool_call",
                        "id": "tool-1",
                        "name": "read_file",
                        "arguments": {"path": "calc.py"},
                    }
                ],
            },
            {
                "stop_reason": "tool_use",
                "content": [
                    {
                        "type": "tool_call",
                        "id": "tool-2",
                        "name": "apply_patch",
                        "arguments": {
                            "path": "calc.py",
                            "replacements": [{"old_text": "return a - b", "new_text": "return a + b"}],
                        },
                    }
                ],
            },
            {
                "stop_reason": "tool_use",
                "content": [
                    {
                        "type": "tool_call",
                        "id": "tool-3",
                        "name": "run_tests",
                        "arguments": {"args": ["-q", "tests/test_calc.py"]},
                    }
                ],
            },
            {
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "Fixed the bug and the targeted test passed."}],
            },
        ],
        "feature-title-case": [
            {
                "stop_reason": "tool_use",
                "content": [
                    {
                        "type": "tool_call",
                        "id": "tool-1",
                        "name": "read_file",
                        "arguments": {"path": "text_utils.py"},
                    }
                ],
            },
            {
                "stop_reason": "tool_use",
                "content": [
                    {
                        "type": "tool_call",
                        "id": "tool-2",
                        "name": "apply_patch",
                        "arguments": {
                            "path": "text_utils.py",
                            "replacements": [
                                {
                                    "old_text": 'return " ".join(words)',
                                    "new_text": 'return " ".join(word.title() for word in words)',
                                }
                            ],
                        },
                    }
                ],
            },
            {
                "stop_reason": "tool_use",
                "content": [
                    {
                        "type": "tool_call",
                        "id": "tool-3",
                        "name": "run_tests",
                        "arguments": {"args": ["-q", "tests/test_text_utils.py"]},
                    }
                ],
            },
            {
                "stop_reason": "end_turn",
                "content": [
                    {
                        "type": "text",
                        "text": "Implemented title case support and the targeted test passed.",
                    }
                ],
            },
        ],
        "analysis-timeout": [
            {
                "stop_reason": "tool_use",
                "content": [
                    {
                        "type": "tool_call",
                        "id": "tool-1",
                        "name": "read_file",
                        "arguments": {"path": "service.py"},
                    }
                ],
            },
            {
                "stop_reason": "end_turn",
                "content": [
                    {
                        "type": "text",
                        "text": "Default timeout is 45. Enabled features are search, history, and export.",
                    }
                ],
            },
        ],
        "bugfix-targeted-report": [
            {
                "stop_reason": "tool_use",
                "content": [
                    {
                        "type": "tool_call",
                        "id": "tool-1",
                        "name": "read_file",
                        "arguments": {"path": "calc.py"},
                    }
                ],
            },
            {
                "stop_reason": "tool_use",
                "content": [
                    {
                        "type": "tool_call",
                        "id": "tool-2",
                        "name": "apply_patch",
                        "arguments": {
                            "path": "calc.py",
                            "replacements": [{"old_text": "return a - b", "new_text": "return a + b"}],
                        },
                    }
                ],
            },
            {
                "stop_reason": "tool_use",
                "content": [
                    {
                        "type": "tool_call",
                        "id": "tool-3",
                        "name": "run_tests",
                        "arguments": {"args": ["-q", "tests/test_calc.py"]},
                    }
                ],
            },
            {
                "stop_reason": "end_turn",
                "content": [
                    {
                        "type": "text",
                        "text": "Fixed add() and passed the targeted test tests/test_calc.py.",
                    }
                ],
            },
        ],
        "analysis-readme-no-write": [
            {
                "stop_reason": "tool_use",
                "content": [
                    {
                        "type": "tool_call",
                        "id": "tool-1",
                        "name": "read_file",
                        "arguments": {"path": "README.md"},
                    }
                ],
            },
            {
                "stop_reason": "end_turn",
                "content": [
                    {
                        "type": "text",
                        "text": "Analysis Repo answers repository questions without modifying files.",
                    }
                ],
            },
        ],
    }
    if task_id not in scripts:
        raise ValueError(f"unknown scripted eval task: {task_id}")
    return llm.FakeProvider(script=scripts[task_id])


def _provider_factory(args: argparse.Namespace) -> Callable[[EvalTask], object]:
    if args.live:
        config = _load_runtime_config(args.profile)
        return lambda task: _build_live_provider(config)
    return lambda task: _build_scripted_eval_provider(task.task_id)


def build_default_runtime(
    profile: str = "code_assistant",
    *,
    ablations: set[str] | None = None,
) -> AgentRuntime:
    config = _load_runtime_config(profile)
    paths = config.paths
    profile_config = build_code_assistant_profile(workspace_root=config.workspace_root, ablations=ablations)
    registry = ToolRegistry(
        lambda: [
            tool
            for tool in load_builtin_tools(profile_config.workspace_root)
            if tool.definition.name in profile_config.active_tool_names
        ]
    )
    if config.live_provider is not None and config.live_provider.reasoning_enabled:
        provider_extra = profile_config.provider_extra
    else:
        provider_extra = {}
    return AgentRuntime(
        provider=_build_live_provider(config),
        system_prompt=build_system_prompt(profile_config.prompt_layers),
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
        policy_engine=(
            None
            if ablations is not None and "policy_engine" in ablations
            else build_code_assistant_policy(disabled_tools=set(profile_config.disabled_tools))
        ),
        provider_extra=provider_extra,
    )


def _build_runtime_with_profile(
    runtime_factory: Callable[..., RuntimeRunner],
    profile: str,
) -> RuntimeRunner:
    try:
        signature = inspect.signature(runtime_factory)
    except (TypeError, ValueError):
        return runtime_factory()

    parameters = tuple(signature.parameters.values())
    if not parameters:
        return runtime_factory()
    if "profile" in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters
    ):
        return runtime_factory(profile=profile)
    first_parameter = parameters[0]
    if first_parameter.kind in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.VAR_POSITIONAL,
    ):
        return runtime_factory(profile)
    return runtime_factory()


def _run_command(
    args: argparse.Namespace,
    runtime_factory: Callable[..., RuntimeRunner],
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    try:
        result = _build_runtime_with_profile(runtime_factory, args.profile).run(
            session_id=args.session,
            user_input=args.input,
        )
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


def _write_task_result(result: object, stdout: TextIO) -> None:
    stdout.write(f"task={result.task_id}\n")
    stdout.write(f"passed={1 if result.score.passed else 0}\n")
    for name, value in result.score.dimensions.items():
        stdout.write(f"{name}={value}\n")
        evidence = result.score.evidence.get(name, "")
        if evidence and value != 1.0:
            stdout.write(f"evidence_{name}={evidence}\n")
    for name, value in result.cost_metrics.items():
        stdout.write(f"cost_{name}={value}\n")
    for attribution in result.failure_attributions:
        stdout.write(f"failure_attribution={attribution}\n")


def _write_suite_result(result: object, stdout: TextIO) -> None:
    stdout.write(f"suite={result.suite_name}\n")
    stdout.write(f"passed_tasks={result.passed_tasks}/{result.total_tasks}\n")
    for name, value in result.dimension_averages.items():
        stdout.write(f"avg_{name}={value}\n")
    for name, value in result.cost_averages.items():
        stdout.write(f"avg_cost_{name}={value}\n")
    for run_result in result.results:
        stdout.write(f"task_status={run_result.task_id}:{1 if run_result.score.passed else 0}\n")


def _write_comparison_result(result: object, stdout: TextIO) -> None:
    baseline_passed = sum(1 for run_result in result.baseline if run_result.score.passed)
    ablation_passed = sum(1 for run_result in result.ablation if run_result.score.passed)
    stdout.write(f"suite={result.suite_name}\n")
    stdout.write(f"compare_ablation={result.ablation_name}\n")
    stdout.write(f"baseline_passed={baseline_passed}/{len(result.baseline)}\n")
    stdout.write(f"ablation_passed={ablation_passed}/{len(result.ablation)}\n")
    for name, value in result.delta_by_dimension.items():
        stdout.write(f"delta_{name}={value}\n")
    for name, value in result.delta_by_cost.items():
        stdout.write(f"delta_cost_{name}={value}\n")
    stdout.write(f"recommendation={result.recommendation}\n")
    stdout.write(f"changed_tasks={','.join(result.changed_tasks)}\n")


def _eval_command(args: argparse.Namespace, stdout: TextIO, stderr: TextIO) -> int:
    tasks_by_id = {task.task_id: task for task in load_default_tasks()}
    ablations = set(args.ablate)

    if args.compare_ablation and not args.suite:
        stderr.write("error: --compare-ablation requires --suite\n")
        return 1

    try:
        provider_factory = _provider_factory(args)
        if args.task:
            if args.task not in tasks_by_id:
                stderr.write(f"error: unknown task {args.task}\n")
                return 1
            task = tasks_by_id[args.task]
            result = run_eval_task(
                task,
                provider=provider_factory(task),
                fixtures_root=Path("tests/evals/fixtures"),
                tmp_root=Path(".agenth") / "evals",
                ablations=ablations,
            )
            _write_task_result(result, stdout)
            return 0

        if args.suite != "default":
            stderr.write(f"error: unknown suite {args.suite}\n")
            return 1

        tasks = tuple(load_default_tasks())
        if args.compare_ablation:
            baseline = run_eval_suite(
                suite_name="default",
                tasks=tasks,
                provider_factory=provider_factory,
                fixtures_root=Path("tests/evals/fixtures"),
                tmp_root=Path(".agenth") / "evals",
                ablations=None,
            )
            ablation = run_eval_suite(
                suite_name="default",
                tasks=tasks,
                provider_factory=provider_factory,
                fixtures_root=Path("tests/evals/fixtures"),
                tmp_root=Path(".agenth") / "evals",
                ablations=ablations | {args.compare_ablation},
            )
            comparison = compare_suite_results(
                "default",
                args.compare_ablation,
                baseline.results,
                ablation.results,
            )
            _write_comparison_result(comparison, stdout)
            return 0

        suite_result = run_eval_suite(
            suite_name="default",
            tasks=tasks,
            provider_factory=provider_factory,
            fixtures_root=Path("tests/evals/fixtures"),
            tmp_root=Path(".agenth") / "evals",
            ablations=ablations,
        )
        _write_suite_result(suite_result, stdout)
        return 0
    except Exception as exc:
        stderr.write(f"error: {exc}\n")
        return 1


def main(
    argv: list[str] | None = None,
    *,
    runtime_factory: Callable[..., RuntimeRunner] = build_default_runtime,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    if args.command == "run":
        return _run_command(args, runtime_factory, stdout, stderr)
    if args.command == "eval":
        return _eval_command(args, stdout, stderr)
    return _cancel_command(args, stdout, stderr)


if __name__ == "__main__":
    raise SystemExit(main())
