from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

from code_agent_harness.engine.cancellation import CancellationToken
from code_agent_harness.engine.loop import AgentRuntime
from code_agent_harness.engine.state_machine import EngineStateMachine
from code_agent_harness.evals.scoring import EvalScore
from code_agent_harness.evals.scoring import score_eval_run
from code_agent_harness.evals.tasks import EvalTask
from code_agent_harness.policies.code_assistant import build_code_assistant_policy
from code_agent_harness.profiles.code_assistant import build_code_assistant_profile
from code_agent_harness.prompts.builders import build_system_prompt
from code_agent_harness.storage.checkpoints import CheckpointStore
from code_agent_harness.storage.sessions import SessionStore
from code_agent_harness.tools.builtins import load_builtin_tools
from code_agent_harness.tools.executor import ToolExecutor
from code_agent_harness.tools.registry import ToolRegistry
from code_agent_harness.types.state import SessionState


@dataclass(frozen=True)
class EvalRunResult:
    workspace_root: Path
    score: EvalScore
    output_text: str
    tool_names: tuple[str, ...]


def run_eval_task(
    task: EvalTask,
    *,
    provider: object,
    fixtures_root: Path,
    tmp_root: Path,
    ablations: set[str] | None = None,
) -> EvalRunResult:
    source_root = fixtures_root / task.fixture_name
    workspace_root = tmp_root / task.task_id
    runtime_root = tmp_root / ".agenth" / task.task_id

    if workspace_root.exists():
        shutil.rmtree(workspace_root)
    if runtime_root.exists():
        shutil.rmtree(runtime_root)
    shutil.copytree(source_root, workspace_root)

    profile = build_code_assistant_profile(workspace_root=workspace_root, ablations=ablations)
    registry = ToolRegistry(
        lambda: [
            tool
            for tool in load_builtin_tools(profile.workspace_root)
            if tool.definition.name in profile.active_tool_names
        ]
    )
    runtime = AgentRuntime(
        provider=provider,
        system_prompt=build_system_prompt(profile.prompt_layers),
        sessions=SessionStore(runtime_root / "sessions"),
        checkpoints=CheckpointStore(runtime_root / "checkpoints"),
        registry=registry,
        executor=ToolExecutor(registry=registry, blob_store_root=runtime_root),
        cancellation=CancellationToken(signal_root=runtime_root / "cancellations"),
        state_machine=EngineStateMachine(SessionState.IDLE),
        context_window_tokens=128_000,
        policy_engine=(
            None
            if ablations is not None and "policy_engine" in ablations
            else build_code_assistant_policy(disabled_tools=set(profile.disabled_tools))
        ),
        provider_extra=profile.provider_extra,
    )
    runtime_result = runtime.run(session_id=task.task_id, user_input=task.user_input)

    tool_names = tuple(
        str(block["tool_name"])
        for message in runtime_result.messages
        if isinstance(message, dict) and isinstance(message.get("content"), list)
        for block in message["content"]
        if isinstance(block, dict) and block.get("type") == "tool_result" and "tool_name" in block
    )
    repository_state_ok = all(
        expected_text in (workspace_root / relative_path).read_text(encoding="utf-8")
        for relative_path, expected_text in task.repo_assertions
    )
    output_text = runtime_result.output_text
    score = score_eval_run(
        tool_choice_ok=tool_names == task.expected_tool_names,
        tool_arguments_ok=True,
        repository_state_ok=repository_state_ok,
        tests_ok=task.task_class == "analysis" or "passed" in output_text.lower(),
        response_content_ok=all(
            fragment.lower() in output_text.lower() for fragment in task.required_response_substrings
        ),
        workflow_ok=True,
    )
    return EvalRunResult(
        workspace_root=workspace_root,
        score=score,
        output_text=output_text,
        tool_names=tool_names,
    )
