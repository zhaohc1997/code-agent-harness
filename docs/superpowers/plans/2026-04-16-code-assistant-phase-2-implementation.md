# Code Assistant Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Narrow the phase 1 runtime into a measurable `code_assistant` profile with real code-assistant tools, policy interception, layered prompts, DeepSeek live support, and a repeatable evaluation harness.

**Architecture:** Keep phase 1 orchestration in `engine`, then add a scenario layer through `profiles`, `prompts`, `policies`, and `evals`. Implement real code-assistant tools bound to a workspace root, filter them through the `code_assistant` profile, and preserve the existing provider/tool/session boundaries so live provider support and offline evaluation can share the same runtime contract.

**Tech Stack:** Python 3.11+, `pytest`, `dataclasses`, `pathlib`, `argparse`, `json`, `os`, `tempfile`, `shutil`, `subprocess`, `urllib.request`

---

## File Structure

- Modify: `README.md`
- Modify: `src/code_agent_harness/cli.py`
- Modify: `src/code_agent_harness/config.py`
- Modify: `src/code_agent_harness/engine/loop.py`
- Modify: `src/code_agent_harness/llm/openai_compatible.py`
- Modify: `src/code_agent_harness/tools/builtins.py`
- Create: `src/code_agent_harness/profiles/__init__.py`
- Create: `src/code_agent_harness/profiles/code_assistant.py`
- Create: `src/code_agent_harness/prompts/__init__.py`
- Create: `src/code_agent_harness/prompts/layers.py`
- Create: `src/code_agent_harness/prompts/builders.py`
- Create: `src/code_agent_harness/policies/__init__.py`
- Create: `src/code_agent_harness/policies/engine.py`
- Create: `src/code_agent_harness/policies/code_assistant.py`
- Create: `src/code_agent_harness/evals/__init__.py`
- Create: `src/code_agent_harness/evals/tasks.py`
- Create: `src/code_agent_harness/evals/scoring.py`
- Create: `src/code_agent_harness/evals/runner.py`
- Create: `src/code_agent_harness/tools/read_file.py`
- Create: `src/code_agent_harness/tools/search_text.py`
- Create: `src/code_agent_harness/tools/list_files.py`
- Create: `src/code_agent_harness/tools/apply_patch_tool.py`
- Create: `src/code_agent_harness/tools/run_tests.py`
- Create: `src/code_agent_harness/tools/git_status.py`
- Create: `src/code_agent_harness/tools/ask_confirmation.py`
- Create: `tests/test_cli_phase2.py`
- Create: `tests/test_prompt_builder.py`
- Create: `tests/test_code_assistant_profile.py`
- Create: `tests/test_code_assistant_tools.py`
- Create: `tests/test_policy_engine.py`
- Create: `tests/test_live_provider.py`
- Modify: `tests/test_engine_loop.py`
- Create: `tests/evals/test_eval_tasks.py`
- Create: `tests/evals/test_eval_scoring.py`
- Create: `tests/evals/test_eval_runner.py`
- Create: `tests/evals/test_eval_ablations.py`
- Create: `tests/evals/fixtures/bugfix_repo/calc.py`
- Create: `tests/evals/fixtures/bugfix_repo/tests/test_calc.py`
- Create: `tests/evals/fixtures/feature_repo/text_utils.py`
- Create: `tests/evals/fixtures/feature_repo/tests/test_text_utils.py`
- Create: `tests/evals/fixtures/analysis_repo/service.py`
- Create: `tests/evals/fixtures/analysis_repo/README.md`

### Task 1: Add Phase 2 Config And CLI Surface

**Files:**
- Modify: `src/code_agent_harness/config.py`
- Modify: `src/code_agent_harness/cli.py`
- Create: `tests/test_cli_phase2.py`

- [ ] **Step 1: Write the failing tests for live config loading and phase 2 CLI parsing**

```python
from pathlib import Path

from code_agent_harness.cli import build_parser
from code_agent_harness.config import RuntimeConfig


def test_runtime_config_reads_deepseek_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CODE_AGENT_HARNESS_LIVE", "1")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("DEEPSEEK_MODEL", "DeepSeek-V3.2")

    config = RuntimeConfig.from_env(
        root=tmp_path / ".agenth",
        workspace_root=tmp_path / "workspace",
        profile="code_assistant",
    )

    assert config.live is True
    assert config.profile == "code_assistant"
    assert config.workspace_root == tmp_path / "workspace"
    assert config.live_provider is not None
    assert config.live_provider.base_url == "https://api.deepseek.com"
    assert config.live_provider.model == "DeepSeek-V3.2"
    assert config.live_provider.reasoning_enabled is True


def test_cli_parser_accepts_phase2_profile_and_eval_command() -> None:
    parser = build_parser()

    run_args = parser.parse_args(
        ["run", "--session", "s1", "--input", "fix the bug", "--profile", "code_assistant"]
    )
    eval_args = parser.parse_args(
        ["eval", "--profile", "code_assistant", "--task", "bugfix-basic", "--ablate", "policy_engine"]
    )

    assert run_args.profile == "code_assistant"
    assert eval_args.command == "eval"
    assert eval_args.task == "bugfix-basic"
    assert eval_args.ablate == ["policy_engine"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/zhaohaichao/learn-claude-code/code-agent-harness && pytest tests/test_cli_phase2.py -q`
Expected: FAIL with `ImportError`, `AttributeError`, or parser errors because `RuntimeConfig.from_env()` and the `eval` subcommand do not exist yet.

- [ ] **Step 3: Write the minimal config and CLI surface**

```python
# src/code_agent_harness/config.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LiveProviderConfig:
    api_key: str
    base_url: str
    model: str
    reasoning_enabled: bool = True
    context_window_tokens: int = 128_000
    output_tokens: int = 32_000
    max_output_tokens: int = 64_000


@dataclass(frozen=True)
class RuntimeConfig:
    root: Path
    workspace_root: Path
    profile: str = "code_assistant"
    live: bool = False
    live_provider: LiveProviderConfig | None = None
    system_prompt: str = "You are code-agent-harness."
    context_window_tokens: int = 12000
    auto_summary_trigger_ratio: float = 0.65
    auto_summary_keep_recent: int = 4

    @classmethod
    def from_env(cls, *, root: Path, workspace_root: Path, profile: str) -> "RuntimeConfig":
        live = os.getenv("CODE_AGENT_HARNESS_LIVE") == "1"
        provider = None
        if live:
            api_key = os.environ["DEEPSEEK_API_KEY"]
            provider = LiveProviderConfig(
                api_key=api_key,
                base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                model=os.getenv("DEEPSEEK_MODEL", "DeepSeek-V3.2"),
                reasoning_enabled=os.getenv("DEEPSEEK_REASONING", "1") != "0",
            )
        return cls(root=root, workspace_root=workspace_root, profile=profile, live=live, live_provider=provider)
```

```python
# src/code_agent_harness/cli.py
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
    eval_parser.add_argument("--task")
    eval_parser.add_argument("--ablate", action="append", default=[])
    eval_parser.add_argument("--live", action="store_true")

    return parser
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/zhaohaichao/learn-claude-code/code-agent-harness && pytest tests/test_cli_phase2.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/zhaohaichao/learn-claude-code/code-agent-harness
git add src/code_agent_harness/config.py src/code_agent_harness/cli.py tests/test_cli_phase2.py
git commit -m "feat: add phase 2 config and cli surface"
```

### Task 2: Add Layered Prompts And The Code Assistant Profile

**Files:**
- Create: `src/code_agent_harness/prompts/__init__.py`
- Create: `src/code_agent_harness/prompts/layers.py`
- Create: `src/code_agent_harness/prompts/builders.py`
- Create: `src/code_agent_harness/profiles/__init__.py`
- Create: `src/code_agent_harness/profiles/code_assistant.py`
- Create: `tests/test_prompt_builder.py`
- Create: `tests/test_code_assistant_profile.py`

- [ ] **Step 1: Write the failing tests for prompt composition and tool narrowing**

```python
from pathlib import Path

from code_agent_harness.profiles.code_assistant import build_code_assistant_profile
from code_agent_harness.prompts.builders import build_system_prompt
from code_agent_harness.prompts.layers import PromptLayers


def test_prompt_builder_stacks_layers_in_order() -> None:
    prompt = build_system_prompt(
        PromptLayers(system="SYSTEM", scenario="SCENARIO", execution="EXECUTION")
    )
    assert prompt == "SYSTEM\n\nSCENARIO\n\nEXECUTION"


def test_prompt_builder_supports_ablation() -> None:
    prompt = build_system_prompt(
        PromptLayers(system="SYSTEM", scenario="SCENARIO", execution="EXECUTION"),
        enabled_layers={"system", "execution"},
    )
    assert "SYSTEM" in prompt
    assert "EXECUTION" in prompt
    assert "SCENARIO" not in prompt


def test_code_assistant_profile_narrows_tools_and_enables_reasoning(tmp_path: Path) -> None:
    profile = build_code_assistant_profile(workspace_root=tmp_path)

    assert profile.name == "code_assistant"
    assert "read_file" in profile.active_tool_names
    assert "run_tests" in profile.active_tool_names
    assert "shell" in profile.disabled_tools
    assert profile.provider_extra["thinking"]["enabled"] is True
    assert "code assistant for one local repository" in profile.prompt_layers.system.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/zhaohaichao/learn-claude-code/code-agent-harness && pytest tests/test_prompt_builder.py tests/test_code_assistant_profile.py -q`
Expected: FAIL with `ModuleNotFoundError` for the new prompt/profile modules.

- [ ] **Step 3: Write the minimal prompt-layer and profile implementation**

```python
# src/code_agent_harness/prompts/layers.py
from dataclasses import dataclass


@dataclass(frozen=True)
class PromptLayers:
    system: str
    scenario: str
    execution: str
```

```python
# src/code_agent_harness/prompts/builders.py
from code_agent_harness.prompts.layers import PromptLayers


def build_system_prompt(
    layers: PromptLayers,
    *,
    enabled_layers: set[str] | None = None,
) -> str:
    enabled = enabled_layers or {"system", "scenario", "execution"}
    parts: list[str] = []
    if "system" in enabled:
        parts.append(layers.system)
    if "scenario" in enabled:
        parts.append(layers.scenario)
    if "execution" in enabled:
        parts.append(layers.execution)
    return "\n\n".join(parts)
```

```python
# src/code_agent_harness/profiles/code_assistant.py
from dataclasses import dataclass
from pathlib import Path

from code_agent_harness.prompts.layers import PromptLayers


@dataclass(frozen=True)
class CodeAssistantProfile:
    name: str
    workspace_root: Path
    active_tool_names: tuple[str, ...]
    disabled_tools: tuple[str, ...]
    prompt_layers: PromptLayers
    provider_extra: dict[str, object]


def build_code_assistant_profile(*, workspace_root: Path) -> CodeAssistantProfile:
    return CodeAssistantProfile(
        name="code_assistant",
        workspace_root=workspace_root,
        active_tool_names=(
            "read_file",
            "search_text",
            "list_files",
            "apply_patch",
            "run_tests",
            "git_status",
            "ask_confirmation",
        ),
        disabled_tools=("shell",),
        prompt_layers=PromptLayers(
            system="You are a code assistant for one local repository.",
            scenario="Use only the active tools. Prefer small, targeted investigation and tests.",
            execution="Reply in the user's language and report concrete code and test outcomes.",
        ),
        provider_extra={"thinking": {"enabled": True}},
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/zhaohaichao/learn-claude-code/code-agent-harness && pytest tests/test_prompt_builder.py tests/test_code_assistant_profile.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/zhaohaichao/learn-claude-code/code-agent-harness
git add src/code_agent_harness/prompts src/code_agent_harness/profiles tests/test_prompt_builder.py tests/test_code_assistant_profile.py
git commit -m "feat: add code assistant prompt layers and profile"
```

### Task 3: Implement The Narrowed Code Assistant Tools

**Files:**
- Create: `src/code_agent_harness/tools/read_file.py`
- Create: `src/code_agent_harness/tools/search_text.py`
- Create: `src/code_agent_harness/tools/list_files.py`
- Create: `src/code_agent_harness/tools/apply_patch_tool.py`
- Create: `src/code_agent_harness/tools/run_tests.py`
- Create: `src/code_agent_harness/tools/git_status.py`
- Create: `src/code_agent_harness/tools/ask_confirmation.py`
- Modify: `src/code_agent_harness/tools/builtins.py`
- Create: `tests/test_code_assistant_tools.py`

- [ ] **Step 1: Write the failing tests for real tool behavior and strengthened schemas**

```python
from pathlib import Path

from code_agent_harness.tools.builtins import load_builtin_tools


def test_read_file_supports_line_ranges(tmp_path: Path) -> None:
    sample = tmp_path / "sample.py"
    sample.write_text("line1\nline2\nline3\n", encoding="utf-8")

    tools = {tool.definition.name: tool for tool in load_builtin_tools(tmp_path)}
    result = tools["read_file"].handler({"path": "sample.py", "start_line": 2, "end_line": 3})

    assert result == "line2\nline3\n"


def test_apply_patch_replaces_expected_text(tmp_path: Path) -> None:
    sample = tmp_path / "calc.py"
    sample.write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")

    tools = {tool.definition.name: tool for tool in load_builtin_tools(tmp_path)}
    result = tools["apply_patch"].handler(
        {
            "path": "calc.py",
            "replacements": [{"old": "return a - b", "new": "return a + b"}],
        }
    )

    assert "applied 1 replacement" in result
    assert sample.read_text(encoding="utf-8") == "def add(a, b):\n    return a + b\n"


def test_run_tests_uses_argument_arrays(tmp_path: Path) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tmp_path / "mod.py").write_text("def value():\n    return 1\n", encoding="utf-8")
    (tests_dir / "test_mod.py").write_text(
        "from mod import value\n\n\ndef test_value():\n    assert value() == 1\n",
        encoding="utf-8",
    )

    tools = {tool.definition.name: tool for tool in load_builtin_tools(tmp_path)}
    output = tools["run_tests"].handler({"args": ["-q", "tests/test_mod.py"]})

    assert "1 passed" in output


def test_strengthened_tool_schema_lists_required_fields(tmp_path: Path) -> None:
    tools = {tool.definition.name: tool for tool in load_builtin_tools(tmp_path)}

    read_file = tools["read_file"].definition
    apply_patch = tools["apply_patch"].definition

    assert "start_line" in read_file.description
    assert apply_patch.input_schema["required"] == ["path", "replacements"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/zhaohaichao/learn-claude-code/code-agent-harness && pytest tests/test_code_assistant_tools.py -q`
Expected: FAIL because `load_builtin_tools()` does not accept a workspace root and the tool handlers are still stubs.

- [ ] **Step 3: Write the minimal real tools and wire them into the builtins loader**

```python
# src/code_agent_harness/tools/read_file.py
from pathlib import Path


def build_read_file_handler(root: Path):
    def handler(arguments: dict[str, object]) -> str:
        path = root / str(arguments["path"])
        start_line = int(arguments.get("start_line", 1))
        end_line = int(arguments.get("end_line", 10**9))
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        return "".join(lines[start_line - 1:end_line])

    return handler
```

```python
# src/code_agent_harness/tools/apply_patch_tool.py
from pathlib import Path


def build_apply_patch_handler(root: Path):
    def handler(arguments: dict[str, object]) -> str:
        path = root / str(arguments["path"])
        content = path.read_text(encoding="utf-8")
        replacements = arguments["replacements"]
        count = 0
        for replacement in replacements:
            old = replacement["old"]
            new = replacement["new"]
            if old not in content:
                raise ValueError(f"patch target not found: {old!r}")
            content = content.replace(old, new, 1)
            count += 1
        path.write_text(content, encoding="utf-8")
        return f"applied {count} replacement(s) to {arguments['path']}"

    return handler
```

```python
# src/code_agent_harness/tools/run_tests.py
import subprocess
import sys
from pathlib import Path


def build_run_tests_handler(root: Path):
    def handler(arguments: dict[str, object]) -> str:
        args = [str(value) for value in arguments.get("args", ["-q"])]
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", *args],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        return completed.stdout + completed.stderr

    return handler
```

```python
# src/code_agent_harness/tools/builtins.py
from pathlib import Path

from code_agent_harness.tools.apply_patch_tool import build_apply_patch_handler
from code_agent_harness.tools.ask_confirmation import build_ask_confirmation_handler
from code_agent_harness.tools.git_status import build_git_status_handler
from code_agent_harness.tools.list_files import build_list_files_handler
from code_agent_harness.tools.read_file import build_read_file_handler
from code_agent_harness.tools.run_tests import build_run_tests_handler
from code_agent_harness.tools.search_text import build_search_text_handler
from code_agent_harness.tools.registry import RegisteredTool
from code_agent_harness.types.tools import ToolDefinition


def load_builtin_tools(root: Path) -> list[RegisteredTool]:
    return [
        RegisteredTool(
            definition=ToolDefinition(
                name="read_file",
                description="Read a UTF-8 file. Required field path:string. Optional start_line:int and end_line:int.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "start_line": {"type": "integer"},
                        "end_line": {"type": "integer"},
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
            ),
            handler=build_read_file_handler(root),
        ),
        RegisteredTool(
            definition=ToolDefinition(
                name="apply_patch",
                description="Apply structured replacements. Required path:string and replacements:array[{old:string,new:string}].",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "replacements": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "old": {"type": "string"},
                                    "new": {"type": "string"},
                                },
                                "required": ["old", "new"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["path", "replacements"],
                    "additionalProperties": False,
                },
            ),
            handler=build_apply_patch_handler(root),
        ),
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/zhaohaichao/learn-claude-code/code-agent-harness && pytest tests/test_code_assistant_tools.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/zhaohaichao/learn-claude-code/code-agent-harness
git add src/code_agent_harness/tools tests/test_code_assistant_tools.py
git commit -m "feat: implement code assistant tools"
```

### Task 4: Add Policy Interception And Confirmation Flow To The Runtime

**Files:**
- Create: `src/code_agent_harness/policies/__init__.py`
- Create: `src/code_agent_harness/policies/engine.py`
- Create: `src/code_agent_harness/policies/code_assistant.py`
- Modify: `src/code_agent_harness/engine/loop.py`
- Modify: `tests/test_engine_loop.py`
- Create: `tests/test_policy_engine.py`

- [ ] **Step 1: Write the failing tests for blocked tools, reminders, and confirmation**

```python
from pathlib import Path

import code_agent_harness.llm as llm
from code_agent_harness.policies.code_assistant import build_code_assistant_policy
from code_agent_harness.types.state import SessionState


def test_policy_blocks_disabled_tool(tmp_path: Path) -> None:
    from conftest import _build_runtime_dependencies
    from code_agent_harness.engine.loop import AgentRuntime

    runtime_dependencies = _build_runtime_dependencies(tmp_path)
    provider = llm.FakeProvider(
        script=[
            {
                "stop_reason": "tool_use",
                "content": [{"type": "tool_call", "id": "tool-1", "name": "shell", "arguments": {"command": "ls"}}],
            },
            {"stop_reason": "end_turn", "content": [{"type": "text", "text": "blocked"}]},
        ]
    )
    runtime = AgentRuntime(
        provider=provider,
        policy_engine=build_code_assistant_policy(disabled_tools={"shell"}),
        **runtime_dependencies,
    )

    result = runtime.run(session_id="s1", user_input="list files")

    assert result.state == SessionState.COMPLETED
    assert any(
        block.get("content", {}).get("reason") == "disabled_tool"
        for message in result.messages
        if isinstance(message.get("content"), list)
        for block in message["content"]
        if isinstance(block, dict) and block.get("type") == "tool_result"
    )


def test_policy_requires_confirmation_for_delete_patch(tmp_path: Path) -> None:
    from conftest import _build_runtime_dependencies
    from code_agent_harness.engine.loop import AgentRuntime

    runtime_dependencies = _build_runtime_dependencies(tmp_path)
    provider = llm.FakeProvider(
        script=[
            {
                "stop_reason": "tool_use",
                "content": [
                    {
                        "type": "tool_call",
                        "id": "tool-1",
                        "name": "apply_patch",
                        "arguments": {
                            "path": "calc.py",
                            "replacements": [{"old": "return a + b", "new": ""}],
                        },
                    }
                ],
            }
        ]
    )
    runtime = AgentRuntime(
        provider=provider,
        policy_engine=build_code_assistant_policy(disabled_tools={"shell"}),
        **runtime_dependencies,
    )

    result = runtime.run(session_id="s1", user_input="delete the function")

    assert result.state == SessionState.AWAITING_CONFIRMATION
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/zhaohaichao/learn-claude-code/code-agent-harness && pytest tests/test_policy_engine.py tests/test_engine_loop.py -q`
Expected: FAIL because `AgentRuntime` does not accept a policy engine and there is no policy module.

- [ ] **Step 3: Write the minimal policy dataclasses and runtime hook**

```python
# src/code_agent_harness/policies/engine.py
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PolicyDecision:
    outcome: str
    reason: str
    message: str
    metadata: dict[str, object] = field(default_factory=dict)
```

```python
# src/code_agent_harness/policies/code_assistant.py
from code_agent_harness.policies.engine import PolicyDecision


class CodeAssistantPolicy:
    def __init__(self, *, disabled_tools: set[str]) -> None:
        self._disabled_tools = disabled_tools

    def evaluate(self, tool_name: str, arguments: dict[str, object]) -> PolicyDecision:
        if tool_name in self._disabled_tools:
            return PolicyDecision("block", "disabled_tool", f"Tool {tool_name} is disabled in code_assistant.")
        if tool_name == "run_tests" and not arguments.get("args"):
            return PolicyDecision("remind", "broad_test_run", "Prefer targeted tests before the full suite.")
        if tool_name == "apply_patch" and any(
            replacement.get("new") == "" for replacement in arguments.get("replacements", [])
        ):
            return PolicyDecision(
                "require_confirmation",
                "destructive_patch",
                "Deleting code requires user confirmation in practical mode.",
            )
        return PolicyDecision("execute", "allowed", "Allowed.")


def build_code_assistant_policy(*, disabled_tools: set[str]) -> CodeAssistantPolicy:
    return CodeAssistantPolicy(disabled_tools=disabled_tools)
```

```python
# src/code_agent_harness/engine/loop.py
@dataclass
class AgentRuntime:
    provider: LLMProvider
    system_prompt: str
    sessions: SessionStore
    checkpoints: CheckpointStore
    registry: ToolRegistry
    executor: ToolExecutor
    cancellation: CancellationToken
    state_machine: EngineStateMachine
    observability: Observability | None = None
    cycle_guard: CycleGuard = field(default_factory=CycleGuard)
    context_window_tokens: int = 12000
    auto_summary_trigger_ratio: float = 0.65
    auto_summary_keep_recent: int = 4
    policy_engine: object | None = None
    provider_extra: dict[str, object] = field(default_factory=dict)

    def _execute_tool_calls(
        self,
        *,
        session_id: str,
        session: dict[str, Any],
        turn_count: int,
        tool_calls: list[dict[str, Any]],
        tool_snapshot: ToolRegistrySnapshot,
    ) -> list[dict[str, object]]:
        results: list[dict[str, object]] = []
        for block in tool_calls:
            tool_name = self._require_tool_name(block.get("name"))
            arguments = self._require_arguments(block.get("arguments"))
            if self.policy_engine is not None:
                decision = self.policy_engine.evaluate(tool_name, arguments)
                if decision.outcome == "block":
                    results.append(self._policy_tool_result(block, tool_name, decision))
                    continue
                if decision.outcome == "remind":
                    results.append(self._policy_tool_result(block, tool_name, decision))
                    continue
                if decision.outcome == "require_confirmation":
                    session["queued_interventions"] = [{"tool_call": block, "decision": decision.reason}]
                    self._state_machine_transition(
                        session_id=session_id,
                        turn_id=turn_count,
                        next_state=SessionState.AWAITING_CONFIRMATION,
                    )
                    results.append(self._policy_tool_result(block, tool_name, decision))
                    break

    def _policy_tool_result(
        self,
        block: dict[str, Any],
        tool_name: str,
        decision: PolicyDecision,
    ) -> dict[str, object]:
        return {
            "type": "tool_result",
            "tool_use_id": block.get("id"),
            "tool_name": tool_name,
            "content": {
                "status": decision.outcome,
                "reason": decision.reason,
                "message": decision.message,
            },
            "is_error": decision.outcome in {"block", "require_confirmation"},
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/zhaohaichao/learn-claude-code/code-agent-harness && pytest tests/test_policy_engine.py tests/test_engine_loop.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/zhaohaichao/learn-claude-code/code-agent-harness
git add src/code_agent_harness/policies src/code_agent_harness/engine/loop.py tests/test_policy_engine.py tests/test_engine_loop.py
git commit -m "feat: add code assistant policy interception"
```

### Task 5: Implement The DeepSeek OpenAI-Compatible Provider

**Files:**
- Modify: `src/code_agent_harness/llm/openai_compatible.py`
- Modify: `src/code_agent_harness/cli.py`
- Modify: `src/code_agent_harness/config.py`
- Create: `tests/test_live_provider.py`

- [ ] **Step 1: Write the failing tests for request/response normalization and env-gated runtime creation**

```python
import json
from pathlib import Path

from code_agent_harness.config import LiveProviderConfig
from code_agent_harness.llm import LLMRequest
from code_agent_harness.llm.openai_compatible import OpenAICompatibleProvider


class RecordingClient:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.requests: list[dict[str, object]] = []

    def send_json(self, payload: dict[str, object]) -> dict[str, object]:
        self.requests.append(payload)
        return self.payload


def test_openai_compatible_provider_maps_tools_and_thinking() -> None:
    client = RecordingClient(
        {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "read_file", "arguments": json.dumps({"path": "calc.py"})},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 7},
        }
    )
    provider = OpenAICompatibleProvider(
        client=client,
        model="DeepSeek-V3.2",
        base_url="https://api.deepseek.com",
        api_key="secret",
    )

    response = provider.generate(
        LLMRequest(
            system_prompt="system",
            messages=[{"role": "user", "content": "read calc.py"}],
            tools=[{"name": "read_file"}],
            extra={"thinking": {"enabled": True}},
        )
    )

    assert client.requests[0]["model"] == "DeepSeek-V3.2"
    assert client.requests[0]["tools"] == [{"name": "read_file"}]
    assert client.requests[0]["thinking"] == {"enabled": True}
    assert response.stop_reason == "tool_use"
    assert response.content[0]["name"] == "read_file"


def test_openai_compatible_provider_raises_on_missing_choices() -> None:
    provider = OpenAICompatibleProvider(
        client=RecordingClient({}),
        model="DeepSeek-V3.2",
        base_url="https://api.deepseek.com",
        api_key="secret",
    )
    try:
        provider.generate(LLMRequest(system_prompt="sys", messages=[], tools=[], extra={}))
    except ValueError as exc:
        assert "choices" in str(exc)
    else:
        raise AssertionError("expected malformed response failure")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/zhaohaichao/learn-claude-code/code-agent-harness && pytest tests/test_live_provider.py -q`
Expected: FAIL because `OpenAICompatibleProvider` is still a stub.

- [ ] **Step 3: Write the minimal DeepSeek-compatible adapter**

```python
# src/code_agent_harness/llm/openai_compatible.py
from __future__ import annotations

import json
import urllib.request

from code_agent_harness.llm.base import LLMRequest, LLMResponse


class JsonHttpClient:
    def __init__(self, *, base_url: str, api_key: str, timeout_seconds: float = 60.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds

    def send_json(self, payload: dict[str, object]) -> dict[str, object]:
        request = urllib.request.Request(
            f"{self._base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self._timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))


class OpenAICompatibleProvider:
    def __init__(self, *, client: object, model: str, base_url: str, api_key: str) -> None:
        self.client = client
        self.model = model
        self.base_url = base_url
        self.api_key = api_key

    def generate(self, request: LLMRequest) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": request.system_prompt}, *request.messages],
            "tools": request.tools,
            **request.extra,
        }
        raw = self.client.send_json(payload)
        choices = raw.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("provider response must include choices")
        choice = choices[0]
        message = choice["message"]
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            content = [
                {
                    "type": "tool_call",
                    "id": call["id"],
                    "name": call["function"]["name"],
                    "arguments": json.loads(call["function"]["arguments"]),
                }
                for call in tool_calls
            ]
            stop_reason = "tool_use"
        else:
            content = [{"type": "text", "text": message.get("content", "") or ""}]
            stop_reason = "end_turn"
        usage = raw.get("usage")
        return LLMResponse(content=content, stop_reason=stop_reason, usage=usage)
```

```python
# src/code_agent_harness/cli.py
def build_default_runtime() -> AgentRuntime:
    config = RuntimeConfig.from_env(
        root=Path.cwd() / ".agenth",
        workspace_root=Path.cwd(),
        profile="code_assistant",
    )
    if config.live_provider is None:
        raise RuntimeError("Live provider disabled. Set CODE_AGENT_HARNESS_LIVE=1 and DeepSeek env vars.")
    registry = ToolRegistry(lambda: load_builtin_tools(config.workspace_root))
    provider = OpenAICompatibleProvider(
        client=JsonHttpClient(
            base_url=config.live_provider.base_url,
            api_key=config.live_provider.api_key,
        ),
        model=config.live_provider.model,
        base_url=config.live_provider.base_url,
        api_key=config.live_provider.api_key,
    )
    return AgentRuntime(
        provider=provider,
        system_prompt=config.system_prompt,
        sessions=SessionStore(config.paths.sessions),
        checkpoints=CheckpointStore(config.paths.checkpoints),
        registry=registry,
        executor=ToolExecutor(registry=registry, blob_store_root=config.paths.root),
        cancellation=CancellationToken(signal_root=config.paths.cancellations),
        state_machine=EngineStateMachine(SessionState.IDLE),
        context_window_tokens=config.context_window_tokens,
        auto_summary_trigger_ratio=config.auto_summary_trigger_ratio,
        auto_summary_keep_recent=config.auto_summary_keep_recent,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/zhaohaichao/learn-claude-code/code-agent-harness && pytest tests/test_live_provider.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/zhaohaichao/learn-claude-code/code-agent-harness
git add src/code_agent_harness/llm/openai_compatible.py src/code_agent_harness/cli.py src/code_agent_harness/config.py tests/test_live_provider.py
git commit -m "feat: add deepseek live provider adapter"
```

### Task 6: Add Eval Task Schema And Dimensioned Scoring

**Files:**
- Create: `src/code_agent_harness/evals/__init__.py`
- Create: `src/code_agent_harness/evals/tasks.py`
- Create: `src/code_agent_harness/evals/scoring.py`
- Create: `tests/evals/test_eval_tasks.py`
- Create: `tests/evals/test_eval_scoring.py`

- [ ] **Step 1: Write the failing tests for task validation and dimensioned scoring**

```python
from code_agent_harness.evals.scoring import score_eval_run
from code_agent_harness.evals.tasks import EvalTask, load_default_tasks


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/zhaohaichao/learn-claude-code/code-agent-harness && pytest tests/evals/test_eval_tasks.py tests/evals/test_eval_scoring.py -q`
Expected: FAIL with `ModuleNotFoundError` for `code_agent_harness.evals`.

- [ ] **Step 3: Write the minimal task and scoring modules**

```python
# src/code_agent_harness/evals/tasks.py
from dataclasses import dataclass


@dataclass(frozen=True)
class EvalTask:
    task_id: str
    task_class: str
    fixture_name: str
    user_input: str
    expected_tool_names: tuple[str, ...]
    required_response_substrings: tuple[str, ...]
    repo_assertions: tuple[tuple[str, str], ...]
    live_eligible: bool

    def __post_init__(self) -> None:
        if self.task_class not in {"bugfix", "feature", "analysis"}:
            raise ValueError(f"task_class must be one of bugfix, feature, analysis, got {self.task_class!r}")


def load_default_tasks() -> tuple[EvalTask, ...]:
    return (
        EvalTask(
            task_id="bugfix-basic",
            task_class="bugfix",
            fixture_name="bugfix_repo",
            user_input="Fix the add() function and run the smallest relevant test.",
            expected_tool_names=("read_file", "apply_patch", "run_tests"),
            required_response_substrings=("fixed", "passed"),
            repo_assertions=(("calc.py", "return a + b"),),
            live_eligible=True,
        ),
        EvalTask(
            task_id="feature-title-case",
            task_class="feature",
            fixture_name="feature_repo",
            user_input="Make title_case_words() capitalize each input word and run the targeted test.",
            expected_tool_names=("read_file", "apply_patch", "run_tests"),
            required_response_substrings=("implemented", "passed"),
            repo_assertions=(("text_utils.py", "word.title()"),),
            live_eligible=True,
        ),
        EvalTask(
            task_id="analysis-timeout",
            task_class="analysis",
            fixture_name="analysis_repo",
            user_input="What is the default timeout and which features are enabled? Do not modify files.",
            expected_tool_names=("read_file",),
            required_response_substrings=("45", "search", "history", "export"),
            repo_assertions=(),
            live_eligible=True,
        ),
    )
```

```python
# src/code_agent_harness/evals/scoring.py
from dataclasses import dataclass


@dataclass(frozen=True)
class EvalScore:
    passed: bool
    dimensions: dict[str, float]


def score_eval_run(
    *,
    tool_choice_ok: bool,
    tool_arguments_ok: bool,
    repository_state_ok: bool,
    tests_ok: bool,
    response_content_ok: bool,
    workflow_ok: bool,
) -> EvalScore:
    dimensions = {
        "tool_choice": 1.0 if tool_choice_ok else 0.0,
        "tool_arguments": 1.0 if tool_arguments_ok else 0.0,
        "repository_state": 1.0 if repository_state_ok else 0.0,
        "tests": 1.0 if tests_ok else 0.0,
        "response_content": 1.0 if response_content_ok else 0.0,
        "workflow": 1.0 if workflow_ok else 0.0,
    }
    return EvalScore(passed=all(value == 1.0 for value in dimensions.values()), dimensions=dimensions)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/zhaohaichao/learn-claude-code/code-agent-harness && pytest tests/evals/test_eval_tasks.py tests/evals/test_eval_scoring.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/zhaohaichao/learn-claude-code/code-agent-harness
git add src/code_agent_harness/evals tests/evals/test_eval_tasks.py tests/evals/test_eval_scoring.py
git commit -m "feat: add eval task schema and scoring"
```

### Task 7: Add Fixture Repositories And The Eval Runner

**Files:**
- Create: `tests/evals/fixtures/bugfix_repo/calc.py`
- Create: `tests/evals/fixtures/bugfix_repo/tests/test_calc.py`
- Create: `tests/evals/fixtures/feature_repo/text_utils.py`
- Create: `tests/evals/fixtures/feature_repo/tests/test_text_utils.py`
- Create: `tests/evals/fixtures/analysis_repo/service.py`
- Create: `tests/evals/fixtures/analysis_repo/README.md`
- Create: `src/code_agent_harness/evals/runner.py`
- Create: `tests/evals/test_eval_runner.py`

- [ ] **Step 1: Write the failing tests for fixture isolation and end-to-end eval collection**

```python
from pathlib import Path

import code_agent_harness.llm as llm
from code_agent_harness.evals.runner import run_eval_task
from code_agent_harness.evals.tasks import EvalTask


def test_eval_runner_copies_fixture_into_isolated_workspace(tmp_path: Path) -> None:
    task = EvalTask(
        task_id="bugfix-basic",
        task_class="bugfix",
        fixture_name="bugfix_repo",
        user_input="Fix the add() function and run the smallest relevant test.",
        expected_tool_names=("read_file", "apply_patch", "run_tests"),
        required_response_substrings=("fixed", "passed"),
        repo_assertions=(("calc.py", "return a + b"),),
        live_eligible=True,
    )

    provider = llm.FakeProvider(
        script=[
            {
                "stop_reason": "tool_use",
                "content": [{"type": "tool_call", "id": "tool-1", "name": "read_file", "arguments": {"path": "calc.py"}}],
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
                            "replacements": [{"old": "return a - b", "new": "return a + b"}],
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
        ]
    )

    result = run_eval_task(
        task,
        provider=provider,
        fixtures_root=Path("tests/evals/fixtures"),
        tmp_root=tmp_path,
    )

    assert result.score.passed is True
    assert result.workspace_root != Path("tests/evals/fixtures/bugfix_repo")
    assert "return a - b" in (Path("tests/evals/fixtures/bugfix_repo") / "calc.py").read_text(encoding="utf-8")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/zhaohaichao/learn-claude-code/code-agent-harness && pytest tests/evals/test_eval_runner.py -q`
Expected: FAIL because the fixtures and `run_eval_task()` do not exist yet.

- [ ] **Step 3: Create the fixtures and the runner**

```python
# tests/evals/fixtures/bugfix_repo/calc.py
def add(a, b):
    return a - b
```

```python
# tests/evals/fixtures/bugfix_repo/tests/test_calc.py
from calc import add


def test_add() -> None:
    assert add(2, 3) == 5
```

```python
# tests/evals/fixtures/feature_repo/text_utils.py
def title_case_words(words: list[str]) -> str:
    return " ".join(words)
```

```python
# tests/evals/fixtures/feature_repo/tests/test_text_utils.py
from text_utils import title_case_words


def test_title_case_words() -> None:
    assert title_case_words(["hello", "world"]) == "Hello World"
```

```python
# tests/evals/fixtures/analysis_repo/service.py
DEFAULT_TIMEOUT_SECONDS = 45
ENABLED_FEATURES = ["search", "history", "export"]
```

```markdown
<!-- tests/evals/fixtures/analysis_repo/README.md -->
# Analysis Repo

Answer questions from the repository without modifying files.
```

```python
# src/code_agent_harness/evals/runner.py
from dataclasses import dataclass
from pathlib import Path
import shutil

from code_agent_harness.evals.scoring import score_eval_run


@dataclass(frozen=True)
class EvalRunResult:
    workspace_root: Path
    score: object
    output_text: str
    tool_names: tuple[str, ...]


def run_eval_task(task, *, provider, fixtures_root: Path, tmp_root: Path):
    source = fixtures_root / task.fixture_name
    workspace = tmp_root / task.task_id
    shutil.copytree(source, workspace)
    from code_agent_harness.profiles.code_assistant import build_code_assistant_profile
    from code_agent_harness.prompts.builders import build_system_prompt
    from code_agent_harness.tools.builtins import load_builtin_tools
    from code_agent_harness.tools.executor import ToolExecutor
    from code_agent_harness.tools.registry import ToolRegistry
    from code_agent_harness.storage.checkpoints import CheckpointStore
    from code_agent_harness.storage.sessions import SessionStore
    from code_agent_harness.engine.loop import AgentRuntime
    from code_agent_harness.engine.cancellation import CancellationToken
    from code_agent_harness.engine.state_machine import EngineStateMachine
    from code_agent_harness.types.state import SessionState

    profile = build_code_assistant_profile(workspace_root=workspace)
    registry = ToolRegistry(lambda: load_builtin_tools(workspace))
    runtime = AgentRuntime(
        provider=provider,
        system_prompt=build_system_prompt(profile.prompt_layers),
        sessions=SessionStore(tmp_root / ".agenth" / "sessions"),
        checkpoints=CheckpointStore(tmp_root / ".agenth" / "checkpoints"),
        registry=registry,
        executor=ToolExecutor(registry=registry, blob_store_root=tmp_root / ".agenth"),
        cancellation=CancellationToken(signal_root=tmp_root / ".agenth" / "cancellations"),
        state_machine=EngineStateMachine(SessionState.IDLE),
    )
    runtime_result = runtime.run(session_id=task.task_id, user_input=task.user_input)
    tool_names = tuple(
        block["tool_name"]
        for message in runtime_result.messages
        if isinstance(message.get("content"), list)
        for block in message["content"]
        if isinstance(block, dict) and block.get("type") == "tool_result"
    )
    repo_state_ok = all(
        expected in (workspace / relative_path).read_text(encoding="utf-8")
        for relative_path, expected in task.repo_assertions
    )
    score = score_eval_run(
        tool_choice_ok=tool_names == task.expected_tool_names,
        tool_arguments_ok=True,
        repository_state_ok=repo_state_ok,
        tests_ok="passed" in runtime_result.output_text.lower(),
        response_content_ok=all(
            part.lower() in runtime_result.output_text.lower()
            for part in task.required_response_substrings
        ),
        workflow_ok=True,
    )
    return EvalRunResult(
        workspace_root=workspace,
        score=score,
        output_text=runtime_result.output_text,
        tool_names=tool_names,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/zhaohaichao/learn-claude-code/code-agent-harness && pytest tests/evals/test_eval_runner.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/zhaohaichao/learn-claude-code/code-agent-harness
git add src/code_agent_harness/evals/runner.py tests/evals
git commit -m "feat: add code assistant eval fixtures and runner"
```

### Task 8: Add Ablations, Eval CLI Wiring, And Final Docs

**Files:**
- Modify: `src/code_agent_harness/cli.py`
- Modify: `src/code_agent_harness/profiles/code_assistant.py`
- Modify: `src/code_agent_harness/prompts/builders.py`
- Create: `tests/evals/test_eval_ablations.py`
- Modify: `README.md`

- [ ] **Step 1: Write the failing tests for ablation toggles and eval CLI output**

```python
import io

from code_agent_harness.cli import build_parser, main


def test_cli_parser_accepts_multiple_ablations() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["eval", "--profile", "code_assistant", "--task", "bugfix-basic", "--ablate", "policy_engine", "--ablate", "prompt_layers"]
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/zhaohaichao/learn-claude-code/code-agent-harness && pytest tests/evals/test_eval_ablations.py tests/test_cli_phase2.py -q`
Expected: FAIL because the CLI does not yet run eval tasks or ablations.

- [ ] **Step 3: Add ablation-aware profile assembly, eval CLI wiring, and README usage docs**

```python
# src/code_agent_harness/profiles/code_assistant.py
def build_code_assistant_profile(*, workspace_root: Path, ablations: set[str] | None = None) -> CodeAssistantProfile:
    disabled = {"shell"}
    provider_extra = {"thinking": {"enabled": True}}
    layers = PromptLayers(
        system="You are a code assistant for one local repository.",
        scenario="Use only the active tools. Prefer small, targeted investigation and tests.",
        execution="Reply in the user's language and report concrete code and test outcomes.",
    )
    ablations = ablations or set()
    if "tool_narrowing" in ablations:
        active = (
            "read_file",
            "search_text",
            "list_files",
            "apply_patch",
            "run_tests",
            "git_status",
            "ask_confirmation",
            "shell",
        )
        disabled = set()
    else:
        active = (
            "read_file",
            "search_text",
            "list_files",
            "apply_patch",
            "run_tests",
            "git_status",
            "ask_confirmation",
        )
    if "prompt_layers" in ablations:
        layers = PromptLayers(system=layers.system, scenario="", execution=layers.execution)
    if "reasoning_mode" in ablations:
        provider_extra = {}
    return CodeAssistantProfile(
        name="code_assistant",
        workspace_root=workspace_root,
        active_tool_names=tuple(active),
        disabled_tools=tuple(sorted(disabled)),
        prompt_layers=layers,
        provider_extra=provider_extra,
    )
```

```python
# src/code_agent_harness/cli.py
def build_default_runtime(*, profile_name: str = "code_assistant", ablations: set[str] | None = None) -> AgentRuntime:
    from code_agent_harness.llm.openai_compatible import JsonHttpClient, OpenAICompatibleProvider
    from code_agent_harness.policies.code_assistant import build_code_assistant_policy
    from code_agent_harness.profiles.code_assistant import build_code_assistant_profile
    from code_agent_harness.prompts.builders import build_system_prompt
    from code_agent_harness.tools.builtins import load_builtin_tools
    from code_agent_harness.tools.registry import ToolRegistry

    config = RuntimeConfig.from_env(
        root=Path.cwd() / ".agenth",
        workspace_root=Path.cwd(),
        profile=profile_name,
    )
    profile = build_code_assistant_profile(workspace_root=config.workspace_root, ablations=ablations)
    registry = ToolRegistry(
        lambda: [
            tool
            for tool in load_builtin_tools(profile.workspace_root)
            if tool.definition.name in profile.active_tool_names
        ]
    )
    policy_engine = None if ablations and "policy_engine" in ablations else build_code_assistant_policy(
        disabled_tools=set(profile.disabled_tools)
    )
    if config.live_provider is None:
        raise RuntimeError("Live provider disabled. Set CODE_AGENT_HARNESS_LIVE=1 and DeepSeek env vars.")
    provider = OpenAICompatibleProvider(
        client=JsonHttpClient(
            base_url=config.live_provider.base_url,
            api_key=config.live_provider.api_key,
        ),
        model=config.live_provider.model,
        base_url=config.live_provider.base_url,
        api_key=config.live_provider.api_key,
    )
    return AgentRuntime(
        provider=provider,
        system_prompt=build_system_prompt(profile.prompt_layers),
        sessions=SessionStore(config.paths.sessions),
        checkpoints=CheckpointStore(config.paths.checkpoints),
        registry=registry,
        executor=ToolExecutor(registry=registry, blob_store_root=config.paths.root),
        cancellation=CancellationToken(signal_root=config.paths.cancellations),
        state_machine=EngineStateMachine(SessionState.IDLE),
        policy_engine=policy_engine,
        provider_extra=profile.provider_extra,
    )


def _run_command(
    args: argparse.Namespace,
    runtime_factory: Callable[..., RuntimeRunner],
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    try:
        result = runtime_factory(profile_name=args.profile, ablations=set()).run(
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


def _eval_command(args: argparse.Namespace, stdout: TextIO, stderr: TextIO) -> int:
    from code_agent_harness.evals.runner import run_eval_task
    from code_agent_harness.evals.tasks import load_default_tasks
    from code_agent_harness.llm.fake_provider import FakeProvider

    tasks_by_id = {task.task_id: task for task in load_default_tasks()}
    if args.task not in tasks_by_id:
        stderr.write(f"error: unknown task {args.task}\n")
        return 1

    task = tasks_by_id[args.task]
    provider = FakeProvider(
        script=[
            {"stop_reason": "end_turn", "content": [{"type": "text", "text": "Evaluation completed."}]}
        ]
    )
    result = run_eval_task(
        task,
        provider=provider,
        fixtures_root=Path("tests/evals/fixtures"),
        tmp_root=Path(".agenth") / "evals",
    )
    for name, value in result.score.dimensions.items():
        stdout.write(f"{name}={value}\n")
    return 0
```

```markdown
<!-- README.md -->
## Phase 2 Code Assistant

Run a live code-assistant session:

Ensure `DEEPSEEK_API_KEY` is already exported in the shell before running these commands.

```bash
export CODE_AGENT_HARNESS_LIVE=1
export DEEPSEEK_BASE_URL=https://api.deepseek.com
export DEEPSEEK_MODEL=DeepSeek-V3.2
code-agent-harness run --profile code_assistant --session demo --input "Fix the failing test"
```

Run an eval task with an ablation:

```bash
code-agent-harness eval --profile code_assistant --task bugfix-basic --ablate policy_engine
```
```

- [ ] **Step 4: Run the phase 2 test suite and the full repository tests**

Run: `cd /Users/zhaohaichao/learn-claude-code/code-agent-harness && pytest tests/test_cli_phase2.py tests/test_prompt_builder.py tests/test_code_assistant_profile.py tests/test_code_assistant_tools.py tests/test_policy_engine.py tests/test_live_provider.py tests/evals/test_eval_tasks.py tests/evals/test_eval_scoring.py tests/evals/test_eval_runner.py tests/evals/test_eval_ablations.py -q`
Expected: PASS

Run: `cd /Users/zhaohaichao/learn-claude-code/code-agent-harness && pytest -q`
Expected: PASS

Optional live smoke after the offline suite is green:

Run: `cd /Users/zhaohaichao/learn-claude-code/code-agent-harness && CODE_AGENT_HARNESS_LIVE=1 pytest tests/test_live_provider.py -q`
Expected: PASS if DeepSeek credentials are set and the network path is healthy; otherwise SKIP or fail with a clear provider/network error.

- [ ] **Step 5: Commit**

```bash
cd /Users/zhaohaichao/learn-claude-code/code-agent-harness
git add README.md src/code_agent_harness/cli.py src/code_agent_harness/profiles/code_assistant.py src/code_agent_harness/prompts/builders.py tests/evals/test_eval_ablations.py tests/test_cli_phase2.py
git commit -m "feat: wire code assistant eval cli and ablations"
```
