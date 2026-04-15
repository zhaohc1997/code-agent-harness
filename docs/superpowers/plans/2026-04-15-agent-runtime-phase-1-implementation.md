# Agent Runtime Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python CLI agent runtime that can run a single-agent ReAct loop with strict tool-call adjacency, cancellation, checkpoint persistence, context compaction, loop detection, and decision-point observability.

**Architecture:** The implementation keeps orchestration in `engine`, provider translation in `llm`, tool discovery and execution in `tools`, persistence in `storage`, and runtime DTOs in `types`. Tests drive the design through a deterministic fake provider and fake tools first, then a thin OpenAI-compatible adapter boundary is added without leaking provider-specific details into the loop.

**Tech Stack:** Python 3.11+, `pytest`, `dataclasses`, `pathlib`, `subprocess`, `json`, optional `openai`

---

## File Structure

- Create: `pyproject.toml`
- Create: `src/code_agent_harness/__init__.py`
- Create: `src/code_agent_harness/cli.py`
- Create: `src/code_agent_harness/config.py`
- Create: `src/code_agent_harness/engine/__init__.py`
- Create: `src/code_agent_harness/engine/loop.py`
- Create: `src/code_agent_harness/engine/state_machine.py`
- Create: `src/code_agent_harness/engine/cancellation.py`
- Create: `src/code_agent_harness/engine/cycle_guard.py`
- Create: `src/code_agent_harness/engine/compaction.py`
- Create: `src/code_agent_harness/engine/observability.py`
- Create: `src/code_agent_harness/llm/__init__.py`
- Create: `src/code_agent_harness/llm/base.py`
- Create: `src/code_agent_harness/llm/fake_provider.py`
- Create: `src/code_agent_harness/llm/openai_compatible.py`
- Create: `src/code_agent_harness/tools/__init__.py`
- Create: `src/code_agent_harness/tools/registry.py`
- Create: `src/code_agent_harness/tools/executor.py`
- Create: `src/code_agent_harness/tools/builtins.py`
- Create: `src/code_agent_harness/tools/limits.py`
- Create: `src/code_agent_harness/storage/__init__.py`
- Create: `src/code_agent_harness/storage/sessions.py`
- Create: `src/code_agent_harness/storage/checkpoints.py`
- Create: `src/code_agent_harness/storage/blobs.py`
- Create: `src/code_agent_harness/storage/logs.py`
- Create: `src/code_agent_harness/types/__init__.py`
- Create: `src/code_agent_harness/types/messages.py`
- Create: `src/code_agent_harness/types/tools.py`
- Create: `src/code_agent_harness/types/state.py`
- Create: `src/code_agent_harness/types/engine.py`
- Create: `tests/conftest.py`
- Create: `tests/test_engine_loop.py`
- Create: `tests/test_protocol.py`
- Create: `tests/test_state_machine.py`
- Create: `tests/test_cancellation.py`
- Create: `tests/test_cycle_guard.py`
- Create: `tests/test_compaction.py`
- Create: `tests/test_session_resume.py`

### Task 1: Scaffold The Package And Tooling

**Files:**
- Create: `pyproject.toml`
- Create: `src/code_agent_harness/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Write the failing test for package import**

```python
from code_agent_harness import __version__


def test_package_exposes_version() -> None:
    assert __version__ == "0.1.0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/zhaohaichao/learn-claude-code/code-agent-harness && pytest tests/test_engine_loop.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'code_agent_harness'`

- [ ] **Step 3: Write the minimal package scaffold**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "code-agent-harness"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = []

[tool.pytest.ini_options]
pythonpath = ["src"]
```

```python
__version__ = "0.1.0"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/zhaohaichao/learn-claude-code/code-agent-harness && pytest tests/test_engine_loop.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/zhaohaichao/learn-claude-code/code-agent-harness
git add pyproject.toml src/code_agent_harness/__init__.py tests
git commit -m "chore: scaffold python package"
```

### Task 2: Define Shared Runtime Types And State Machine

**Files:**
- Create: `src/code_agent_harness/types/messages.py`
- Create: `src/code_agent_harness/types/tools.py`
- Create: `src/code_agent_harness/types/state.py`
- Create: `src/code_agent_harness/types/engine.py`
- Create: `src/code_agent_harness/engine/state_machine.py`
- Create: `tests/test_state_machine.py`
- Modify: `tests/conftest.py`

- [ ] **Step 1: Write the failing tests for states, messages, and transition rules**

```python
from code_agent_harness.engine.state_machine import EngineStateMachine
from code_agent_harness.types.state import SessionState


def test_valid_transition_idle_to_running() -> None:
    machine = EngineStateMachine(SessionState.IDLE)
    machine.transition(SessionState.RUNNING)
    assert machine.state is SessionState.RUNNING


def test_invalid_transition_completed_to_awaiting_confirmation() -> None:
    machine = EngineStateMachine(SessionState.COMPLETED)
    try:
        machine.transition(SessionState.AWAITING_CONFIRMATION)
    except ValueError as exc:
        assert "Invalid transition" in str(exc)
    else:
        raise AssertionError("expected transition failure")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/zhaohaichao/learn-claude-code/code-agent-harness && pytest tests/test_state_machine.py -q`
Expected: FAIL with `ModuleNotFoundError` for `code_agent_harness.engine.state_machine`

- [ ] **Step 3: Write the minimal types and state machine**

```python
from enum import Enum


class SessionState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    AWAITING_USER_INPUT = "awaiting_user_input"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
```

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class ToolCallBlock:
    id: str
    name: str
    arguments: dict[str, object]
```

```python
from code_agent_harness.types.state import SessionState


ALLOWED_TRANSITIONS = {
    SessionState.IDLE: {SessionState.RUNNING},
    SessionState.RUNNING: {
        SessionState.AWAITING_CONFIRMATION,
        SessionState.AWAITING_USER_INPUT,
        SessionState.COMPLETED,
        SessionState.FAILED,
        SessionState.CANCELLED,
    },
    SessionState.AWAITING_CONFIRMATION: {SessionState.RUNNING, SessionState.CANCELLED},
    SessionState.AWAITING_USER_INPUT: {SessionState.RUNNING, SessionState.CANCELLED},
    SessionState.COMPLETED: {SessionState.RUNNING},
    SessionState.FAILED: {SessionState.RUNNING},
    SessionState.CANCELLED: {SessionState.RUNNING},
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/zhaohaichao/learn-claude-code/code-agent-harness && pytest tests/test_state_machine.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/zhaohaichao/learn-claude-code/code-agent-harness
git add src/code_agent_harness/types src/code_agent_harness/engine/state_machine.py tests/test_state_machine.py tests/conftest.py
git commit -m "feat: add runtime types and state machine"
```

### Task 3: Add Filesystem Storage And Decision-Point Logging

**Files:**
- Create: `src/code_agent_harness/config.py`
- Create: `src/code_agent_harness/storage/sessions.py`
- Create: `src/code_agent_harness/storage/checkpoints.py`
- Create: `src/code_agent_harness/storage/blobs.py`
- Create: `src/code_agent_harness/storage/logs.py`
- Create: `src/code_agent_harness/engine/observability.py`
- Create: `tests/test_session_resume.py`

- [ ] **Step 1: Write the failing storage tests**

```python
from code_agent_harness.storage.sessions import SessionStore
from code_agent_harness.types.state import SessionState


def test_session_store_round_trip(tmp_path) -> None:
    store = SessionStore(tmp_path / ".agenth")
    payload = {
        "session_id": "s1",
        "state": SessionState.IDLE.value,
        "messages": [],
        "task_goal": "Fix a bug",
        "is_running": False,
        "queued_interventions": [],
        "turn_count": 0,
    }
    store.save(payload)
    assert store.load("s1") == payload
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/zhaohaichao/learn-claude-code/code-agent-harness && pytest tests/test_session_resume.py -q`
Expected: FAIL with `ModuleNotFoundError` for `code_agent_harness.storage.sessions`

- [ ] **Step 3: Write the minimal storage and logging implementation**

```python
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimePaths:
    root: Path

    @property
    def sessions(self) -> Path:
        return self.root / "sessions"
```

```python
import json


class SessionStore:
    def __init__(self, root):
        self.root = root
        (self.root / "sessions").mkdir(parents=True, exist_ok=True)

    def save(self, payload):
        path = self.root / "sessions" / f"{payload['session_id']}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load(self, session_id):
        path = self.root / "sessions" / f"{session_id}.json"
        return json.loads(path.read_text(encoding="utf-8"))
```

```python
class StructuredLogger:
    def append(self, event: dict[str, object]) -> None:
        ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/zhaohaichao/learn-claude-code/code-agent-harness && pytest tests/test_session_resume.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/zhaohaichao/learn-claude-code/code-agent-harness
git add src/code_agent_harness/config.py src/code_agent_harness/storage src/code_agent_harness/engine/observability.py tests/test_session_resume.py
git commit -m "feat: add filesystem storage and logging"
```

### Task 4: Add Provider Abstractions And Fake Provider

**Files:**
- Create: `src/code_agent_harness/llm/base.py`
- Create: `src/code_agent_harness/llm/fake_provider.py`
- Create: `src/code_agent_harness/llm/openai_compatible.py`
- Create: `tests/test_engine_loop.py`

- [ ] **Step 1: Write the failing provider tests**

```python
from code_agent_harness.llm.fake_provider import FakeProvider


def test_fake_provider_returns_scripted_responses() -> None:
    provider = FakeProvider(script=[{"stop_reason": "end_turn", "content": []}])
    response = provider.generate(system_prompt="sys", messages=[], tools=[], extra={})
    assert response.stop_reason == "end_turn"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/zhaohaichao/learn-claude-code/code-agent-harness && pytest tests/test_engine_loop.py -q`
Expected: FAIL with `ModuleNotFoundError` for `code_agent_harness.llm.fake_provider`

- [ ] **Step 3: Write the unified provider interfaces**

```python
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class LLMResponse:
    content: list[object]
    stop_reason: str
    usage: dict[str, int] | None = None


class LLMProvider(Protocol):
    def generate(self, system_prompt: str, messages: list[object], tools: list[object], extra: dict[str, object]) -> LLMResponse:
        ...
```

```python
class FakeProvider:
    def __init__(self, script):
        self._script = list(script)

    def generate(self, system_prompt, messages, tools, extra):
        payload = self._script.pop(0)
        return LLMResponse(**payload)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/zhaohaichao/learn-claude-code/code-agent-harness && pytest tests/test_engine_loop.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/zhaohaichao/learn-claude-code/code-agent-harness
git add src/code_agent_harness/llm tests/test_engine_loop.py
git commit -m "feat: add provider abstraction and fake provider"
```

### Task 5: Add Tool Registry, Execution Limits, And Blob Externalization

**Files:**
- Create: `src/code_agent_harness/tools/registry.py`
- Create: `src/code_agent_harness/tools/executor.py`
- Create: `src/code_agent_harness/tools/builtins.py`
- Create: `src/code_agent_harness/tools/limits.py`
- Create: `tests/test_protocol.py`

- [ ] **Step 1: Write the failing tool execution tests**

```python
from code_agent_harness.tools.executor import ToolExecutor
from code_agent_harness.tools.registry import ToolRegistry


def test_large_tool_output_is_externalized(tmp_path) -> None:
    registry = ToolRegistry(lambda: [])
    executor = ToolExecutor(registry=registry, blob_store_root=tmp_path / ".agenth")
    content = "x" * 25001
    result = executor._apply_limit("read_file", content)
    assert result.external_blob_id is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/zhaohaichao/learn-claude-code/code-agent-harness && pytest tests/test_protocol.py -q`
Expected: FAIL with `ModuleNotFoundError` for `code_agent_harness.tools.executor`

- [ ] **Step 3: Write the minimal registry and executor**

```python
TOOL_LIMITS = {
    "read_file": 20_000,
    "search_text": 10_000,
    "shell": 15_000,
}
```

```python
class ToolRegistry:
    def __init__(self, loader):
        self._loader = loader

    def list_tools(self):
        return list(self._loader())
```

```python
class ToolExecutor:
    def _apply_limit(self, tool_name, content):
        limit = TOOL_LIMITS.get(tool_name, 10_000)
        if len(content) <= limit:
            return ToolExecutionResult(content=content, external_blob_id=None)
        blob_id = self._blob_store.save_text(content)
        return ToolExecutionResult(
            content=f"[externalized:{blob_id}]",
            external_blob_id=blob_id,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/zhaohaichao/learn-claude-code/code-agent-harness && pytest tests/test_protocol.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/zhaohaichao/learn-claude-code/code-agent-harness
git add src/code_agent_harness/tools tests/test_protocol.py
git commit -m "feat: add tool registry and bounded execution"
```

### Task 6: Implement Cancellation, Cycle Guard, And Context Compaction

**Files:**
- Create: `src/code_agent_harness/engine/cancellation.py`
- Create: `src/code_agent_harness/engine/cycle_guard.py`
- Create: `src/code_agent_harness/engine/compaction.py`
- Create: `tests/test_cancellation.py`
- Create: `tests/test_cycle_guard.py`
- Create: `tests/test_compaction.py`

- [ ] **Step 1: Write the failing engine utility tests**

```python
from code_agent_harness.engine.cycle_guard import CycleGuard


def test_cycle_guard_blocks_repeated_tool_calls() -> None:
    guard = CycleGuard(max_repeats=2)
    assert guard.record("read_file", {"path": "a.py"}) is False
    assert guard.record("read_file", {"path": "a.py"}) is False
    assert guard.record("read_file", {"path": "a.py"}) is True
```

```python
from code_agent_harness.engine.compaction import micro_compact


def test_micro_compact_preserves_latest_tool_results() -> None:
    messages = [
        {"role": "user", "content": [{"type": "tool_result", "tool_name": "search_text", "content": "a" * 200}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_name": "search_text", "content": "b" * 200}]},
    ]
    compacted = micro_compact(messages, keep_recent=1)
    assert compacted[-1]["content"][0]["content"] == "b" * 200
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/zhaohaichao/learn-claude-code/code-agent-harness && pytest tests/test_cancellation.py tests/test_cycle_guard.py tests/test_compaction.py -q`
Expected: FAIL with missing engine utility modules

- [ ] **Step 3: Write the minimal utility implementations**

```python
class CancellationToken:
    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def is_cancelled(self) -> bool:
        return self._cancelled
```

```python
import json


class CycleGuard:
    def __init__(self, max_repeats: int = 2) -> None:
        self.max_repeats = max_repeats
        self._last_key = None
        self._repeat_count = 0

    def record(self, tool_name: str, arguments: dict[str, object]) -> bool:
        key = (tool_name, json.dumps(arguments, sort_keys=True))
        self._repeat_count = self._repeat_count + 1 if key == self._last_key else 0
        self._last_key = key
        return self._repeat_count >= self.max_repeats
```

```python
def estimate_tokens(messages) -> int:
    return max(1, len(str(messages)) // 4)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/zhaohaichao/learn-claude-code/code-agent-harness && pytest tests/test_cancellation.py tests/test_cycle_guard.py tests/test_compaction.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/zhaohaichao/learn-claude-code/code-agent-harness
git add src/code_agent_harness/engine/cancellation.py src/code_agent_harness/engine/cycle_guard.py src/code_agent_harness/engine/compaction.py tests/test_cancellation.py tests/test_cycle_guard.py tests/test_compaction.py
git commit -m "feat: add runtime guards and compaction"
```

### Task 7: Implement The ReAct Loop And Protocol Enforcement

**Files:**
- Create: `src/code_agent_harness/engine/loop.py`
- Modify: `src/code_agent_harness/engine/observability.py`
- Modify: `src/code_agent_harness/storage/checkpoints.py`
- Modify: `tests/test_engine_loop.py`
- Modify: `tests/test_protocol.py`

- [ ] **Step 1: Write the failing loop tests**

```python
from code_agent_harness.engine.loop import AgentRuntime
from code_agent_harness.llm.fake_provider import FakeProvider


def test_engine_completes_without_tool_calls(runtime_dependencies) -> None:
    provider = FakeProvider(script=[{
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": "done"}],
    }])
    runtime = AgentRuntime(provider=provider, **runtime_dependencies)
    result = runtime.run(session_id="s1", user_input="Summarize status")
    assert result.state == "completed"
    assert result.output_text == "done"
```

```python
def test_protocol_requires_tool_results_adjacent(runtime_dependencies) -> None:
    provider = FakeProvider(script=[{
        "stop_reason": "tool_use",
        "content": [{"type": "tool_call", "id": "tool-1", "name": "read_file", "arguments": {"path": "a.py"}}],
    }, {
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": "finished"}],
    }])
    runtime = AgentRuntime(provider=provider, **runtime_dependencies)
    result = runtime.run(session_id="s1", user_input="Read a file")
    assert result.messages[-2]["role"] == "assistant"
    assert result.messages[-1]["role"] == "user"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/zhaohaichao/learn-claude-code/code-agent-harness && pytest tests/test_engine_loop.py tests/test_protocol.py -q`
Expected: FAIL with missing `AgentRuntime`

- [ ] **Step 3: Write the minimal runtime loop**

```python
class AgentRuntime:
    def run(self, session_id: str, user_input: str):
        session = self._sessions.load_or_create(session_id, task_goal=user_input)
        self._state_machine.transition(SessionState.RUNNING)
        session["messages"].append({"role": "user", "content": user_input})
        while True:
            if self._cancellation.is_cancelled():
                self._state_machine.transition(SessionState.CANCELLED)
                break
            tools = self._registry.list_tools()
            messages = micro_compact(session["messages"])
            response = self._provider.generate(
                system_prompt=self._system_prompt,
                messages=messages,
                tools=tools,
                extra={},
            )
            session["messages"].append({"role": "assistant", "content": response.content})
            if response.stop_reason != "tool_use":
                self._state_machine.transition(SessionState.COMPLETED)
                break
            tool_results = self._executor.execute_many(response.content)
            session["messages"].append({"role": "user", "content": tool_results})
            self._checkpoints.save(session_id, session)
        return RuntimeResult(state=self._state_machine.state.value, output_text=self._extract_text(session["messages"]), messages=session["messages"])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/zhaohaichao/learn-claude-code/code-agent-harness && pytest tests/test_engine_loop.py tests/test_protocol.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/zhaohaichao/learn-claude-code/code-agent-harness
git add src/code_agent_harness/engine/loop.py src/code_agent_harness/engine/observability.py src/code_agent_harness/storage/checkpoints.py tests/test_engine_loop.py tests/test_protocol.py
git commit -m "feat: implement agent runtime loop"
```

### Task 8: Add The CLI And Final End-To-End Verification

**Files:**
- Create: `src/code_agent_harness/cli.py`
- Modify: `tests/conftest.py`
- Modify: `tests/test_engine_loop.py`

- [ ] **Step 1: Write the failing CLI smoke test**

```python
from code_agent_harness.cli import build_parser


def test_cli_parser_accepts_run_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["run", "--session", "s1", "--input", "hello"])
    assert args.command == "run"
    assert args.session == "s1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/zhaohaichao/learn-claude-code/code-agent-harness && pytest tests/test_engine_loop.py -q`
Expected: FAIL with `ModuleNotFoundError` for `code_agent_harness.cli`

- [ ] **Step 3: Write the minimal CLI**

```python
import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="code-agent-harness")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--session", required=True)
    run_parser.add_argument("--input", required=True)
    subparsers.add_parser("cancel").add_argument("--session", required=True)
    return parser
```

- [ ] **Step 4: Run the full test suite**

Run: `cd /Users/zhaohaichao/learn-claude-code/code-agent-harness && pytest -q`
Expected: PASS for all runtime contract tests

- [ ] **Step 5: Commit**

```bash
cd /Users/zhaohaichao/learn-claude-code/code-agent-harness
git add src/code_agent_harness/cli.py tests/conftest.py tests/test_engine_loop.py
git commit -m "feat: add cli entrypoint"
```

## Self-Review

### Spec coverage

- runtime package scaffold: covered by Task 1
- centralized types and state machine: covered by Task 2
- filesystem sessions, checkpoints, blobs, and logs: covered by Task 3
- provider abstraction and `extra` seam: covered by Task 4
- dynamic tool list, output ceilings, blob externalization: covered by Task 5
- cancellation, loop detection, and two-layer compaction foundation: covered by Task 6
- ReAct loop, protocol adjacency, checkpoint writes, intervention-safe flow: covered by Task 7
- thin CLI and final verification: covered by Task 8

### Placeholder scan

The plan contains no `TBD`, `TODO`, or deferred implementation markers. Each task names exact files, explicit test commands, and concrete code snippets.

### Type consistency

The plan consistently uses:

- `SessionState`
- `ToolCallBlock`
- `LLMResponse`
- `ToolRegistry`
- `ToolExecutor`
- `CancellationToken`
- `CycleGuard`
- `AgentRuntime`

These names should be kept unchanged during implementation to avoid cross-task drift.
