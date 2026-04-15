from __future__ import annotations

import pytest

from code_agent_harness.config import RuntimePaths
from code_agent_harness.engine.cancellation import CancellationToken
from code_agent_harness.engine.observability import Observability
from code_agent_harness.engine.state_machine import EngineStateMachine
from code_agent_harness.storage.checkpoints import CheckpointStore
from code_agent_harness.storage.logs import StructuredLogger
from code_agent_harness.storage.sessions import SessionStore
from code_agent_harness.tools.executor import ToolExecutor
from code_agent_harness.tools.registry import RegisteredTool, ToolRegistry
from code_agent_harness.types.state import SessionState
from code_agent_harness.types.tools import ToolDefinition


def _build_runtime_dependencies(tmp_path, *, registry: ToolRegistry | None = None):
    paths = RuntimePaths(tmp_path / ".agenth")
    active_registry = registry or ToolRegistry(
        lambda: [
            RegisteredTool(
                definition=ToolDefinition(name="read_file"),
                handler=lambda arguments: f"read:{arguments['path']}",
            )
        ]
    )
    return {
        "system_prompt": "You are a test agent.",
        "sessions": SessionStore(paths.sessions),
        "checkpoints": CheckpointStore(paths.checkpoints),
        "registry": active_registry,
        "executor": ToolExecutor(registry=active_registry, blob_store_root=tmp_path / ".agenth"),
        "cancellation": CancellationToken(signal_root=paths.cancellations),
        "state_machine": EngineStateMachine(SessionState.IDLE),
        "observability": Observability(StructuredLogger(paths.logs)),
    }


@pytest.fixture
def runtime_dependencies(tmp_path):
    return _build_runtime_dependencies(tmp_path)
