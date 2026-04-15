import json

import pytest

import code_agent_harness.llm as llm
from code_agent_harness.config import RuntimePaths
from code_agent_harness.engine.observability import DecisionPointEvent, Observability
from code_agent_harness.storage.blobs import BlobStore
from code_agent_harness.storage.checkpoints import CheckpointStore
from code_agent_harness.storage.logs import StructuredLogger
from code_agent_harness.storage.sessions import SessionStore
from code_agent_harness.types.state import SessionState


def test_runtime_paths_return_leaf_directories(tmp_path) -> None:
    paths = RuntimePaths(tmp_path / ".agenth")

    assert paths.sessions == tmp_path / ".agenth" / "sessions"
    assert paths.checkpoints == tmp_path / ".agenth" / "checkpoints"
    assert paths.blobs == tmp_path / ".agenth" / "blobs"
    assert paths.logs == tmp_path / ".agenth" / "logs"


def test_session_store_round_trip(tmp_path) -> None:
    store = SessionStore(RuntimePaths(tmp_path / ".agenth").sessions)
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


def test_checkpoint_store_round_trip(tmp_path) -> None:
    store = CheckpointStore(RuntimePaths(tmp_path / ".agenth").checkpoints)
    payload = {
        "session_id": "s1",
        "turn_count": 3,
        "state": SessionState.RUNNING.value,
    }

    store.save("s1", 3, payload)

    assert store.load("s1", 3) == payload


def test_blob_store_round_trip(tmp_path) -> None:
    store = BlobStore(RuntimePaths(tmp_path / ".agenth").blobs)

    store.save("blob-1", "payload text")

    assert store.load("blob-1") == "payload text"


def test_structured_logger_appends_jsonl_line(tmp_path) -> None:
    logger = StructuredLogger(RuntimePaths(tmp_path / ".agenth").logs)
    event = {"event_name": "checkpoint_write", "session_id": "s1", "status": "executed"}

    logger.append(event)

    assert (RuntimePaths(tmp_path / ".agenth").logs / "events.jsonl").read_text(encoding="utf-8") == (
        json.dumps(event, sort_keys=True) + "\n"
    )


def test_observability_emit_writes_structured_event(tmp_path) -> None:
    logger = StructuredLogger(RuntimePaths(tmp_path / ".agenth").logs)
    observability = Observability(logger)
    event = DecisionPointEvent(
        event_name="tool_registry_reload",
        session_id="s1",
        turn_id=2,
        component="engine",
        status="executed",
        metadata={"tool_count": 4},
        timestamp="2026-04-15T00:00:00+00:00",
    )

    observability.emit(event)

    written = (RuntimePaths(tmp_path / ".agenth").logs / "events.jsonl").read_text(encoding="utf-8").strip()
    assert json.loads(written) == event.as_dict()


def test_logger_rejects_traversal_filename(tmp_path) -> None:
    with pytest.raises(ValueError, match="filename must be a simple leaf name"):
        StructuredLogger(RuntimePaths(tmp_path / ".agenth").logs, filename="../escape.jsonl")


def test_blob_store_rejects_parent_directory_blob_id(tmp_path) -> None:
    store = BlobStore(RuntimePaths(tmp_path / ".agenth").blobs)

    with pytest.raises(ValueError, match="blob_id must be a simple leaf name"):
        store.save("..", "payload text")


def test_checkpoint_store_rejects_parent_directory_session_id(tmp_path) -> None:
    store = CheckpointStore(RuntimePaths(tmp_path / ".agenth").checkpoints)

    with pytest.raises(ValueError, match="session_id must be a simple leaf name"):
        store.save("..", 1, {"session_id": "s1", "turn_count": 1, "state": SessionState.RUNNING.value})


def test_logger_rejects_parent_directory_filename(tmp_path) -> None:
    with pytest.raises(ValueError, match="filename must be a simple leaf name"):
        StructuredLogger(RuntimePaths(tmp_path / ".agenth").logs, filename="..")


def test_logger_rejects_non_serializable_event(tmp_path) -> None:
    logger = StructuredLogger(RuntimePaths(tmp_path / ".agenth").logs)

    with pytest.raises(ValueError, match="event must be JSON serializable"):
        logger.append({"bad": {1, 2, 3}})


def test_runtime_resumes_persisted_running_session(tmp_path) -> None:
    from conftest import _build_runtime_dependencies
    from code_agent_harness.engine.loop import AgentRuntime

    runtime_dependencies = _build_runtime_dependencies(tmp_path)
    runtime_dependencies["sessions"].save(
        {
            "session_id": "s1",
            "state": SessionState.RUNNING.value,
            "messages": [
                {"role": "user", "content": "Read a file"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "id": "tool-1",
                            "name": "read_file",
                            "arguments": {"path": "a.py"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tool-1",
                            "tool_name": "read_file",
                            "content": "read:a.py",
                        }
                    ],
                },
            ],
            "task_goal": "Read a file",
            "is_running": True,
            "queued_interventions": [],
            "turn_count": 1,
        }
    )
    provider = llm.FakeProvider(
        script=[
            {
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "finished"}],
            }
        ]
    )
    runtime = AgentRuntime(provider=provider, **runtime_dependencies)

    result = runtime.run(session_id="s1", user_input="Read a file")

    assert result.state == SessionState.COMPLETED
    assert [message for message in result.messages if message["role"] == "user" and isinstance(message["content"], str)] == [
        {"role": "user", "content": "Read a file"}
    ]
    checkpoint = runtime_dependencies["checkpoints"].load("s1", 2)
    assert checkpoint["state"] == SessionState.COMPLETED.value
    assert checkpoint["messages"][-1] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "finished"}],
    }
