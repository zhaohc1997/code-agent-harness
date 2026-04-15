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
