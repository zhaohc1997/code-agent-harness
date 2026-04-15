import json
from pathlib import Path
from typing import TypeAlias


JsonPrimitive: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]


def _require_leaf_name(value: str, label: str) -> str:
    candidate = Path(value)
    if (
        value in {"", ".", ".."}
        or candidate.is_absolute()
        or len(candidate.parts) != 1
        or "/" in value
        or "\\" in value
    ):
        raise ValueError(f"{label} must be a simple leaf name, got {value!r}")
    return value


def _dump_json(payload: dict[str, JsonValue]) -> str:
    try:
        return json.dumps(payload, indent=2, sort_keys=True)
    except TypeError as exc:
        raise ValueError("payload must be JSON serializable") from exc


class CheckpointStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        session_id: str,
        turn_count: int | dict[str, JsonValue],
        payload: dict[str, JsonValue] | None = None,
    ) -> None:
        session_id = _require_leaf_name(session_id, "session_id")
        resolved_turn_count, resolved_payload = self._resolve_save_arguments(turn_count, payload)
        session_dir = self.root / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        path = session_dir / f"turn-{resolved_turn_count}.json"
        path.write_text(_dump_json(resolved_payload), encoding="utf-8")

    def load(self, session_id: str, turn_count: int) -> dict[str, JsonValue]:
        session_id = _require_leaf_name(session_id, "session_id")
        path = self.root / session_id / f"turn-{turn_count}.json"
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _resolve_save_arguments(
        turn_count: int | dict[str, JsonValue],
        payload: dict[str, JsonValue] | None,
    ) -> tuple[int, dict[str, JsonValue]]:
        if isinstance(turn_count, dict):
            if payload is not None:
                raise TypeError("payload must be omitted when turn_count is provided by payload")
            candidate_payload = turn_count
            candidate_turn_count = candidate_payload.get("turn_count")
        else:
            if payload is None:
                raise TypeError("payload is required when turn_count is provided explicitly")
            candidate_payload = payload
            candidate_turn_count = turn_count

        if not isinstance(candidate_turn_count, int):
            raise ValueError("turn_count must be an int")
        return candidate_turn_count, candidate_payload
