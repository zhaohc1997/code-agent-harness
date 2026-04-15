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


class SessionStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, payload: dict[str, JsonValue]) -> None:
        session_id = _require_leaf_name(str(payload["session_id"]), "session_id")
        path = self.root / f"{session_id}.json"
        path.write_text(_dump_json(payload), encoding="utf-8")

    def load(self, session_id: str) -> dict[str, JsonValue]:
        session_id = _require_leaf_name(session_id, "session_id")
        path = self.root / f"{session_id}.json"
        return json.loads(path.read_text(encoding="utf-8"))
