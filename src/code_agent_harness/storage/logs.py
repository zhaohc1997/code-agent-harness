import json
from pathlib import Path


def _require_leaf_name(value: str, label: str) -> str:
    candidate = Path(value)
    if candidate.name != value or candidate.is_absolute() or candidate.parent != Path("."):
        raise ValueError(f"{label} must be a simple leaf name, got {value!r}")
    return value


class StructuredLogger:
    def __init__(self, root: Path | str, filename: str = "events.jsonl") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / _require_leaf_name(filename, "filename")

    def append(self, event: dict[str, object]) -> None:
        try:
            payload = json.dumps(event, sort_keys=True)
        except TypeError as exc:
            raise ValueError("event must be JSON serializable") from exc
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(payload)
            handle.write("\n")
