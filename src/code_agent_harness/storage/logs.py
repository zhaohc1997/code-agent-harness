import json
from pathlib import Path


class StructuredLogger:
    def __init__(self, root: Path | str, filename: str = "events.jsonl") -> None:
        self.root = Path(root)
        self.logs_dir = self.root / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.logs_dir / filename

    def append(self, event: dict[str, object]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True))
            handle.write("\n")
