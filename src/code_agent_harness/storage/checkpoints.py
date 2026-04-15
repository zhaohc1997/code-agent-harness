import json
from pathlib import Path


class CheckpointStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.checkpoints_dir = self.root / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def save(self, session_id: str, turn_count: int, payload: dict[str, object]) -> None:
        session_dir = self.checkpoints_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        path = session_dir / f"turn-{turn_count}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def load(self, session_id: str, turn_count: int) -> dict[str, object]:
        path = self.checkpoints_dir / session_id / f"turn-{turn_count}.json"
        return json.loads(path.read_text(encoding="utf-8"))
