import json
from pathlib import Path


class SessionStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.sessions_dir = self.root / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def save(self, payload: dict[str, object]) -> None:
        path = self.sessions_dir / f"{payload['session_id']}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def load(self, session_id: str) -> dict[str, object]:
        path = self.sessions_dir / f"{session_id}.json"
        return json.loads(path.read_text(encoding="utf-8"))
