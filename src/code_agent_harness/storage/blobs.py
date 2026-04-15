from pathlib import Path


def _require_leaf_name(value: str, label: str) -> str:
    candidate = Path(value)
    if candidate.name != value or candidate.is_absolute() or candidate.parent != Path("."):
        raise ValueError(f"{label} must be a simple leaf name, got {value!r}")
    return value


class BlobStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, blob_id: str, content: str) -> None:
        blob_id = _require_leaf_name(blob_id, "blob_id")
        path = self.root / blob_id
        path.write_text(content, encoding="utf-8")

    def load(self, blob_id: str) -> str:
        blob_id = _require_leaf_name(blob_id, "blob_id")
        path = self.root / blob_id
        return path.read_text(encoding="utf-8")
