from pathlib import Path


class BlobStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.blobs_dir = self.root / "blobs"
        self.blobs_dir.mkdir(parents=True, exist_ok=True)

    def save(self, blob_id: str, content: str) -> None:
        path = self.blobs_dir / blob_id
        path.write_text(content, encoding="utf-8")

    def load(self, blob_id: str) -> str:
        path = self.blobs_dir / blob_id
        return path.read_text(encoding="utf-8")
