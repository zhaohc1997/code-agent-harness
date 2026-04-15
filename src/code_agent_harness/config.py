from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimePaths:
    root: Path

    @property
    def sessions(self) -> Path:
        return self.root / "sessions"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def blobs(self) -> Path:
        return self.root / "blobs"

    @property
    def logs(self) -> Path:
        return self.root / "logs"
