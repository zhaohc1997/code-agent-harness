from __future__ import annotations

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

    @property
    def cancellations(self) -> Path:
        return self.root / "cancellations"


@dataclass(frozen=True)
class RuntimeConfig:
    root: Path
    system_prompt: str = "You are code-agent-harness."
    context_window_tokens: int = 12000
    auto_summary_trigger_ratio: float = 0.65
    auto_summary_keep_recent: int = 4

    @property
    def paths(self) -> RuntimePaths:
        return RuntimePaths(self.root)
