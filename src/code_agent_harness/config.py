from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os


@dataclass(frozen=True)
class LiveProviderConfig:
    api_key: str
    base_url: str = "https://api.deepseek.com"
    model: str = "DeepSeek-V3.2"
    reasoning_enabled: bool = True
    context_window_tokens: int = 128_000
    output_tokens: int = 32_000
    max_output_tokens: int = 64_000


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
    workspace_root: Path = field(default_factory=Path.cwd)
    profile: str = "code_assistant"
    live: bool = False
    live_provider: LiveProviderConfig | None = None
    system_prompt: str = "You are code-agent-harness."
    context_window_tokens: int = 12000
    auto_summary_trigger_ratio: float = 0.65
    auto_summary_keep_recent: int = 4

    @classmethod
    def from_env(cls, root: Path, workspace_root: Path, profile: str) -> "RuntimeConfig":
        live = os.getenv("CODE_AGENT_HARNESS_LIVE") == "1"
        live_provider: LiveProviderConfig | None = None
        if live:
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                raise ValueError("DEEPSEEK_API_KEY is required when CODE_AGENT_HARNESS_LIVE=1")
            live_provider = LiveProviderConfig(
                api_key=api_key,
                base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                model=os.getenv("DEEPSEEK_MODEL", "DeepSeek-V3.2"),
                reasoning_enabled=os.getenv("DEEPSEEK_REASONING") != "0",
            )
        return cls(
            root=root,
            workspace_root=workspace_root,
            profile=profile,
            live=live,
            live_provider=live_provider,
            context_window_tokens=(
                live_provider.context_window_tokens if live_provider is not None else cls.context_window_tokens
            ),
        )

    @property
    def paths(self) -> RuntimePaths:
        return RuntimePaths(self.root)
