from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from code_agent_harness.storage.blobs import BlobStore

from code_agent_harness.tools.limits import get_tool_limit
from code_agent_harness.tools.registry import ToolRegistry


@dataclass(frozen=True)
class ToolExecutionResult:
    content: str
    external_blob_id: str | None = None


class ToolExecutor:
    def __init__(self, registry: ToolRegistry, blob_store_root: Path | str) -> None:
        self._registry = registry
        self._blob_store = BlobStore(Path(blob_store_root) / "blobs")

    def _apply_limit(self, tool_name: str, content: str) -> ToolExecutionResult:
        limit = get_tool_limit(tool_name)
        if len(content) <= limit:
            return ToolExecutionResult(content=content)

        blob_id = self._save_external_blob(tool_name, content)
        return ToolExecutionResult(
            content=f"[externalized:{blob_id}]",
            external_blob_id=blob_id,
        )

    def _save_external_blob(self, tool_name: str, content: str) -> str:
        digest = sha256(content.encode("utf-8")).hexdigest()[:16]
        blob_id = f"{tool_name}-{digest}"
        self._blob_store.save(blob_id, content)
        return blob_id
