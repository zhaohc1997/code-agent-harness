import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from code_agent_harness.storage.blobs import BlobStore

from code_agent_harness.tools.limits import get_tool_limit
from code_agent_harness.tools.registry import RegisteredTool, ToolRegistry


@dataclass(frozen=True)
class ToolExecutionResult:
    content: str
    external_blob_id: str | None = None


class ToolExecutor:
    def __init__(self, registry: ToolRegistry, blob_store_root: Path | str) -> None:
        self._registry = registry
        self._blob_store = BlobStore(Path(blob_store_root) / "blobs")

    def execute(self, tool_name: str, arguments: dict[str, object]) -> ToolExecutionResult:
        registered_tool = self._registry.resolve_tool(tool_name)
        return self.execute_registered(registered_tool, arguments)

    def execute_registered(
        self,
        registered_tool: RegisteredTool,
        arguments: dict[str, object],
    ) -> ToolExecutionResult:
        output = registered_tool.handler(arguments)
        return self._apply_limit(registered_tool.definition.name, self._normalize_output(output))

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

    def _normalize_output(self, output: object) -> str:
        if isinstance(output, str):
            return output
        try:
            return json.dumps(output, sort_keys=True, ensure_ascii=False)
        except TypeError:
            return str(output)
