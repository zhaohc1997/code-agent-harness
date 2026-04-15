from dataclasses import dataclass


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str | None = None
    input_schema: dict[str, object] | None = None
