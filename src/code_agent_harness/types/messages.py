from dataclasses import dataclass


@dataclass(frozen=True)
class ToolCallBlock:
    id: str
    name: str
    arguments: dict[str, object]
