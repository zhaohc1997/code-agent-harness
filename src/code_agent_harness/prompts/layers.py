from dataclasses import dataclass


@dataclass(frozen=True)
class PromptLayers:
    system: str
    scenario: str
    execution: str
