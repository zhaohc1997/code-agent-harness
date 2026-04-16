from code_agent_harness.prompts.layers import PromptLayers

DEFAULT_ENABLED_LAYERS = frozenset({"system", "scenario", "execution"})


def build_system_prompt(
    layers: PromptLayers,
    *,
    enabled_layers: set[str] | frozenset[str] | None = None,
) -> str:
    active_layers = DEFAULT_ENABLED_LAYERS if enabled_layers is None else enabled_layers
    parts: list[str] = []

    if "system" in active_layers:
        parts.append(layers.system)
    if "scenario" in active_layers:
        parts.append(layers.scenario)
    if "execution" in active_layers:
        parts.append(layers.execution)

    return "\n\n".join(parts)
