from code_agent_harness.prompts.layers import PromptLayers

PROMPT_LAYER_ORDER = ("system", "scenario", "execution")
DEFAULT_ENABLED_LAYERS = frozenset(PROMPT_LAYER_ORDER)


def build_system_prompt(
    layers: PromptLayers,
    *,
    enabled_layers: set[str] | frozenset[str] | None = None,
) -> str:
    active_layers = DEFAULT_ENABLED_LAYERS if enabled_layers is None else frozenset(enabled_layers)
    unknown_layers = sorted(active_layers - DEFAULT_ENABLED_LAYERS)
    if unknown_layers:
        unknown_layers_text = ", ".join(unknown_layers)
        raise ValueError(f"unknown prompt layers: {unknown_layers_text}")

    parts = [getattr(layers, layer_name) for layer_name in PROMPT_LAYER_ORDER if layer_name in active_layers]

    return "\n\n".join(parts)
