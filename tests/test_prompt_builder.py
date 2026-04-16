import pytest

from code_agent_harness.prompts.builders import build_system_prompt
from code_agent_harness.prompts.layers import PromptLayers


def test_prompt_builder_stacks_layers_in_order() -> None:
    prompt = build_system_prompt(
        PromptLayers(system="SYSTEM", scenario="SCENARIO", execution="EXECUTION")
    )

    assert prompt == "SYSTEM\n\nSCENARIO\n\nEXECUTION"


def test_prompt_builder_supports_ablation() -> None:
    prompt = build_system_prompt(
        PromptLayers(system="SYSTEM", scenario="SCENARIO", execution="EXECUTION"),
        enabled_layers={"system", "execution"},
    )

    assert prompt == "SYSTEM\n\nEXECUTION"


def test_prompt_builder_rejects_unknown_layer_names() -> None:
    with pytest.raises(ValueError, match="unknown prompt layers: typo"):
        build_system_prompt(
            PromptLayers(system="SYSTEM", scenario="SCENARIO", execution="EXECUTION"),
            enabled_layers={"system", "typo"},
        )
