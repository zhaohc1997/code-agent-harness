from __future__ import annotations

import json

import pytest

from code_agent_harness.cli import build_default_runtime
from code_agent_harness.config import LiveProviderConfig
from code_agent_harness.llm import LLMRequest
from code_agent_harness.llm.openai_compatible import OpenAICompatibleProvider
from code_agent_harness.types.tools import ToolDefinition


class RecordingClient:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.requests: list[dict[str, object]] = []

    def send_json(self, payload: dict[str, object]) -> dict[str, object]:
        self.requests.append(payload)
        return self.payload


def test_openai_compatible_provider_prefers_tool_compatible_model_for_tool_requests() -> None:
    client = RecordingClient(
        {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "reasoning_content": "I should inspect the file first.",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "index": 0,
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": json.dumps({"path": "calc.py"}),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 7},
        }
    )
    provider = OpenAICompatibleProvider(
        client=client,
        model="DeepSeek-V3.2",
        base_url="https://api.deepseek.com",
        api_key="secret",
    )

    response = provider.generate(
        LLMRequest(
            system_prompt="system",
            messages=[{"role": "user", "content": "read calc.py"}],
            tools=[{"name": "read_file"}],
            extra={"thinking": {"enabled": True}},
        )
    )

    assert client.requests[0]["model"] == "deepseek-chat"
    assert client.requests[0]["tools"] == [{"name": "read_file"}]
    assert "thinking" not in client.requests[0]
    assert response.stop_reason == "tool_use"
    assert response.content[0] == {"type": "reasoning", "text": "I should inspect the file first."}
    assert response.content[1]["name"] == "read_file"
    assert response.content[1]["index"] == 0
    assert response.content[1]["arguments"] == {"path": "calc.py"}


def test_openai_compatible_provider_raises_on_missing_choices() -> None:
    provider = OpenAICompatibleProvider(
        client=RecordingClient({}),
        model="DeepSeek-V3.2",
        base_url="https://api.deepseek.com",
        api_key="secret",
    )

    with pytest.raises(ValueError, match="choices"):
        provider.generate(LLMRequest(system_prompt="sys", messages=[], tools=[], extra={}))


def test_openai_compatible_provider_converts_tool_definitions_to_openai_tools() -> None:
    client = RecordingClient(
        {
            "choices": [
                {
                    "message": {"content": "done"},
                    "finish_reason": "stop",
                }
            ]
        }
    )
    provider = OpenAICompatibleProvider(
        client=client,
        model="DeepSeek-V3.2",
        base_url="https://api.deepseek.com",
        api_key="secret",
    )

    provider.generate(
        LLMRequest(
            system_prompt="system",
            messages=[],
            tools=[
                ToolDefinition(
                    name="read_file",
                    description="Read a file",
                    input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
                )
            ],
            extra={},
        )
    )

    assert client.requests[0]["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            },
        }
    ]


def test_openai_compatible_provider_uses_reasoner_for_toolless_thinking_requests() -> None:
    client = RecordingClient(
        {
            "choices": [
                {
                    "message": {"content": "done"},
                    "finish_reason": "stop",
                }
            ]
        }
    )
    provider = OpenAICompatibleProvider(
        client=client,
        model="DeepSeek-V3.2",
        base_url="https://api.deepseek.com",
        api_key="secret",
    )

    provider.generate(
        LLMRequest(
            system_prompt="system",
            messages=[{"role": "user", "content": "Summarize the repository."}],
            tools=[],
            extra={"thinking": {"enabled": True}},
        )
    )

    assert client.requests[0]["model"] == "deepseek-reasoner"
    assert client.requests[0]["thinking"] == {"type": "enabled"}


def test_openai_compatible_provider_replays_reasoning_content_in_follow_up_messages() -> None:
    client = RecordingClient(
        {
            "choices": [
                {
                    "message": {"content": "done"},
                    "finish_reason": "stop",
                }
            ]
        }
    )
    provider = OpenAICompatibleProvider(
        client=client,
        model="DeepSeek-V3.2",
        base_url="https://api.deepseek.com",
        api_key="secret",
    )

    provider.generate(
        LLMRequest(
            system_prompt="system",
            messages=[
                {
                    "role": "assistant",
                    "content": [
                        {"type": "reasoning", "text": "Need file contents before answering."},
                        {
                            "type": "tool_call",
                            "id": "call-1",
                            "index": 0,
                            "name": "read_file",
                            "arguments": {"path": "calc.py"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call-1",
                            "tool_name": "read_file",
                            "content": "def add(a, b): return a + b",
                        }
                    ],
                },
            ],
            tools=[{"name": "read_file"}],
            extra={"thinking": {"enabled": True}},
        )
    )

    assistant_message = client.requests[0]["messages"][1]
    tool_message = client.requests[0]["messages"][2]

    assert assistant_message["role"] == "assistant"
    assert assistant_message["reasoning_content"] == "Need file contents before answering."
    assert assistant_message["content"] == ""
    assert assistant_message["tool_calls"][0]["id"] == "call-1"
    assert assistant_message["tool_calls"][0]["index"] == 0
    assert "thinking" not in client.requests[0]
    assert tool_message == {
        "role": "tool",
        "tool_call_id": "call-1",
        "name": "read_file",
        "content": "def add(a, b): return a + b",
    }


def test_openai_compatible_provider_collapses_older_tool_loops_for_follow_up_requests() -> None:
    client = RecordingClient(
        {
            "choices": [
                {
                    "message": {"content": "done"},
                    "finish_reason": "stop",
                }
            ]
        }
    )
    provider = OpenAICompatibleProvider(
        client=client,
        model="DeepSeek-V3.2",
        base_url="https://api.deepseek.com",
        api_key="secret",
    )

    provider.generate(
        LLMRequest(
            system_prompt="system",
            messages=[
                {"role": "user", "content": "Find the timeout."},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "reasoning", "text": "List files first."},
                        {
                            "type": "tool_call",
                            "id": "call-1",
                            "index": 0,
                            "name": "list_files",
                            "arguments": {"path": "."},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call-1",
                            "tool_name": "list_files",
                            "content": "README.md\nservice.py",
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "reasoning", "text": "Read service.py next."},
                        {
                            "type": "tool_call",
                            "id": "call-2",
                            "index": 0,
                            "name": "read_file",
                            "arguments": {"path": "service.py"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call-2",
                            "tool_name": "read_file",
                            "content": "DEFAULT_TIMEOUT_SECONDS = 45",
                        }
                    ],
                },
            ],
            tools=[{"name": "read_file"}],
            extra={"thinking": {"enabled": True}},
        )
    )

    messages = client.requests[0]["messages"]

    assert all(message.get("tool_call_id") != "call-1" for message in messages if isinstance(message, dict))
    assert messages[-2]["tool_calls"][0]["id"] == "call-2"
    assert messages[-1]["tool_call_id"] == "call-2"
    assert messages[2]["role"] == "assistant"
    assert messages[2]["content"] != ""


def test_build_default_runtime_uses_live_provider_config(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CODE_AGENT_HARNESS_LIVE", "1")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("DEEPSEEK_MODEL", "DeepSeek-V3.2")

    runtime = build_default_runtime("code_assistant")

    assert isinstance(runtime.provider, OpenAICompatibleProvider)
    assert runtime.provider.model == "DeepSeek-V3.2"
    assert runtime.provider.base_url == "https://api.deepseek.com"
    assert runtime.provider.api_key == "secret"
    assert runtime.provider_extra == {"thinking": {"enabled": True}}


def test_live_provider_config_has_phase2_token_defaults() -> None:
    config = LiveProviderConfig(api_key="secret")

    assert config.context_window_tokens == 128_000
    assert config.output_tokens == 32_000
    assert config.max_output_tokens == 64_000
