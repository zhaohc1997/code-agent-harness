import io

import pytest

from code_agent_harness import __version__
from code_agent_harness.cli import build_parser, main
import code_agent_harness.llm as llm
from code_agent_harness.types.state import SessionState
from code_agent_harness.types.engine import RuntimeResult
from code_agent_harness.types.tools import ToolDefinition


def test_package_exposes_version() -> None:
    assert __version__ == "0.1.0"


def test_cli_parser_accepts_run_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["run", "--session", "s1", "--input", "hello"])
    assert args.command == "run"
    assert args.session == "s1"


def test_llm_package_exports_are_usable() -> None:
    request = llm.LLMRequest(system_prompt="sys", messages=[], tools=[], extra={})
    provider = llm.FakeProvider(script=[{"stop_reason": "end_turn", "content": []}])
    response = provider.generate(request)
    assert isinstance(request, llm.LLMRequest)
    assert isinstance(response, llm.LLMResponse)


def test_fake_provider_preserves_scripted_usage() -> None:
    provider = llm.FakeProvider(
        script=[
            {
                "stop_reason": "end_turn",
                "content": [],
                "usage": {"input_tokens": 3, "output_tokens": 5},
            }
        ]
    )
    request = llm.LLMRequest(system_prompt="sys", messages=[], tools=[], extra={})
    response = provider.generate(request)
    assert response.stop_reason == "end_turn"
    assert response.usage == {"input_tokens": 3, "output_tokens": 5}


def test_fake_provider_script_exhaustion_raises_targeted_error() -> None:
    provider = llm.FakeProvider(script=[])
    request = llm.LLMRequest(system_prompt="sys", messages=[], tools=[], extra={})
    with pytest.raises(llm.FakeProviderScriptExhausted):
        provider.generate(request)


def test_engine_completes_without_tool_calls(runtime_dependencies) -> None:
    from code_agent_harness.engine.loop import AgentRuntime

    provider = llm.FakeProvider(
        script=[
            {
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "done"}],
            }
        ]
    )
    runtime = AgentRuntime(provider=provider, **runtime_dependencies)

    result = runtime.run(session_id="s1", user_input="Summarize status")

    assert result.state == SessionState.COMPLETED
    assert result.output_text == "done"


def test_engine_checkpoints_successful_terminal_turn(runtime_dependencies) -> None:
    from code_agent_harness.engine.loop import AgentRuntime

    provider = llm.FakeProvider(
        script=[
            {
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "done"}],
            }
        ]
    )
    runtime = AgentRuntime(provider=provider, **runtime_dependencies)

    runtime.run(session_id="s1", user_input="Summarize status")

    checkpoint = runtime_dependencies["checkpoints"].load("s1", 1)
    assert checkpoint["state"] == SessionState.COMPLETED.value
    assert checkpoint["messages"][-1] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "done"}],
    }


def test_engine_checkpoints_clean_cancellation(runtime_dependencies) -> None:
    from code_agent_harness.engine.loop import AgentRuntime

    runtime_dependencies["cancellation"].cancel()
    provider = llm.FakeProvider(
        script=[
            {
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "should not run"}],
            }
        ]
    )
    runtime = AgentRuntime(provider=provider, **runtime_dependencies)

    result = runtime.run(session_id="s1", user_input="Stop now")

    assert result.state == SessionState.CANCELLED
    checkpoint = runtime_dependencies["checkpoints"].load("s1", 0)
    assert checkpoint["state"] == SessionState.CANCELLED.value
    assert checkpoint["messages"] == [{"role": "user", "content": "Stop now"}]


def test_engine_fails_fast_on_repeated_tool_use_loop(tmp_path) -> None:
    from code_agent_harness.engine.loop import AgentRuntime
    from conftest import _build_runtime_dependencies

    runtime_dependencies = _build_runtime_dependencies(tmp_path)
    provider = llm.FakeProvider(
        script=[
            {
                "stop_reason": "tool_use",
                "content": [
                    {
                        "type": "tool_call",
                        "id": "tool-1",
                        "name": "read_file",
                        "arguments": {"path": "a.py"},
                    }
                ],
            },
            {
                "stop_reason": "tool_use",
                "content": [
                    {
                        "type": "tool_call",
                        "id": "tool-2",
                        "name": "read_file",
                        "arguments": {"path": "a.py"},
                    }
                ],
            },
            {
                "stop_reason": "tool_use",
                "content": [
                    {
                        "type": "tool_call",
                        "id": "tool-3",
                        "name": "read_file",
                        "arguments": {"path": "a.py"},
                    }
                ],
            },
        ]
    )
    runtime = AgentRuntime(provider=provider, **runtime_dependencies)

    with pytest.raises(RuntimeError, match="cycle detected"):
        runtime.run(session_id="s1", user_input="Read a file repeatedly")

    session = runtime_dependencies["sessions"].load("s1")
    assert session["state"] == SessionState.FAILED.value


def test_engine_persists_failed_state_when_provider_raises(tmp_path) -> None:
    from code_agent_harness.engine.loop import AgentRuntime
    from conftest import _build_runtime_dependencies

    class ExplodingProvider:
        def generate(self, request):
            raise RuntimeError("provider blew up")

    runtime_dependencies = _build_runtime_dependencies(tmp_path)
    runtime = AgentRuntime(provider=ExplodingProvider(), **runtime_dependencies)

    with pytest.raises(RuntimeError, match="provider blew up"):
        runtime.run(session_id="s1", user_input="Summarize status")

    session = runtime_dependencies["sessions"].load("s1")
    assert session["state"] == SessionState.FAILED.value
    assert session["messages"] == [{"role": "user", "content": "Summarize status"}]


def test_engine_persists_failed_state_when_tool_execution_raises(tmp_path) -> None:
    from code_agent_harness.engine.loop import AgentRuntime
    from conftest import _build_runtime_dependencies
    from code_agent_harness.tools.registry import RegisteredTool, ToolRegistry

    registry = ToolRegistry(
        lambda: [
            RegisteredTool(
                definition=ToolDefinition(name="read_file"),
                handler=lambda arguments: (_ for _ in ()).throw(RuntimeError("tool blew up")),
            )
        ]
    )
    runtime_dependencies = _build_runtime_dependencies(tmp_path, registry=registry)
    provider = llm.FakeProvider(
        script=[
            {
                "stop_reason": "tool_use",
                "content": [
                    {
                        "type": "tool_call",
                        "id": "tool-1",
                        "name": "read_file",
                        "arguments": {"path": "a.py"},
                    }
                ],
            }
        ]
    )
    runtime = AgentRuntime(provider=provider, **runtime_dependencies)

    with pytest.raises(RuntimeError, match="tool blew up"):
        runtime.run(session_id="s1", user_input="Read a file")

    session = runtime_dependencies["sessions"].load("s1")
    assert session["state"] == SessionState.FAILED.value
    assert session["messages"][-1]["role"] == "assistant"


def test_cli_main_runs_runtime_and_reports_result() -> None:
    calls: list[tuple[str, str]] = []

    class FakeRuntime:
        def run(self, session_id: str, user_input: str) -> RuntimeResult:
            calls.append((session_id, user_input))
            return RuntimeResult(
                state=SessionState.COMPLETED,
                output_text="done",
                messages=[{"role": "assistant", "content": [{"type": "text", "text": "done"}]}],
            )

    stdout = io.StringIO()
    stderr = io.StringIO()

    exit_code = main(
        ["run", "--session", "s1", "--input", "hello"],
        runtime_factory=lambda: FakeRuntime(),
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 0
    assert calls == [("s1", "hello")]
    assert stderr.getvalue() == ""
    assert "state=completed" in stdout.getvalue()
    assert "done" in stdout.getvalue()
