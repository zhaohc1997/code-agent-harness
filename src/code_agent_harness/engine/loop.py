from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any

from code_agent_harness.engine.cancellation import CancellationToken
from code_agent_harness.engine.compaction import micro_compact
from code_agent_harness.engine.cycle_guard import CycleGuard
from code_agent_harness.engine.observability import Observability
from code_agent_harness.engine.state_machine import EngineStateMachine
from code_agent_harness.llm.base import LLMProvider
from code_agent_harness.storage.checkpoints import CheckpointStore
from code_agent_harness.storage.sessions import SessionStore
from code_agent_harness.tools.executor import ToolExecutor
from code_agent_harness.tools.registry import ToolRegistry
from code_agent_harness.types.engine import LLMRequest, RuntimeResult
from code_agent_harness.types.state import SessionState


@dataclass
class AgentRuntime:
    provider: LLMProvider
    system_prompt: str
    sessions: SessionStore
    checkpoints: CheckpointStore
    registry: ToolRegistry
    executor: ToolExecutor
    cancellation: CancellationToken
    state_machine: EngineStateMachine
    observability: Observability | None = None
    cycle_guard: CycleGuard = field(default_factory=CycleGuard)

    def run(self, session_id: str, user_input: str) -> RuntimeResult:
        session = self._load_or_create_session(session_id=session_id, task_goal=user_input)
        self._state_machine_transition(SessionState(session["state"]), SessionState.RUNNING)
        session["messages"].append({"role": "user", "content": user_input})
        self._save_session(session)

        try:
            while True:
                if self.cancellation.is_cancelled():
                    self._state_machine_transition(self.state_machine.state, SessionState.CANCELLED)
                    break

                session["turn_count"] += 1
                turn_count = session["turn_count"]
                tools = self.registry.list_tools()
                response = self.provider.generate(
                    LLMRequest(
                        system_prompt=self.system_prompt,
                        messages=micro_compact(session["messages"]),
                        tools=tools,
                        extra={},
                    )
                )
                session["messages"].append({"role": "assistant", "content": response.content})
                self._emit(
                    event_name="llm_turn",
                    session_id=session_id,
                    turn_id=turn_count,
                    status="executed",
                    metadata={"stop_reason": response.stop_reason, "tool_count": len(tools)},
                )

                if response.stop_reason != "tool_use":
                    self._state_machine_transition(self.state_machine.state, SessionState.COMPLETED)
                    break

                tool_calls = self._extract_tool_calls(response.content)
                tool_results = self._execute_tool_calls(tool_calls)
                session["messages"].append({"role": "user", "content": tool_results})
                self.checkpoints.save(session_id, turn_count, session)
                self._save_session(session)
        except Exception:
            self._persist_failed_state(session_id, session)
            raise

        self._save_session(session)
        return RuntimeResult(
            state=self.state_machine.state,
            output_text=self._extract_output_text(session["messages"]),
            messages=session["messages"],
        )

    def _load_or_create_session(self, session_id: str, task_goal: str) -> dict[str, Any]:
        try:
            session = self.sessions.load(session_id)
        except FileNotFoundError:
            session = {
                "session_id": session_id,
                "state": SessionState.IDLE.value,
                "messages": [],
                "task_goal": task_goal,
                "is_running": False,
                "queued_interventions": [],
                "turn_count": 0,
            }
            self.sessions.save(session)
        return session

    def _execute_tool_calls(self, tool_calls: list[dict[str, Any]]) -> list[dict[str, object]]:
        results: list[dict[str, object]] = []
        for block in tool_calls:
            tool_name = str(block["name"])
            arguments = self._coerce_arguments(block.get("arguments"))
            if self.cycle_guard.record(tool_name, arguments):
                raise RuntimeError(f"cycle detected for tool call: {tool_name}")
            execution = self.executor.execute(tool_name, arguments)
            tool_result = {
                "type": "tool_result",
                "tool_use_id": block.get("id"),
                "tool_name": tool_name,
                "content": execution.content,
            }
            if execution.external_blob_id is not None:
                tool_result["external_blob_id"] = execution.external_blob_id
            results.append(tool_result)
        return results

    def _save_session(self, session: dict[str, Any]) -> None:
        session["state"] = self.state_machine.state.value
        session["is_running"] = self.state_machine.state == SessionState.RUNNING
        self.sessions.save(session)

    def _persist_failed_state(self, session_id: str, session: dict[str, Any]) -> None:
        if self.state_machine.state != SessionState.FAILED:
            self._state_machine_transition(self.state_machine.state, SessionState.FAILED)
        self._save_session(session)
        turn_count = session.get("turn_count")
        if isinstance(turn_count, int):
            self.checkpoints.save(session_id, turn_count, session)

    def _state_machine_transition(self, current_state: SessionState, next_state: SessionState) -> None:
        self.state_machine.state = current_state
        self.state_machine.transition(next_state)

    def _emit(
        self,
        *,
        event_name: str,
        session_id: str,
        turn_id: int,
        status: str,
        metadata: dict[str, object] | None = None,
    ) -> None:
        if self.observability is None:
            return
        self.observability.emit_decision_point(
            event_name=event_name,
            session_id=session_id,
            turn_id=turn_id,
            component="engine.loop",
            status=status,
            metadata=metadata,
        )

    @staticmethod
    def _coerce_arguments(arguments: object) -> dict[str, object]:
        if isinstance(arguments, dict):
            return arguments
        return {}

    @staticmethod
    def _extract_tool_calls(content: list[object]) -> list[dict[str, Any]]:
        tool_calls = [
            block
            for block in content
            if isinstance(block, dict) and block.get("type") == "tool_call"
        ]
        if not tool_calls:
            raise ValueError("tool_use response must include at least one tool_call")
        return tool_calls

    @staticmethod
    def _extract_output_text(messages: list[dict[str, Any]]) -> str:
        for message in reversed(messages):
            if message.get("role") != "assistant":
                continue
            content = message.get("content")
            if isinstance(content, list):
                text_parts = [
                    block["text"]
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)
                ]
                if text_parts:
                    return "\n".join(text_parts)
            if isinstance(content, str):
                return content
        return ""
