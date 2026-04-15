from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any

from code_agent_harness.engine.cancellation import CancellationToken
from code_agent_harness.engine.compaction import auto_compact
from code_agent_harness.engine.cycle_guard import CycleGuard
from code_agent_harness.engine.observability import Observability
from code_agent_harness.engine.state_machine import EngineStateMachine
from code_agent_harness.llm.base import LLMProvider
from code_agent_harness.storage.checkpoints import CheckpointStore
from code_agent_harness.storage.sessions import SessionStore
from code_agent_harness.tools.executor import ToolExecutor
from code_agent_harness.tools.registry import ToolRegistry, ToolRegistrySnapshot
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
    context_window_tokens: int = 12000
    auto_summary_trigger_ratio: float = 0.65
    auto_summary_keep_recent: int = 4

    def run(self, session_id: str, user_input: str) -> RuntimeResult:
        self.cancellation.bind(session_id)
        self.cycle_guard.reset()
        session = self._load_or_create_session(session_id=session_id, task_goal=user_input)
        resume_session = self._enter_running_state(session_id, SessionState(session["state"]))
        if not resume_session:
            session["messages"].append({"role": "user", "content": user_input})
        self._save_session(session)

        try:
            while True:
                cancelled = self.cancellation.is_cancelled()
                self._emit(
                    event_name="cancellation_check",
                    session_id=session_id,
                    turn_id=int(session.get("turn_count", 0)),
                    status="executed",
                    metadata={"cancelled": cancelled},
                )
                if cancelled:
                    self.cancellation.acknowledge()
                    self._state_machine_transition(
                        session_id=session_id,
                        turn_id=int(session.get("turn_count", 0)),
                        next_state=SessionState.CANCELLED,
                    )
                    break

                session["turn_count"] += 1
                turn_count = session["turn_count"]
                tool_snapshot = self.registry.snapshot()
                tools = tool_snapshot.list_tools()
                self._emit(
                    event_name="tool_registry_reload",
                    session_id=session_id,
                    turn_id=turn_count,
                    status="executed",
                    metadata={"tool_count": len(tools)},
                )
                compacted = auto_compact(
                    session["messages"],
                    system_prompt=self.system_prompt,
                    task_goal=str(session.get("task_goal", user_input)),
                    max_tokens=self.context_window_tokens,
                    trigger_ratio=self.auto_summary_trigger_ratio,
                    keep_recent=self.auto_summary_keep_recent,
                )
                self._emit(
                    event_name="context_micro_compaction",
                    session_id=session_id,
                    turn_id=turn_count,
                    status="executed" if compacted.applied_micro_compaction else "skipped",
                    metadata={
                        "input_tokens": compacted.input_token_estimate,
                        "output_tokens": compacted.output_token_estimate,
                    },
                )
                self._emit(
                    event_name="context_auto_summary",
                    session_id=session_id,
                    turn_id=turn_count,
                    status="executed" if compacted.applied_auto_summary else "skipped",
                    metadata={
                        "threshold_tokens": int(self.context_window_tokens * self.auto_summary_trigger_ratio),
                        "output_tokens": compacted.output_token_estimate,
                    },
                )
                request = LLMRequest(
                    system_prompt=self.system_prompt,
                    messages=compacted.messages,
                    tools=tools,
                    extra={},
                )
                try:
                    response = self.provider.generate(request)
                except Exception as exc:
                    self._emit(
                        event_name="llm_turn",
                        session_id=session_id,
                        turn_id=turn_count,
                        status="error",
                        metadata={"error": str(exc)},
                    )
                    raise
                session["messages"].append({"role": "assistant", "content": response.content})
                self._emit(
                    event_name="llm_turn",
                    session_id=session_id,
                    turn_id=turn_count,
                    status="executed",
                    metadata={"stop_reason": response.stop_reason, "tool_count": len(tools)},
                )

                if response.stop_reason != "tool_use":
                    self._state_machine_transition(
                        session_id=session_id,
                        turn_id=turn_count,
                        next_state=SessionState.COMPLETED,
                    )
                    break

                tool_calls = self._extract_tool_calls(response.content)
                tool_results = self._execute_tool_calls(
                    session_id=session_id,
                    turn_count=turn_count,
                    tool_calls=tool_calls,
                    tool_snapshot=tool_snapshot,
                )
                session["messages"].append({"role": "user", "content": tool_results})
                self._checkpoint_session(session_id, session, turn_count=turn_count)
        except Exception:
            self._persist_failed_state(session_id, session)
            raise

        self._checkpoint_session(session_id, session, turn_count=int(session.get("turn_count", 0)))
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

    def _execute_tool_calls(
        self,
        *,
        session_id: str,
        turn_count: int,
        tool_calls: list[dict[str, Any]],
        tool_snapshot: ToolRegistrySnapshot,
    ) -> list[dict[str, object]]:
        results: list[dict[str, object]] = []
        for block in tool_calls:
            tool_name = self._require_tool_name(block.get("name"))
            arguments = self._require_arguments(block.get("arguments"))
            if self.cycle_guard.record(tool_name, arguments):
                self._emit(
                    event_name="loop_detection",
                    session_id=session_id,
                    turn_id=turn_count,
                    status="blocked",
                    metadata={"tool_name": tool_name, "tool_use_id": str(block.get("id"))},
                )
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.get("id"),
                        "tool_name": tool_name,
                        "content": {
                            "status": "blocked",
                            "reason": "cycle_detected",
                            "message": "Repeated tool call was blocked by the cycle guard.",
                        },
                        "is_error": True,
                    }
                )
                continue
            self._emit(
                event_name="loop_detection",
                session_id=session_id,
                turn_id=turn_count,
                status="executed",
                metadata={"tool_name": tool_name, "tool_use_id": str(block.get("id"))},
            )
            execution = self.executor.execute_registered(tool_snapshot.resolve_tool(tool_name), arguments)
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

    def _checkpoint_session(self, session_id: str, session: dict[str, Any], *, turn_count: int) -> None:
        self._save_session(session)
        persisted_turn_count = session.get("turn_count")
        if isinstance(persisted_turn_count, int):
            self.checkpoints.save(session_id, turn_count, session)
            self._emit(
                event_name="checkpoint_write",
                session_id=session_id,
                turn_id=turn_count,
                status="executed",
                metadata={"checkpoint_turn": turn_count},
            )
            return
        self._emit(
            event_name="checkpoint_write",
            session_id=session_id,
            turn_id=turn_count,
            status="skipped",
            metadata={"reason": "missing_turn_count"},
        )

    def _persist_failed_state(self, session_id: str, session: dict[str, Any]) -> None:
        if self.state_machine.state != SessionState.FAILED:
            self._state_machine_transition(
                session_id=session_id,
                turn_id=int(session.get("turn_count", 0)),
                next_state=SessionState.FAILED,
            )
        self._checkpoint_session(session_id, session, turn_count=int(session.get("turn_count", 0)))

    def _enter_running_state(self, session_id: str, current_state: SessionState) -> bool:
        self.state_machine.state = current_state
        if current_state == SessionState.RUNNING:
            self._emit(
                event_name="state_transition",
                session_id=session_id,
                turn_id=0,
                status="skipped",
                metadata={"from": current_state.value, "to": current_state.value, "reason": "resume_running_session"},
            )
            return True
        previous_state = self.state_machine.state
        self.state_machine.transition(SessionState.RUNNING)
        self._emit(
            event_name="state_transition",
            session_id=session_id,
            turn_id=0,
            status="executed",
            metadata={"from": previous_state.value, "to": SessionState.RUNNING.value},
        )
        return False

    def _state_machine_transition(self, *, session_id: str, turn_id: int, next_state: SessionState) -> None:
        current_state = self.state_machine.state
        self.state_machine.transition(next_state)
        self._emit(
            event_name="state_transition",
            session_id=session_id,
            turn_id=turn_id,
            status="executed",
            metadata={"from": current_state.value, "to": next_state.value},
        )

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
        raise ValueError("tool_call arguments must be an object")

    @staticmethod
    def _require_arguments(arguments: object) -> dict[str, object]:
        return AgentRuntime._coerce_arguments(arguments)

    @staticmethod
    def _require_tool_name(name: object) -> str:
        if isinstance(name, str) and name:
            return name
        raise ValueError("tool_call name must be a non-empty string")

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
