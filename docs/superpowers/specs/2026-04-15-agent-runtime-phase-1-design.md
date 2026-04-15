# Agent Runtime Phase 1 Design

**Date:** 2026-04-15

**Status:** Drafted for review

## Goal

Build the first phase of a Python CLI agent runtime in `code-agent-harness/` that can run a single-agent ReAct loop with strict tool-call protocol, state transitions, cancellation, checkpoint persistence, context compaction, loop detection, and decision-point observability.

This phase is a runtime foundation, not a fully narrowed code-assistant product. It must be stable under tests, resumable from disk, and structured so later provider, tool, and scenario-specific work can be added without rewriting the loop.

## Scope

This phase includes:

- A Python package with a CLI entrypoint
- A single-agent runtime
- A unified LLM interface with provider adapters
- A tool registry and executor
- Filesystem-backed session, checkpoint, blob, and log storage
- Centralized type definitions for messages, tool signatures, state, and runtime payloads
- ReAct loop orchestration
- Multi-turn state handling
- Immediate cancellation checks at turn boundaries
- Strict `assistant tool_call -> user tool_result` adjacency
- Two-layer context compaction
- Loop detection for repeated tool use
- Intervention message queuing
- Decision-point observability events
- Unit tests for the runtime contract

This phase does not include:

- Multi-agent coordination
- Frontend integration
- Business-specific tool narrowing
- Database storage
- Full production concurrency management beyond single-runtime ownership rules
- Broad real-world tool coverage beyond the minimum runtime scaffolding

## Delivery Shape

The deliverable is a `Python CLI` project rooted at `code-agent-harness/`. The CLI is intentionally thin. Its job is to create or resume a session, submit user input, start the engine, and report current runtime state. The core behavior must remain inside the runtime modules so it is testable without the CLI.

The first phase optimizes for correctness of the runtime contract rather than raw task-solving capability. Real provider support is designed as an adapter boundary, but tests should primarily use a deterministic fake provider.

## Architecture

The runtime is split into three primary modules and two supporting modules:

- `engine`: owns the ReAct loop, state transitions, cancellation checks, loop detection, compaction triggers, protocol enforcement, and orchestration
- `llm`: exposes a provider-neutral interface and adapts provider-specific request and response formats
- `tools`: owns tool registration, per-turn tool reloading, execution, output limiting, and externalization of oversized outputs
- `storage`: persists sessions, checkpoints, blobs, and structured logs on the filesystem
- `types`: defines message blocks, tool calls, tool results, state enums, and shared runtime DTOs

The key architectural rule is that `engine` orchestrates but does not own provider-specific logic or storage internals. Provider translation stays in `llm`, persistence stays in `storage`, and tool execution stays in `tools`.

## Repository Layout

The initial file layout should be:

```text
code-agent-harness/
├── pyproject.toml
├── src/code_agent_harness/
│   ├── cli.py
│   ├── config.py
│   ├── engine/
│   │   ├── loop.py
│   │   ├── state_machine.py
│   │   ├── cancellation.py
│   │   ├── cycle_guard.py
│   │   ├── compaction.py
│   │   └── observability.py
│   ├── llm/
│   │   ├── base.py
│   │   ├── fake_provider.py
│   │   └── openai_compatible.py
│   ├── tools/
│   │   ├── registry.py
│   │   ├── executor.py
│   │   ├── builtins.py
│   │   └── limits.py
│   ├── storage/
│   │   ├── sessions.py
│   │   ├── checkpoints.py
│   │   ├── blobs.py
│   │   └── logs.py
│   └── types/
│       ├── messages.py
│       ├── tools.py
│       ├── state.py
│       └── engine.py
└── tests/
    ├── test_engine_loop.py
    ├── test_protocol.py
    ├── test_state_machine.py
    ├── test_cancellation.py
    ├── test_cycle_guard.py
    ├── test_compaction.py
    └── test_session_resume.py
```

Additional directories under `.agenth/` are created at runtime:

```text
.agenth/
├── sessions/
├── checkpoints/
├── blobs/
└── logs/
```

## Module Responsibilities

### `types`

`types` is the single source of truth for runtime structures. It must define:

- message roles
- assistant text blocks
- tool call blocks
- tool result blocks
- session state enum
- tool schema definitions
- runtime request and response containers

Rules:

- no filesystem logic
- no provider logic
- no orchestration logic
- no duplicated ad hoc dictionaries outside the type layer unless adapting external libraries

### `llm`

`llm` must expose one unified interface for the engine, for example:

- model id
- system prompt
- normalized messages
- tool definitions
- provider-specific `extra`

Provider adapters handle vendor-specific differences such as:

- tool-calling field names
- thinking or reasoning flags
- response block mapping
- token accounting shape

The engine must not know whether the underlying provider is OpenAI-compatible, Anthropic-like, or fake. It only consumes normalized assistant outputs.

Phase 1 will implement:

- a `FakeProvider` for deterministic testing
- one real adapter boundary, preferably `openai_compatible`, kept intentionally thin

### `tools`

`tools.registry` returns the current tool list on every turn. The engine must not cache tool definitions across turns because newly registered tools from a hook must become visible on the next turn.

`tools.executor` is responsible for:

- validating tool existence
- executing tool handlers
- applying per-tool output limits
- externalizing oversized output to blob storage
- returning normalized `tool_result` payloads

Phase 1 should support a minimal set of built-in tools sufficient to exercise the runtime contract in tests. The contract matters more than breadth.

### `storage`

`storage` is filesystem-backed in phase 1. It stores:

- session snapshots
- checkpoints after each turn
- blob payloads for oversized tool results
- structured logs for observability events

Storage must be replaceable later. The engine should depend on storage interfaces or focused classes instead of reading and writing files directly.

### `engine`

`engine` is the only module that decides when to:

- change state
- stop the loop
- compact context
- call the provider
- execute tools
- record checkpoints
- flush queued interventions

It is the runtime coordinator, not the implementation home for everything else.

## Runtime State Model

The minimum state machine for phase 1 is:

- `idle`
- `running`
- `awaiting_user_input`
- `awaiting_confirmation`
- `completed`
- `failed`
- `cancelled`

Transitions:

- new session starts as `idle`
- submitting a user task moves `idle -> running`
- requesting a confirmation moves `running -> awaiting_confirmation`
- user approval or rejection moves `awaiting_confirmation -> running`
- a response with no tool calls and a final answer moves `running -> completed`
- uncaught runtime exceptions move `running -> failed`
- cancellation at the start of a turn moves `running -> cancelled`
- resuming work after new input can move `completed -> running` or `awaiting_user_input -> running`

The state machine should reject invalid transitions instead of silently accepting them.

## ReAct Loop Contract

The main engine loop must remain structurally simple:

1. Load the current session state
2. Check cancellation signal at turn start
3. Reload the current tool list
4. Run `micro` compaction
5. Estimate context size and trigger `auto` summary if threshold is exceeded
6. Reinject system identity and task goal if compaction occurred
7. Call the LLM provider
8. Append the assistant message
9. If there are no tool calls, mark the session completed and return the assistant answer
10. Validate tool-call adjacency requirements
11. Run loop detection
12. Execute tools
13. Append a single immediate `tool_result` user message
14. Flush any queued intervention messages after tool results are recorded
15. Persist checkpoint and continue

This loop must not be complicated by provider-specific conditions or storage-specific file code.

## Message Protocol

This is a hard invariant:

- when the assistant emits tool calls, the next message must be the corresponding `tool_result` container
- no unrelated intervention or reminder message may appear between those two steps

To preserve that invariant:

- intervention messages are queued during the turn
- they are flushed only after the `tool_result` message is appended
- assistant and user message blocks must use normalized internal structures so tests can validate adjacency precisely

If the engine detects a protocol-breaking state before the provider call, it must fail loudly and log the decision point rather than silently continue.

## Cancellation Model

Cancellation is checked at the beginning of every turn. This ensures `stop` becomes effective before the next provider call and before new tool execution starts.

Phase 1 does not need full interruptibility for arbitrary long-running work in the middle of a tool call, but the design must leave room for cooperative cancellation later. The current contract is:

- cancellation requested while `running`
- next turn start sees the signal
- engine transitions to `cancelled`
- checkpoint and logs are written
- no further provider or tool actions are started

## Context Compaction

Phase 1 uses two compaction layers.

### Micro compaction

This runs before each provider call. It removes or replaces older low-value tool results while preserving:

- current task goal
- system identity
- recent critical decisions
- latest failure reasons
- the newest relevant tool results

The first compaction target is stale tool output. The goal is to shrink context without losing the chain of reasoning that explains the current situation.

### Auto summary

When estimated tokens reach 60-70% of the configured context window, the engine triggers auto-summary using a dedicated summary model or the fake summary provider in tests.

The summary output must be injected back into the context together with explicit reintroduction of:

- task goal
- system identity
- current constraints
- current progress

This is mandatory. Compaction without reintroducing identity and goal causes long-run drift.

## Tool System Rules

Phase 1 tool execution must satisfy these rules:

- tools are reloaded every turn
- tool handlers are addressed by structured name, schema, and callable
- each tool has a hard output ceiling
- oversized output is written to blob storage and replaced with a blob reference result
- tool errors are returned as structured failures, not raised raw into the transcript

Initial output ceilings:

- `read_file`: 20k characters
- `search_text`: 10k characters
- `shell`: 15k characters

The exact built-in tool set may stay small in phase 1, but the limit enforcement and blob externalization behavior must exist now because it is part of the runtime contract.

## Shell and Git Safety

Any shell or git-backed tool must use `execFile`-style subprocess execution with argument arrays, not shell string concatenation.

In Python terms, phase 1 should implement this through subprocess calls equivalent to:

- `shell=False`
- argv list input

Rules:

- reject raw shell strings for execution tools
- reject commands that require shell parsing
- perform workspace path validation before execution

This is a design requirement even if only a minimal shell-like test tool is used in phase 1.

## Loop Detection

The engine must detect repeated failed tool attempts. The minimum rule is:

- same tool name
- same normalized arguments
- repeated consecutively
- repeated up to a configured threshold

When the threshold is reached:

- the engine blocks execution
- records an observability event with `blocked`
- emits a structured tool result explaining that the strategy is repeating

The model should then be forced to choose a different approach on the next step.

## Intervention Queue

Intervention messages include:

- confirmation requests
- policy reminders
- engine-generated nudges

These messages must never be inserted between assistant tool calls and user tool results. They are collected during execution and appended only after the tool result message is safely written.

## Observability

Observability starts in phase 1 and is recorded at decision points, not only execution points.

Each event should include:

- event name
- timestamp
- session id
- turn id
- component
- status
- relevant metadata

The required statuses are:

- `executed`
- `skipped`
- `blocked`
- `error`

Missing events at a critical decision point are themselves a diagnostic problem. The design goal is to make “the path never ran” distinguishable from “the path ran and chose not to act”.

Key decision points to instrument:

- cancellation check
- compaction trigger decision
- auto-summary trigger decision
- tool registry reload
- loop detection decision
- intervention queue flush
- state transition attempt
- checkpoint write

## Session and Checkpoint Persistence

Session persistence stores the latest truth for:

- session id
- current state
- last messages
- task goal
- whether the engine is running
- queued interventions
- current turn counter

Checkpoint persistence stores per-turn snapshots and should allow:

- inspection of prior turns
- recovery after failure
- debugging of protocol or compaction issues

The backend runtime is the single source of truth. If a frontend is added later, it must fetch `last_messages` and `is_running` from storage instead of maintaining an independent state snapshot.

## CLI Behavior

The CLI should remain thin and support:

- creating a new session
- resuming an existing session
- submitting a user prompt
- requesting cancellation
- printing the latest assistant output and state

The CLI must not embed its own runtime state beyond transient argument parsing and display formatting.

## Configuration

Phase 1 should centralize configuration in `config.py` and environment variables. The minimum settings are:

- `MODEL_ID`
- `API_KEY`
- `MAX_CONTEXT_TOKENS`
- `SUMMARY_MODEL_ID`
- `WORKSPACE_ROOT`
- `TEST_COMMAND_ALLOWLIST`
- `.agenth` storage root

No business logic should hardcode these values.

## Testing Strategy

Phase 1 is test-first at the runtime contract level. The test suite must cover:

- loop completes when the assistant returns no tool calls
- multiple tool calls execute in order and results are written back correctly
- `assistant tool_call -> user tool_result` adjacency remains valid
- invalid state transitions are rejected
- cancellation changes behavior at the next turn start
- loop detection blocks repeated identical tool calls
- oversized tool output is written to blob storage and replaced by a blob reference
- micro compaction removes stale tool output while preserving required context
- auto-summary preserves task goal and system identity after compaction
- session restore can resume work from persisted storage

Tests should prefer deterministic fake providers and fake tools. Only a minimal smoke test, if any, should touch a real provider adapter.

## Risks and Non-Goals

The biggest risk in this phase is trying to make the runtime look feature-rich before the contract is stable. The design explicitly avoids broad tool coverage and multi-agent behavior to keep the first phase narrow.

Another risk is letting provider quirks leak into the engine. The engine must consume normalized messages only. If provider-specific fields spread into loop logic, later adapters will be expensive.

This phase is intentionally not optimizing for:

- benchmark accuracy on real coding tasks
- UI ergonomics
- production-grade concurrent scheduling
- provider parity across multiple vendors

## Implementation Recommendation

Implement the runtime with deterministic seams first:

- write types before orchestration logic
- write the fake provider before the real provider adapter
- write storage classes before the CLI
- exercise the loop through tests before broadening tool coverage

The order matters because phase 1 is about enforcing runtime invariants, not maximizing model capability.

## Acceptance Criteria

Phase 1 is considered complete when:

- all runtime contract unit tests pass
- the CLI can create, resume, and run a session
- the engine persists sessions and checkpoints to disk
- cancellation works at turn boundaries
- protocol adjacency is guaranteed by tests
- compaction preserves task identity and system identity
- loop detection and observability emit structured artifacts
- there are no known silent failure paths in the core decision points

## Open Items Resolved in This Spec

To avoid ambiguity, the following choices are fixed now:

- implementation language: Python
- delivery shape: CLI
- provider strategy: fake provider first, thin real adapter boundary second
- storage strategy: filesystem only
- runtime scope: single agent only
- first phase priority: correctness of runtime contract over tool breadth
