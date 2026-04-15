# Code Assistant Phase 2 Design

**Date:** 2026-04-16

**Status:** Drafted for review

## Goal

Build phase 2 of `code-agent-harness/` by narrowing the phase 1 general-purpose runtime into a single-scenario `code_assistant` system for one local repository.

This phase must turn the existing runtime into something that is measurable and optimizable for a concrete business-like workflow:

- bugfix tasks
- small feature tasks
- read-only analysis tasks

The result is not a more general runtime. It is a sharper `code_assistant` profile with evaluation fixtures, narrowed tools, layered prompts, policy interception, live provider support, and ablation-ready scoring.

## Scope

Phase 2 includes:

- one built-in `code_assistant` profile
- a narrowed tool surface for the code-assistant scenario
- explicit disabled-tool handling
- strengthened tool descriptions for the active business tools
- layered prompts for system, scenario, and execution constraints
- a policy engine for pre-tool reminders, interception, blocking, and confirmation escalation
- an evaluation harness with at least 3 Python fixture repositories using `pytest`
- an initial evaluation set of 3-5 tasks distributed across those repositories
- task definitions that encode expected tool usage, expected repository state, and expected answer content
- dimensioned scoring instead of a single pass/fail result
- ablation toggles for major mechanisms
- one real OpenAI-compatible live provider integration for DeepSeek, gated behind explicit live configuration

Phase 2 does not include:

- multi-scenario framework generalization
- multi-agent orchestration
- frontend integration
- database-backed persistence
- cross-language fixture repositories
- broad new general-purpose runtime features unrelated to code-assistant narrowing

## Product Shape

The product remains a Python CLI project rooted at `code-agent-harness/`.

Phase 1 already provided the stable runtime foundation:

- ReAct engine loop
- cancellation and checkpointing
- tool protocol enforcement
- context compaction
- loop detection
- observability

Phase 2 adds a scenario layer around that runtime. The runtime should stay largely stable. The scenario-specific behavior should be introduced through profile configuration, prompts, policy decisions, provider configuration, and evaluation tooling.

The intended shape is:

- the CLI can select a `code_assistant` profile
- the profile can assemble a narrowed execution context
- the evaluation runner can run repeatable tasks against fixture repositories
- the same runtime can be exercised offline with fake providers and online with an explicit live provider

## User Scenario

The single supported scenario in phase 2 is a one-repository code assistant operating in a local Python repository.

The phase 2 assistant must optimize for this workflow:

1. understand a user task
2. inspect repository files and symbols
3. choose the correct narrowed tools
4. avoid unnecessary or unsafe actions
5. propose or apply targeted code changes
6. run the smallest relevant tests first
7. continue until the task is complete or blocked
8. report an answer that clearly states what changed, what passed, and what remains

The scenario explicitly covers three task classes:

- `bugfix`
- `feature`
- `analysis`

These classes must all be present in the phase 2 evaluation set.

## Acceptance Criteria

Phase 2 is complete only when all of the following are true:

- `code_assistant` can be selected explicitly from the CLI or equivalent runtime entrypoint
- the runtime uses a narrowed active tool set for this profile rather than exposing the full general-purpose surface
- disabled tools are represented explicitly and are not merely "discouraged"
- tool descriptions for active business tools encode required field names, types, and enumerations clearly enough to reduce argument guessing
- layered prompt construction is in place and can be toggled in evaluation
- policy interception can block, remind, or require confirmation before selected tool actions
- at least 3 Python `pytest` fixture repositories are runnable through the evaluation harness and cover bugfix, feature, and analysis work
- an initial evaluation set of 3-5 tasks runs across those repositories
- evaluation results are dimensioned across tool choice, tool arguments, repository state, tests, and final response content
- at least 3 ablation toggles are available and can be run against the same task set
- the DeepSeek live provider can run behind an explicit live flag without affecting default offline tests
- phase 1 behavior does not regress

## Architecture

Phase 2 keeps the phase 1 layering:

- `engine`: orchestration and runtime state
- `llm`: provider abstraction and adapter logic
- `tools`: tool registration and execution
- `storage`: sessions, checkpoints, blobs, and logs
- `types`: shared DTOs and state

Phase 2 adds a code-assistant scenario layer:

- `profiles`: scenario assembly and runtime narrowing
- `prompts`: layered prompt definitions and composition
- `policies`: decision rules applied before tool execution
- `evals`: scenario evaluation, scoring, and ablation workflows

The main architectural rule for phase 2 is:

**Do not grow `engine` into a business-logic bucket.**

The engine should continue to run a generic loop. Business-specific narrowing belongs in the profile, prompts, policy engine, and eval harness.

## Repository Additions

The expected additions are:

```text
code-agent-harness/
├── docs/
│   └── superpowers/
│       └── specs/
│           └── 2026-04-16-code-assistant-phase-2-design.md
├── src/code_agent_harness/
│   ├── cli.py
│   ├── config.py
│   ├── profiles/
│   │   └── code_assistant.py
│   ├── prompts/
│   │   ├── layers.py
│   │   └── builders.py
│   ├── policies/
│   │   ├── engine.py
│   │   └── code_assistant.py
│   ├── evals/
│   │   ├── runner.py
│   │   ├── scoring.py
│   │   └── tasks.py
│   └── llm/
│       └── openai_compatible.py
└── tests/
    ├── test_code_assistant_profile.py
    └── evals/
        ├── fixtures/
        │   ├── bugfix_repo/
        │   ├── feature_repo/
        │   └── analysis_repo/
        ├── test_eval_runner.py
        ├── test_eval_scoring.py
        └── test_eval_tasks.py
```

Additional fixture repositories can be added later, but phase 2 should start with the minimum set needed to cover the three task classes.

## Module Responsibilities

### `profiles/code_assistant.py`

This module owns the single phase 2 scenario profile. It should define:

- active tools for the scenario
- explicit `disabled_tools`
- default prompt layer inputs
- default policy configuration
- default evaluation switches
- whether reasoning mode is enabled for the live provider

It is acceptable for phase 2 to hard-code one profile rather than building a general profile registry. The goal is scenario depth, not scenario breadth.

### `prompts/layers.py`

This module should define the content units for the three prompt layers:

- `system`: role, capability boundary, and identity
- `scenario`: code-assistant rules, prohibited behavior, and scenario constraints
- `execution`: output format, confirmation expectations, language consistency, and result reporting requirements

Each layer should remain independently testable and inspectable. The design goal is to make prompt changes local rather than editing one giant string.

### `prompts/builders.py`

This module should compose the prompt layers into a final runtime prompt. It should also support ablation by allowing one or more layers to be omitted through configuration.

The builder must be deterministic so that prompt assembly can be asserted in unit tests and evaluation runs.

### `policies/engine.py`

This module should define the generic policy decision surface applied before tool execution. It should support outcomes such as:

- execute
- remind
- require_confirmation
- block

The policy engine should run at the decision point, not after the side effect has already happened.

### `policies/code_assistant.py`

This module should encode code-assistant-specific rules such as:

- prefer targeted tests before broad test runs
- require confirmation for destructive or high-risk patch behavior
- block or escalate dangerous git write operations
- prevent disabled tools from being used

These rules should produce structured results so they can be evaluated and logged.

### `evals/tasks.py`

This module should define the task schema for phase 2 evaluation. Each task must include:

- task id
- task class
- fixture repository identifier
- user input
- active and disabled tool configuration
- expected tool call sequence or allowed sequence constraints
- expected repository-state assertions
- expected final response assertions
- whether the task is eligible for live runs

The task definition format must be rich enough to evolve alongside the codebase and evaluation rules.

### `evals/runner.py`

This module should orchestrate evaluation runs by:

1. copying a fixture repository into a temporary work directory
2. creating the runtime/profile configuration for the task
3. executing the task with either a fake or live provider
4. collecting runtime transcript, tool calls, test output, and final repository state
5. passing collected artifacts into scoring

The runner should support single-task runs, full-suite runs, and ablation runs.

### `evals/scoring.py`

This module should score tasks by dimension rather than collapsing everything into one boolean.

Required dimensions:

- tool choice correctness
- tool argument correctness
- final repository state correctness
- test outcome correctness
- final response completeness
- workflow correctness when a multi-step process is required

This scoring model is critical. A task should be able to fail because the repository was fixed but the assistant did not report the required facts, or because the answer looked plausible while the repository remained wrong.

### `llm/openai_compatible.py`

This module should evolve from a stub into a real OpenAI-compatible provider adapter for DeepSeek. It must:

- read live configuration from environment-driven runtime config
- map normalized messages and tools to the provider request format
- map provider responses back into normalized content blocks and stop reasons
- pass provider-specific reasoning or thinking configuration through `extra`
- report usage when present
- fail clearly on malformed provider responses

The phase 2 adapter is intentionally single-provider in practice, but it must keep the phase 1 provider boundary intact.

## Narrowed Tool Strategy

The phase 1 runtime is intentionally general enough to grow. Phase 2 must do the opposite for the active scenario.

The `code_assistant` profile should expose only the tools required for local one-repository development work. The active set should stay within the narrow band suggested by the guidance:

- enough to complete the tasks
- small enough to reduce tool-selection entropy

The exact tool list can evolve during implementation, but the design target is approximately 8-15 active tools.

All out-of-scope tools must be represented as explicitly disabled rather than simply omitted from documentation. The profile should be able to surface that a tool is disabled for this scenario, and policies should block attempts to use disabled tools.

## Tool Description Strategy

For the active tools in the `code_assistant` profile, descriptions should be strengthened with concrete schema guidance:

- exact field names
- required vs optional fields
- field types
- enumerated values when relevant
- short usage constraints

The goal is to reduce one of the highest-ROI failure modes in tool use: model-side argument guessing.

This applies most strongly to tools whose parameters are easy to guess wrong, such as:

- paths
- line ranges
- patch payloads
- command scopes
- confirmation request structures

## Prompt Layering

Phase 2 prompt construction should be divided into three layers:

### System Layer

Defines:

- assistant identity
- high-level capability boundary
- the fact that it is a code assistant for one local repository

### Scenario Layer

Defines:

- allowed and disallowed behavior for the code-assistant workflow
- preference for targeted investigation
- safety constraints
- operational rules such as minimizing unnecessary writes or test scope

### Execution Layer

Defines:

- expected answer structure
- confirmation behavior
- same-language reply requirement when enabled
- requirement to report concrete test results and code-change outcomes

The layers must be separable so that ablations can disable one mechanism at a time.

## Policy Engine

Business and workflow rules must not rely only on the prompt. Phase 2 should introduce a policy engine that evaluates tool intentions before execution.

For code-assistant behavior, the policy engine should support at least these interventions:

- **remind** before broad test execution to encourage minimal relevant tests
- **require confirmation** before destructive or high-risk patch behavior
- **block** disallowed tools or dangerous write-oriented git actions
- **pass through** safe read/search/test behavior in the default practical mode

Policy outcomes must become structured runtime artifacts so that:

- the model can recover from them
- the logs can record them
- the eval runner can score them

These interventions should respect the phase 1 protocol constraints. Structured reminders or blocked results must not break the required adjacency between assistant tool calls and user tool results.

## Runtime Flow

The phase 2 runtime flow should be:

1. CLI or caller selects the `code_assistant` profile
2. profile assembles active tools, disabled tools, prompt-layer inputs, policy configuration, and provider configuration
3. prompt builder composes the final prompt
4. engine begins a turn and reloads the current active tools for the profile
5. LLM returns either a direct answer or a tool call
6. before tool execution, policy engine evaluates the intended action
7. policy outcome is applied:
   - execute directly
   - return a structured reminder artifact
   - transition to confirmation
   - return a structured blocked artifact
8. if execution happens, tool results are written back in strict protocol order
9. engine continues until completion, failure, cancellation, or confirmation wait
10. final response is checked against execution-layer expectations in eval runs

The runtime should continue to respect phase 1 constraints:

- cancellation checked at turn boundaries
- tool registry reloaded per turn
- `tool_call -> tool_result` adjacency preserved
- compaction and observability retained

## Evaluation Strategy

Evaluation is the center of phase 2, not a final afterthought.

The initial evaluation set should contain 3-5 tasks across at least 3 small Python fixture repositories. Each repository should be self-contained and deterministic. Each task should have an automated pass/fail basis and dimensioned scoring.

The first three fixtures should cover:

- `bugfix_repo`
- `feature_repo`
- `analysis_repo`

Additional tasks can be added later, but phase 2 only needs enough coverage to establish useful optimization pressure.

Each evaluation task should define:

- task input
- fixture repository
- expected relevant tool calls
- expected repository-state outcome
- expected tests to pass or expected read-only state to remain unchanged
- expected response content requirements

The evaluation harness should copy fixtures into temporary directories so runs are isolated and repeatable.

## Ablation Strategy

Phase 2 must support ablation because not every plausible mechanism helps.

At minimum, the system should support ablation toggles for:

- tool narrowing
- layered prompts
- policy engine
- reasoning mode
- strengthened tool descriptions

The evaluation runner should be able to re-run the same tasks with one mechanism disabled and produce comparable scores. This is how phase 2 will decide whether a mechanism provides real value rather than only sounding correct.

## Scoring Model

The scoring model must be dimensioned. Required dimensions are:

- `tool_choice`
- `tool_arguments`
- `repository_state`
- `tests`
- `response_content`
- `workflow`

The model may aggregate these into a summary score, but raw dimensions must remain visible. This prevents false positives such as:

- repository fixed but required response facts missing
- plausible response but repository still incorrect
- correct files modified but wrong workflow used

## Provider Integration

Phase 2 should integrate one real live provider using DeepSeek's OpenAI-compatible endpoint.

Runtime configuration should come from environment variables rather than committed secrets. The expected inputs are:

- `DEEPSEEK_API_KEY`
- `DEEPSEEK_BASE_URL`
- `DEEPSEEK_MODEL`
- `CODE_AGENT_HARNESS_LIVE`

The design assumptions for the live provider are:

- model: `DeepSeek-V3.2`
- base URL: `https://api.deepseek.com`
- reasoning or thinking mode should be enabled through provider-specific `extra`
- context length target: `128K`
- output default target: `32K`
- output maximum target: `64K`

Live runs must be explicit opt-in. Offline tests and default local development must remain deterministic and not require network access.

## Configuration

Phase 2 should expand runtime configuration so that both offline and live behavior can be selected predictably.

Configuration should cover at least:

- runtime storage root
- selected profile
- selected provider mode (`fake` vs `live`)
- live provider credentials and base URL
- live model id
- reasoning-mode toggle
- ablation flags
- evaluation selection

Configuration should remain centralized rather than scattering environment access across the codebase.

## Error Handling

Phase 2 should separate two failure classes:

### Runtime Failures

Examples:

- provider timeout
- invalid provider response
- tool execution crash
- evaluation harness crash
- fixture test command failure when the task expected success

These should preserve the phase 1 runtime failure semantics while recording structured evaluation artifacts.

### Policy Failures

Examples:

- disabled tool attempted
- destructive change requires confirmation
- disallowed git write action
- malformed high-risk tool parameters

These should not necessarily collapse the whole session. They should usually become structured blocked or intercepted results that the model can react to.

This distinction is necessary so evaluation can tell whether the model chose badly, the runtime broke, or the policy correctly intervened.

## Observability

Phase 2 should continue phase 1 decision-point observability and extend it to scenario-level decisions, including:

- profile selection
- active vs disabled tool assembly
- policy decisions
- evaluation task start and finish
- ablation switches in effect
- provider mode selection

No-event paths remain dangerous. If a key decision is missing from logs, the implementation should treat that as an observability gap.

## Testing Strategy

Phase 2 should use three layers of testing:

### Unit Tests

Offline tests for:

- profile assembly
- prompt layer composition
- policy decisions
- task definitions
- scoring rules
- config loading
- provider request/response normalization with mocked payloads

### Offline Evaluation Tests

Fixture-based tests using fake or scripted providers to validate:

- evaluation runner behavior
- repository copy isolation
- scoring dimensions
- ablation toggles
- expected repository-state assertions

### Live Smoke And Eval Tests

Explicit opt-in tests that validate:

- DeepSeek configuration loading
- request execution against the live endpoint
- response normalization under real provider behavior
- optional small live evaluation runs

These tests must be skipped cleanly unless live mode is enabled.

## Completion Definition

Phase 2 is done when:

- the code-assistant profile is implemented and selectable
- the narrowed tool strategy is enforced
- policy interception is active and logged
- prompt layering is active and ablatable
- at least 3 Python fixture repositories are evaluated repeatably
- scoring reports dimensioned outcomes
- at least 3 ablation toggles can be exercised
- the DeepSeek live adapter can run under explicit live configuration
- offline and live test boundaries are clear
- phase 1 tests remain green

## Implementation Notes

The implementation should prefer minimal, testable additions over broad abstractions. In particular:

- do not build a multi-scenario framework for future scenarios yet
- do not move business logic into the engine
- do not make live provider access mandatory for development
- do not let evaluation rely on one opaque pass/fail bit

Phase 2 is successful if it produces a measurable `code_assistant` system whose accuracy can be improved through targeted narrowing and ablation, not if it produces the most abstract design.
