# Code Assistant Phase 2.5 Design

**Date:** 2026-04-16

**Status:** Drafted for review

## Goal

Build phase 2.5 of `code-agent-harness/` by tightening the `code_assistant` evaluation layer that was introduced in phase 2.

Phase 2 established the single-scenario code-assistant profile, narrowed tools, layered prompts, policy interception, fixture repositories, and a first evaluation harness. Phase 2.5 does not broaden the runtime. It makes evaluation strict enough to support serious scenario narrowing work:

- task-level tool argument validation
- task-level workflow validation
- suite-level baseline evaluation
- suite-level ablation comparison
- dimensioned failure evidence for offline and live runs

The intended result is that the phase 2 code-assistant system can be judged against the eight-step narrowing method with much less hand-waving. The runtime should remain generic. The evaluation contract should become explicit.

## Scope

Phase 2.5 includes:

- a richer evaluation task schema with structured tool, workflow, and outcome expectations
- trace extraction from runtime messages so scoring can use evidence instead of hard-coded booleans
- strict `tool_arguments` scoring
- strict `workflow` scoring
- suite execution for the default task set
- baseline-vs-ablation comparison output
- stable failure evidence surfaced through the CLI
- matching semantics that tolerate reasonable provider variation without collapsing into vague pass/fail checks
- tests and documentation for the stricter evaluation layer

Phase 2.5 does not include:

- new business scenarios beyond `code_assistant`
- multi-agent orchestration
- frontend or visualization work
- a general experiment tracking platform
- JSON report export
- a database-backed outcome abstraction
- provider protocol redesign
- changes to the generic engine loop beyond reading already-produced runtime messages

## Problem Statement

The current phase 2 evaluation harness has four useful properties:

- it can run fixture repositories in isolated workspaces
- it can execute both offline scripted runs and explicit live runs
- it can score multiple dimensions
- it supports ablation flags

However, it still has four material weaknesses:

1. `tool_arguments` is effectively a placeholder and does not validate anything.
2. `workflow` is effectively a placeholder and does not validate anything.
3. task definitions are too weak to describe process requirements with precision.
4. ablation comparison is possible only through manual reruns and manual reading of output.

That means phase 2 is only partially aligned with the narrowing method it is supposed to support. Phase 2.5 closes those gaps without turning the evaluation layer into a research platform.

## Product Shape

The product remains a Python CLI project rooted at `code-agent-harness/`.

Phase 2.5 does not introduce a new runtime profile. It strengthens the existing `code_assistant` evaluation contract.

The intended user-visible shape is:

- `code-agent-harness eval --task <task-id>` still works
- `code-agent-harness eval --task <task-id> --ablate <name>` still works
- `code-agent-harness eval --task <task-id> --live` still works
- `code-agent-harness eval --suite default` runs the default task set
- `code-agent-harness eval --suite default --compare-ablation <name>` runs baseline and ablation and prints deltas
- failed dimensions print short evidence so a user can see what was wrong without opening internal logs

## Acceptance Criteria

Phase 2.5 is complete only when all of the following are true:

- each evaluation task expresses structured tool expectations, workflow expectations, and outcome expectations
- `tool_arguments` is scored from actual runtime trace evidence and is no longer hard-coded
- `workflow` is scored from actual runtime trace evidence and is no longer hard-coded
- offline scripted runs and live runs use the same task schema and the same scoring semantics
- a single-task eval run prints dimensioned scores plus failure evidence when a dimension fails
- the default suite can be run through one CLI command
- ablation comparison can be run through one CLI command and prints baseline, ablation, and delta summaries
- the evaluation layer still works against the existing fixture repositories for `bugfix`, `feature`, and `analysis`
- phase 2 runtime behavior does not regress

## Architecture

Phase 2.5 keeps the phase 2 architectural rule:

**Do not grow `engine` into a business-logic bucket.**

The generic engine already persists assistant messages, user messages, tool calls, and tool results in the runtime transcript. Phase 2.5 should read those artifacts and derive evaluation evidence from them. It should not add scenario-specific branching into the engine loop.

Phase 2.5 primarily strengthens three modules and one CLI surface:

- `evals/tasks.py`: richer task schema and default task definitions
- `evals/runner.py`: runtime trace extraction and suite orchestration
- `evals/scoring.py`: dimension scoring plus evidence
- `cli.py`: suite and comparison entrypoints plus stable output formatting

The profile, prompt, policy, tool, storage, and provider layers are reused as they already exist.

## Data Model

### `EvalTask`

`EvalTask` should move from a weak sample description to an executable evaluation contract.

Each task should include four categories of information:

1. `metadata`
2. `tool_expectations`
3. `workflow_expectations`
4. `outcome_expectations`

The minimum fields are:

- `task_id`
- `task_class`
- `fixture_name`
- `user_input`
- `tool_expectations`
- `workflow_expectations`
- `outcome_expectations`
- `live_eligible`

The three task classes remain:

- `bugfix`
- `feature`
- `analysis`

### Tool Expectations

Each expected tool interaction should be represented as a structured entry rather than a plain name string.

Minimum fields:

- `name`
- `required`
- `argument_expectations`

Optional fields:

- `must_appear_before`
- `must_appear_after`

The schema must support both precise and flexible matching so that live-model variation is tolerated where appropriate.

Each argument expectation should define:

- `field_path`
- `match_mode`
- `expected`

### Workflow Expectations

Workflow constraints describe process correctness, not exact argument equality. They express the minimal process the scenario requires.

Examples:

- `must_read_before_patch`
- `must_run_tests_before_finish`
- `forbid_patch`
- `forbid_test_runs`
- `allow_confirmation`
- `require_response_summary`

The schema should allow task-level overrides on top of task-class defaults.

### Outcome Expectations

Outcome expectations define what must be true at the end of the run.

Minimum supported areas:

- repository assertions
- test assertions
- response assertions

This keeps the existing final-state checks while making them more explicit and easier to extend later.

## Matching Semantics

Phase 2.5 should not require exact byte-for-byte equality for every argument. That would make live evaluation brittle for no product benefit.

Instead, the evaluation layer should support a small explicit set of matching modes.

### Supported Match Modes

- `exact`
  Use when a field must equal a specific value.

- `contains`
  Use when a list or string must include a required element or fragment.

- `unordered_contains`
  Use when a list must contain items but order does not matter.

- `regex`
  Use when a value can vary but must match a stable pattern.

- `non_empty`
  Use when the field must be present and non-empty.

- `length_at_least`
  Use when list-size or string-size matters more than exact contents.

The initial implementation should stay small. If a match mode is not needed by the default task set, it should not be added.

### Argument Matching Rules

Argument matching is applied only after a tool call is matched to an expected tool entry by tool name and positional constraints.

Rules:

- if a task requires `read_file` on a specific file, the path should usually use `exact`
- if a task requires a targeted test run, `run_tests.args` should usually use `contains` with the expected test target
- if an argument is optional for correctness, it should not be required in the task definition
- extra model-supplied arguments should not fail matching unless they conflict with the expectation

This keeps matching strict on intent without punishing benign verbosity from the model.

## Workflow Semantics

Workflow correctness is not the same thing as replaying one exact script. It should measure whether the agent followed the required process for the task.

### Task-Class Defaults

The scorer should apply these defaults unless a task overrides them.

#### `analysis`

- must not modify repository files
- must not use `apply_patch`
- must not claim test execution unless tests were actually run
- should ground its answer in inspected repository content

#### `bugfix`

- must inspect at least one relevant file before patching
- must run the targeted test before successful completion
- must not finish with a success claim if the required test did not run

#### `feature`

- must inspect the target implementation file before patching
- must run the targeted test before successful completion
- must report implementation plus verification in the final answer

### Workflow Evaluation Philosophy

Workflow scoring should answer:

- did the agent use the minimal correct process for this task class
- did it violate a scenario rule
- did it terminate with claims unsupported by its own actions

Workflow scoring should not answer:

- did the model reproduce the exact same trace as the scripted provider
- did it take the shortest possible path

## Runtime Trace Extraction

`evals/runner.py` should stop collapsing runtime behavior into only `tool_names` and `output_text`.

Instead, it should derive an `EvalTrace` from `RuntimeResult.messages`.

Minimum trace contents:

- ordered tool call records
- ordered tool result records
- final assistant output
- final runtime state
- workspace root

Each extracted tool call record should include:

- index within the run
- `tool_name`
- `arguments`
- `tool_use_id`
- whether a corresponding `tool_result` exists
- a summarized result status such as `ok`, `blocked`, `require_confirmation`, or `error`

This trace becomes the single evidence source for scoring.

## Scoring Model

### Dimensions

The phase 2 dimension list remains:

- `tool_choice`
- `tool_arguments`
- `repository_state`
- `tests`
- `response_content`
- `workflow`

Phase 2.5 changes how these dimensions are produced.

### Dimension Evaluation

`tool_choice`

- passes when required tools are present and required ordering constraints are satisfied
- fails when a required tool is missing or an ordering rule is violated

`tool_arguments`

- passes when matched tool calls satisfy all declared argument expectations
- fails when a required matched call has missing or conflicting arguments

`repository_state`

- passes when all repository assertions are satisfied
- fails when any required file content assertion fails

`tests`

- passes when required test execution expectations are satisfied
- fails when the task requires test execution and the trace or outcome does not show the required run

`response_content`

- passes when required answer fragments or patterns are present
- fails when the answer omits required facts or claims unsupported success language

`workflow`

- passes when task-class defaults and task-level workflow expectations are satisfied
- fails when process rules are violated

### Evidence Model

Each dimension should return:

- numeric value `1.0` or `0.0`
- short evidence text

Examples of acceptable evidence:

- `missing required tool run_tests`
- `run_tests.args did not contain tests/test_calc.py`
- `analysis task used apply_patch`
- `final response omitted timeout value`

The overall pass value remains the conjunction of dimension passes. Phase 2.5 does not introduce weighted scoring.

## Suite and Ablation Execution

### Single Task Runs

The existing single-task flow stays in place:

- load one task
- choose offline scripted provider or live provider
- run the task
- score the task
- print dimensions and evidence

### Suite Runs

Phase 2.5 adds a default suite entrypoint that runs all default tasks in sequence.

Requirements:

- suite execution must reuse the same task definitions as single-task execution
- each task still runs in an isolated copied fixture workspace
- output must include per-task pass/fail plus aggregate counts

### Ablation Comparison

Phase 2.5 adds a comparison mode that runs:

1. baseline
2. one ablation configuration

The output should show:

- baseline pass count
- ablation pass count
- per-dimension aggregate delta
- task ids whose pass/fail status changed

The initial implementation should support one ablation name per comparison run. Multi-ablation matrix execution is intentionally out of scope.

## CLI Design

The CLI should remain backward-compatible with existing phase 2 usage.

### Supported Eval Shapes

- `code-agent-harness eval --task bugfix-basic`
- `code-agent-harness eval --task bugfix-basic --ablate policy_engine`
- `code-agent-harness eval --task analysis-timeout --live`
- `code-agent-harness eval --suite default`
- `code-agent-harness eval --suite default --live`
- `code-agent-harness eval --suite default --compare-ablation policy_engine`

`--task` and `--suite` should be mutually exclusive. One of them must be provided.

`--compare-ablation` should only be accepted with `--suite`. Single-task ablations should continue to use `--ablate`.

### Output Contract

Single-task output should include:

- task id
- overall pass/fail
- one line per dimension
- evidence lines for failed dimensions

Suite output should include:

- suite id
- passed task count
- total task count
- per-dimension aggregate summary
- one short line per task

Comparison output should include:

- compared ablation name
- baseline vs ablation pass counts
- per-dimension deltas
- task ids with changed pass/fail outcomes

The output format should stay plain text and deterministic so unit tests can assert it without fragile string scraping.

## Testing Strategy

Phase 2.5 should be implemented through TDD.

The tests should be expanded in three groups.

### Schema and Matching Tests

Add tests for:

- task schema validation
- supported match modes
- argument expectation success and failure cases
- workflow expectation validation

### Runner and Scoring Tests

Add tests for:

- extracting tool calls and tool results from runtime messages
- matching required tools and arguments
- detecting workflow violations such as patching before reading
- detecting analysis-task mutation attempts
- producing evidence strings for failures
- scoring suite summaries and comparison deltas

### CLI Tests

Add tests for:

- parser support for `--suite`
- parser support for `--compare-ablation`
- single-task output with evidence
- suite output
- comparison output

## Documentation Changes

Phase 2.5 documentation should be additive, not a rewrite of phase 2.

Required documentation updates:

- this design document
- a new implementation plan document
- README command examples for suite execution and ablation comparison

The README should stay practical. It should document how to run the new commands and what kind of output to expect. It should not become a long methodology essay.

## Risks and Tradeoffs

### Risk: Overfitting the Eval Schema to Scripted Providers

If the schema is too rigid, live runs will fail for harmless variation. This is why argument matching must support a small set of flexible modes instead of exact equality everywhere.

### Risk: Making Workflow Too Vague

If workflow is only described in natural language, the scorer will quietly regress back to placeholders. This is why workflow expectations must be structured and task-class defaults must be explicit.

### Risk: Making Comparison Output Too Heavy

If comparison mode tries to become a full experiment platform, phase 2.5 will sprawl. This is why the first version should compare baseline against one ablation only and keep output textual.

## Non-Goals

Phase 2.5 intentionally does not solve:

- benchmark expansion beyond the current fixture set
- automated benchmark generation
- multi-provider score normalization
- skill-learning or memory-learning research loops
- dashboarding or historical experiment storage
- database state abstractions for non-code scenarios

## Completion Standard

Phase 2.5 should be considered done when:

- the evaluation task schema is structured enough to encode tool, workflow, and outcome requirements
- `tool_arguments` and `workflow` are computed from trace evidence
- `eval --suite default` works
- `eval --suite default --compare-ablation <name>` works
- failed dimensions print readable evidence
- the same semantics apply to offline and live runs
- the full test suite remains green

At that point, phase 2 is no longer only "narrowed in spirit". It becomes narrow enough to measure, compare, and iterate with discipline.
