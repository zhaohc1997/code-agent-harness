# Code Assistant Phase 3 Design

**Date:** 2026-04-18

**Status:** Drafted for review

## Goal

Build phase 3 of `code-agent-harness/` by deepening the existing `code_assistant` scenario with a cost-directed optimization loop.

Phase 1 built the generic runtime. Phase 2 narrowed it into a code-assistant scenario. Phase 2.5 made the evaluation contract strict enough to support real ablations. Phase 3 uses that foundation to answer a different question:

**Where should we spend engineering and model budget first for this specific code assistant?**

The answer should be driven by eval evidence, not intuition.

## Core Principle

Resources should not be spread evenly across every mechanism. Phase 3 follows this priority order:

1. Expand and strengthen the evaluation set first.
2. Strengthen tool descriptions next because argument mistakes waste whole agent turns.
3. Strengthen pre-tool policy reminders after that because models forget global rules in long runs.
4. Treat reasoning mode as a task-dependent decision variable, not a universal default.
5. Keep model upgrades last because they are expensive and often hide harness problems.
6. Treat infrastructure such as compaction, observability, loop detection, locks, and cancellation as entry tickets, not the main phase 3 investment.

Negative-contribution mechanisms must be made visible and easy to disable.

## Scope

Phase 3 includes:

- expanded `code_assistant` eval tasks, with at least five default tasks covering `bugfix`, `feature`, `analysis`, and boundary cases
- stronger built-in tool descriptions written as field-level interface contracts for the model
- more specific code-assistant prompt layers aligned with inspect, patch, test, and report workflows
- richer pre-tool policy reminders and confirmations for code-assistant behavior
- eval cost metrics derived from runtime traces
- failure attribution derived from dimension scores and workflow evidence
- suite-level cost averages in CLI output
- baseline-vs-ablation cost deltas in CLI output
- simple mechanism recommendations for ablation comparisons
- documentation for the phase 3 optimization order and commands

Phase 3 does not include:

- new scenarios beyond `code_assistant`
- a reusable multi-scenario narrowing framework
- multi-agent orchestration
- GUI or visualization
- automatic task classification for the `run` command
- automatic model selection
- real token billing or provider cost accounting
- a database-backed experiment tracker
- new generic engine-loop behavior unless existing trace data is insufficient

## Product Shape

The product remains a Python CLI project rooted at `code-agent-harness/`.

The main user-visible change is that evaluation output becomes useful for optimization decisions:

- `code-agent-harness eval --suite default` reports pass rate, dimension averages, and cost averages
- `code-agent-harness eval --suite default --compare-ablation <mechanism>` reports dimension deltas, cost deltas, changed tasks, and a recommendation
- single-task eval output reports dimension scores, failure evidence, and task-level cost metrics

The `run` command should benefit from low-risk mechanisms such as stronger tool descriptions, stronger prompt layers, and safer policy reminders. It should not gain broad experimental behavior such as automatic task-class routing or dynamic model selection in this phase.

## Architecture

Phase 3 keeps the phase 2 architectural rule:

**Do not grow `engine` into a business-logic bucket.**

The runtime already emits messages, tool calls, policy decisions, tool results, and final state. Phase 3 should read and score those artifacts in `evals`, not add scenario-specific branches inside the loop.

The main implementation areas are:

- `evals/tasks.py`
  Expands default task coverage and encodes boundary cases.

- `evals/trace.py`
  Exposes enough trace evidence to count tool attempts, blocked calls, reminders, confirmations, test calls, patch calls, and assistant turns.

- `evals/scoring.py`
  Keeps phase 2.5 dimension scoring and adds failure attribution.

- `evals/runner.py`
  Aggregates task-level cost metrics into suite averages and comparison deltas.

- `tools/builtins.py`
  Upgrades built-in tool descriptions into explicit interface contracts.

- `policies/code_assistant.py`
  Adds more specific pre-tool reminders and confirmation rules without changing the generic policy interface.

- `profiles/code_assistant.py`
  Defines phase 3 default prompt wording and mechanism ablation names.

- `cli.py`
  Prints cost metrics, cost deltas, and mechanism recommendations.

## Mechanism Design

### 1. Evaluation Set First

The default `code_assistant` suite should grow from three tasks to at least five.

It must include:

- at least two `bugfix` tasks
- at least one `feature` task
- at least one `analysis` task
- at least two boundary cases that test process failures rather than only final correctness

Good boundary cases include:

- an analysis task where patching or test execution is a workflow failure
- a code-change task where broad tests are less desirable than targeted tests
- a code-change task where final response content must explicitly report the test target

The goal is not to make a large benchmark. The goal is to make every later mechanism change measurable.

### 2. Tool Description ROI

Built-in tool descriptions must be strengthened for the active code-assistant tools:

- `read_file`
  Must state that `path` is a workspace-relative string, `start_line` and `end_line` are optional inclusive integers, and the tool is for inspecting file content before editing.

- `search_text`
  Must state that `pattern` is an exact text fragment, not a regex, and `path` is an optional workspace-relative search scope.

- `list_files`
  Must state that `path` is optional, workspace-relative, and defaults to the workspace root.

- `apply_patch`
  Must state that `path` is a workspace-relative string and `replacements` is an array of objects with exact field names `old_text`, `new_text`, and optional `replace_all`.

- `run_tests`
  Must state that `args` is a required `array[string]` of pytest arguments and shell command strings are invalid.

- `ask_confirmation`
  Must state that `message` is required and `default` is an optional boolean.

This mechanism is low risk and should be active in both eval and `run`.

### 3. Policy Rule Injection

The policy layer should remain pre-tool and decision-point based. It should not execute tools or mutate repository state.

Phase 3 policy behavior should include:

- block disabled tools
- remind on broad or empty `run_tests` arguments
- require confirmation for destructive patches
- remind on large or ambiguous patch attempts
- remind when patching before any read evidence is available if trace context is available later

The existing `PolicyDecision` outcomes remain:

- `execute`
- `remind`
- `require_confirmation`
- `block`

If a rule cannot be implemented safely without engine changes, it should stay in eval scoring first rather than forcing scenario state into the engine.

### 4. Reasoning Mode Allocation

Reasoning mode stays available through provider `extra` configuration and the existing `reasoning_mode` ablation.

Phase 3 should make reasoning mode visible as a mechanism in evaluation:

- comparisons should show when the `reasoning_mode` ablation is being tested
- output should include whether the baseline and ablation used reasoning configuration
- recommendations should not blindly prefer reasoning if pass rate and dimensions do not improve

The `run` command should not gain automatic task-class-based reasoning selection in this phase.

### 5. Cost Metrics

Each eval run should expose trace-derived cost metrics.

Minimum task-level metrics:

- `tool_call_count`
- `successful_tool_call_count`
- `assistant_turn_count`
- `test_invocation_count`
- `patch_invocation_count`
- `confirmation_count`
- `blocked_call_count`
- `reminder_count`
- `targeted_test_ratio`

Optional metrics may be added only if they are derived from existing trace evidence without provider-specific billing assumptions.

Suite results should average numeric metrics across tasks.

### 6. Failure Attribution

Failure attribution should explain why a task failed without replacing dimension scoring.

Supported attribution buckets:

- `tool_selection_failure`
- `tool_argument_failure`
- `workflow_failure`
- `repo_state_failure`
- `test_failure`
- `response_failure`
- `cost_regression`

The mapping can be simple:

- failed `tool_choice` maps to `tool_selection_failure`
- failed `tool_arguments` maps to `tool_argument_failure`
- failed `workflow` maps to `workflow_failure`
- failed `repository_state` maps to `repo_state_failure`
- failed `tests` maps to `test_failure`
- failed `response_content` maps to `response_failure`
- worse cost metrics in comparison can add `cost_regression`

The first version should favor transparent rules over complex heuristics.

### 7. Mechanism Recommendation

`compare_suite_results()` should produce a simple recommendation for the ablated mechanism.

Recommended labels:

- `keep`
- `neutral`
- `disable_or_rework`

Initial rule:

- `keep` if ablation lowers pass rate or key dimension averages without reducing cost enough to justify the loss
- `disable_or_rework` if ablation improves pass rate or materially reduces cost without hurting dimensions
- `neutral` if pass rate, dimension averages, and cost averages are effectively unchanged

This is intentionally simple. It should make negative-contribution mechanisms visible without pretending to be a full experiment platform.

## Data Flow

1. `EvalTask` defines task class, fixture, user input, tool expectations, workflow expectations, and outcome expectations.
2. `run_eval_task()` copies the fixture, builds the `code_assistant` profile, executes the runtime, and extracts an `EvalTrace`.
3. `score_eval_task()` computes dimension scores and failure evidence.
4. Phase 3 metrics code computes cost metrics and attribution from the same trace and score.
5. `run_eval_suite()` aggregates pass rate, dimension averages, and cost averages.
6. `compare_suite_results()` computes dimension deltas, cost deltas, changed tasks, and mechanism recommendation.
7. `cli.py` prints a compact text report that can be used in normal terminal workflows.

No part of this flow should require a GUI, database, or provider-specific cost API.

## Online Runtime Boundary

The following phase 3 mechanisms may be active in `run` by default:

- stronger tool descriptions
- stronger code-assistant prompt layer wording
- safer and more specific policy reminders
- stricter disabled-tool consistency

The following mechanisms remain eval-only in phase 3:

- failure attribution reports
- suite cost averages
- ablation recommendation
- task-class-specific reasoning allocation
- automatic mechanism disabling
- automatic model upgrade decisions

This boundary keeps online behavior stable while still allowing eval evidence to guide future changes.

## Acceptance Criteria

Phase 3 is complete only when all of the following are true:

- the default eval suite contains at least five `code_assistant` tasks
- the suite covers `bugfix`, `feature`, `analysis`, and at least two boundary cases
- tool descriptions for active code-assistant tools include field names, field types, and major constraints
- policy tests cover broad test reminders and destructive or risky patch handling
- eval task results include cost metrics
- eval suite results include cost averages
- ablation comparison includes cost deltas
- ablation comparison includes a mechanism recommendation
- failure attribution is computed from dimension scores and exposed in eval data
- CLI output includes task-level cost metrics for single-task evals
- CLI suite output includes cost averages
- CLI comparison output includes cost deltas and recommendation
- README documents phase 3 cost-priority order and commands
- the full repository test suite passes
- phase 1, phase 2, and phase 2.5 commands continue to work

## Non-Goals

Phase 3 intentionally does not build:

- a general scenario framework
- a visual dashboard
- a long-term experiment database
- automatic prompt optimization
- automatic model switching
- multi-agent routing
- browser or OS automation
- provider billing reconciliation

Those features may become relevant later, but they are not needed to make the current `code_assistant` scenario measurably better.

## Risks

- Adding too many metrics can make CLI output noisy. The first version should print compact key-value lines.
- Cost metrics can be mistaken for real money. The documentation must call them trace-derived proxy metrics.
- Policy reminders can create more turns if overused. This is exactly why reminder counts and ablation comparisons are part of the acceptance criteria.
- Expanded eval tasks can become brittle if they overfit to scripted fake providers. The task schema should continue to tolerate reasonable live-model variation through the existing matcher.

## Implementation Direction

The implementation should proceed in this order:

1. Expand eval tasks and scripted providers.
2. Add trace-derived cost metrics and failure attribution.
3. Upgrade tool descriptions and prompt wording.
4. Extend code-assistant policy reminders.
5. Add suite cost averages, comparison cost deltas, and recommendations.
6. Update CLI output and README.
7. Run focused and full test suites.

This order follows the cost-structure principle: measure first, then optimize the highest-ROI mechanism, then decide what survives by ablation evidence.
