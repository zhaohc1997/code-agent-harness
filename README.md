# code-agent-harness

Python CLI agent runtime for experimenting with a single-agent ReAct loop, provider abstraction, tool execution, session persistence, checkpointing, context compaction, and runtime observability.

Current project docs:

- `docs/superpowers/specs/2026-04-15-agent-runtime-phase-1-design.md`
- `docs/superpowers/plans/2026-04-15-agent-runtime-phase-1-implementation.md`
- `docs/superpowers/specs/2026-04-16-code-assistant-phase-2-design.md`
- `docs/superpowers/plans/2026-04-16-code-assistant-phase-2-implementation.md`

## Phase 2 Code Assistant

Run a live code-assistant session:

Ensure `DEEPSEEK_API_KEY` is already exported before running these commands.

```bash
export CODE_AGENT_HARNESS_LIVE=1
export DEEPSEEK_BASE_URL=https://api.deepseek.com
export DEEPSEEK_MODEL=DeepSeek-V3.2
code-agent-harness run --profile code_assistant --session demo --input "Fix the failing test"
```

Run an eval task with an ablation:

```bash
code-agent-harness eval --profile code_assistant --task bugfix-basic --ablate policy_engine
```
