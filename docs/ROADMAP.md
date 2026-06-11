# Linch SDK Roadmap

This file is intentionally kept as a lightweight roadmap/status page.

---


## Future Development Directions

Linch should keep its center of gravity as an explicit, embeddable runtime for
context-heavy agent workflows. It does not need to copy every part of a broader
agent ecosystem SDK. The strategic advantage is runtime transparency: clear
context construction, policy-aware tool execution, inspectable events,
resource-aware scheduling, durable state, and a small surface that application
developers can embed without surrendering control.

---

## Phase 4 — Budgets, compaction ladder, workflow engine (2026-06)

- **A. Budget primitive** — `RunBudget` token/USD caps shared across the agent tree;
  `BudgetEvent` warning/exceeded; graceful error stop.
  Primary files: src/linch/budget.py, src/linch/loop.py, src/linch/session.py,
  src/linch/subagents/runner.py, src/linch/events.py
- **B. Compaction ladder** — micro-compact (LLM-free tool-result elision), reactive
  recovery on ContextLengthError, forced-compaction circuit breaker; opt-in via
  `Agent(compaction_ladder=CompactionLadder())`.
  Primary files: src/linch/compaction.py, src/linch/loop.py
- **C. Workflow engine** — `agent.run_workflow(fn)`: WorkflowContext
  (agent/parallel/pipeline/phase/budget), content-addressed journal on RunStore,
  resume replay, WorkflowEvent.
  Primary files: src/linch/workflow/{context,journal,engine}.py, src/linch/events.py,
  src/linch/agent.py
