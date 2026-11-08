# Coding Agent Instructions

Read `EXPERIMENT_BUILD_SPEC.md` in full, especially Sections 3, 5, 7, 8, 12, 16 and 18.

## Mission
Implement the next unchecked step of Experiment 1: evidence selection in stateful AI safety monitoring. Continue, test, and document the existing source code. Do not replace it without a documented reason.

## Required rules
- Keep raw benchmark data outside version control and reports.
- Follow evaluation-only access conditions for DecomposedHarm. Do not train on it.
- Never generate or execute a harmful objective from benchmark content. Only classify preexisting requests.
- Do not let original goals, future turns, labels or harmful indices enter monitor inputs.
- Calibrate on validation and evaluate once on held-out test.
- Do not conflate synthetic mock outputs with scientific findings.
- Confirm harmful_index numbering and metric semantics with original implementation before a real evaluation.
- If GPU, model authorization, or data access is missing, continue CPU-safe work and state the blocker.
- After every change run `python -m pytest -q` and add regression tests.
- At the end write or update `docs/experiment_log.md` with completed work and test status.

## Start now
1. Inspect repository and tests.
2. Run `python -m pytest -q`.
3. Read Phase 0 and Phase 1 tasks in the spec.
4. Implement the first unchecked task and test it.
5. Report exact commands to resume on an authorized GPU machine.
