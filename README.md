# Stateful AI Safety Monitoring: Experiment 1

**Status (tag `phase3-v1.1`, 2026-11-08): Phases 0 to 3 complete; stopped after Phase 3. Phases 4 and 5 deferred, not skipped.** Version 1.1 revises the manuscript after an internal review and adds post-hoc input checks (`scripts/phase3_overlap.py`, `docs/phase3_overlap.md`); no experiment was rerun. The pre-registration, amendment and freeze files are unchanged from the original 2026-11-03 run.

- Phase 2: reproduction of the published Llama-3.1-8B sequential monitor, `docs/reproduction_report.md`.
- Phase 3: pre-registered history-selection study, `docs/phase3_report.md`.
- Manuscript: `docs/manuscript/` (Word; current version `history-selection-sequential-monitor-phase3-v1.1.docx`, original v1 kept; figures in `docs/figures/`, built by `scripts/make_figures.py` and `docs/manuscript/build_manuscript.js`).
- Full change and decision history: `docs/experiment_log.md`.

This repository tests whether selecting earlier authorized requests improves detection by a fixed safety monitor. It starts with a data adapter, causal replay, current/recent/full/BM25 history, optional dense retrieval, validation-only calibration and test evaluation.

## Start with Claude Code or Codex

Read `EXPERIMENT_BUILD_SPEC.md`. Both `CLAUDE.md` and `AGENTS.md` are included, so a coding agent can begin with the next unchecked task.

## Quick start on a developer computer

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,science]'
python -m pytest -q
bash scripts/ci.sh          # lint + tests + fixture smoke CLI
safety-study audit --data fixtures/synthetic_fixture.jsonl --index-base 0
safety-study evaluate --validation fixtures/synthetic_validation_fixture.jsonl \
  --test fixtures/synthetic_test_fixture.jsonl --index-base 0 \
  --backend mock --mode full --fpr-budget 0.05
```

The mock backend uses arbitrary fictional markers and produces **smoke tests only**, not AI safety findings.

## Real benchmark is not bundled

If `python3.11` is missing: `uv venv --python 3.11 .venv` (set `UV_PYTHON_PREFERENCE=only-managed` if a broken system Python is picked up).


Accept gated access terms at https://huggingface.co/datasets/YuehHanChen/DecomposedHarm and respect the creators' evaluation-only request. Never use this dataset to train a model. Actual data cannot be committed to this repository. Confirm `harmful_index` numbering before data ingestion.

## Data integrity

The monitor view contains current request and earlier requests only, not labels, future turns, or original high-level harmful goals. A separated evaluation command fits score thresholds on validation and reports test metrics without raw text.

## Current limitations

- See `docs/phase3_report.md` §6 and `docs/reproduction_report.md` §4–4b. Raw benchmark data, model weights and per-sequence result files are not in this repository.
- Oracle evidence (Phase 4) is planned, not implemented.
- Sequence bootstrap only; task-family grouping is not verified.
- Do not use smoke test numbers in a PhD statement or paper.

See `EXPERIMENT_BUILD_SPEC.md` for tasks, metrics, ethics and acceptance criteria.
