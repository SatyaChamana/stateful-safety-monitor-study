# How Much History Does a Sequential Misuse Monitor Need?

A reproduction of a published sequential misuse monitor and a controlled comparison of which earlier requests a fixed safety judge should see. Independent study by Satya Chamana; manuscript draft, not peer reviewed.

**Release:** tag `phase3-v1.1` (code and protocol as run on 2026-11-03; v1.1 adds post-hoc checks, no rerun). `main` also carries later manuscript wording fixes and integrity hardening (see Changelog).

## Research question

Decomposition attacks split a harmful goal into steps that each look harmless. A sequential monitor judges each new request together with earlier ones. This study asks how much of that history a fixed judge needs, and how it should be chosen: the current request alone, the most recent steps, full history, or earlier steps selected by BM25, dense embeddings or a hybrid of the two.

## Design

- **Judge:** Llama-3.1-8B-Instruct (revision pinned) with the upstream Agent-ICL prompt, bfloat16, local inference on an NVIDIA H100 NVL.
- **Benchmark:** DecomposedHarm (gated, evaluation only), the authors' 264 validation and 528 test sequences, pinned by ID.
- **Phase 2:** reproduce the published monitor under the authors' scoring rule.
- **Phase 3:** hold judge and prompt fixed and vary only the history it sees, across ten conditions. The analysis plan (`configs/phase3_plan.json`), one amendment and the thresholds were fixed before the single test run; thresholds were calibrated on validation to a 5% false-alarm budget and frozen with SHA-256 hashes of the plan and every validation file. These files were not deposited with an external registry, so their timing rests on this project's own records.
- **Statistics:** paired sequence bootstrap with counts and denominators; non-inferiority margin 0.05 in timely recall.

## Findings (test, 528 sequences)

| Result | Evidence | Status |
|---|---|---|
| Reproduction: F1 matches the authors' released outputs to within 0.001 under their scoring rule (Agent-ICL test F1 0.898) | `docs/reproduction_report.md` | Aggregate agreement; confusion matrices differ slightly, so not identical per-sequence behavior |
| Full history vs current request: timely recall 0.798 vs 0.684 (+0.114, 95% CI +0.068 to +0.156), but benign FPR 0.068 vs 0.019 | `docs/phase3_report.md` §3 | Positive on the primary readout, not separable from the false-alarm difference |
| At equal test false-alarm counts, the overall gain disappears | `docs/phase3_report.md` §5 | Exploratory (thresholds tuned on test) |
| Top-3 BM25, dense and hybrid retrieval are non-inferior to full history in timely recall (lower 95% bounds ≥ −0.011 unadjusted, about −0.015 Bonferroni; margin 0.05); no increase in false alarms detected | `docs/phase3_report.md` §3 | Benchmark-specific: sequences are short (median 7 steps), and top-3 input equals full history whenever 3 or fewer earlier steps exist |
| Retrieval does not beat keeping the three most recent steps | `docs/phase3_report.md` §3, `docs/phase3_overlap.md` | Inputs to retrieval and recency were identical in about half of the decisions that decide timely detection |
| Cost: top-3 input averages about 1,525 tokens per decision vs 1,556 for full history | `docs/phase3_report.md` §4 | About 2% fewer tokens; this study shows no meaningful efficiency gain |

The main limitation is construct validity: the benchmark is too short for history selection to matter much. See the manuscript's Limitations section and `docs/phase3_report.md` §6.

## Reproduce

Requirements: Python 3.11. Exact versions for the CPU development environment are in `constraints/cpu-dev-py311-macos-arm64.txt`. GPU runs used CPython 3.11.18, torch 2.15.0+cu132 (CUDA 13.2) on an H100 NVL; the full GPU lock file is written by `scripts/clipper/setup.sh`.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
python -m pip install -e '.[dev,science]'
bash scripts/ci.sh          # lint, 120 tests (9 skip without the upstream clone or encoder), fixture smoke CLI
```

The real benchmark is not bundled. Accept the gated terms at https://huggingface.co/datasets/YuehHanChen/DecomposedHarm and respect the creators' evaluation-only request; never train on it. GPU runs: `docs/clipper.md` and `scripts/slurm/`. The commands that regenerate every table and figure from result files are in the manuscript's Appendix B. Per-sequence result files are not in this repository.

## Integrity safeguards in code

- The runner refuses any history item from the current or a later step.
- The test stage refuses to run without a frozen threshold file whose protocol version, margin and false-alarm budget match; the analysis plan must hash to the value recorded in that file, and a real run's judge model, revision, dtype, prompt, split file and encoder must match the plan.
- Every scored turn must have a finite continuous score in [0, 1]; a missing score stops the run instead of counting as "not flagged".
- Raw benchmark text, labels and model weights are never committed; tests use fictional fixtures.

## Repository map

| Path | Contents |
|---|---|
| `src/stateful_safety/` | Dataset adapter, history selectors, paper-mode judge, stage runner, metrics, statistics |
| `configs/` | Analysis plan, amendment, frozen thresholds, split IDs, exclusions |
| `scripts/` | Summaries, exploratory and post-hoc analyses, figures, SLURM and cluster setup |
| `docs/` | Reproduction and Phase 3 reports, overlap checks, dataset manifest, experiment log, manuscript |
| `tests/`, `fixtures/` | Unit and integration tests on fictional data |

## Changelog

- **phase3-v1.1 (2026-11-08):** manuscript revised after an internal review; post-hoc input-overlap checks (`scripts/phase3_overlap.py`). No experiment rerun.
- **main after phase3-v1.1:** manuscript wording fixes from a second review; "pre-registered" replaced by "analysis plan fixed before the test run" (no external registry was used); test stage now checks inputs against the frozen plan; per-turn score validation; GitHub Actions CI.

## How this was built

The code was written with AI coding assistants under a written build specification (`EXPERIMENT_BUILD_SPEC.md`; agent instructions in `CLAUDE.md` and `AGENTS.md`). The author designed the study, specified each component and its tests, ran every job and wrote the analysis.
