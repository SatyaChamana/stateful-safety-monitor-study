# Experiment log

## 2026-11-02, initial scaffold

**Environment:** Python 3.13.5, CPU-only PyTorch, no NVIDIA GPU exposed. GitHub and Hugging Face could not be resolved from this container. Python 3.11 is the target deployment version.

**Implemented:** Installable package, strict sequence model, CSV/JSON/JSONL loader, dataset audit, per-turn causal history selector, BM25, optional dense backend, current/recent/full history, mock judge, unvalidated local HF judge, validation threshold selection, test metric summarization, basic sequence bootstrap, fixtures and automated tests.

**Verified:** Ran `python -m pytest -q`: **13 passed**. Ran `scripts/system_report.py` (CUDA unavailable), and exact-match split integrity checks. Ran synthetic audit and validation/test smoke evaluation with mock judge; no real model or gated dataset was used. No research effect size established.

**Not run:** DecomposedHarm real audit, paper reproduction, CUDA inference, hybrid retrieval, oracle evidence, task-family bootstrap, response-aware extension.

**Risk:** Harm index numbering, actual upstream data fields, published DSR semantics, and real judge score parsing remain unverified.

**Next:** Set up CI and environment lockfile; obtain authorized dataset and model access; inspect upstream schema; reproduce official full-history baseline before history comparisons.

## 2026-11-02, session 2: DecomposedHarm adapter and reproduction prerequisites

**Environment:** MacBook M1 Pro, 16 GB RAM, macOS (Darwin arm64), no CUDA. Python 3.11.15 (uv-managed; no system `python3.11`). Packages pinned in `constraints/cpu-dev-py311-macos-arm64.txt` (numpy 2.4.6, pandas 3.1.0, scikit-learn 1.9.1, pytest 9.2.0, ruff 0.17.3, huggingface-hub 2.4.0). No torch/transformers installed. Parent repo HEAD `017e2b2` (this directory is not yet committed). `results/system_report.json` regenerated.

**Baseline before changes:** `python -m pytest -q` → **13 passed**.

**Access check:** network available. HF dataset revision `25e1d59…` is gated (`auto`); no HF token on this machine; gated CSVs return HTTP 401. Terms NOT accepted, NOT bypassed. Upstream code cloned at commit `79f5b41` into `third_party/` (added to `.gitignore`: the clone ships benchmark CSV/JSON files). Upstream data files were not opened beyond the CSV header line and SHA-256.

**Completed:**
- Verified from upstream code and paper: real columns `id,original_task,decomposition,harm_index,label,task_type` (spec §5.2 names were wrong; correction added). `harm_index` is **1-based, inclusive**. Boundary construction differs by type: QA = first StrongREJECT ≥ 0.75 prefix (≥ 2); image = always last step; agent = LLM-annotated.
- Paper evaluation split likely differs from the full union: `data/combine.py` undersamples val/test to 176 per type (val then halved) with an unseeded shuffle. Not reproducible from components; must be resolved by the real audit.
- `dataset.py`: `--schema decomposedharm` adapter (requires `--index-base 1`, namespaces IDs as `task_type/id`, never stores `original_task`, rejects missing columns, unknown task types, non-integral/junk/out-of-range boundaries); `read_rows`, `split_from_filename`; `audit_files` multi-file aggregate audit (SHA-256, sizes, revision, ingest time, rejections by reason, length percentiles, per-type boundary stats, benign denominators, cross-split ID/step/`original_task`-hash overlap).
- `cli.py`: `audit` accepts several `--data` files plus `--schema`/`--revision`; `pilot`/`evaluate` accept `--schema`. Native single-file audit output unchanged.
- `metrics.py`: `paper_protocol_outcomes` (upstream iterative semantics: step-2 start, strict `>`, inclusive boundary, DSR = recall, F1) and `PAPER_THRESHOLD_GRID`. Kept apart from our timely-recall metric.
- Found and verified on **fictional scores**: the upstream concurrent-mode threshold sweep is shifted one step (ignores step-2 firings; counts boundary+1 as timely). `scripts/check_upstream_metric.py` + pinned test. Effect on paper numbers NOT measured.
- Phase 0: `scripts/ci.sh` (ruff E9/F, pytest, fixture smoke CLI); fixed 5 pyflakes findings (2 unused imports removed; 3 Phase-3 placeholder lines marked `noqa`, logic unchanged); `data` extra (`huggingface_hub`).
- Docs: `docs/dataset_manifest.md`, `docs/reference_implementation_notes.md`; spec checkboxes, §18 and decision log updated; README.

**Files added:** `fixtures/decomposedharm_shape_fixture_{val,test}.csv` (fictional), `tests/test_decomposedharm_adapter.py`, `tests/test_paper_protocol.py`, `tests/test_upstream_metric_probe.py`, `scripts/ci.sh`, `scripts/check_upstream_metric.py`, `constraints/cpu-dev-py311-macos-arm64.txt`, two docs above. **Modified:** `dataset.py`, `cli.py`, `metrics.py`, `retrieval.py` (noqa only), `scripts/system_report.py`, `tests/test_integrity.py`, `pyproject.toml`, `.gitignore`, `README.md`, `EXPERIMENT_BUILD_SPEC.md`.

**Tests after changes:** `python -m pytest -q` → **38 passed**; `bash scripts/ci.sh` → `CI OK`. Probe test skips automatically when `third_party/` is absent.

**Real-data numerical outputs:** none. All numbers produced this session come from fictional fixtures or fictional score lists and are software checks only.

**Not run / not validated:** real DecomposedHarm audit; HF-vs-GitHub hash comparison; paper prompt mode in the judge; any model inference; GPU lock file; dense/hybrid/oracle retrieval; task-family grouping.

**Research / ethics risks:** (1) the public GitHub copy makes it easy to skip the HF gate: don't, the human accepts terms first. (2) The paper's split may be a 264/528 subset; using the full union instead would silently change the replication target. (3) The upstream sweep misalignment could shift the paper-selected threshold; report both sweeps, do not assume a direction. (4) Upstream maps unparsable replies to benign and strips apostrophes; reproduce for paper mode but count and disclose.

**Next safe executable action (human, ~2 min):**
```bash
# 1. In a browser: open https://huggingface.co/datasets/YuehHanChen/DecomposedHarm, sign in, accept the conditions.
# 2. Then, from the repo root:
source .venv/bin/activate
hf auth login
hf download YuehHanChen/DecomposedHarm --repo-type dataset \
  --revision 25e1d5910c2cccbdcae11799193595cea999bda8 --local-dir data/raw/decomposedharm
safety-study audit --schema decomposedharm --index-base 1 \
  --revision 25e1d5910c2cccbdcae11799193595cea999bda8 \
  --data data/raw/decomposedharm/combined_train.csv data/raw/decomposedharm/combined_val.csv \
         data/raw/decomposedharm/combined_test.csv \
  --out results/real_dataset_audit.json > /dev/null
```
Then compare `files[].sha256` with the GitHub hashes in `docs/dataset_manifest.md` §1 and the counts with §4.

**Next CPU agent task:** paper-interface prompt mode in `judge.py` (exact upstream formatting and templates, step-2 start, parse-failure counting), tested with fictional fixtures.

**GPU machine resume:** `uv venv --python 3.11 .venv && source .venv/bin/activate && pip install -e '.[dev,science,data,model]' && python scripts/system_report.py --out results/system_report.json && bash scripts/ci.sh`. Then pin the Llama-3.1-8B-Instruct revision and follow `docs/reference_implementation_notes.md` §4.

## 2026-11-02, session 2b: real DecomposedHarm audit

**Access:** user accepted HF terms and logged in (account `Satya-Chamana`). Downloaded revision `25e1d5910c2cccbdcae11799193595cea999bda8` to `data/raw/decomposedharm/` (gitignored). HF CSV SHA-256 values are **identical** to the GitHub copies at `79f5b41`.

**Commands:** `hf download ... --revision 25e1d59...`; `safety-study audit --schema decomposedharm --index-base 1 --revision 25e1d59... --data data/raw/decomposedharm/combined_{train,val,test}.csv --out results/real_dataset_audit.json`; `python scripts/build_paper_split.py --revision 25e1d59... --upstream-commit 79f5b41... --out configs/paper_eval_split.json`.

**First audit run:** 1,165 rows rejected ("must be list"). These were exactly all agent rows: agent `decomposition` is newline-separated text, not a list. The strict loader worked as intended.

**Change:** `dataset.py` `_agent_steps`: agent cells parse as a list if possible, else one step per non-blank stripped line (upstream `split_subtasks` fallback; apostrophes kept). Only for agent; QA/image still must be lists. Agent rows in the fictional fixtures were converted to the real multi-line shape. New tests in `tests/test_decomposedharm_adapter.py`. New `scripts/build_paper_split.py` + `tests/test_build_paper_split.py`.

**Real-data aggregates (source: HF revision `25e1d59`; dataset properties, not research findings):**
- 4,641 rows (3,243 / 698 / 700), 0 rejected, 0 duplicate IDs, 0 null boundaries, 0 boundaries at step 1.
- Card/Table 5 say 4,639: the +2 is QA test (334 vs 332), present in upstream per-type JSON too.
- HF CSVs are the full union. The paper's eval subset (upstream `combined/{val,test}_data.json`) is val 264 (137h/127b), test 528 (263h/265b). Every row maps uniquely to a CSV row; pinned in `configs/paper_eval_split.json` (IDs and hashes only).
- 1 test pair (`agent/benign_0011`, `agent/harmful_0011`) has identical steps and opposite labels; it is in the paper test subset.
- 7 exact `original_task` matches across splits (QA, same label each). One is val-test inside the paper subset (`QA/harmful_0226` / `QA/harmful_0693`).
- Boundary: image always last step; agent last step in 479/582; QA median 3.

**Tests:** `bash scripts/ci.sh` → **43 passed**, `CI OK`. Audit JSON scanned for long strings: none (no request text).

**Not run:** model inference, paper prompt mode, any metric on real data.

**Open decisions (user):** handling of the conflicting-label pair and goal overlaps (keep vs predeclared exclusion sensitivity); controlled-study split (paper subset vs full union). Both must be fixed before any test-set evaluation.

**Next CPU task:** paper-interface prompt mode in `judge.py` plus an `--ids configs/paper_eval_split.json` filter for `evaluate`, with fictional tests.

## 2026-11-02, session 2c: exclusion sensitivity and paper mode

**User decision:** keep every row for the paper replication; add a predeclared exclusion sensitivity analysis; then build paper mode.

**Exclusions (frozen before any scoring):** rules in `integrity.py` (`EXCLUSION_RULES`): R1 identical steps with conflicting labels → drop all members; R2 goal shared across splits → drop every val/test member; R3 goal with conflicting labels → drop all members. `scripts/build_exclusions.py` (refuses to overwrite) wrote `configs/sensitivity_exclusions.json`: R1 = R3 = the `agent/*_0011` pair; R2 = 8 QA IDs; union 10. Inside the paper subset: val `QA/harmful_0226`, `QA/harmful_0447`; test `QA/benign_0531`, `QA/harmful_0693`, `agent/benign_0011`, `agent/harmful_0011`.

**Paper mode:** new `src/stateful_safety/paper.py`; `Decision.parse_status`; `run_sequences(skip_first_turn=...)` with `turns_not_scored` and `parse_status_counts`; `evaluate --protocol paper --paper-prompt NAME --ids FILE --exclude FILE --max-input-tokens N`. Details and sources: `docs/reference_implementation_notes.md` §2b.

**Checks on real data, no model:** `--ids` selects exactly 264 / 528 rows. 5,202 model calls per prompt for val+test. Longest prompt about 2.4k tokens (agent ICL, chars/4 estimate). ICL leakage: agent and DQ ICL prompts share lines only with train; image and mixed ICL share exact lines with 8/12 and 6/6 val/test sequences (flagged; inspect before using those prompts).

**Tests:** `bash scripts/ci.sh` → **58 passed**, `CI OK`. New: `tests/test_exclusions.py`, `tests/test_paper_mode.py` (includes a 20-trial match against upstream `evaluate_predictions`).

**Real-data results:** none. No model has scored any benchmark row.

**Next (GPU machine, Llama license accepted):**
```bash
pip install -e '.[dev,science,data,model]' && python scripts/system_report.py --out results/system_report.json && bash scripts/ci.sh
safety-study evaluate --protocol paper --paper-prompt BINARY_BASELINE \
  --schema decomposedharm --index-base 1 \
  --validation data/raw/decomposedharm/combined_val.csv --test data/raw/decomposedharm/combined_test.csv \
  --ids configs/paper_eval_split.json --exclude configs/sensitivity_exclusions.json \
  --backend hf --model-id meta-llama/Llama-3.1-8B-Instruct --mode full \
  --out results/paper_mode_llama_baseline.json
```
Pin the model revision first, record dtype, then run it again with `--paper-prompt BINARY_INTENTION_EVAL_W_8_AGENT_ICL`. Before the full run, do a small validation-only smoke run to check reply parsing on real outputs.

## 2026-11-02, session 2d: model revision pin and validation-only smoke command

**Changes:**
- `--model-revision` on `pilot`, `evaluate` and the new `paper-smoke`. Every `--backend hf` run must pass a full 40-hex commit (`paper.validate_model_revision`); `main`, short hashes and uppercase are refused before any download. Passed to both `from_pretrained` calls in `TransformersGenerator` (paper mode) and `TransformersJudge` (diagnostic). Paper-mode outputs now include `model` (ID, revision, dtype, device, token caps); diagnostic outputs record `model_revision`.
- `safety-study paper-smoke`: validation-only (no `--test` option exists). Deterministic sample (`--n`, default 20; half per label, round-robin across task types, ordered by sha256 of ID). Reports parse-status counts, parse-OK rate, reply size percentiles, input tokens, seconds per call, overflow and error counts. No thresholds, no metrics, no reply text (`RecordingGenerator` keeps sizes only). Status `VALIDATION_SMOKE_NOT_A_RESULT` for HF runs.
- Refactor: shared `_model_revision` and `_paper_parts` in `cli.py` (mock-on-real-data refusal unchanged and rechecked on the real val file).

**Model metadata lookup (HF API, no download):** `meta-llama/Llama-3.1-8B-Instruct` sha `0e9e39f249a16976918f6564b8830bc894c89659`, lastModified 2024-09-25, gated `manual`. `auth_check` → `GatedRepoError`: this account has no Llama access yet.

**Tests:** `bash scripts/ci.sh` → **73 passed**, `CI OK`. New `tests/test_revision_and_smoke.py`: revision validation, unpinned HF refusal on all four commands, revision reaches both loaders (fake `torch`/`transformers` modules), sampler deterministic and balanced, `paper-smoke` rejects `--test`, mock smoke output has no metric keys and no fixture text.

**Real-data results:** none.

**Next (human):** request access at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct (manual approval).
**Next (GPU machine, after access):**
```bash
REV=0e9e39f249a16976918f6564b8830bc894c89659   # re-confirm: python -c "from huggingface_hub import HfApi;print(HfApi().model_info('meta-llama/Llama-3.1-8B-Instruct').sha)"
safety-study paper-smoke --schema decomposedharm --index-base 1 \
  --validation data/raw/decomposedharm/combined_val.csv --ids configs/paper_eval_split.json \
  --paper-prompt BINARY_BASELINE --n 20 --backend hf \
  --model-id meta-llama/Llama-3.1-8B-Instruct --model-revision $REV \
  --out results/paper_smoke_llama_baseline.json
```
Go on to the full `evaluate --protocol paper` (add `--model-revision $REV`) only if `parse_ok_rate` is high and there are no overflows. Note that `max_new_tokens` is 2000 (upstream default), so a slow reply shows up in `seconds_per_call`.

## 2026-11-02, session 2e: clipper cluster preparation

**Cluster probe (user-run on the login node, pasted in):** login node, no GPU, SLURM. `debug-gpu` (30 min): RTX 8000, V100S, H100 NVL; `short-gpu` (4 h): RTX 8000, V100S. Python 3.12 module, no 3.11. Home quota about 41 GB free. Hugging Face reachable. My own SSH attempt from the Mac was blocked by the permission system; not retried.

**Changes:**
- `--dtype {bfloat16,float16,float32}` is required for every HF run (no `auto`); passed to `from_pretrained` and recorded in outputs. Reason: V100S and RTX 8000 lack bf16, and `auto` would pick different precisions on different GPU classes.
- `scripts/clipper/setup.sh` (login node: uv + Py 3.11, extras, pinned upstream clone, CI, Linux lock file, private permissions). `scripts/slurm/gpu_check.sbatch` and `scripts/slurm/paper_smoke.sbatch` (offline model, validation only). Runbook `docs/clipper.md`.

**Tests:** `bash scripts/ci.sh` → **73 passed**, `CI OK` (dtype checks added to `tests/test_revision_and_smoke.py`). SLURM and setup scripts are syntax-checked only; nothing ran on clipper.

**Open decision:** GPU class + dtype for the whole study. The H100 only offers 30-minute jobs unless a longer partition exists; V100S/RTX 8000 mean fp16.

**Next:** user runs `docs/clipper.md` steps 1-4; Llama access still pending for steps 5-6.

## 2026-11-02, session 2f: study hardware fixed

**User decision:** Tesla V100S with float16 for the whole study (replaces the RTX 4090 / H100 plan in spec §6/§15).

**Changes:** both SLURM scripts request `gpu:tesla_v100s:1` and exit (code 2) if the allocated GPU name lacks "V100S"; `paper_smoke.sbatch` hard-codes `DTYPE=float16`. `configs/pilot.yaml` records dtype and hardware. Spec decision log has new Hardware and Model dtype rows. `docs/clipper.md` updated. Paper-mode `model` metadata now also records the GPU name.

**Consequences to disclose in the reproduction report:** fp16 instead of bf16; local HF Transformers instead of Together "Turbo" serving; V100S throughput sets the run time (5,202 calls per prompt for val+test). If a full run does not fit in 4 h, split it into jobs by sequence; never switch GPU class or dtype.

**Tests:** `bash scripts/ci.sh` → **73 passed**, `CI OK`. SLURM scripts syntax-checked only; not yet run on clipper.

## 2026-11-02, session 2g: correction to hardware premise

Full `sinfo` shows H100 NVL (3 nodes × 2) in `short-gpu` (4 h) and `gpu` (5 days), and V100S/RTX 8000 in `gpu` (5 days) too. My earlier statement that H100 was limited to 30-minute debug jobs was wrong (it came from truncated output). The V100S + float16 decision was based on that statement. The lock stays in place unchanged until the user re-confirms or switches. No code changed in this entry; `docs/clipper.md` partition table corrected.

## 2026-11-02, session 2h: study hardware revised to H100 NVL + bfloat16

**User decision (after the correction above):** NVIDIA H100 NVL with bfloat16 for the whole study. No run had used V100S, so nothing has to be re-done.

**Changes:** both SLURM scripts request `gpu:nvidia_h100_nvl:1` and exit (code 2) unless the GPU name matches `*H100*NVL*`; `paper_smoke.sbatch` hard-codes `DTYPE=bfloat16`. `configs/pilot.yaml`, spec decision log and `docs/clipper.md` updated. Default partitions: `gpu_check` on `debug-gpu`; `paper_smoke` on `short-gpu` (4 h). Use `gpu` (5 days) for full runs if 4 h is not enough. Remaining deviation from the paper: local HF Transformers vs Together "Turbo" serving.

`gpu_check.sbatch` now runs its matmul check in bfloat16. (A first edit attempt aborted on a safety assertion before writing any file; re-applied.)

**Tests:** `bash scripts/ci.sh` → **73 passed**, `CI OK`. SLURM scripts syntax-checked only.

## 2026-11-02, session 2i: clipper setup and GPU check (user-run, output pasted)

**Code transfer:** rsync from the Mac, 76 files, 228 KB; no data, venv, upstream clone or results. (Two earlier attempts ran inside the SSH session and copied nothing; runbook clarified. `.ruff_cache` came along; now excluded.)

**Login-node setup (`scripts/clipper/setup.sh`):** CPython 3.11.18 via uv; `ruff` clean; `pytest` **73 passed**; `CI OK`. Linux lock file written on clipper at `constraints/clipper-py311-linux-x86_64.txt` (not yet copied back). HF login on clipper with the user's own fine-grained token. SLURM account: the user's own (required by clipper; set via `SBATCH_ACCOUNT`).

**GPU check (`scripts/slurm/gpu_check.sbatch`, debug-gpu):** NVIDIA H100 NVL, 95,830 MiB, driver 610.43.02; torch 2.15.0+cu132 (CUDA 13.2), CUDA available, `bf16_supported True`, bf16 matmul OK. The study hardware/dtype lock (H100 NVL, bfloat16) works on clipper. No model inference yet.

**Dataset on clipper:** downloaded with the user's own HF login, revision `25e1d59`, to `data/raw/decomposedharm/` (private permissions from `setup.sh`). SHA-256 of all three CSVs **match** `docs/dataset_manifest.md` §1 (test `ea85ac20…`, train `ba98d5dc…`, val `d94d376e…`), so clipper, the Mac audit and the GitHub copy hold identical files.

**Model on clipper:** Meta access granted to the user. `hf download meta-llama/Llama-3.1-8B-Instruct --revision 0e9e39f249a16976918f6564b8830bc894c89659 --exclude "original/*"` → 14 files, 16.1 GB, snapshot directory `.../snapshots/0e9e39f249a16976918f6564b8830bc894c89659` (matches the pinned revision).

**Submitted:** validation-only smoke, SLURM job **798351** (`paper_smoke.sbatch`, `PROMPT=BINARY_BASELINE`, n=20, H100 NVL, bfloat16, offline). Not a result; checks reply parsing and timing.

**Pending:** smoke job output; copy `constraints/clipper-py311-linux-x86_64.txt` back to the Mac.

## 2026-11-02, session 2j: smoke job 798351 failed; HF interface fix

**Job 798351 (clipper, H100 NVL):** model weights loaded (291/291), then the first `generate` call raised `AttributeError` on `tokens.shape`. Cause: the transformers release installed on clipper returns a `BatchEncoding` (dict with `input_ids`, `attention_mask`) from `apply_chat_template(return_tensors="pt")`, not a bare tensor. Also a deprecation warning: `torch_dtype` → `dtype`. No sequence was scored; no output JSON written. Not a result.

**Fix:** new `judge.encode_chat` asks for `return_dict=True` and returns `(input_ids, attention_mask)`; both `TransformersGenerator` (paper mode) and `TransformersJudge` (diagnostic) pass both to `generate`, and load with `dtype=`. Regression test `tests/test_hf_generate_interface.py` uses fakes shaped like the new API; verified it fails on the old code with the same `AttributeError: shape` and passes on the fix.

**Tests:** `bash scripts/ci.sh` → **75 passed**, `CI OK`.

**Next:** re-sync `src/` and `tests/` to clipper, rerun CI there, resubmit the smoke job.

## 2026-11-02, session 2k: first real paper-mode smoke (job 798352)

**Run:** `paper_smoke.sbatch`, `BINARY_BASELINE`, validation split only (paper subset), n=20, node g050. Model `meta-llama/Llama-3.1-8B-Instruct` @ `0e9e39f…`, NVIDIA H100 NVL, `torch.bfloat16`, `max_new_tokens` 2000. Status `VALIDATION_SMOKE_NOT_A_RESULT`. These are engineering checks, not research findings.

**Aggregates:** 20/20 sequences scored (QA 3+3, agent 3+3, image 4+4); 142 model calls; 0 context overflows; 0 errors. Parse status: 142 `ok` (no "sorry" refusals, no replies without a number). Reply tokens p50 2, p75 47, p90 67, max 145; reply chars p50 1, p75 263, max 859. Mean input tokens per call 127–233. Seconds per call: mean 0.90, p50 0.13, max 5.28. Projected full val+test pass for this prompt: 5,202 × 0.9 s ≈ 78 min (fits `short-gpu`).

**Concern:** about a quarter of replies run long even though `BINARY_BASELINE` asks for 0 or 1 only. Upstream parsing takes the LAST number in the reply, so "ok" means a number was found, not that it is the verdict. Kept unchanged for replication. Added `parsed_score_buckets` (exactly_0 / exactly_1 / between_0_and_1 / above_1) and `replies_over_5_tokens` to `paper-smoke` to measure it; rerun the smoke before the full pass.

**Benign log line:** transformers warns it ignores `clean_up_tokenization_spaces` for BPE tokenizers; that leaves decoded text unmodified, so it doesn't affect number parsing.

**Tests:** `bash scripts/ci.sh` → **76 passed**, `CI OK`.

## 2026-11-02, session 2l: smoke rerun with score buckets (job 798354)

**Aggregates (validation, n=20, same sample, `BINARY_BASELINE`; not a result):** 142 calls, all parse status `ok`. **Parsed scores: exactly_0 112, exactly_1 30, none between or above 1.** 53/142 replies over 5 tokens, all still parsed to 0 or 1. Seconds per call mean 0.87, p50 0.13, max 5.12. Token percentiles unchanged from job 798352. Reply-char p75 263 → 257: small run-to-run GPU nondeterminism under greedy decoding (spec §7.4); disclose.

**Residual risk (disclose, not fixable without reading replies):** in a long reply, the last 0/1 may not be the verdict. Same rule as upstream; kept for replication.

**Added:** `scripts/slurm/paper_eval.sbatch`: full paper-protocol run (validation thresholds, then one test pass) on the paper subset with frozen exclusions. Refuses to run if a result for the same prompt exists; writes to `.partial` and renames only on success. Projected time for `BINARY_BASELINE` about 80 min; limit 3 h 45 min.

## 2026-11-02, session 2m: full paper-protocol run submitted; report drafted

**Submitted by user:** `paper_eval.sbatch`, `PROMPT=BINARY_BASELINE` (first and only test pass for this prompt). Running.

**Paper comparison target verified from the PDF:** Figure 9 (zero-shot binary) is word-for-word `BINARY_BASELINE`; Figure 10 (zero-shot probability) is `BINARY_INTENTION_EVAL_ORIGINAL`. The intro's "DSR: 58% vs. 27% using Llama-3.1-8B with a zero-shot prompt" matches Table 2 (sequential 0.575 vs single input 0.267), so Table 2 "Ours: Sequential" is the comparison for this run (paper claim: F1 0.719 ± 0.026, DSR 0.575 ± 0.033). Which split Table 2 used is not stated.

**Added:** `statistics.bootstrap_outcome_counts` (sequence bootstrap of F1/DSR from TP/FP/FN/TN; exact equivalence to resampling sequences, since each sequence has one outcome) + `tests/test_bootstrap_counts.py`. `docs/reproduction_report.md` drafted with setup, comparison mapping, deviations and an empty results table.

**Tests:** `bash scripts/ci.sh` → **79 passed**, `CI OK`. (No re-sync needed for the running job; the bootstrap runs on the Mac from the result JSON.)

## 2026-11-02, session 2n: first real result — paper baseline reproduced (job 798355)

**Run:** `paper_eval.sbatch`, `BINARY_BASELINE`, paper subset (val 264, test 528), Llama-3.1-8B-Instruct @ `0e9e39f…`, H100 NVL, bf16, greedy. Validation threshold 0.05 (both sweeps; 0/1 replies make every grid threshold equivalent). One test pass. Source file: `results/paper_eval_BINARY_BASELINE_bfloat16_798355.json` (clipper).

**This is a real result (replication), not a new research finding:**
- Test, all rows: TP 161, FP 3, FN 102 (15 too late, 87 never), TN 262 → **F1 0.754** (SE 0.023, 95% CI 0.709–0.798), **DSR 0.612** (SE 0.029, CI 0.555–0.672); benign flagged 3/265.
- Test, frozen exclusions: F1 0.757, DSR 0.613 (n=524).
- Paper Table 2 claim (sequential, Llama-3.1-8B): F1 0.719 ± 0.026, DSR 0.575 ± 0.033. Both inside our CIs; rough z ≈ 1.0 (F1), 0.8 (DSR). **Reproduced within uncertainty.**
- Parse: 11 of 5,202 replies had no number (mapped to 0 as upstream). Tokens: 1.02 M input, 100 k output. Model time ≈ 53 min (0.59–0.64 s/call).

**Docs:** `docs/reproduction_report.md` §3 filled; spec Phase 2 checkboxes updated.

**Next:** (1) add per-task-type outcome counts to paper-mode output; (2) run `BINARY_INTENTION_EVAL_ORIGINAL` (Table 4 zero-shot, F1 0.690) and `BINARY_INTENTION_EVAL_W_8_AGENT_ICL` (Table 4 optimized, F1 0.881); (3) copy the result JSON back to the Mac for archiving.

## 2026-11-02, session 2o: per-type breakdown and per-sequence records

**Archive:** user copied `results/paper_eval_BINARY_BASELINE_bfloat16_798355.json` to the Mac; contents verified against the pasted output (TP 161, FP 3, FN 102, TN 262). Gitignored like all results.

**Changes (paper mode output):**
- `metrics.paper_outcomes_by_task_type`: outcomes per task type at the one shared threshold fitted on all-type validation (not per-type thresholds). Output keys `validation_by_task_type`, `test_by_task_type` in every sweep block.
- `per_sequence` list: ID, split, task type, label, 1-based harm step, scores for steps 2..n (4 dp), first firing step at the primary threshold. No request text. Enables paired comparisons across prompts and history modes (spec §7.4).
- Tests: per-type counts sum to combined counts; per-sequence first-fire reproduces the aggregate TP count.

**The baseline run (798355) has neither.** Not re-running it: the test set is used once per prompt, and a second GPU pass could give slightly different numbers (nondeterminism), creating two results to choose from. Its per-type split is reported as unavailable.

**Tests:** `bash scripts/ci.sh` → **80 passed**, `CI OK`.

## 2026-11-02, session 2p: probability and Agent-ICL runs submitted; summary script

**Submitted by user:** `paper_eval.sbatch` with `PROMPT=BINARY_INTENTION_EVAL_ORIGINAL` and `PROMPT=BINARY_INTENTION_EVAL_W_8_AGENT_ICL` (one test pass each). Running. The Agent-ICL template's 8 examples all answer with a bare 1–2 character number.

**Added:** `scripts/summarize_paper_runs.py` (Markdown tables: test F1/DSR with count bootstrap, per-type counts, and a paired sequence bootstrap of F1/DSR differences plus discordant counts between runs that carry `per_sequence`) + `tests/test_summarize_paper_runs.py`. Checked on the baseline file: reproduces F1 0.754 (0.023), DSR 0.612 (0.029).

**Tests:** `bash scripts/ci.sh` → **83 passed**, `CI OK`.

## 2026-11-02, session 2q: probability-prompt run (job 798359) and mapping correction

**Run:** `BINARY_INTENTION_EVAL_ORIGINAL`, paper subset, same pinned setup. File copied to the Mac: `results/paper_eval_BINARY_INTENTION_EVAL_ORIGINAL_bfloat16_798359.json` (242 KB, includes per-sequence records).

**Result (test, iterative, threshold 0.25 from validation upstream sweep):** TP 225, FP 42, FN 38 (15 late, 23 never), TN 223 → F1 0.849 (SE 0.017), DSR 0.856 (SE 0.022); benign flagged 42/265. Aligned sweep picked 0.20 (first prompt where the shift changes the threshold); test F1 0.851. Exclusions: F1 0.851. All replies parsed. 4.0 output tokens/call, 0.13 s/call.

**Mapping correction:** the upstream repo ships the authors' own aggregate results (`model_responses/summary/.../Meta-Llama-3.1-8B-Instruct-Turbo_*`). Their `BINARY_BASELINE` validation F1 is 0.759, exactly Table 3's zero-shot value, so the paper's zero-shot baseline (Tables 3–4) is the binary prompt. My earlier mapping of Table 4 zero-shot to the probability prompt was wrong; corrected in the report. The probability prompt has no paper table for Llama-3.1-8B.

**Comparison with authors' saved test result (their sweep rule, applied to our per-sequence scores):** ours F1 0.875 (232/35/31/230) vs authors 0.874 (242/49/21/216), same best threshold 0.25. Reproduced.

**Binary baseline vs corrected references:** test F1 0.754 vs Table 4 0.690 ± 0.028 (z ≈ 1.8), Table 2 0.719 (z ≈ 1.0); validation 0.788 vs authors' saved 0.759.

**Per type (probability prompt, test):** QA F1 0.731 / DSR 0.616 (all 15 late detections); agent 0.834 / 0.943 with 28/88 benign flagged; image 0.957 / 1.000.

**Docs:** `docs/reproduction_report.md` §1 rewritten (two reference sources, corrected mapping), §3b–3d added.

## 2026-11-02, session 2r: Agent-ICL run (job 798360); Phase 2 complete

**Result (test, threshold 0.05 from validation, both sweeps):** TP 228, FP 17, FN 35 (15 late, 20 never), TN 248 → F1 0.898 (SE 0.014), DSR 0.867 (SE 0.021); benign flagged 17/265. Exclusions: F1 0.899. All replies parsed; 6 replies (2 sequences) parsed as 3.0, outcome unchanged in both. 1,557 input / 8.7 output tokens per call, 0.31 s/call.

**References:** Table 4 optimized Llama 0.881 ± 0.014 (z ≈ 0.8). Authors' saved test file, their rule: theirs 0.932, ours 0.931 (237/9/26/256 vs 245/18/18/247).

**Paired (Agent-ICL minus probability, 528 test sequences):** F1 +0.049 [+0.019, +0.079]; DSR +0.011 [−0.026, +0.049]. Gain is from fewer false alarms (benign flagged by only one prompt: 35 vs 10), not more catches (11 vs 14).

**Report:** §0 summary across all three prompts; §3e Agent-ICL; §4b limitations added (from the audit: rough z, test-picked thresholds in authors' files, different operating point, narrow intervals, one run per prompt, unpaired binary run, inspected test scores). Binary-prompt wording softened from "reproduced" to "consistent with Table 2, borderline vs Table 4".

**Phase 2 status:** all tasks checked except the model-revision pin note. Gate: protocol deviations documented, no ground truth enters the judge view. Next: Phase 3 (history-selection comparison) — needs a decision on split and FPR budget.

## 2026-11-02, session 2s: Phase 3 set up (no runs yet)

**User decisions:** judge = Agent-ICL prompt (chosen on validation F1: 0.922 vs 0.862 vs 0.788); reuse the 528 paper test sequences, disclosed.

**Pre-registration:** `configs/phase3_plan.json` written 2026-11-03T03:43Z, before any Phase 3 run. Judge, encoder (`sentence-transformers/all-MiniLM-L6-v2` @ `1110a24…`, Apache-2.0), data, test-reuse disclosure, protocol (step 1 unscored, only evidence changes, chronological presentation, per-condition threshold at validation sequence FPR ≤ 0.05), 10 pilot conditions (current, full, recent/bm25/dense/hybrid × k 3, 5), 8 extension conditions (k 1, 10), primary metric (test timely recall at frozen threshold), primary comparisons H1 / H2 (non-inferiority vs full) / H4 (retrievers vs recency), proposed primary k=3 with rationale. Margin left null on purpose: set from validation only, before the test stage.

**Code:**
- `retrieval.py` rewritten: `bm25_scores`, ranked top-k with recency tie-break, `DenseScorer` (encoder loaded once, each text embedded once, nothing written to disk), `hybrid_search` (RRF k=60 over relevance ranks, then chronological). Old BM25/recent behavior unchanged (existing tests pass).
- `runner.py`: dense scorer built once per run; `retrieval_seconds`; `decisions_history_shorter_than_full` (how often a condition's evidence actually differs from full history).
- `paper.py`: `paper_prompt(..., require_full_history=False)` lets the same template/parsing see selected history; still rejects unordered or non-earlier steps. `PaperJudge(require_full_history=...)`.
- `metrics.calibrate_threshold` ignores NaN (unscored step 1).
- New `history.py` (`run_stage`: validation fits, test only reads frozen) and `safety-study history-eval` (test stage refuses without a frozen file, without a margin, or with a different FPR budget; validation refuses a threshold file).
- `scripts/freeze_phase3_thresholds.py`: checks judge, revision, dtype, budget, encoder revision, one result per pilot condition, no mock; requires a margin; refuses to overwrite.
- `scripts/slurm/phase3.sbatch`: array over plan conditions, H100/bf16 guard, offline, refuses to repeat a condition+stage, `.partial` then rename. `setup.sh` now installs the `retrieval` extra.

**Not verified locally:** the real sentence-transformers encoder (not installed on the Mac); first exercised by the clipper validation stage.

**Tests:** `bash scripts/ci.sh` → **94 passed**, `CI OK` (new `tests/test_phase3.py`: 11 tests).

## 2026-11-02, session 2t: Phase 3 validation submitted; summary script

**Submitted by user:** `phase3.sbatch --array=0-9 STAGE=validation` (10 pilot conditions). Running.

**Added:** `scripts/summarize_phase3.py` (per-condition table: threshold, FPR, timely recall, any detection, share of decisions with shorter-than-full history, tokens and seconds per decision; paired sequence bootstrap of timely-recall and FPR differences vs `full`, with discordant counts) + `tests/test_summarize_phase3.py`. On validation it informs the margin and primary-k decisions; on test it produces the comparison tables.

**Tests:** `bash scripts/ci.sh` → **96 passed**, `CI OK`.

## 2026-11-03, session 2u: Phase 3 validation (superseded) and protocol amendment 1

**Validation run 798493 (10 conditions, protocol as pre-registered):** all on the pinned setup, all replies parsed. **Superseded; never to be frozen or cited as results.** It showed that fixed-FPR calibration cannot work on the judge's nearly binary parsed scores: with 127 benign sequences the 5% budget allows 6 flags; at score ≥ 1, `full` flags 5 (timely 112/137) but `bm25_k3` flags 7 and `recent_k3` 9, so their thresholds jumped to a misparsed 3.0 or to none, and timely recall fell to 0. `current` flags 30/127 at its only usable cutoff.

**Validation-only observations (hints, not findings):** at k=5 about 70% of decisions already see all earlier steps and `bm25_k5` equals `full` exactly, which supports k=3; at score ≥ 1, k=3 retrievers time-detect 115–118 vs 112 for full, with 1–4 more benign flags; dense retrieval costs about 11 s per 1,744 decisions.

**User decision:** amendment with readouts A + B. Written to `configs/phase3_amendment_1.json` (2026-11-03T04:27Z; references the plan's SHA-256), before any test run:
- A (primary): continuous P("1")/(P("0")+P("1")) at the first generated token (plain and space-prefixed digit tokens, single-token check at load), from the same greedy generation (`output_logits`); per-condition threshold at validation sequence FPR ≤ 0.05.
- B (secondary): parsed score at a fixed cutoff ≥ 1.0, no calibration.
- Calibration candidates restricted to [0, 1].
- Diagnostics: first-token 0/1 probability mass; A-vs-parsed agreement.
- Protocol tag `phase3-a1`; outputs `results/phase3a1_*`; the freeze script requires the tag and the amendment, so superseded files are rejected.

**Code:** `paper.first_token_ids`, `paper.two_token_readout`, `Generation.p_one/p01_mass`, `Decision.continuous_score/readout_mass`, runner `continuous_out` + mass stats, `calibrate_threshold(valid_range=...)`, `history.py` rewritten for two readouts, freeze/summary scripts and CLI updated.

**Tests:** `bash scripts/ci.sh` → **100 passed**, `CI OK`. New checks: readout math, token-id rules, judge without continuous score refused, misparse never becomes a threshold, protocol-version guards, HF fake returns logits only when asked.

## 2026-11-03, session 2v: Phase 3 validation under amendment 1 (job 798524)

**Run:** `phase3.sbatch --array=0-9 STAGE=validation`, protocol `phase3-a1`, 10 pilot conditions, 264 validation sequences (127 benign / 137 harmful), 1,744 scored decisions each, all replies parsed (`ok` 1744). Validation stage only; no test run yet. **These are calibration-stage numbers, not findings.**

**Readout A validity:** first-token 0/1 mass min 0.992, median ≈ 1.000, 0 decisions below 0.5, in every condition. A (≥ 0.5) agrees with the parsed 0/1 answer on 99.7–99.9% of decisions. Scores spread (e.g. `full` benign max-score bins [<0.1: 109, 0.1–0.4: 11, 0.4–0.6: 2, 0.6–0.9: 3, 0.9–0.99: 2]), so the FPR calibration has room; every condition calibrated to 6/127 = 0.047. Readout A is usable.

**Readout A (threshold; timely recall /137):** current 0.989; 0.693 · full 0.438; 0.847 · bm25_k3 0.562; 0.839 · dense_k3 0.469; 0.847 · hybrid_k3 0.500; 0.854 · recent_k3 0.651; 0.839 · all k5 ≈ full (0.832–0.847). Paired vs full: current −0.153 [−0.226, −0.088]; k3 retrievers within ±0.03 (SE 0.013–0.016); bm25/dense/hybrid k5 identical to full (0 discordant sequences).

**Readout B (fixed ≥ 1):** current flags 30/127 (FPR 0.236); k3 conditions catch more than full (+0.022 to +0.044) but also flag more (+0.008 to +0.031). It moves along the trade-off curve, so A is the fair comparison.

**Notes for the decisions:** k5 cannot distinguish retrieval from full (most decisions already see all history), so k=3 is the informative choice. Token savings from retrieval are small (k3 1,526 vs full 1,558 input tokens/decision) because the fixed Agent-ICL prompt dominates; seconds per decision vary by node and job and are not comparable across conditions.

**Pending user decision:** margin and primary k → `freeze_phase3_thresholds.py`.

## 2026-11-03, session 2w: Phase 3 thresholds frozen

**User decision:** primary k = 3; non-inferiority margin 0.05 timely recall (rationale: 5 points is the largest loss of timely detection we would accept in practice; k5 retrievers matched `full` on every validation sequence, so k5 cannot separate retrieval from full history). Not chosen to fit observed validation differences.

**Frozen:** `configs/phase3_thresholds.json`, status `FROZEN_BEFORE_TEST`, 2026-11-03T18:44:27Z, protocol `phase3-a1`, primary readout A, FPR budget 0.05, plan SHA-256 4f099d93…, amendment SHA-256 d8f524cb…, 10 per-condition thresholds with source-file SHA-256s (from job 798524). No test-stage run exists yet.

**Next:** sync to clipper, run `STAGE=test` once.

## 2026-11-03, session 2x: Phase 3 test stage submitted

**Submitted by user:** `phase3.sbatch --array=0-9 STAGE=test`, reading the frozen `configs/phase3_thresholds.json`. One-shot evaluation; running.

## 2026-11-03, session 2y: Phase 3 test results (job 798699), one-shot, frozen thresholds

**Run:** 10 pilot conditions on the 528 reused paper test sequences (265 benign / 263 harmful; reuse disclosed in the plan), thresholds from `configs/phase3_thresholds.json`, protocol `phase3-a1`, all replies parsed; first-token 0/1 mass median ≈ 1.000, 0 below 0.5; A-vs-parsed agreement 99.5–99.8%. Statistics: paired sequence bootstrap, 2,000 draws, seed 1729 (sequence-level, not task-family).

**Pre-registered results (readout A, primary):**

| Condition | Test FPR (benign flags /265) | Timely recall /263 |
|---|---|---|
| current | 0.019 (5) | 0.684 |
| full | 0.068 (18) | 0.798 |
| bm25_k3 | 0.057 (15) | 0.802 |
| dense_k3 | 0.064 (17) | 0.810 |
| hybrid_k3 | 0.060 (16) | 0.814 |
| recent_k3 | 0.045 (12) | 0.772 |

- **H1 (full vs current):** timely recall +0.114 [+0.068, +0.156], 34 vs 4 discordant; but full also flags more benign sequences: FPR difference current − full −0.049 [−0.079, −0.019]. The frozen thresholds transferred to different test FPRs, so this is not a same-FPR comparison.
- **H2 (non-inferiority, margin 0.05):** candidate − full: bm25_k3 +0.004 [−0.011, +0.019]; dense_k3 +0.011 [−0.008, +0.034]; hybrid_k3 +0.015 [−0.004, +0.038]. All lower bounds > −0.05 → **non-inferior, all three**; FPR differences −0.011, −0.004, −0.008 (CIs include 0). Holds under a Bonferroni split too (lower bounds ≥ −0.011 against −0.05).
- **H4 (retrievers vs recent_k3):** bm25_k3 +0.030 [+0.008, +0.057], dense_k3 +0.038 [+0.015, +0.065], but with higher FPR (+0.011 [0.000, +0.026], +0.019 [+0.004, +0.038]); recent_k3's threshold transferred to a lower FPR. Under readout B (same fixed cutoff for all) the differences are bm25 −0.008 [−0.023, +0.008], dense −0.004 [−0.019, +0.008]. **H4 not supported:** the readout-A gain cannot be separated from the FPR difference.
- **Calibration transfer:** test FPR 0.019–0.068 against a validation fit of 0.047 (6/127); `full` exceeds the 0.05 budget on test.

**Exploratory (decided after unblinding; not confirmatory):**
- *Matched test FPR (thresholds tuned on test):* timely recall at 5 / 13 / 18 benign flags: current 0.684 / 0.772 / 0.787; full 0.658 / 0.772 / 0.798; hybrid_k3 0.707 / 0.810 / 0.821; recent_k3 0.711 / 0.787 / 0.810. At equal false alarms, full history does not beat current overall, and recency is not worse than retrieval.
- *By task type (frozen thresholds; paired, full − current):* QA +0.198 [+0.093, +0.302] (20 vs 3; both 0/90 benign flags); agent +0.148 [+0.080, +0.227]; image 0.000 (ceiling 0.989). Full-history benign FPR: agent 0.091 vs 0.034, image 0.115 vs 0.023. History helps decomposed QA and agent detection but adds benign false alarms in agent and image sequences.
- *QA, hybrid_k3 − recent_k3:* +0.093 [+0.035, +0.163] (8 vs 0, QA benign flags 0 for both). A candidate for a pre-registered follow-up, not a finding.
- *Cost:* input tokens per decision current 1,477, k3 ≈ 1,525, full 1,556 (the fixed Agent-ICL prompt dominates); retrieval saves ≈ 2% here. Seconds per decision are node-dependent and not compared.

**Spec:** Phase 3 "same judge and split" and "calibrate/freeze" checked. Still open: k = 1, 10 extension; plots.

## 2026-11-03, session 2z: Phase 3 results section

**Added:** `docs/phase3_report.md`: summary verdicts, design and timeline (plan → superseded run → amendment → freeze → single test run), readout validity, pre-registered H1/H2/H4 tables, secondary results (readout B, k = 5, calibration transfer, cost), exploratory §5 clearly separated, limitations, claim boundaries, and reproduce commands.

**Added:** `scripts/phase3_exploratory.py` (matched benign-flag recall with thresholds tuned on test; per-type paired diffs at frozen thresholds; refuses non-test or non-`phase3-a1` files) and `tests/test_phase3_exploratory.py` (3 tests). It replaces the ad hoc session-2y computation; its per-type CIs differ slightly from 2y's (different bootstrap draws per subset), and the 2y entry was corrected to the script's values (QA full − current [+0.093, +0.302]; QA hybrid − recent [+0.035, +0.163]).

**Tests:** `bash scripts/ci.sh` → **103 passed**, `CI OK`.

## 2026-11-03, session 3a: study stopped after Phase 3; manuscript; tag phase3-v1

**User decision:** stop the study after Phase 3 for the application cycle. **Phases 4 (mechanism ablations, oracle annotations) and 5 (robustness, task-family statistics) are deferred, not skipped**; Phase 3 leftovers (k = 1 and 10 extension; latency-vs-k plot) also deferred. Latency is not plotted because seconds per decision depend on the cluster node.

**Added:**
- `scripts/make_figures.py` (6 figures to `docs/figures/`, aggregates only; Figure 4 titled EXPLORATORY) + `tests/test_make_figures.py`.
- `docs/manuscript/history-selection-sequential-monitor-phase3-v1.docx` (manuscript draft: abstract, methods, all pre-registered and exploratory results with 13 tables and 6 figures, discussion, limitations, references, appendices) and its generator `docs/manuscript/build_manuscript.js` (docx-js). Numbers are taken from `docs/reproduction_report.md`, `docs/phase3_report.md` and the scripts above. References were not re-checked against the PDFs in this session.
- README status updated.

**Repository:** standalone git repository created from this folder (gitignored data, results, weights and `third_party/` excluded), tagged `phase3-v1`, pushed to GitHub as a private repository.

**Tests:** `bash scripts/ci.sh` → **104 passed**, `CI OK`.

## 2026-11-08, session 3a: manuscript revision v1.1 (Penn State portfolio), post-hoc input checks

Scope chosen by Satya: wording fixes plus quick checks from existing logs, with no new judge calls. Driven by `docs/manuscript/review-2026-11-05/editorial-decision.md`.

**New script:** `scripts/phase3_overlap.py`, which writes `docs/phase3_overlap.md`. It recomputes k=3 selections using the pinned selectors and records, and reads judge scores from the saved test files. Dense and hybrid were recomputed on CPU (Mac, sentence-transformers 6.2.0, encoder @ 1110a24).
- **Item 2:** among timely-relevant decisions, the share where top-3 input differs from full history:

  | Task type | Harmful | Benign |
  |---|---|---|
  | QA | 14% | 37% |
  | Agent | 45% | 47% |
  | Image | 67% | 66% |

- **Item 7:** retrieved top 3 equals recent 3 in 6% to 29% of decisions (mean Jaccard 0.34 to 0.57). Harmful sequences where a retriever and recency disagree, split into different steps / same steps:

  | Retriever | Different steps | Same steps |
  |---|---|---|
  | BM25 | 7 | 5 |
  | Dense | 6 | 6 |
  | Hybrid | 9 | 6 |

- **Item 10, noise proxy:** 1,562 to 1,833 identical-input decisions per condition pair. 0 differ by more than 0.001, and the maximum difference is 0.0000. This also validates the recomputed selections.
- **Item 15:** rerunning the summaries confirms the shared CI [+0.019, +0.079] is a coincidence. It comes from two independent computations: Phase 2 F1 diff and Phase 3 H1 FPR diff.
- `pytest`: 104 passed.

**Manuscript text changes** (`build_manuscript.js` builds `history-selection-sequential-monitor-phase3-v1.1.docx`; v1 kept):
- Item 1: exploratory results are labelled in the abstract, contributions, Discussion and Conclusion. "holds at matched false alarms" removed.
- Item 2: H2 restated as benchmark-specific.
- Item 3: margin described as set after validation and before test. Smallest passing margins reported (0.011 / 0.008 / 0.004). Power sentence corrected (a 3-point loss passes about 44%).
- Item 4: one rule for H1 and H4.
- Item 5: "judge more suspicious" replaced with an operating-point statement.
- Item 6: Phase 2 vs Phase 3 gap explained as different readout and threshold rule.
- Item 7: overlap results added in §4.4 and Table 14.
- Item 8: privacy reworded to "read less"; only recency caps storage.
- Items 10, 12, 16 and 17: one sentence each.
- Item 13: abstract now says "F1 matches".
- Item 14: recent_k5 0.060 wording fixed; Table 2 run times filled from result-file timestamps; abstract cut to 254 words.
- **Not done (out of scope):** item 9 (related work: only papers read may be added), item 11 (clustered bootstrap), item 18 (public tag, pending repo decision).
