# Experiment 1: Evidence Selection in Stateful AI Safety Monitoring

**Executable research engineering specification for Claude Code or OpenAI Codex**  
**Spec version:** 1.0, 2026-11-02  
**Research direction:** Satya Chamana, PhD Research Statement V5  
**Project status:** Core scaffold, synthetic smoke test, and DecomposedHarm schema adapter implemented. Real data audit and benchmark replication NOT RUN.  
**First task for your coding agent:** Follow `AGENTS.md`, inspect the repo, run all tests, then perform the *next unchecked* task in Section 18.

> **Instruction to the coding agent:** You are the research software engineer for a reproducible AI safety monitoring study. Continue the existing repository rather than replacing it. Maintain the distinction between (1) previously published results, (2) unvalidated software smoke tests, and (3) results from our controlled experiments. Do not invent any experimental result or infer that the hypothesis is true. Stop and report when missing dataset authorization, a GPU, required licenses, or paper-defined metrics prevent a scientifically valid run. Implement and test what is possible without them.

---

## 1. Research purpose

### 1.1 Central question

What historical evidence must a safety monitor retain and retrieve to detect coordinated misuse before harm occurs, while controlling computational cost, false positives, memory use, and information access?

**Experiment 1 (E1)** tests the **request-history** part of that question. We do not need to create an attacker, train a new model, or produce actionable harmful outputs to perform this evaluation.

### 1.2 Scientific claim to test, NOT assume

A fixed safety judge may approach full-history detection performance when it receives a small set of relevant earlier requests. The performance gap may arise from (a) evidence excluded by access policy, (b) failure to retrieve relevant history, (c) loss of chronological structure, or (d) errors by the judge.

**The strongest initial contribution is diagnostic:** identify where information is lost in the monitoring pipeline, rather than simply claim a new defense outperforms previous work.

### 1.3 Relationship to the longer PhD plan

- **Aim 1:** Establish when stateful history helps at a common false-positive and cost budget.
- **Aim 2, primary scientific mechanism:** Explain which evidence matters, and whether missing retrieval evidence or judge reasoning causes failure.
- **Aim 3, later study:** Evaluate retention, deletion, access constraints, and residual privacy exposure.

This repository implements Aim 1 foundations and Aim 2 prerequisites. Do **not** implement Aim 3 before E1 passes the baseline reproduction gates.

### 1.4 Main hypotheses, with counterfactuals

| ID | Research hypothesis | Null / failure case | Required evaluation |
| --- | --- | --- | --- |
| H1 | A full-history monitor improves timely detection over current-only classification. | No improvement or higher benign FPR. | Paired tests at identical thresholds calibration protocol. |
| H2 | Top-k retrieval recovers much of full-history detection at lower complete-system cost. | Missed steps, overhead, or poor judge performance cancels gain. | Vary retrieval k; report tokens, GPU work, retrieval and total latency. |
| H3 | Relevant evidence selection partly explains decisions independently of judge identity. | Oracle evidence does not improve decisions. | Controlled removal, restoration, and oracle-selected context. |
| H4 | Different retrievers fail on different kinds of decomposition. | Simple recency works as well as dense or hybrid. | BM25 vs dense vs recency; hard lexical-mismatch and benign controls. |

Do not prespecify a desired effect size as a guaranteed result. Decide practical equivalence margins on the **validation** protocol before looking at test outcomes.

---

## 2. Primary sources and verification

Read these sources before changing the evaluator. Record the **exact repository commit**, benchmark revision, model revision, date accessed, and source of each methodological choice.

1. **Yueh-Han Chen et al.** *Monitoring Decomposition Attacks in LLMs with Lightweight Sequential Monitors*. ICLR 2026, arXiv:2506.10949. Paper: <https://arxiv.org/abs/2506.10949>. Repository: <https://github.com/YuehHanChen/Monitoring-Decomposition-Attack>. Dataset: <https://huggingface.co/datasets/YuehHanChen/DecomposedHarm>.
2. **Brown et al.** *Stateful Online Monitoring Catches Distributed Agent Attacks*. arXiv:2605.31593. Use for later agent and online clustering comparisons, **not** for the first reproduction.
3. **Sun et al.** *TwinGate: Stateful Defense against Decompositional Jailbreaks in Untraceable Traffic via Asymmetric Contrastive Learning*. arXiv:2604.27861. Motivates testing lexical versus intent evidence. It is a later stronger baseline, not a required E1 implementation.
4. **Isak and Dressman.** *Magnet: Detecting Cross-Session AI Misuse Through Capability Accumulation*. arXiv:2608.02518. Motivates **E2** comparison of previous requests, responses, and tool outputs; do not falsely claim those outputs exist in DecomposedHarm.
5. **Pramod et al.** *Towards a Unified Misuse Monitoring Benchmark*. arXiv:2610.07089. Use to challenge novelty claims and to compare intervention timing definitions.
6. **Shen et al.** *One Turn Too Late*. arXiv:2605.05630. Motivates later response-aware timing tests. Do not treat a post-response monitor as equivalent to a pre-response monitor.
7. **Sun et al.** *Decomposition Attacks Across Unlinkable Identities: Limits of Stateful Defenses for LLM Services*. arXiv:2608.17445. Disallows unsupported claims about detecting hidden coordination across truly unlinkable identities.
8. **Morris et al.** *Text Embeddings Reveal (Almost) As Much As Text*. arXiv:2310.06816. Relevant for future retention risk, **not E1**.

**Source rules:** Use paper claims as claims from the paper. Never copy published performance figures into our result tables. Verify defense success rate (DSR), first intervention, and `harmful_index` semantics in the paper and its code. In particular, confirm whether the harmful boundary is inclusive, and whether indices are zero- or one-based. The current loader refuses to guess indexing.

---

## 3. Scope and threat model

### In scope for E1

- Textual request sequences with a common, *authorized* history key.
- Offline, pre-response safety monitoring: the current request arrives, the monitor selects **previous** requests, and the monitor issues a risk score before any next response is released.
- Evaluating monitor performance against existing published benchmark traces.
- Public benchmark splits and benign, synthetic software test fixtures.
- Evaluating scores, not executing any task described inside a benchmark request.

### Explicitly out of scope for E1

- Generating new harmful decomposition plans.
- Executing cyber, biological, physical, or other potentially harmful benchmark tasks.
- Using a target model to complete the benchmark's harmful goals.
- Cross-account identity deanonymization, covert tracking, or unapproved data pooling.
- Training or fine-tuning on DecomposedHarm. Its authors request evaluation-only use.
- Claiming deletion compliance, privacy guarantees, or accurate prevention of real-world harm.
- Running MTID response-aware experiments before request-only E1 passes.

The dataset contains sensitive evaluation prompts. Keep data local, do not print raw text to logs or share prompts to remote services. Use only local, licensed model inference unless a separate ethics/access review authorizes another approach.

---

## 4. Experiment design

### 4.1 Unit of observation

- **Sequence:** An ordered list of user requests with a sequence-level benign/harmful label.
- **Turn:** The request at position `t`, with the view restricted to earlier turns `0..t-1`.
- **Harm-enabling boundary:** Published `harmful_index` annotation, if verified for the selected benchmark task type.
- **Task family:** A group of related tasks. The benchmark may not provide a reliable group identifier. Derive and audit it **only if a defensible, predeclared rule is available**. Do not pretend that task-type labels are independent task families.

### 4.2 Main configurations

All conditions must use the same judge checkpoint, prompt semantics, decoding options, context treatment, dataset, and label handling. The only changed input is the historical evidence. Thresholds are fitted **separately per configuration** on validation data to meet the same prespecified sequence-level benign FPR budget.

| Condition | Monitor input at turn t | Purpose |
| --- | --- | --- |
| C0 `current` | Current request only. | Stateless baseline. |
| C1 `recent` | Current request + most recent k prior requests. | Strong cheap baseline. |
| C2 `full` | Current request + all prior requests. | Paper-style contextual upper reference, subject to context window. |
| C3 `bm25` | Current request + top-k BM25 earlier requests in chronological presentation order. | Lexical retrieval. |
| C4 `dense` | Current request + top-k general dense retrieval over earlier requests. | Semantic retrieval. |
| C5 `hybrid` | BM25 + dense reciprocal-rank fusion, then chronological presentation. | Add after C3 and C4 pass. |
| C6 `oracle` | Annotated relevant earlier evidence only, under a fixed budget. | Diagnostic upper reference; never a deployed system. |

**Pilot retrieval depths:** k = 1, 3, 5, 10. Begin with k = 3 and 5. The full condition ignores k.

**Crucial fair-comparison distinction:** `full` uses all history, which may exceed the model context limit. Do not silently truncate it. Record the overflow rate and define a comparison under a matched token budget separately from the unconstrained full-history comparison.

### 4.3 Evidence views

E1 uses **previous requests only** because that is directly supported by DecomposedHarm's released fields. In E2, on a suitable trace dataset, add:

1. prior requests only;
2. prior request/response pairs;
3. prior requests, responses, and authorized tool results.

Never make a response at turn t available before it would exist in a real deployment. Never compare pre-response and post-response monitoring without labeling the difference.

### 4.4 Ablations after baseline

- Remove the highest-ranked retrieved request while preserving all other inputs.
- Restore it and repeat with the same judge and decoding seed.
- Replace it with a *matched benign* request and repeat.
- Randomly replace a retrieved item with an authorized earlier request.
- Shuffle the presentation order in a separately labeled experiment.
- Give annotated oracle evidence under the same k budget.

Interpret these as **decision dependence** on a specific monitor setup. They do not prove that one item is necessary for every possible model.

### 4.5 Benign controls and distractors

A monitor should not flag all decomposition. Legitimate agents split ordinary tasks into subtasks. Use matched benign benchmark controls; add benign multi-agent traces only after checking source permissions, labels, and comparability. Do not report an estimate of benign agent FPR from ordinary chat alone.

For a robustness check, add benign distractors at predeclared rates, with deterministic random seeds. Do not allow distractors to move the ground-truth harm boundary without updating how that boundary is evaluated. Report both original and transformed sequence lengths.

---

## 5. Dataset access, integrity, and handling

### 5.1 DecomposedHarm requires user access approval

The Hugging Face dataset page states that users must sign in and accept conditions to access files. Its authors specifically request **evaluation-only** use and provide a canary policy for public excerpts. Respect these conditions.

On a machine with approved access:

```bash
# Use your own Hugging Face account. Do not put tokens in source files.
pip install -U huggingface_hub
hf auth login

# After accepting the dataset's access conditions on the website:
hf download YuehHanChen/DecomposedHarm \
  --repo-type dataset \
  --local-dir data/raw/decomposedharm
```

Some files may instead be available through the paper's repository. Do not silently substitute differently processed splits. Record the selected files and checksums.

### 5.2 Expected dataset card fields

> **Correction (2026-11-02):** the real upstream CSV header is `id,original_task,decomposition,harm_index,label,task_type`, with `task_type` in `QA`/`image`/`agent`. The names below are the original planning assumption. See `docs/dataset_manifest.md`.

- `id` stable sequence identifier;
- `original_task` high-level task text (**never** passed to monitor);
- `decomposed_subtasks` ordered list of previous/current requests;
- `label` equal to benign/harmful;
- `harmful_index` earliest harm-enabling step, **semantics must be verified**;
- `task_type` one of jailbreak / image / agent.

The dataset card lists 4,639 examples distributed across three task types and train/validation/test partitions. Validate local counts rather than hard-coding them, because file layout and revisions can change.

### 5.3 Required dataset audit

Implement a CLI that reports aggregate only:

- files, hashes, sizes, dataset revision, ingestion time;
- count by original split, task type, and label;
- sequence-length percentiles and missing values;
- boundary range, null-rate, and whether the index appears valid;
- duplicate sequence IDs and identical subtask hashes;
- potential same-family overlap across splits;
- whether the exact paper evaluation split can be reproduced;
- task-family identification limitations;
- validation/test availability and size of the benign denominator.

**Never** rely on `original_task` as an input feature, retrieved item, or judge prompt. Never use test data to tune retrieval k, prompts, or thresholds.

### 5.4 Predefined partitions

- Use paper train split for **inspection only** unless explicit permission and protocol permit other evaluation development.
- Use validation for prompt selection, retrieval depth, threshold calibration, and statistical design.
- Freeze all parameters and evaluate held-out test only after validation is complete.
- If paper split contains related or generated paired tasks across partitions, report the limitation; build an additional task-family holdout if possible, without contaminating the original replication protocol.
- A test set is not a development set. After inspecting test outcomes, any major method change requires a new test set or a clearly disclosed exploratory analysis.

---

## 6. Judge model and resource budget

### Preferred reference model

**Llama-3.1-8B-Instruct** with a pinned model revision, if Meta's license is accepted. The paper also reports other monitors. Do not claim exact reproduction until its **prompting procedure, calibration, score parsing, and evaluation script** are matched.

Recommended local hardware:

- **Pilot:** one NVIDIA RTX 4090 with 24 GB VRAM, 64 GB RAM preferred, 40 GB free disk as a practical starter allowance.
- **Full benchmark:** university H100 under SLURM if institution grants an allocation. Larger context and extra runs may need it.
- **CPU-only machine:** perform data audits, unit tests, BM25 retrieval, metrics, and synthetic smoke tests. Do not expect local 8B model throughput.

These are **planning allowances**, not observed benchmark resource use.

### Technical configurations to record

- Python, CUDA, driver, GPU model, VRAM;
- `torch`, `transformers`, tokenizer, inference library versions;
- model ID and exact revision;
- dtype / quantization choice and max context;
- prompt template exact version and decoding parameters;
- tokenizer input tokens, generated tokens, wall time, GPU utilization if available;
- retrieval encoder revision, index build duration and memory size;
- exact dataset manifest and selected task split.

Do not modify the paper's original baseline until a reference reproduction is saved. Keep `third_party/` outside the core implementation or pinned as a submodule, and never copy hard-coded vendor API keys from upstream code. Use environment variables and local inference when possible.

### Reproduction hierarchy

1. **Software sanity:** synthetic fixture + mock judge. No scientific conclusion.
2. **Paper-interface reproduction:** verify original paper's prompt formatting, scores, output parsing, and DSR definition.
3. **Paper-like baseline:** matching open judge on exact DecomposedHarm split; report deviations.
4. **Controlled main study:** same judge across current/recent/full/retrieval conditions.
5. **Mechanism analysis:** controlled evidence interventions.

Never substitute Step 1 for Step 3 in a publication or PhD statement.

---

## 7. Metrics and their interpretation

### 7.1 Primary metric

**Timely detection recall at a fixed sequence-level benign FPR.** For a harmful sequence `i`, `t*_i` is the verified first harm-enabling turn, and `t_hat_i` is the first turn at which the score crosses threshold `theta`.

```
TimelyRecall(theta) = mean_i[ t_hat_i exists and t_hat_i <= t*_i ]
```

This must be paired with intervention timing, because a monitor that blocks very early can have high timely recall while blocking benign early steps. Use the benchmark's exact DSR implementation for official replication, and keep our diagnostic metric distinctly named if definitions differ.

### 7.2 False positives

**Sequence-level FPR:** among benign sequences, the fraction with at least one blocked turn.

```
FPR_sequence(theta) = benign_sequences_with_any_block / benign_sequences_total
```

Thresholds must be fit **only on validation benign sequences**. Use the same target budgets across monitors, e.g., 5% during initial pilots and 1% in the final comparison if enough benign data are available. At 1% FPR, a small benign test denominator supports very little precision. Report numerator and denominator and uncertainty.

### 7.3 Secondary metrics

| Metric | Definition / caution |
| --- | --- |
| Any-detection recall | Share of harmful sequences flagged at least once; NOT the same as timely detection. |
| First intervention | First threshold-crossing turn. Respect index base and paper definition. |
| Detection delay | First intervention relative to verified harm boundary. Positive delays can indicate too-late intervention. |
| Step recall@k | Share of annotated earlier attack-associated steps retrieved, if **reliable step relevance labels** exist. Without them, mark unavailable. |
| Evidence sufficiency | Whether oracle or manually selected relevant evidence changes the monitor decision at fixed t. Requires separate annotation. |
| Tokens per decision | Tokenizer-reported input/output, not characters; mock backend must report null. |
| End-to-end latency | Include retrieval, embedding and LLM inference, not just judge time. |
| GPU peak memory | Record allocated and reserved memory under a specified backend. |
| Index size / build cost | Account for retrieval preprocessing and maintenance. |
| Abstention/parse failure | Report unparsable model outputs; never silently map them to benign. |

### 7.4 Statistics

- Report 95% confidence intervals, ideally **cluster bootstrap by independent task family**, where groups can be justified.
- Current scaffold implements a **sequence bootstrap only** as a software placeholder; do not call it a task-family bootstrap.
- For paired monitor comparisons, resample the **same groups** and calculate paired outcome differences.
- Show counts alongside percentages. Show confidence intervals for FPR and timely recall.
- Run multiple seeds for stochastic decisions and retrieval when needed. Deterministic decoding does not by itself eliminate hardware variability.
- Define a primary comparison and limit secondary comparisons. Separate confirmatory and exploratory claims.

### 7.5 Diagnostic failure categories

For every monitored turn, without exposing raw prompt text in exported reports, categorize failures as:

1. history unavailable due to authorized scope;
2. relevant prior evidence absent from top-k;
3. retrieved evidence present but order or context malformed;
4. judge returns incorrect or late safety decision;
5. judge output parse error or context overflow;
6. ground-truth ambiguity / inadequate annotation.

Categories are only inferable when the needed annotation exists. Do not guess them from aggregate labels.

---

## 8. Research validity controls

**No temporal leakage.** At turn t, the judge receives the current request and only earlier history. All retrieval indexes are restricted to earlier authorized records. No future turn, target task description, task label, or harmful index may enter the prompt.

**No train-test leakage.** Never fit a threshold on test data; never tune retrieval k on test. Use benchmark training content only within allowed evaluation conditions.

**Chronology.** After selecting top-k history by relevance, present the selected items in original time order. Record a separate ablation if order is changed.

**Scope.** In E1, history is limited to each synthetic sequence or linked single-session benchmark trace. Cross-user linkage is not implied or simulated silently.

**No hidden truncation.** Context overflow is recorded and handled according to a predefined protocol; do not quietly delete history from `full` mode.

**Prompt injection.** Treat benchmark requests as untrusted content. The judge only emits a score and must not follow instructions in data. Failure to enforce this is itself a monitor reliability issue.

**Mock separation.** The repository includes synthetic markers with artificial labels to test data flow. These are *not* harmful intent classes, not a benchmark, and not a measured safety improvement. File output must state `SMOKE_TEST_ONLY`.

**Security and consent.** Do not print or commit sensitive prompts; do not place API keys in code or logs; check dataset license, model terms, and IRB/institutional review if nonpublic interactions are introduced.

---

## 9. Target architecture

```text
DecomposedHarm (accepted access; local-only)
            |
       Strict loader
            |
   Audit + split validation
            |
    for each sequence i:
      for t = 0..T-1:
        current = steps[t]
        prior   = steps[:t]              # never steps[t+1:]
            |
        select history by mode
            |
        fixed safety judge
            |
        risk score + token/time telemetry
            |
    VALIDATION threshold calibration
            |
    frozen threshold, TEST evaluation
            |
      metrics + confidence intervals
            |
    aggregated result tables and figures
```

### Repository layout

```text
stateful-safety-experiment/
  EXPERIMENT_BUILD_SPEC.md   # This document
  AGENTS.md                  # Agent execution rules
  CLAUDE.md                  # Claude Code entry point
  README.md                  # Developer quick start
  pyproject.toml
  configs/pilot.yaml
  fixtures/                   # Fictional synthetic markers ONLY
  src/stateful_safety/
    dataset.py                # strict adapter and audit
    schema.py                 # isolated monitor view, ground truth outside it
    retrieval.py              # BM25, recent, full, optional dense
    judge.py                  # mock software test and unvalidated local HF adapter
    runner.py                 # per-turn causal replay
    metrics.py                # sequence metrics; validation calibration
    statistics.py             # preliminary sequence bootstrap
    cli.py                    # audit / pilot / evaluate
  tests/
  scripts/
  results/                    # local aggregate results; gitignored
```

### Core software already implemented

- Strict dataset loader for JSONL, JSON, CSV.
- Explicit requirement to declare `harmful_index` index base.
- Non-text aggregate dataset audit.
- Per-turn current, full, recent, BM25, and optional dense selection.
- A deterministic mock judge for fictional fixtures.
- An unvalidated **local** Transformers inference adapter.
- Validation-only threshold calibration and test-only application in CLI `evaluate`.
- Sequence-level detection metrics and preliminary bootstrap.
- Tests for parser behavior, temporal selection, calibration, and smoke evaluation.

**Current limits:** The Hugging Face model backend and optional dense encoder have not been executed in this CPU-only, offline environment. The model judge prompt is not the paper's validated prompt, the dataset adapter has not yet been run on the gated benchmark, hybrid retrieval and oracle retrieval are not implemented, and current statistics are not publishable task-family confidence intervals.

---

## 10. Fast commands on your machine

Run these from the extracted repository root.

### Local software setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate           # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,science]'
python -m pytest -q
```

### Run the built-in synthetic smoke test

```bash
safety-study audit \
  --data fixtures/synthetic_fixture.jsonl \
  --index-base 0 \
  --out results/synthetic_audit.json

safety-study pilot \
  --data fixtures/synthetic_fixture.jsonl \
  --index-base 0 \
  --mode full --backend mock --threshold 0.9 \
  --out results/SMOKE_TEST_RESULTS.json

# Separate calibration and held-out evaluation, with fictional fixtures only:
safety-study evaluate \
  --validation fixtures/synthetic_validation_fixture.jsonl \
  --test fixtures/synthetic_test_fixture.jsonl \
  --index-base 0 --mode full --backend mock --fpr-budget 0.05 \
  --out results/synthetic_heldout_smoke.json
```

Outputs from these commands are **engineering verification only** and cannot go in the statement as benchmark results.

### Real data audit, after access approval

First discover the local benchmark's actual file layout and verify index semantics. Then run on a selected CSV or JSONL file:

```bash
safety-study audit \
  --data data/raw/decomposedharm/EXACT_SOURCE_FILE.csv \
  --index-base VERIFIED_0_OR_1 \
  --out results/real_dataset_audit.json
```

`EXACT_SOURCE_FILE.csv` and `VERIFIED_0_OR_1` are placeholders. Do **not** run literally, and do not guess these values. If the upstream file has different columns, create a tested adapter while preserving the original label meanings.

### Local model prototype, once GPU and model access are ready

```bash
python -m pip install -e '.[model,retrieval]'

safety-study pilot \
  --data data/processed/SMALL_APPROVED_EVAL_SUBSET.jsonl \
  --index-base VERIFIED_0_OR_1 \
  --mode full --backend hf \
  --model-id meta-llama/Llama-3.1-8B-Instruct \
  --out results/llama_full_unvalidated_pilot.json
```

The existing local HF adapter is a **prototype**, not a paper reproduction. The coding agent must first validate its model loading, token counts, prompt template, score parsing, generation length, and context limit. On a 4090, memory may require a documented quantized configuration. Do not silently change dtype or quantization midway through the study.

---

## 11. Paper reproduction procedure

The original author repository includes `src/1_to_k_binary_eval.py`, prompt templates, and latency scripts. Replicate the intended **cumulative-context** method with the original evaluation semantics, with these steps:

1. Clone the original repo into `third_party/Monitoring-Decomposition-Attack` only after network and license access are available. Pin its Git revision.
2. Read the paper and implementation. Document the official DSR definition, `harmful_index` interpretation, input format, prompting sweep, thresholds, and model configuration.
3. Run a small set of explicitly authorized, non-executing monitor classifications on a development partition. Check exact prompt formatting and output parsing.
4. Compute the official metric on the paper's published evaluation split, if reproducible. Record model configuration, seed, judge prompt, exceptions, timeouts, and costs.
5. Write `docs/reproduction_report.md` with our result, the paper's reported result, exact methodological differences, and uncertainty. A mismatch is a legitimate finding, not something to hide.
6. Freeze the reproduced full-history monitor before comparing history-selection mechanisms.

**Do not** assume that the simple score-formatting prompt in this starter matches the author's optimized monitor. The two should be separately named in results.

---

## 12. Detailed implementation tasks for Claude Code / Codex

### Phase 0: Make the environment reproducible

- [x] Create installable Python package and CLI.
- [x] Add strict synthetic fixtures and tests.
- [x] Run synthetic dataset audit and full-history smoke test.
- [x] Add separate validation-to-test evaluation path.
- [~] Add a pinned lockfile or a tested requirements constraints file after the deployment environment is known. *(CPU dev: `constraints/cpu-dev-py311-macos-arm64.txt`. GPU: `constraints/clipper-py311-linux-x86_64.txt` generated on clipper 2026-11-02; copy back to version it.)*
- [x] Add `scripts/system_report.py` to record CPU/GPU, Python, CUDA, PyTorch and package versions without secrets.
- [x] Add GitHub Actions or local CI for `pytest` and static checks. *(local `scripts/ci.sh`: ruff E9/F, pytest, fixture smoke CLI. No GitHub Actions: project lives inside a larger repo.)*

**Gate:** Automated tests pass on a CPU without credentials or a GPU.

### Phase 1: Build validated DecomposedHarm data adapter

- [x] Accept dataset access terms as the **human user**, not via bypass. *(2026-11-02, account Satya-Chamana.)*
- [x] Download authorized data and original code to local working directory. *(HF revision `25e1d59` in `data/raw/decomposedharm/`; code `79f5b41` in `third_party/`; both gitignored. CSV hashes identical across sources.)*
- [x] Create `docs/dataset_manifest.md` with all hashes, revisions, and column definitions.
- [x] Inspect the benchmark index base, harmful boundary, and task-type specific logic using documented small examples; do not print sensitive task text in reports. *(From upstream code + paper: 1-based, inclusive; per-type construction differs. See manifest §3.)*
- [x] Implement schema adapters for the actual upstream file layout; reject missing or inconsistent fields. *(`--schema decomposedharm`; validated on fictional fixtures only.)*
- [x] Audit duplicates, split/family leakage, null indices, and total counts. *(Real audit 2026-11-02: 4,641 rows, 0 rejected; 1 same-steps/opposite-label pair; 7 cross-split goal matches; paper subset pinned in `configs/paper_eval_split.json`. Manifest §5-6.)*
- [x] Add fixture-based tests for each upstream schema shape, using fictional substituted content. *(CSV: list cells for QA/image, multi-line text for agent. Paper-subset JSON covered by `tests/test_build_paper_split.py`.)*

**Gate:** A reproducible data audit matches selected source files and exposes unresolved discrepancies.

### Phase 2: Baseline reproduction

- [x] Match the paper's sequential monitor prompt, label conventions, and DSR metric. *(`--protocol paper`, CPU-tested; sweep matches upstream code on fictional scores. Notes §2b.)*
- [x] Pin a Llama-3.1-8B-Instruct model revision. *(`0e9e39f249a16976918f6564b8830bc894c89659`, used for all runs.)*
- [x] Validate local inference on an RTX 4090 or H100. If CUDA unavailable, do not claim run success. *(clipper H100 NVL, bf16; full run job 798355.)*
- [x] Compute actual token counts, decoding failures, context overflow, and latency. *(Report §3: 5,202 calls, 11 no-number replies, 0 overflows, ~0.6 s/call.)*
- [x] Calibrate per paper protocol on validation only.
- [x] Evaluate exact held-out split; compare with paper without forcing agreement. *(3 prompts; report §0.)*
- [x] Write `docs/reproduction_report.md` and aggregate JSON.

**Gate:** Published protocol deviations are documented. No leaked ground-truth data enters the judge view.

### Phase 3: Controlled evidence-selection comparison

- [x] Finish C0 current, C1 recent, C2 full, C3 BM25, C4 dense, and C5 hybrid modes. *(2026-11-02; CPU-tested with fakes; real encoder runs on clipper.)*
- [x] For dense retrieval, instantiate the encoder once and batch/cache embeddings by sequence; do not reload for every turn. *(`DenseScorer`: one load per run, each text embedded once.)*
- [x] For hybrid RRF, use **relevance-sorted** ranks, then reorder selected items chronologically for judging. *(RRF k=60.)*
- [ ] Add k = 1, 3, 5, 10. Compute actual tokenizer budget.
- [x] Run same judge and evaluation split in each condition. *(Validation 798524, test 798699; 10 pilot conditions; Agent-ICL; 264 / 528 sequences.)*
- [x] Calibrate thresholds on validation. Freeze before testing. *(Frozen 2026-11-03T18:44Z, `configs/phase3_thresholds.json`; test run once, 798699.)*
- [ ] Produce plots of timely recall vs input tokens, FPR vs recall, and latency vs k.

**Gate:** Every comparison has its own validated threshold, held-out results, and complete-system cost.

### Phase 4: Mechanism ablations

> **Deferred 2026-11-03** (study stopped after Phase 3 at tag `phase3-v1`; see experiment log session 3a). Phase 5 likewise.

- [ ] Create independent evidence annotations on a small, prespecified subset.
- [ ] Add oracle-selected top-k history using annotations; oracle cannot access future turn evidence.
- [ ] Add remove/restore/replace ablations with matched benign controls.
- [ ] Attribute differences to retrieval vs interpretation only where evidence supports the distinction.
- [ ] Include cases in which full history or oracle evidence makes the judge worse.

**Gate:** At least one diagnostic plot or table explains *why* a monitor decision changes, including negative cases.

### Phase 5: Robustness, generalization, and statistical quality

- [ ] Audit and add matched benign, legitimate decomposition workflows where justified.
- [ ] Introduce bounded benign distractors without changing label timing incorrectly.
- [ ] Create task-family grouping and cluster-bootstrap CI, if defensible. Otherwise describe limits and use conservative alternatives.
- [ ] Compare paired effects and show denominator counts.
- [ ] Calculate index build and embedding amortized costs.
- [ ] Freeze confirmatory analyses before opening any new external test set.

**Gate:** A paper-ready experiment table exists with limitations and no unverified novelty assertions.

### Phase 6: Later study, do not begin automatically

- E2: request-only versus request-response versus authorized tool-result evidence on a benchmark that actually includes those outputs.
- E3: bounded retention, deletion, access control, and residual leakage under a defined privacy threat model.
- Adaptive threat models only if they remain within approved, controlled evaluation and do not execute harmful tasks.

---

## 13. Acceptance criteria

A complete **E1 baseline** must meet all of these:

- [ ] Source, data revision, split, model revision, prompt, and threshold protocol are recorded.
- [ ] The judge sees no `original_task`, `label`, `harmful_index`, or future requests.
- [ ] Harm boundary and index base have documented source validation.
- [ ] Validation and test roles remain strictly separate.
- [ ] Paper DSR and our diagnostic metrics are clearly distinguished.
- [ ] At least C0/current and C2/full run with the same fixed local model.
- [ ] At least C1/recent, C3/BM25 and C4/dense run on the same held-out split.
- [ ] Results report denominators, 95% CI where supportable, FPR, timely recall, first block, token use, and full costs.
- [ ] Raw harmful prompts and private data are not committed or included in exported reports.
- [ ] Every failed parse, overflow, or unsuccessful run is counted and disclosed.
- [ ] The final report identifies one strength, one failure mode, one limitation, and one next discriminating experiment.

No single numeric detection target is a valid acceptance criterion. A clear, reproducible negative result is acceptable.

---

## 14. Proposed result files

```
results/
  data_audit_YYYYMMDD.json
  reference_baseline_YYYYMMDD.json
  comparisons_YYYYMMDD.csv
  bootstrap_intervals_YYYYMMDD.csv
  resource_telemetry_YYYYMMDD.json
  figures/
    timely_recall_vs_cost.png
    fpr_tradeoff.png
    k_sensitivity.png
    failure_attribution.png

docs/
  dataset_manifest.md
  reproduction_report.md
  experiment_log.md
  research_findings.md
```

Files in `results/` and local data directories must not contain the raw benchmark prompts. Use hashed IDs for diagnostic joins and obtain institutional approval before sharing private traces. Aggregate statistics can be exported after review.

---

## 15. Decision log to update during implementation

| Decision | Current choice | Why | Review before |
| --- | --- | --- | --- |
| First benchmark | DecomposedHarm, start with QA | Relevant sequence labels; published reference | Real download |
| First monitoring stage | Pre-response request-only | Makes history effects identifiable | E2 |
| Judge | Llama-3.1-8B-Instruct | Open model and published reference | GPU run |
| Primary metric | Timely recall at 5% sequence FPR | Interventions matter more than any block | Baseline protocol check |
| Main comparator | Full history vs recent and retrieved | Direct mechanism question | Test freeze |
| Retrieval depths | 1, 3, 5, 10 | Sensitivity analysis | Validation |
| Hardware | **NVIDIA H100 NVL, clipper `short-gpu` (4 h) / `gpu` (5 days), all runs** (decided 2026-11-02; briefly V100S on a wrong premise) | Native bf16, long jobs available | Fixed |
| Model dtype | **bfloat16 for every run** | Llama 3.1 native precision; one dtype study-wide; remaining deviation: local HF vs Together Turbo | Fixed |
| Dataset handling | Evaluation only, local | Respect authors' terms | Any publication |
| Index base | **1-based, inclusive** (`harm_index`) | Verified from upstream code + paper §3.1, 2026-11-02 | Real-file audit |
| Task-family grouping | **Not yet established** | Avoid false cluster-bootstrap claim | Inference |
| Replication split | Paper subset `configs/paper_eval_split.json` (val 264, test 528) | Matches upstream `combined/*.json` exactly | Phase 2 |
| Conflicting-label pair, goal overlaps | Keep all rows (primary); rule-based exclusion sensitivity, frozen `configs/sensitivity_exclusions.json` | User decision 2026-11-02 | Fixed |
| Paper threshold selector | Primary `upstream_sweep`; `aligned_sweep` as sensitivity | Replicate the paper's actual procedure; disclose the shift | Fixed |
| Controlled-study split | **OPEN** | Paper subset vs full union (larger benign denominators) | Phase 3 |
| Hybrid and oracle | Hybrid implemented (RRF); oracle not yet | Oracle needs annotations | Phase 4 |
| Phase 3 judge | Agent-ICL prompt, paper format, selected history | Best on validation (0.922); user decision 2026-11-02 | Fixed |
| Phase 3 test set | Reuse the 528 paper test sequences, disclosed | Fresh hold-out would be 172 unbalanced rows | Fixed |
| Phase 3 primary k / margin | k=3; non-inferiority margin 0.05 timely recall | k5 indistinguishable from full on validation; 5 points = largest practically acceptable loss; frozen in `configs/phase3_thresholds.json` 2026-11-03T18:44Z | Fixed |

Update this table in the code repository when a decision changes.

---

## 16. Ethics and publication rules

- Do not run tasks the benchmark describes. Classify requests without responding to their substantive content.
- Do not train on DecomposedHarm unless the authors explicitly authorize it, contrary to their current evaluation-only request.
- No generation of new harmful examples is needed for E1. Reuse approved static evaluation traces.
- Do not claim 0% false positives based on a small sample. State denominators and uncertainty.
- Keep all high-level novelty statements narrow: the proposed mechanism study examines evidence selection and decision dependence; earlier papers already explored sequential monitors, clustering, timing, intent encoders, and capability accumulation.
- For future multi-user experiments, use approved, authorized linkage only. Semantic similarity does not grant permission to use another person's data.
- Keep paper result tables, mock outputs, and actual results clearly separated.
- If credentials or approvals are missing, implement the **next safe step** and report the blocker. Do not bypass it.

---

## 17. How to work with a coding agent

### Claude Code

Open the extracted folder in your terminal, then run:

```bash
claude
```

Paste:

```text
Read CLAUDE.md and EXPERIMENT_BUILD_SPEC.md completely. Continue the existing
codebase. First run the current tests and synthetic smoke evaluation. Then
start the next unchecked Phase 0 or Phase 1 task. Follow the research validity,
benchmark licensing and data protection rules. Make small tested commits.
Update docs/experiment_log.md after each milestone. Never claim mock results
are scientific results. If Hugging Face approval or CUDA blocks a real run,
finish the data-independent work and report the exact next command.
```

### OpenAI Codex

Open the repository in Codex or run your installed Codex CLI from the root:

```bash
codex
```

Paste:

```text
Read AGENTS.md and EXPERIMENT_BUILD_SPEC.md. Inspect the existing code and
run pytest. Continue with the first unchecked Phase 0 or Phase 1 task.
Keep changes incremental, tested, and reproducible. Do not fabricate results
or bypass dataset restrictions. Show the final changed files, tests run,
research risks, and next step requiring the user's GPU or approved dataset.
```

A coding agent should ask you only for truly missing permissions, file access, or hardware. It should not reinterpret research claims silently or change ground truth to make a metric look better.

---

## 18. CURRENT STATUS and exact next work

### Verified in this ChatGPT container (2026-11-02)

- Python 3.13.5 (the deployment target is Python 3.11).
- CPU-only PyTorch; `nvidia-smi` is not available.
- GitHub and Hugging Face network resolution was unavailable here.
- Basic synthetic fixture, audit, BM25/recent/full/current history selection, mock judge, validation/test score pipeline, and strict `harmful_index` handling implemented.
- All automated tests passed after test corrections, and the synthetic CLI produced aggregate results.
- A local Transformers inference backend was **written but not executed**. The paper-accurate model prompt/evaluation protocol still needs implementing.

### Next actions, in order

1. **Completed:** `scripts/system_report.py` reports versions and CUDA availability, without secrets.
2. **Completed:** exact-match cross-split checks reject repeated IDs and repeated step sequences; tests cover both.
3. **Completed (2026-11-02):** local CI (`scripts/ci.sh`); CPU dev constraints. GPU lock still pending.
4. **Completed (2026-11-02):** terms accepted by user, data downloaded, real audit run. Gate met with open items in manifest §6.
5. **Completed from code (2026-11-02):** upstream columns are `decomposition`/`harm_index`; 1-based inclusive. Adapter + fictional tests done. Real-file audit still to run.
6. **Completed (2026-11-02):** reference repo pinned at `79f5b41`; protocol, DSR and deviations in `docs/reference_implementation_notes.md`.
7. **Completed (2026-11-02):** paper mode, `--ids`, `--exclude`, frozen exclusion list.
8. **Completed (2026-11-02):** `--model-revision` (required, full commit) and `paper-smoke` (validation-only reply check).
9. **Next (needs GPU + Llama license):** pin a Llama-3.1-8B-Instruct revision, run `--protocol paper` on the paper subset with `BINARY_BASELINE` then `BINARY_INTENTION_EVAL_W_8_AGENT_ICL`.

### Handoff success report format

At the end of each coding-agent session, write `docs/experiment_log.md`:

- Date / Git commit / machine / package versions;
- Task completed, modified files, and test commands;
- Aggregate numerical outputs from real data, if any, with source identifiers;
- What was NOT run or validated;
- Detected research or ethics risks;
- The next safe executable action.

**End of build specification.**
