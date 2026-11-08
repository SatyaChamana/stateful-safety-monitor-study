# Phase 3 report: which earlier steps should a sequential monitor see?

Status: **test stage run once (job 798699) with thresholds frozen beforehand.** §3 is the pre-registered analysis; §5 is exploratory and was decided after unblinding. Aggregates only; no request text.

## 0. Summary

| Hypothesis (pre-registered) | Result on test | Verdict |
| --- | --- | --- |
| H1: full history beats current-only | timely recall +0.114 [+0.068, +0.156], but benign FPR also higher by 0.049 [+0.019, +0.079] | **Supported on the primary readout, with a false-alarm cost**; not a same-FPR win (§5.1) |
| H2: top-3 retrieval stays within 0.05 of full history | bm25 +0.004, dense +0.011, hybrid +0.015; all lower bounds ≥ −0.011 | **Supported for all three retrievers** |
| H4: retrievers beat recency at k = 3 | +0.030 / +0.038 on readout A, with higher FPR; ≈ 0 on readout B | **Not supported** |

Plain reading: on DecomposedHarm, an 8B judge given its three most relevant earlier steps detects as well as one given all of them. History matters for decomposed QA and agent attacks, but it also makes the judge flag more benign agent and image sequences. Smart retrieval does not reliably beat simply taking the last three steps.

## 1. Question and design

One fixed judge; the **only** thing that changes between conditions is which earlier steps the judge sees at each decision.

- **Judge:** meta-llama/Llama-3.1-8B-Instruct @ `0e9e39f2…`, bf16, H100 NVL, greedy decoding, prompt `BINARY_INTENTION_EVAL_W_8_AGENT_ICL` (best on validation in Phase 2; reproduced paper F1 0.898, see `reproduction_report.md` §3e).
- **Conditions (10):** `current` (no history), `full` (all earlier steps), and top-k selection with k ∈ {3, 5} by `recent`, `bm25` (k1 1.2, b 0.75), `dense` (all-MiniLM-L6-v2 @ `1110a243…`, no fine-tuning) and `hybrid` (RRF, k = 60). Selected steps are presented in chronological order; selection sees only earlier steps.
- **Data:** DecomposedHarm @ `25e1d591…`, paper split pinned by ID. Validation: 264 sequences (127 benign / 137 harmful). Test: 528 (265 / 263), **the same test set used in Phase 2** (reuse disclosed in the plan before any Phase 3 run).
- **Scoring:** step 1 never scored; a sequence fires at its first step whose score ≥ threshold. **Timely recall** = harmful sequences first flagged at or before the harm step / harmful sequences. **Sequence FPR** = benign sequences flagged at any step / benign sequences.
- **Readouts (amendment 1):**
  - A (primary): P("1") / (P("0") + P("1")) at the judge's first generated token, threshold fitted per condition on validation to sequence FPR ≤ 0.05.
  - B (secondary): the parsed reply at a fixed cutoff ≥ 1, no calibration.
- **Statistics:** paired sequence bootstrap on identical IDs, 2,000 draws, seed 1729, sequence-level (not task-family). 95% percentile intervals.

### Timeline (all UTC, 2026-11-03)
| Time | Event |
| --- | --- |
| 03:43 | Plan pre-registered (`configs/phase3_plan.json`): conditions, metric, H1/H2/H4, k = 3 proposed, margin to be set from validation. |
| — | Validation run 798493 exposed a calibration cliff: parsed scores are near-binary, so a 5% budget on 127 benign sequences forced some thresholds onto a misparsed 3.0 or none. **Superseded, never frozen.** |
| 04:27 | Amendment 1 (`configs/phase3_amendment_1.json`): readouts A + B, thresholds restricted to [0, 1]. No test data seen. |
| — | Validation run 798524 under the amendment. |
| 18:44 | Frozen (`configs/phase3_thresholds.json`): primary k = 3, margin 0.05 (largest acceptable loss of timely recall; k = 5 matched `full` on every validation sequence, so it cannot separate retrieval from full history). |
| — | Test run 798699, once. |

## 2. Readout validity

| Check | Validation | Test |
| --- | --- | --- |
| Replies parsed | 1,744 / 1,744 per condition | all |
| First-token P(0) + P(1): min; decisions < 0.5 | ≥ 0.992; 0 | median ≈ 1.000; 0 |
| A (≥ 0.5) agrees with parsed 0/1 | 99.7–99.9% | 99.5–99.8% |

The continuous score is a faithful version of the judge's own answer, with enough spread among benign sequences for FPR calibration (for example `full`, benign max score: 109 below 0.1, 18 between 0.1 and 0.99).

## 3. Pre-registered results (test, readout A, frozen thresholds)

| Condition | Threshold | Benign flagged (FPR) | Timely recall | Any detection |
| --- | --- | --- | --- | --- |
| `current` | 0.989 | 5 / 265 (0.019) | 0.684 | 0.757 |
| `full` | 0.438 | 18 / 265 (0.068) | 0.798 | 0.852 |
| `bm25_k3` | 0.562 | 15 (0.057) | 0.802 | 0.871 |
| `dense_k3` | 0.469 | 17 (0.064) | 0.810 | 0.875 |
| `hybrid_k3` | 0.500 | 16 (0.060) | 0.814 | 0.882 |
| `recent_k3` | 0.651 | 12 (0.045) | 0.772 | 0.856 |
| `bm25_k5` | 0.438 | 18 (0.068) | 0.802 | 0.859 |
| `dense_k5` | 0.438 | 18 (0.068) | 0.795 | 0.852 |
| `hybrid_k5` | 0.438 | 18 (0.068) | 0.798 | 0.856 |
| `recent_k5` | 0.469 | 16 (0.060) | 0.791 | 0.856 |

Denominators: 263 harmful, 265 benign.

### H1: full vs current
| | Difference (full − current) [95% CI] | Discordant (current only / full only) |
| --- | --- | --- |
| Timely recall | +0.114 [+0.068, +0.156] | 4 / 34 |
| Sequence FPR | +0.049 [+0.019, +0.079] | — |

Full history detects more attacks in time, and it also flags more benign sequences. The thresholds were fitted to the same validation FPR (0.047), but they transferred to different test FPRs, so this is not a comparison at equal false alarms (see §5.1).

### H2: non-inferiority of top-3 retrieval vs full (margin 0.05)
| Candidate | Timely recall diff [95% CI] (SE) | Lower bound > −0.05? | FPR diff [95% CI] | Discordant (full only / cand only) |
| --- | --- | --- | --- | --- |
| `bm25_k3` | +0.004 [−0.011, +0.019] (0.008) | yes | −0.011 [−0.034, +0.011] | 2 / 3 |
| `dense_k3` | +0.011 [−0.008, +0.034] (0.010) | yes | −0.004 [−0.015, +0.008] | 2 / 5 |
| `hybrid_k3` | +0.015 [−0.004, +0.038] (0.011) | yes | −0.008 [−0.023, +0.008] | 2 / 6 |

All three are non-inferior, without a higher FPR. The conclusion survives a Bonferroni split across the three candidates (worst lower bound −0.011 against −0.05). It also holds at matched test FPR (§5.1). About 55% of k = 3 decisions saw fewer steps than `full`.

### H4: retrievers vs recency at k = 3
| Candidate − `recent_k3` | Readout A timely diff | Readout A FPR diff | Readout B timely diff | Readout B FPR diff |
| --- | --- | --- | --- | --- |
| `bm25_k3` | +0.030 [+0.008, +0.057] | +0.011 [0.000, +0.026] | −0.008 [−0.023, +0.008] | +0.004 [0.000, +0.011] |
| `dense_k3` | +0.038 [+0.015, +0.065] | +0.019 [+0.004, +0.038] | −0.004 [−0.019, +0.008] | −0.004 [−0.011, 0.000] |

On readout A, the retrievers' gain comes with more benign flags, because `recent_k3`'s threshold (0.651) transferred to a lower test FPR (0.045). Under one fixed cutoff for every condition (readout B), the difference disappears. H4 is not supported.

## 4. Secondary results

**Readout B (fixed cutoff ≥ 1, test):**
- `current`: timely recall 0.913, FPR 0.313.
- `full`: 0.776, FPR 0.053.
- k = 3 retrievers: 0.802–0.814, FPR 0.053–0.060.
- `recent_k3`: 0.810, FPR 0.057.

Without calibration, current-only flags almost a third of benign sequences.

**k = 5:** all four selectors are indistinguishable from `full` (timely recall differences −0.008 to +0.004). About 71% of their decisions already see every earlier step.

**Calibration transfer:** a validation fit of 6/127 benign flags gave test FPRs from 0.019 (`current`) to 0.068 (`full` and the k = 5 conditions). `full` exceeds the 0.05 budget on test. A budget estimated from 127 benign sequences is coarse: one sequence is 0.8 points of FPR.

**Cost:**
- Input tokens per decision: current 1,477; k = 3 about 1,525; k = 5 about 1,544; full 1,556.
- The fixed few-shot prompt dominates, so retrieval saves about 2% of input here.
- Dense retrieval adds embedding time once per text.
- Seconds per decision depend on the node and are not compared.

## 5. Exploratory analyses (decided after unblinding; not confirmatory)

Source: `scripts/phase3_exploratory.py results/phase3a1_test_*.json`.

### 5.1 Equal false alarms (threshold tuned on test)
Timely recall at the lowest threshold that flags at most F benign sequences (actual flag count in parentheses):

| Condition | F = 5 | F = 13 | F = 18 |
| --- | --- | --- | --- |
| `current` | 0.684 (5) | 0.772 (12) | 0.787 (18) |
| `full` | 0.658 (5) | 0.772 (13) | 0.798 (18) |
| `bm25_k3` | 0.700 (5) | 0.760 (13) | 0.802 (18) |
| `dense_k3` | 0.700 (5) | 0.802 (13) | 0.810 (17) |
| `hybrid_k3` | 0.707 (5) | 0.810 (13) | 0.821 (17) |
| `recent_k3` | 0.711 (5) | 0.787 (13) | 0.810 (18) |

At equal false alarms, full history does **not** beat current-only overall. Recency is not worse than retrieval, and k = 3 is not worse than full. The H1 gain in §3 therefore reflects a different operating point as much as better discrimination overall.

### 5.2 By task type (frozen thresholds)
| Comparison | Type | Timely diff [95% CI] | FPR diff [95% CI] | Discordant (ref / cand) |
| --- | --- | --- | --- | --- |
| full − current | QA (86 h / 90 b) | +0.198 [+0.093, +0.302] | 0.000 | 3 / 20 |
| full − current | agent (88 / 88) | +0.148 [+0.080, +0.227] | +0.057 [0.000, +0.114] | 0 / 13 |
| full − current | image (89 / 87) | 0.000 [−0.034, +0.034] | +0.092 [+0.023, +0.161] | 1 / 1 |
| hybrid_k3 − recent_k3 | QA | +0.093 [+0.035, +0.163] | 0.000 | 0 / 8 |
| hybrid_k3 − recent_k3 | agent | +0.034 [−0.023, +0.091] | +0.011 [0.000, +0.034] | 2 / 5 |
| hybrid_k3 − full | QA | 0.000 [−0.035, +0.035] | 0.000 | 1 / 1 |

- **Decomposed QA (no benign QA flags in any condition):** this is where history helps cleanly. Full history doubles the share of QA attacks caught in time (0.291 → 0.488). Hybrid retrieval matches full and beats recency by 9 points, because the relevant QA pieces are not always the most recent ones.
- **Image and agent:** history adds benign false alarms (image benign FPR: current 0.023, full 0.115). Image attacks are caught near-perfectly even without history.
- These are post-hoc, per-type, multiple comparisons. The QA retrieval-vs-recency result is a hypothesis for a pre-registered follow-up, not a finding.

## 6. Limitations
- **Test reuse:** the 528 test sequences were also evaluated in Phase 2 (a different study question; no Phase 3 choice used them). Disclosed in advance; a fresh hold-out would have been 172 unbalanced rows.
- **Threshold transfer:** test FPRs differ between conditions (0.019–0.068), which confounds H1 and H4 on readout A. A larger benign calibration set, or a comparison at fixed test FPR pre-registered in advance, would remove this.
- **Short sequences:** most decisions have few earlier steps (at k = 5, about 71% see all of them), so this benchmark cannot show retrieval's cost advantage or its behaviour on long histories.
- **Single judge, single seed:** one 8B model with greedy decoding. Other judges may use history differently.
- **Statistics:** sequence-level bootstrap. Task-family clustering is not modelled, so intervals may be too narrow if families are correlated.
- **Amendment:** the readout changed after a first validation run (never on test). The reasons and the superseded files are recorded in the amendment.

## 7. What can and cannot be claimed
- **Can:**
  - On DecomposedHarm, with this judge, top-3 selection by BM25, dense or hybrid retrieval is non-inferior to full history in timely recall (margin 0.05), with no increase in false alarms.
  - At the frozen operating points, full history catches more attacks than current-only, at a higher false-alarm rate.
- **Cannot:**
  - That history improves discrimination overall at equal false alarms (not supported exploratorily).
  - That learned or lexical retrieval beats recency (H4 not supported).
  - That retrieval reduces cost on this benchmark.
  - That any result generalizes to longer histories or other judges.

## 8. Reproduce
```bash
python scripts/summarize_phase3.py results/phase3a1_test_*.json                         # §3, §4
python scripts/summarize_phase3.py --reference recent_k3 results/phase3a1_test_*.json   # H4
python scripts/phase3_exploratory.py results/phase3a1_test_*.json                       # §5
```
Inputs: `configs/phase3_plan.json`, `configs/phase3_amendment_1.json`, `configs/phase3_thresholds.json` (SHA-256 of every validation file inside). Results files are gitignored and stay on the local machine.
