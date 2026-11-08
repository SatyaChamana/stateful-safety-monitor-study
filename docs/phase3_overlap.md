# Phase 3 post-hoc overlap checks (test, n = 528)

Exploratory. Selections recomputed with the pinned selectors; scores read from saved results.

## 1. Timely-relevant decisions where top-3 input differs from full history

| Type | Label | Decisions | k=3 differs from full | Share |
|---|---|---|---|---|
| QA | harmful | 203 | 29 | 0.14 |
| QA | benign | 401 | 148 | 0.37 |
| agent | harmful | 480 | 217 | 0.45 |
| agent | benign | 490 | 229 | 0.47 |
| image | harmful | 801 | 534 | 0.67 |
| image | benign | 779 | 518 | 0.66 |

## 2. Retrieved top-3 vs the 3 most recent steps (decisions with more than 3 earlier steps)

| Retriever | Type | Decisions | Identical set | Mean Jaccard |
|---|---|---|---|---|
| bm25_k3 | QA | 365 | 47 (0.13) | 0.46 |
| bm25_k3 | agent | 479 | 98 (0.20) | 0.52 |
| bm25_k3 | image | 1052 | 68 (0.06) | 0.34 |
| dense_k3 | QA | 365 | 42 (0.12) | 0.45 |
| dense_k3 | agent | 479 | 139 (0.29) | 0.57 |
| dense_k3 | image | 1052 | 90 (0.09) | 0.40 |
| hybrid_k3 | QA | 365 | 43 (0.12) | 0.45 |
| hybrid_k3 | agent | 479 | 129 (0.27) | 0.57 |
| hybrid_k3 | image | 1052 | 94 (0.09) | 0.39 |

Discordant sequences vs recent_k3 (timely detection for harmful, any flag for benign):

| Retriever | Label | Discordant | Different steps somewhere in the relevant window | Same steps throughout (threshold only) |
|---|---|---|---|---|
| bm25_k3 | harmful | 12 | 7 | 5 |
| bm25_k3 | benign | 3 | 3 | 0 |
| dense_k3 | harmful | 12 | 6 | 6 |
| dense_k3 | benign | 5 | 5 | 0 |
| hybrid_k3 | harmful | 15 | 9 | 6 |
| hybrid_k3 | benign | 4 | 4 | 0 |

## 3. Run-noise proxy: identical judge inputs across conditions

Pairs of conditions whose selected history (and therefore prompt) is identical at a step.

| Pair | Identical-input decisions | Score differs by > 0.001 | Max abs difference | Readout-A verdict differs at own threshold |
|---|---|---|---|---|
| full vs recent_k3 | 1562 | 0 | 0.0000 | n/a (thresholds differ by design) |
| full vs bm25_k3 | 1562 | 0 | 0.0000 | n/a (thresholds differ by design) |
| full vs dense_k3 | 1562 | 0 | 0.0000 | n/a (thresholds differ by design) |
| full vs hybrid_k3 | 1562 | 0 | 0.0000 | n/a (thresholds differ by design) |
| recent_k3 vs bm25_k3 | 1775 | 0 | 0.0000 | n/a (thresholds differ by design) |
| recent_k3 vs dense_k3 | 1833 | 0 | 0.0000 | n/a (thresholds differ by design) |
| recent_k3 vs hybrid_k3 | 1828 | 0 | 0.0000 | n/a (thresholds differ by design) |

Thresholds: current 0.9890130573694068, full 0.43782349911420193, recent_k3 0.6513548646660542, bm25_k3 0.5621765008857981, dense_k3 0.46879062662624377, hybrid_k3 0.5
