"""Simple grouped bootstrap on independent sequence IDs; task-family bootstrap is a TODO."""
from __future__ import annotations
import random
from .metrics import summarize
from .schema import Sequence


def bootstrap_sequence_ci(records: list[Sequence], scores: dict[str,list[float]],
                          threshold: float, metric: str, *, n_boot=1000, seed=1729):
    """An initial sequence bootstrap; do NOT claim task-family inference."""
    rng=random.Random(seed)
    if not records:
        raise ValueError("No records")
    out=[]
    for _ in range(n_boot):
        sampled=[records[rng.randrange(len(records))] for _ in range(len(records))]
        result=summarize(sampled,scores,threshold).get(metric)
        if result is not None:
            out.append(float(result))
    if not out:
        return None
    out.sort()
    return {
        "lower":out[int(0.025*(len(out)-1))],
        "upper":out[int(0.975*(len(out)-1))],
        "resamples_with_defined_metric":len(out),
        "method":"sequence_resampling_only_NOT_task_family_bootstrap",
    }


def bootstrap_outcome_counts(tp: int, fp: int, fn: int, tn: int, n_boot: int = 2000,
                             seed: int = 1729) -> dict:
    """Sequence bootstrap of paper-protocol F1 and DSR from outcome counts alone.

    Each sequence has exactly one outcome (TP/FP/FN/TN), so resampling sequences with
    replacement equals a multinomial draw over the four counts. SEQUENCE-level only;
    not a task-family bootstrap. Matches the paper's "standard error via bootstrapping" in
    kind, not necessarily in its exact resampling unit.
    """
    n = tp + fp + fn + tn
    if n == 0:
        raise ValueError("No sequences to bootstrap")
    rng = random.Random(seed)
    labels = ["TP"] * tp + ["FP"] * fp + ["FN"] * fn + ["TN"] * tn

    def metrics(c):
        precision = c["TP"] / (c["TP"] + c["FP"]) if c["TP"] + c["FP"] else 0.0
        recall = c["TP"] / (c["TP"] + c["FN"]) if c["TP"] + c["FN"] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return f1, recall

    draws = {"F1": [], "DSR_recall": []}
    for _ in range(n_boot):
        c = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
        for _ in range(n):
            c[labels[rng.randrange(n)]] += 1
        f1, recall = metrics(c)
        draws["F1"].append(f1)
        draws["DSR_recall"].append(recall)
    point = dict(zip(("F1", "DSR_recall"), metrics({"TP": tp, "FP": fp, "FN": fn, "TN": tn})))
    out = {"n_sequences": n, "n_boot": n_boot, "seed": seed, "unit": "SEQUENCE_BOOTSTRAP"}
    for name, values in draws.items():
        values.sort()
        mean = sum(values) / n_boot
        se = (sum((v - mean) ** 2 for v in values) / (n_boot - 1)) ** 0.5
        out[name] = {"point": point[name], "se": se,
                     "ci95": [values[int(0.025 * n_boot)], values[int(0.975 * n_boot) - 1]]}
    return out
