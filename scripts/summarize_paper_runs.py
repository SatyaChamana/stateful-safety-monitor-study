#!/usr/bin/env python3
"""Summarize paper-mode result JSONs (aggregates only) for docs/reproduction_report.md.

For each file: test F1/DSR with count-based sequence bootstrap, benign flag rate, per-type
counts. For every pair of files that both carry per_sequence records: a paired comparison on
the SAME test sequences (paired sequence bootstrap of the F1 and DSR difference, and
discordant counts). Prints Markdown; reads no request text.
"""
from __future__ import annotations
import argparse
import json
import random
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stateful_safety.statistics import bootstrap_outcome_counts  # noqa: E402

KEYS = ("TP", "FP", "FN", "TN")


def _outcome(rec: dict) -> str:
    fired = rec["first_fire_step"]
    if rec["label"] == "benign":
        return "FP" if fired is not None else "TN"
    return "TP" if fired is not None and fired <= rec["harm_step"] else "FN"


def _f1_dsr(c: dict) -> tuple[float, float]:
    p = c["TP"] / (c["TP"] + c["FP"]) if c["TP"] + c["FP"] else 0.0
    r = c["TP"] / (c["TP"] + c["FN"]) if c["TP"] + c["FN"] else 0.0
    return (2 * p * r / (p + r) if p + r else 0.0), r


def paired(a: dict, b: dict, n_boot: int = 2000, seed: int = 1729) -> dict:
    """a, b: id -> outcome on the same test IDs. Paired bootstrap of (b - a)."""
    if set(a) != set(b):
        raise ValueError("Paired comparison needs identical test IDs")
    ids = sorted(a)
    rng = random.Random(seed)

    def diff(sample):
        ca = {k: 0 for k in KEYS}
        cb = {k: 0 for k in KEYS}
        for i in sample:
            ca[a[i]] += 1
            cb[b[i]] += 1
        (fa, ra), (fb, rb) = _f1_dsr(ca), _f1_dsr(cb)
        return fb - fa, rb - ra

    point = diff(ids)
    draws = [diff([ids[rng.randrange(len(ids))] for _ in ids]) for _ in range(n_boot)]
    out = {"n": len(ids), "unit": "PAIRED_SEQUENCE_BOOTSTRAP"}
    for k, name in enumerate(("F1_diff", "DSR_diff")):
        vals = sorted(d[k] for d in draws)
        out[name] = {"point": point[k], "ci95": [vals[int(0.025 * n_boot)], vals[int(0.975 * n_boot) - 1]]}
    harmful = [i for i in ids if a[i] in ("TP", "FN")]
    out["harmful_caught_only_by_a"] = sum(a[i] == "TP" and b[i] == "FN" for i in harmful)
    out["harmful_caught_only_by_b"] = sum(a[i] == "FN" and b[i] == "TP" for i in harmful)
    benign = [i for i in ids if a[i] in ("FP", "TN")]
    out["benign_flagged_only_by_a"] = sum(a[i] == "FP" and b[i] == "TN" for i in benign)
    out["benign_flagged_only_by_b"] = sum(a[i] == "TN" and b[i] == "FP" for i in benign)
    return out


def summarize(paths: list[Path], block: str = "primary_all_rows") -> str:
    lines, per_seq = [], {}
    lines.append("| Prompt | Threshold | n | TP | FP | FN | TN | F1 (SE) [95% CI] | DSR (SE) [95% CI] | Benign flagged |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    type_lines = []
    for path in paths:
        data = json.loads(path.read_text())
        entry = data[block]["upstream_sweep"]
        t = entry["test"]
        b = bootstrap_outcome_counts(*(t[k] for k in KEYS))
        lines.append(
            f"| `{data['paper_prompt']}` | {entry['threshold_from_validation']} | {b['n_sequences']} | "
            + " | ".join(str(t[k]) for k in KEYS)
            + f" | {t['F1']:.3f} ({b['F1']['se']:.3f}) [{b['F1']['ci95'][0]:.3f}, {b['F1']['ci95'][1]:.3f}]"
            + f" | {t['DSR_recall']:.3f} ({b['DSR_recall']['se']:.3f}) [{b['DSR_recall']['ci95'][0]:.3f}, {b['DSR_recall']['ci95'][1]:.3f}]"
            + f" | {t['FP']}/{t['FP'] + t['TN']} |")
        for ttype, o in sorted((entry.get("test_by_task_type") or {}).items()):
            type_lines.append(f"| `{data['paper_prompt']}` | {ttype} | " + " | ".join(str(o[k]) for k in KEYS)
                              + f" | {o['F1']:.3f} | {o['DSR_recall']:.3f} | {o['too_late']} | {o['never_triggered']} |")
        if data.get("per_sequence"):
            per_seq[data["paper_prompt"]] = {r["id"]: _outcome(r) for r in data["per_sequence"]
                                             if r["split"] == "test"}
        warn = data["test_usage"]["parse_status_counts"]
        lines.append(f"|  | parse: {warn} | calls {data['test_usage']['decisions']} | | | | | | | |")
    out = ["### Test results (primary, all rows)", *lines]
    if type_lines:
        out += ["", "### By task type (test, shared threshold)",
                "| Prompt | Type | TP | FP | FN | TN | F1 | DSR | Too late | Never |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |", *type_lines]
    pairs = list(combinations(sorted(per_seq), 2))
    if pairs:
        out += ["", "### Paired comparisons (same test sequences; b minus a)"]
        for a, b in pairs:
            r = paired(per_seq[a], per_seq[b])
            out.append(f"- a=`{a}`, b=`{b}`, n={r['n']}: F1 diff {r['F1_diff']['point']:+.3f} "
                       f"[{r['F1_diff']['ci95'][0]:+.3f}, {r['F1_diff']['ci95'][1]:+.3f}]; "
                       f"DSR diff {r['DSR_diff']['point']:+.3f} [{r['DSR_diff']['ci95'][0]:+.3f}, {r['DSR_diff']['ci95'][1]:+.3f}]; "
                       f"attacks caught only by a/b: {r['harmful_caught_only_by_a']}/{r['harmful_caught_only_by_b']}; "
                       f"benign flagged only by a/b: {r['benign_flagged_only_by_a']}/{r['benign_flagged_only_by_b']}")
    return "\n".join(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("files", type=Path, nargs="+")
    p.add_argument("--block", default="primary_all_rows", choices=("primary_all_rows", "sensitivity_exclusions"))
    a = p.parse_args()
    print(summarize(a.files, a.block))
