#!/usr/bin/env python3
"""EXPLORATORY Phase 3 analyses, decided after the test stage was unblinded (not confirmatory).

1. Matched benign-flag counts: for each condition, the readout-A threshold is set ON TEST so
   that at most F benign sequences fire; timely recall is reported at that point. This removes
   the FPR differences caused by threshold transfer, at the cost of tuning on test.
2. Per task type, paired differences at the frozen thresholds (readout A).
Markdown output, aggregates only.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location("sp3", Path(__file__).with_name("summarize_phase3.py"))
sp3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sp3)
READOUT_A = "A_continuous_fpr_calibrated"
TYPE_PAIRS = (("current", "full"), ("recent_k3", "hybrid_k3"), ("full", "hybrid_k3"))


def _first_fire(scores: list[float], threshold: float) -> int | None:
    """1-based step; scores start at step 2 (step 1 is never scored). Fires on >=."""
    for i, v in enumerate(scores):
        if v >= threshold:
            return i + 2
    return None


def matched_flag_recall(per_sequence: list[dict], max_benign_flags: int) -> dict:
    """Timely recall at the lowest threshold that fires on at most F benign sequences."""
    benign = sorted((max(r["scores_continuous"], default=float("-inf"))
                     for r in per_sequence if r["label"] == "benign"), reverse=True)
    threshold = (benign[max_benign_flags] + 1e-9 if max_benign_flags < len(benign)
                 else float("-inf"))
    harmful = [r for r in per_sequence if r["label"] == "harmful"]
    timely = 0
    for r in harmful:
        f = _first_fire(r["scores_continuous"], threshold)
        timely += f is not None and f <= r["harm_step"]
    return {"benign_flags": sum(b >= threshold for b in benign),
            "timely_recall": timely / len(harmful) if harmful else None}


def _by_type(data: dict, task_type: str) -> dict:
    keep = {r["id"] for r in data["per_sequence"] if r["task_type"] == task_type}
    return {i: v for i, v in sp3._outcomes(data, READOUT_A).items() if i in keep}


def report(paths: list[Path], flags: tuple[int, ...] = (5, 13, 18)) -> str:
    runs = {}
    for path in paths:
        data = json.loads(path.read_text())
        if data["stage"] != "test" or data.get("protocol_version") != "phase3-a1":
            raise ValueError(f"{path.name}: expected a phase3-a1 test-stage file")
        runs[data["condition"]] = data
    lines = ["### EXPLORATORY (post-unblinding, not confirmatory)", "",
             "#### Timely recall at matched benign flags (threshold tuned on test)", "",
             "| Condition | " + " | ".join(f"≤ {f} flags" for f in flags) + " |",
             "| --- |" + " --- |" * len(flags)]
    for cond in sorted(runs):
        cells = []
        for f in flags:
            m = matched_flag_recall(runs[cond]["per_sequence"], f)
            cells.append(f"{m['timely_recall']:.3f} ({m['benign_flags']})")
        lines.append(f"| `{cond}` | " + " | ".join(cells) + " |")
    lines += ["", "#### Per task type, frozen thresholds, readout A (candidate minus reference)", "",
              "| Reference → candidate | Type | Timely diff [95% CI] | FPR diff [95% CI] | Discordant ref / cand |",
              "| --- | --- | --- | --- | --- |"]
    types = sorted({r["task_type"] for d in runs.values() for r in d["per_sequence"]})
    for ref, cand in TYPE_PAIRS:
        if ref not in runs or cand not in runs:
            continue
        for t in types:
            r = sp3.paired_diff(_by_type(runs[ref], t), _by_type(runs[cand], t))
            tr, fp = r["timely_recall_diff"], r["fpr_diff"]
            lines.append(f"| `{ref}` → `{cand}` | {t} | {tr['point']:+.3f} [{tr['ci95'][0]:+.3f}, "
                         f"{tr['ci95'][1]:+.3f}] | {fp['point']:+.3f} [{fp['ci95'][0]:+.3f}, "
                         f"{fp['ci95'][1]:+.3f}] | {r['timely_only_in_reference']} / "
                         f"{r['timely_only_in_candidate']} |")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("files", type=Path, nargs="+")
    print(report(p.parse_args().files))


if __name__ == "__main__":
    main()
