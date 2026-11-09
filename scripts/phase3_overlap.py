"""Post-hoc checks on the Phase 3 TEST run, requested by the 2026-11-05 manuscript review.

Exploratory and descriptive only. No new judge calls: selections are recomputed with the same
deterministic selectors (src/stateful_safety/retrieval.py) on the same pinned records, and judge
scores are read from the saved per-sequence result files.

  1. Review item 2: share of timely-relevant decisions where k=3 input differs from full history.
  2. Review item 7: overlap between retrieved top-3 sets and the 3 most recent steps, and a split of
     retrieval-vs-recency discordant sequences into "different steps" vs "same steps, threshold only".
  3. Review item 10: a run-noise proxy. Where two conditions sent the judge byte-identical inputs,
     compare the saved continuous scores.

Usage:
  PYTHONPATH=src python scripts/phase3_overlap.py [--no-dense] > docs/phase3_overlap.md
"""
from __future__ import annotations
import argparse, glob, json, statistics, sys
from pathlib import Path

from stateful_safety.dataset import load_records
from stateful_safety.retrieval import select_history, DenseScorer

DATA = "data/raw/decomposedharm/combined_test.csv"
IDS = "configs/paper_eval_split.json"
ENCODER = ("sentence-transformers/all-MiniLM-L6-v2", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41")
TYPES = ("QA", "agent", "image")


def load_runs():
    runs = {}
    for f in glob.glob("results/phase3a1_test_*.json"):
        d = json.loads(Path(f).read_text())
        runs[d["condition"]] = {r["id"]: r for r in d["per_sequence"]}
    return runs


def selections(records, modes, dense):
    """sel[mode][id][t] = tuple of selected earlier-step indices for scored step t (0-based)."""
    sel = {m: {} for m in modes}
    for rec in records:
        for m in modes:
            mode, k = (m.split("_k")[0], int(m.split("_k")[1])) if "_k" in m else (m, 0)
            per = {}
            for t in range(1, len(rec.steps)):
                prior = [(i, rec.steps[i]) for i in range(t)]
                per[t] = tuple(i for i, _ in select_history(mode, rec.steps[t], prior, k, dense=dense))
            sel[m][rec.sequence_id] = per
    return sel


def timely_steps(rec, row):
    """0-based scored steps that matter for timely detection: up to the harm step (harmful), all (benign)."""
    last = len(rec.steps) - 1
    if row["label"] == "harmful":
        last = min(last, row["harm_step"] - 1)
    return range(1, last + 1)


def outcome(row):
    f = row["first_fire_step_A"]
    if row["label"] == "harmful":
        return f is not None and f <= row["harm_step"]
    return f is not None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-dense", action="store_true")
    a = ap.parse_args()
    records = load_records(DATA, index_base=1, default_split="test", schema="decomposedharm")
    wanted = set(json.loads(Path(IDS).read_text())["splits"]["test"]["ids"])
    records = [r for r in records if r.sequence_id in wanted]
    runs = load_runs()
    retr = ["bm25_k3"] + ([] if a.no_dense else ["dense_k3", "hybrid_k3"])
    modes = ["current", "full", "recent_k3"] + retr
    dense = None if a.no_dense else DenseScorer(*ENCODER)
    sel = selections(records, modes, dense)
    rows = {c: runs[c] for c in modes}
    by_id = {r.sequence_id: r for r in records}
    out = print

    out(f"# Phase 3 post-hoc overlap checks (test, n = {len(records)})\n")
    out("Exploratory. Selections recomputed with the pinned selectors; scores read from saved results.\n")

    # 1. Item 2
    out("## 1. Timely-relevant decisions where top-3 input differs from full history\n")
    out("| Type | Label | Decisions | k=3 differs from full | Share |\n|---|---|---|---|---|")
    for ty in TYPES:
        for lab in ("harmful", "benign"):
            n = d = 0
            for sid, rec in by_id.items():
                row = rows["full"][sid]
                if row["task_type"] != ty or row["label"] != lab:
                    continue
                for t in timely_steps(rec, row):
                    n += 1
                    d += t > 3
            out(f"| {ty} | {lab} | {n} | {d} | {d/n:.2f} |" if n else f"| {ty} | {lab} | 0 | 0 | - |")

    # 2. Item 7
    out("\n## 2. Retrieved top-3 vs the 3 most recent steps (decisions with more than 3 earlier steps)\n")
    out("| Retriever | Type | Decisions | Identical set | Mean Jaccard |\n|---|---|---|---|---|")
    for m in retr:
        for ty in TYPES:
            js = []
            for sid, rec in by_id.items():
                if rows["full"][sid]["task_type"] != ty:
                    continue
                for t in range(4, len(rec.steps)):
                    x, y = set(sel[m][sid][t]), set(sel["recent_k3"][sid][t])
                    js.append(len(x & y) / len(x | y))
            if js:
                same = sum(j == 1.0 for j in js)
                out(f"| {m} | {ty} | {len(js)} | {same} ({same/len(js):.2f}) | {statistics.mean(js):.2f} |")
    out("\nDiscordant sequences vs recent_k3 (timely detection for harmful, any flag for benign):\n")
    out("| Retriever | Label | Discordant | Different steps somewhere in the relevant window | Same steps throughout (threshold only) |\n|---|---|---|---|---|")
    for m in retr:
        for lab in ("harmful", "benign"):
            disc = diff = 0
            for sid, rec in by_id.items():
                r1, r0 = rows[m][sid], rows["recent_k3"][sid]
                if r1["label"] != lab or outcome(r1) == outcome(r0):
                    continue
                disc += 1
                diff += any(sel[m][sid][t] != sel["recent_k3"][sid][t] for t in timely_steps(rec, r1))
            out(f"| {m} | {lab} | {disc} | {diff} | {disc - diff} |")

    # 3. Item 10
    out("\n## 3. Run-noise proxy: identical judge inputs across conditions\n")
    out("Pairs of conditions whose selected history (and therefore prompt) is identical at a step.\n")
    out("| Pair | Identical-input decisions | Score differs by > 0.001 | Max abs difference | Readout-A verdict differs at own threshold |\n|---|---|---|---|---|")
    thr = json.loads(Path("configs/phase3_thresholds.json").read_text())["conditions"]
    pairs = [("full", c) for c in ["recent_k3"] + retr] + [("recent_k3", c) for c in retr]
    for a_, b_ in pairs:
        n = big = 0
        mx = 0.0
        for sid, rec in by_id.items():
            sa, sb = rows[a_][sid]["scores_continuous"], rows[b_][sid]["scores_continuous"]
            for t in range(1, len(rec.steps)):
                if sel[a_][sid][t] != sel[b_][sid][t]:
                    continue
                n += 1
                dlt = abs(sa[t - 1] - sb[t - 1])  # saved lists drop unscored step 1
                mx = max(mx, dlt)
                big += dlt > 1e-3
        out(f"| {a_} vs {b_} | {n} | {big} | {mx:.4f} | n/a (thresholds differ by design) |")
    out("\nThresholds: " + ", ".join(f"{c} {thr[c]['threshold']}" for c in modes if c in thr))


if __name__ == "__main__":
    sys.exit(main())
