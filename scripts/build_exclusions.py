#!/usr/bin/env python3
"""Freeze the predeclared sensitivity-exclusion list (IDs and hashes only, no text).

Run once, before any test-set evaluation. The primary replication keeps every row;
the sensitivity analysis drops the union of these IDs. Refuses to overwrite.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stateful_safety.dataset import parse_decomposedharm_row, read_rows, split_from_filename  # noqa: E402
from stateful_safety.integrity import EXCLUSION_RULES, sensitivity_exclusions  # noqa: E402


def collect(paths: list[Path]):
    items, files = [], []
    for path in paths:
        split = split_from_filename(path)
        for row in read_rows(path):
            rec = parse_decomposedharm_row(row, default_split=split, index_base=1)
            goal = hashlib.sha256(str(row["original_task"]).strip().encode("utf-8")).hexdigest()
            items.append((split, rec, goal))
        files.append({"file": path.name, "split": split,
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return items, files


def build(paths: list[Path], revision: str, paper_split: dict | None = None) -> dict:
    items, files = collect(paths)
    rules = sensitivity_exclusions(items)
    union = sorted(set().union(*rules.values()))
    out = {"status": "PREDECLARED_BEFORE_TEST_EVALUATION",
           "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "dataset_revision": revision, "files": files,
           "primary_analysis": "keep all rows (paper replication)",
           "sensitivity_analysis": "drop the union of all rule IDs; labels never changed",
           "rules": EXCLUSION_RULES, "ids_by_rule": rules, "union_ids": union}
    if paper_split:
        out["union_in_paper_subset"] = {
            split: sorted(set(union) & set(info["ids"]))
            for split, info in paper_split["splits"].items()}
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, nargs="+", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--paper-split", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise SystemExit(f"{a.out} exists; the exclusion list is frozen. Do not regenerate.")
    paper = json.loads(a.paper_split.read_text()) if a.paper_split else None
    result = build(a.data, a.revision, paper)
    a.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"ids_by_rule": result["ids_by_rule"],
                      "union_in_paper_subset": result.get("union_in_paper_subset")}, indent=1))
