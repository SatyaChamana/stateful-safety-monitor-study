#!/usr/bin/env python3
"""Pin the paper's evaluation subset (upstream data/combined/{val,test}_data.json) as CSV IDs.

Upstream built these subsets with an unseeded shuffle (data/combine.py), so they can only be
recovered by matching released rows. Each JSON row is matched to exactly one HF CSV row on
(id, decomposition, label, harm_index); any miss or ambiguity aborts. Output holds IDs and
hashes only, never request text.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _harm(value):
    if value in (None, "") or (isinstance(value, float) and value != value):
        return None
    return int(float(value))


def match_split(csv_path: Path, json_path: Path) -> list[str]:
    index = defaultdict(list)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["id"], row["decomposition"], row["label"], _harm(row["harm_index"]))
            index[key].append(f'{row["task_type"]}/{row["id"]}')
    rows = json.loads(json_path.read_text(encoding="utf-8").replace("NaN", "null"))
    ids = []
    for row in rows:
        decomposition = row["decomposition"]
        if not isinstance(decomposition, str):
            decomposition = repr(decomposition)
        found = index.get((row["id"], decomposition, row["label"], _harm(row["harm_index"])), [])
        if len(found) != 1:
            raise ValueError(f"{json_path.name}: row {row['id']} matched {len(found)} CSV rows")
        ids.append(found[0])
    if len(set(ids)) != len(ids):
        raise ValueError(f"{json_path.name}: two JSON rows matched the same CSV row")
    return sorted(ids)


def build(csv_dir: Path, json_dir: Path, revision: str, upstream_commit: str) -> dict:
    out = {"description": "Paper evaluation subset as namespaced CSV IDs (task_type/id).",
           "hf_revision": revision, "upstream_commit": upstream_commit, "splits": {}}
    for split, name in (("val", "validation"), ("test", "test")):
        csv_path = csv_dir / f"combined_{split}.csv"
        json_path = json_dir / f"{split}_data.json"
        ids = match_split(csv_path, json_path)
        out["splits"][name] = {
            "csv_file": csv_path.name, "csv_sha256": _sha256(csv_path),
            "json_file": f"data/combined/{json_path.name}", "json_sha256": _sha256(json_path),
            "n": len(ids),
            "count_by_type_label": dict(sorted(Counter(
                (i.split("/")[0] + ":" + ("harmful" if "/harmful_" in i else "benign"))
                for i in ids).items())),
            "ids": ids,
        }
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv-dir", type=Path, default=Path("data/raw/decomposedharm"))
    p.add_argument("--json-dir", type=Path,
                   default=Path("third_party/Monitoring-Decomposition-Attack/data/combined"))
    p.add_argument("--revision", required=True)
    p.add_argument("--upstream-commit", required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    result = build(a.csv_dir, a.json_dir, a.revision, a.upstream_commit)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "ids"}
                      for k, v in result["splits"].items()}, indent=1))
