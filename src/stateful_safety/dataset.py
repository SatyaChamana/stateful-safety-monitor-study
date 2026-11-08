"""Strict loaders. Evaluation-only data. Never export underlying prompt text."""
from __future__ import annotations
import ast
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Iterable
from .schema import Sequence


def _steps(value):
    if isinstance(value, (tuple, list)):
        return tuple(value)
    if isinstance(value, str):
        for decoder in (json.loads, ast.literal_eval):
            try:
                parsed = decoder(value)
                if isinstance(parsed, (list, tuple)):
                    return tuple(parsed)
            except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
                pass
    raise ValueError("decomposed_subtasks must be list or serialized list")


def parse_row(row: dict, *, default_split: str = "unknown", index_base: int | None = None) -> Sequence:
    """The meaning of harmful_index needs human verification against paper/source.

    index_base=0 or 1 is a *declared* choice. None rejects harmful rows with
    a non-null index instead of silently assuming an indexing convention.
    """
    ident = str(row.get("id", row.get("sequence_id", "")))
    steps = _steps(row.get("decomposed_subtasks", row.get("steps")))
    label = str(row.get("label", "")).strip().lower()
    if label not in ("benign", "harmful"):
        raise ValueError(f"{ident}: expected label benign/harmful; got {label!r}")
    value = row.get("harmful_index")
    if value in (None, "", "null", "None", "nan"):
        harm = None
    else:
        if index_base not in (0, 1):
            raise ValueError("Declare --index-base 0 or --index-base 1 before loading harmful_index")
        harm = int(value) - index_base
    return Sequence(sequence_id=ident, split=str(row.get("split", default_split)),
                    task_type=str(row.get("task_type", "unknown")), label=label,
                    steps=steps, harmful_index=harm)


# Upstream DecomposedHarm layout (HF revision 25e1d59 / GitHub 79f5b41). See docs/dataset_manifest.md.
DECOMPOSEDHARM_COLUMNS = ("id", "original_task", "decomposition", "harm_index", "label", "task_type")
DECOMPOSEDHARM_TASK_TYPES = ("QA", "agent", "image")
# Verified from upstream code and paper (1-based, inclusive). Not a default for other schemas.
DECOMPOSEDHARM_INDEX_BASE = 1
SCHEMAS = ("native", "decomposedharm")
_NULLS = (None, "", "null", "None", "nan", "NaN")


def _integral_or_none(value, ident: str):
    """Accept 3, '3', 3.0, '3.0'; reject non-integral or junk annotations."""
    if value in _NULLS or (isinstance(value, float) and value != value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{ident}: harm_index is not numeric") from None
    if not number.is_integer():
        raise ValueError(f"{ident}: harm_index is not an integer")
    return int(number)


def _agent_steps(value, ident: str) -> tuple[str, ...]:
    """Agent cells are newline-separated text (real-file audit, 2026-11-02). Mirrors the upstream
    `split_subtasks` fallback: one step per non-blank line, whitespace stripped. Apostrophe
    removal is a paper-mode choice and is NOT applied here."""
    if isinstance(value, (list, tuple)):
        return tuple(value)
    try:
        return _steps(value)
    except ValueError:
        pass
    if not isinstance(value, str) or "\n" not in value:
        raise ValueError(f"{ident}: agent decomposition must be a list or multi-line text")
    return tuple(line.strip() for line in value.split("\n") if line.strip())


def parse_decomposedharm_row(row: dict, *, default_split: str = "unknown",
                             index_base: int | None = None) -> Sequence:
    """Map one upstream row to Sequence. `original_task` is checked for presence, never kept."""
    missing = [c for c in DECOMPOSEDHARM_COLUMNS if c not in row]
    if missing:
        raise ValueError(f"DecomposedHarm row missing columns {missing}")
    if index_base != DECOMPOSEDHARM_INDEX_BASE:
        raise ValueError("DecomposedHarm harm_index is 1-based (docs/dataset_manifest.md); "
                         "declare --index-base 1")
    task_type = str(row["task_type"]).strip()
    if task_type not in DECOMPOSEDHARM_TASK_TYPES:
        raise ValueError(f"Unknown DecomposedHarm task_type {task_type!r}")
    raw_id = str(row["id"]).strip()
    if not raw_id:
        raise ValueError("DecomposedHarm row has empty id")
    # Upstream IDs are only unique within a task type; namespace them losslessly.
    ident = f"{task_type}/{raw_id}"
    steps = row["decomposition"]
    if task_type == "agent":
        steps = _agent_steps(steps, ident)
    return parse_row({"id": ident, "steps": steps, "label": row["label"],
                      "harmful_index": _integral_or_none(row["harm_index"], ident),
                      "split": default_split, "task_type": task_type},
                     default_split=default_split, index_base=index_base)


def split_from_filename(path: str | Path) -> str:
    """Upstream files are named combined_{train,val,test}.csv; anything else is 'unknown'."""
    match = re.search(r"(?:^|_)(train|val|test)(?:_data)?$", Path(path).stem)
    return {"train": "train", "val": "validation", "test": "test"}[match.group(1)] if match else "unknown"


def read_rows(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No dataset at {path}. For DecomposedHarm, accept its access terms first.")
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(s) for s in handle if s.strip()]
    elif path.suffix == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError("JSON must be a list of record objects")
    elif path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    else:
        raise ValueError("Supported dataset formats: .jsonl, .json, .csv")
    return rows


def load_records(path: str | Path, *, default_split="unknown", index_base: int | None = None,
                 schema: str = "native") -> list[Sequence]:
    if schema not in SCHEMAS:
        raise ValueError(f"Unknown schema {schema!r}; expected one of {SCHEMAS}")
    parser = parse_decomposedharm_row if schema == "decomposedharm" else parse_row
    rows = read_rows(path)
    records = [parser(row, default_split=default_split, index_base=index_base) for row in rows]
    ids = [r.sequence_id for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate sequence IDs in dataset")
    return records


def audit(records: Iterable[Sequence]) -> dict:
    """Return aggregate metadata only. Never include raw requests in result."""
    recs = list(records)
    counts = Counter((r.split, r.task_type, r.label) for r in recs)
    lengths = sorted(len(r.steps) for r in recs)
    duplicates = Counter(hashlib.sha256(json.dumps(r.steps).encode()).hexdigest() for r in recs)
    by_family = Counter(r.task_type for r in recs)
    unannotated = sum(1 for r in recs if r.label == "harmful" and r.harmful_index is None)
    return {
        "record_count": len(recs),
        "count_by_split_type_label": [
            {"split": k[0], "task_type": k[1], "label": k[2], "count": v}
            for k, v in sorted(counts.items())
        ],
        "count_by_type": dict(sorted(by_family.items())),
        "sequence_length_min": lengths[0] if lengths else None,
        "sequence_length_median": lengths[len(lengths)//2] if lengths else None,
        "sequence_length_max": lengths[-1] if lengths else None,
        "harmful_sequences_missing_boundary": unannotated,
        "duplicate_step_sequences": sum(v-1 for v in duplicates.values() if v>1),
        "research_validity_notes": [
            "Check the harmful_index 0/1-base convention against original annotated examples.",
            "A sequence label is not a turn-level ground-truth label.",
            "Check cross-split task-family leakage before evaluating generalization.",
            "Only aggregate metadata is written to disk.",
        ],
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _text_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False).encode("utf-8")).hexdigest()


def _percentiles(values: list[int]) -> dict:
    if not values:
        return {}
    ordered = sorted(values)
    return {f"p{q}": ordered[min(len(ordered) - 1, (q * len(ordered)) // 100)]
            for q in (0, 10, 25, 50, 75, 90, 100)}


def _rejection_reason(error: Exception) -> str:
    # Messages start with a sequence ID; keep only the reason so reports aggregate cleanly.
    return str(error).split(": ", 1)[-1]


def audit_files(paths: list[str | Path], *, schema: str = "decomposedharm",
                index_base: int | None = None, revision: str | None = None) -> dict:
    """Multi-file aggregate audit (spec 5.3). Writes hashes and counts only, never request text.

    Rows that fail strict parsing are counted by reason instead of aborting, so a real
    audit exposes every discrepancy. Evaluation paths still use the strict `load_records`.
    """
    parser = parse_decomposedharm_row if schema == "decomposedharm" else parse_row
    files, records, rejected = [], [], Counter()
    goal_digests_by_split = defaultdict(set)
    for raw in paths:
        path = Path(raw)
        split = split_from_filename(path)
        rows = read_rows(path)
        accepted = 0
        for row in rows:
            try:
                rec = parser(row, default_split=split, index_base=index_base)
            except ValueError as error:
                rejected[_rejection_reason(error)] += 1
                continue
            records.append(rec)
            accepted += 1
            if row.get("original_task") not in _NULLS:
                goal_digests_by_split[split].add(_text_digest(str(row["original_task"]).strip()))
        files.append({"file": path.name, "split": split, "sha256": _sha256_file(path),
                      "bytes": path.stat().st_size, "rows": len(rows), "rows_accepted": accepted})

    ids = Counter(r.sequence_id for r in records)
    by_split = defaultdict(list)
    for rec in records:
        by_split[rec.split].append(rec)
    step_digests = {s: {_text_digest(list(r.steps)) for r in recs} for s, recs in by_split.items()}
    cross_split = []
    for a, b in combinations(sorted(by_split), 2):
        ids_a = {r.sequence_id for r in by_split[a]}
        ids_b = {r.sequence_id for r in by_split[b]}
        cross_split.append({
            "splits": [a, b],
            "overlap_sequence_ids": len(ids_a & ids_b),
            "overlap_exact_step_sequences": len(step_digests[a] & step_digests[b]),
            "overlap_exact_original_task": len(goal_digests_by_split[a] & goal_digests_by_split[b]),
        })

    boundaries = {}
    for task_type in sorted({r.task_type for r in records}):
        harmful = [r for r in records if r.task_type == task_type and r.label == "harmful"]
        annotated = [r for r in harmful if r.harmful_index is not None]
        positions = [r.harmful_index + 1 for r in annotated]  # back to 1-based for reporting
        boundaries[task_type] = {
            "harmful": len(harmful),
            "boundary_null": len(harmful) - len(annotated),
            "boundary_1based_percentiles": _percentiles(positions),
            "boundary_at_step_1": sum(p == 1 for p in positions),
            "boundary_at_last_step": sum(r.harmful_index == len(r.steps) - 1 for r in annotated),
        }

    summary = audit(records)
    summary.update({
        "schema": schema,
        "index_base_declared": index_base,
        "dataset_revision": revision,
        "ingested_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": files,
        "rows_rejected": sum(rejected.values()),
        "rows_rejected_by_reason": dict(sorted(rejected.items())),
        "duplicate_sequence_ids": sum(v - 1 for v in ids.values() if v > 1),
        "sequence_length_percentiles": _percentiles([len(r.steps) for r in records]),
        "benign_denominator_by_split": {s: sum(r.label == "benign" for r in recs)
                                        for s, recs in sorted(by_split.items())},
        "harm_boundary_by_task_type": boundaries,
        "cross_split_overlap": cross_split,
        "paper_split_reproducible": "UNVERIFIED: compare counts with paper Table 5 and "
                                    "upstream data/combine.py (unseeded shuffle, val halved)",
        "task_family_grouping": "NOT ESTABLISHED: original_task overlap is an exact-match "
                                "proxy; paired benign/harmful variants are not linked by ID",
    })
    return summary
