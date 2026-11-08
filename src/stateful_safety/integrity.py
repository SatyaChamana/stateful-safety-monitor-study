"""Conservative exact-match split leakage checks without exposing text."""
from __future__ import annotations
import hashlib
import json
from collections import defaultdict
from .schema import Sequence


def _digest(record: Sequence) -> str:
    return hashlib.sha256(json.dumps(record.steps, ensure_ascii=False).encode('utf-8')).hexdigest()


def compare_splits(validation: list[Sequence], test: list[Sequence]) -> dict:
    val_ids = {rec.sequence_id for rec in validation}
    test_ids = {rec.sequence_id for rec in test}
    val_content = {_digest(rec) for rec in validation}
    test_content = {_digest(rec) for rec in test}
    overlap_ids = len(val_ids & test_ids)
    overlap_content = len(val_content & test_content)
    return {
        "n_validation": len(validation),
        "n_test": len(test),
        "overlap_sequence_ids": overlap_ids,
        "overlap_exact_step_sequences": overlap_content,
        "family_overlap_unknown": True,
        "valid_for_initial_evaluation": overlap_ids == 0 and overlap_content == 0,
        "notes": "Exact matching does not detect paraphrases or common original task families.",
    }


EXCLUSION_RULES = {
    "R1_identical_steps_conflicting_labels":
        "Exact step sequence occurs with both labels anywhere in the dataset; exclude all members.",
    "R2_goal_shared_across_splits":
        "Exact original_task occurs in more than one split; exclude every validation/test member.",
    "R3_goal_conflicting_labels":
        "Exact original_task occurs with both labels anywhere; exclude all members.",
}
EVAL_SPLITS = ("validation", "test")


def sensitivity_exclusions(items: list[tuple[str, Sequence, str]]) -> dict[str, list[str]]:
    """Predeclared exclusion sets for a sensitivity analysis. Labels are never changed.

    items: (split, record, original_task_digest). Digests only; no text is retained.
    Returns rule -> sorted sequence IDs. The primary replication keeps every row.
    """
    by_steps, by_goal = defaultdict(list), defaultdict(list)
    for split, rec, goal in items:
        by_steps[_digest(rec)].append((split, rec))
        by_goal[goal].append((split, rec))
    r1 = {rec.sequence_id for group in by_steps.values()
          if len({rec.label for _, rec in group}) > 1 for _, rec in group}
    r2 = {rec.sequence_id for group in by_goal.values()
          if len({split for split, _ in group}) > 1
          for split, rec in group if split in EVAL_SPLITS}
    r3 = {rec.sequence_id for group in by_goal.values()
          if len({rec.label for _, rec in group}) > 1 for _, rec in group}
    return {"R1_identical_steps_conflicting_labels": sorted(r1),
            "R2_goal_shared_across_splits": sorted(r2),
            "R3_goal_conflicting_labels": sorted(r3)}
