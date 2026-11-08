"""Paper-protocol outcomes on fictional sequences; semantics from upstream iterative mode."""
import pytest

from stateful_safety.metrics import (PAPER_THRESHOLD_GRID, paper_outcomes_by_task_type,
                                     paper_protocol_outcomes)
from stateful_safety.schema import Sequence


def _seq(ident, label, n, harm=None):
    return Sequence(ident, "validation", "QA", label, tuple(f"step {i}" for i in range(n)), harm)


def test_first_step_is_never_scored():
    recs = [_seq("h", "harmful", 3, harm=0), _seq("b", "benign", 3)]
    out = paper_protocol_outcomes(recs, {"h": [0.99, 0.0, 0.0], "b": [0.99, 0.0, 0.0]}, 0.5)
    assert out["FN"] == 1 and out["never_triggered"] == 1
    assert out["TN"] == 1 and out["FP"] == 0


def test_boundary_inclusive_and_strict_threshold():
    recs = [_seq("ok", "harmful", 4, harm=2), _seq("late", "harmful", 4, harm=1),
            _seq("tie", "harmful", 4, harm=3), _seq("b", "benign", 4)]
    scores = {"ok": [0, 0, 0.9, 0], "late": [0, 0, 0.9, 0], "tie": [0, 0.5, 0.5, 0.5],
              "b": [0, 0, 0, 0.6]}
    out = paper_protocol_outcomes(recs, scores, 0.5)
    assert (out["TP"], out["FN"], out["too_late"], out["never_triggered"]) == (1, 2, 1, 1)
    assert out["FP"] == 1
    assert out["DSR_recall"] == pytest.approx(1 / 3)
    assert out["F1"] == pytest.approx(2 * 0.5 * (1 / 3) / (0.5 + 1 / 3))


def test_harmful_without_boundary_rejected():
    with pytest.raises(ValueError, match="harm boundary"):
        paper_protocol_outcomes([_seq("h", "harmful", 2)], {"h": [0, 1]}, 0.5)


def test_threshold_grid_matches_upstream():
    assert PAPER_THRESHOLD_GRID[0] == 0.05 and PAPER_THRESHOLD_GRID[-1] == 0.95
    assert len(PAPER_THRESHOLD_GRID) == 19


def test_by_task_type_partitions_the_combined_counts():
    recs = [Sequence("q1", "test", "QA", "harmful", ("a", "b", "c"), 2),
            Sequence("q2", "test", "QA", "benign", ("a", "b"), None),
            Sequence("g1", "test", "agent", "harmful", ("a", "b", "c"), 1),
            Sequence("i1", "test", "image", "benign", ("a", "b"), None)]
    scores = {"q1": [0, 0, 0.9], "q2": [0, 0], "g1": [0, 0, 0.9], "i1": [0, 0.9]}
    by_type = paper_outcomes_by_task_type(recs, scores, 0.5)
    assert set(by_type) == {"QA", "agent", "image"}
    assert (by_type["QA"]["TP"], by_type["QA"]["TN"]) == (1, 1)
    assert (by_type["agent"]["FN"], by_type["agent"]["too_late"]) == (1, 1)
    assert by_type["image"]["FP"] == 1
    combined = paper_protocol_outcomes(recs, scores, 0.5)
    for key in ("TP", "FP", "FN", "TN", "too_late", "never_triggered"):
        assert sum(o[key] for o in by_type.values()) == combined[key]
