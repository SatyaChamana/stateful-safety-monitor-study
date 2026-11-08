"""Predeclared sensitivity exclusions on fictional records."""
from stateful_safety.integrity import sensitivity_exclusions
from stateful_safety.schema import Sequence


def _rec(ident, label, steps, harm=None):
    return Sequence(ident, "x", "QA", label, tuple(steps), harm)


def test_rules_select_expected_ids_and_never_touch_labels():
    items = [
        ("test", _rec("QA/benign_1", "benign", ["a", "b"]), "g1"),
        ("test", _rec("QA/harmful_1", "harmful", ["a", "b"], 1), "g2"),       # R1 with benign_1
        ("train", _rec("QA/harmful_2", "harmful", ["c", "d"], 1), "g3"),
        ("validation", _rec("QA/harmful_3", "harmful", ["e", "f"], 1), "g3"),  # R2 (val member)
        ("test", _rec("QA/harmful_4", "harmful", ["g", "h"], 1), "g3"),        # R2 (test member)
        ("validation", _rec("QA/benign_5", "benign", ["i", "j"]), "g4"),
        ("validation", _rec("QA/harmful_5", "harmful", ["k", "l"], 1), "g4"),  # R3 pair
        ("test", _rec("QA/benign_6", "benign", ["m", "n"]), "g5"),             # clean
    ]
    out = sensitivity_exclusions(items)
    assert out["R1_identical_steps_conflicting_labels"] == ["QA/benign_1", "QA/harmful_1"]
    assert out["R2_goal_shared_across_splits"] == ["QA/harmful_3", "QA/harmful_4"]
    assert out["R3_goal_conflicting_labels"] == ["QA/benign_5", "QA/harmful_5"]
    assert all("QA/benign_6" not in ids for ids in out.values())
    assert [rec.label for _, rec, _ in items][:2] == ["benign", "harmful"]
