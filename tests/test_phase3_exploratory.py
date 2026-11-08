"""Exploratory Phase 3 script on fictional per-sequence records."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("p3x", ROOT / "scripts/phase3_exploratory.py")
p3x = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p3x)


def _rec(i, label, scores, harm_step=None, task_type="QA", thr=0.5):
    fire = p3x._first_fire(scores, thr)
    return {"id": i, "label": label, "task_type": task_type, "harm_step": harm_step,
            "scores_continuous": scores, "first_fire_step_A": fire, "first_fire_step_B": fire}


def test_first_fire_starts_at_step_two():
    assert p3x._first_fire([0.1, 0.7], 0.5) == 3
    assert p3x._first_fire([0.1], 0.5) is None


def test_matched_flags_sets_threshold_above_the_next_benign():
    recs = [_rec("b0", "benign", [0.9]), _rec("b1", "benign", [0.6]), _rec("b2", "benign", [0.1]),
            _rec("h0", "harmful", [0.95], harm_step=2), _rec("h1", "harmful", [0.1, 0.7], harm_step=3),
            _rec("h2", "harmful", [0.1, 0.1, 0.99], harm_step=3)]
    m0 = p3x.matched_flag_recall(recs, 0)
    assert m0 == {"benign_flags": 0, "timely_recall": pytest.approx(1 / 3)}
    m1 = p3x.matched_flag_recall(recs, 1)
    assert m1["benign_flags"] == 1 and m1["timely_recall"] == pytest.approx(2 / 3)
    assert p3x.matched_flag_recall(recs, 2) == {"benign_flags": 2, "timely_recall": pytest.approx(2 / 3)}


def test_report_requires_test_stage_and_renders(tmp_path):
    def run(cond, h_scores, stage="test"):
        recs = [_rec("b0", "benign", [0.2]), _rec("h0", "harmful", h_scores, harm_step=2),
                _rec("b1", "benign", [0.3], task_type="agent"),
                _rec("h1", "harmful", h_scores, harm_step=2, task_type="agent")]
        return {"condition": cond, "stage": stage, "protocol_version": "phase3-a1", "per_sequence": recs}
    (tmp_path / "c.json").write_text(json.dumps(run("current", [0.1])))
    (tmp_path / "f.json").write_text(json.dumps(run("full", [0.9])))
    md = p3x.report(sorted(tmp_path.glob("*.json")), flags=(0,))
    assert "EXPLORATORY" in md and "| `full` | 1.000 (0) |" in md
    assert "| `current` → `full` | QA | +1.000" in md
    (tmp_path / "v.json").write_text(json.dumps(run("bm25_k3", [0.9], stage="validation")))
    with pytest.raises(ValueError, match="test-stage"):
        p3x.report(sorted(tmp_path.glob("*.json")))
