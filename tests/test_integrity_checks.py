"""Integrity checks added after the 2026-11-08 external review (no model, no upstream needed)."""
import hashlib, json
from types import SimpleNamespace

import pytest

from stateful_safety import cli, history


def _args(tmp_path, plan, **over):
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    a = dict(backend="hf", plan=str(plan_path), mode="bm25", model_id="m", model_revision="r" * 40,
             dtype="bfloat16", paper_prompt="P", ids="configs/split.json", dense_model=None, dense_revision=None)
    a.update(over)
    return SimpleNamespace(**a), hashlib.sha256(plan_path.read_bytes()).hexdigest()


PLAN = {"judge": {"model_id": "m", "model_revision": "r" * 40, "dtype": "bfloat16", "paper_prompt": "P"},
        "data": {"ids": "configs/split.json"},
        "dense_encoder": {"model": "enc", "revision": "e" * 40}}


def test_matching_inputs_pass(tmp_path):
    args, sha = _args(tmp_path, PLAN)
    cli._check_test_inputs(args, {"plan_sha256": sha})


@pytest.mark.parametrize("field,value", [("model_revision", "x" * 40), ("paper_prompt", "OTHER"),
                                         ("dtype", "float16"), ("model_id", "other"), ("ids", "configs/other.json")])
def test_any_mismatch_with_plan_is_refused(tmp_path, field, value):
    args, sha = _args(tmp_path, PLAN, **{field: value})
    with pytest.raises(SystemExit, match="differ from the frozen plan"):
        cli._check_test_inputs(args, {"plan_sha256": sha})


def test_dense_encoder_must_match_plan(tmp_path):
    args, sha = _args(tmp_path, PLAN, mode="dense", dense_model="enc", dense_revision="f" * 40)
    with pytest.raises(SystemExit, match="--dense-revision"):
        cli._check_test_inputs(args, {"plan_sha256": sha})


def test_edited_plan_is_refused(tmp_path):
    args, sha = _args(tmp_path, PLAN)
    with pytest.raises(SystemExit, match="does not match"):
        cli._check_test_inputs(args, {"plan_sha256": "0" * 64})


def test_real_run_needs_plan_hash(tmp_path):
    args, _ = _args(tmp_path, PLAN)
    with pytest.raises(SystemExit, match="no plan_sha256"):
        cli._check_test_inputs(args, {})


def test_mock_run_without_plan_hash_is_allowed(tmp_path):
    args, _ = _args(tmp_path, PLAN, backend="mock")
    cli._check_test_inputs(args, {})


def test_real_frozen_file_matches_real_plan():
    frozen = json.loads(open("configs/phase3_thresholds.json").read())
    sha = hashlib.sha256(open("configs/phase3_plan.json", "rb").read()).hexdigest()
    assert frozen["plan_sha256"] == sha


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.5, None])
def test_one_invalid_turn_score_stops_the_run(monkeypatch, bad):
    def fake_run_sequences(records, judge, mode, k, dense_model, skip_first_turn, continuous_out):
        continuous_out["s1"] = [float("nan"), 0.2, bad, 0.4]  # step 1 unscored; step 3 invalid
        return {"s1": [None, 0, 0, 0]}, {"elapsed_seconds": 1.0, "retrieval_seconds": 0.0, "decisions": 3}
    monkeypatch.setattr(history, "run_sequences", fake_run_sequences)
    with pytest.raises(ValueError, match="at step 3"):
        history.run_stage([], judge=None, mode="full", k=None, dense=None, stage="test",
                          fpr_budget=0.05, frozen_threshold=0.5)
