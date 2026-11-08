"""Phase 3 machinery on fictional data: retrieval, stages, freeze rules, CLI."""
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from stateful_safety.history import condition_name, run_stage
from stateful_safety.paper import UPSTREAM_PROMPTS, MockPaperGenerator, PaperJudge, paper_prompt
from stateful_safety.retrieval import (RRF_K, DenseScorer, bm25_scores, hybrid_search,
                                       select_history)
from stateful_safety.runner import run_sequences
from stateful_safety.schema import MonitorView, Sequence

ROOT = Path(__file__).resolve().parents[1]
VAL = ROOT / "fixtures/decomposedharm_shape_fixture_val.csv"
TEST = ROOT / "fixtures/decomposedharm_shape_fixture_test.csv"
needs_upstream = pytest.mark.skipif(not UPSTREAM_PROMPTS.exists(), reason="pinned upstream clone absent")


class _FakeEncoder:
    """Embeds by a fixed lookup so ranks are known; counts encode calls."""

    def __init__(self, table):
        self.table, self.calls, self.texts = table, 0, 0

    def encode(self, texts, normalize_embeddings):
        self.calls += 1
        self.texts += len(texts)
        return [self.table[t] for t in texts]


def _scorer(table):
    return DenseScorer("fake", encoder=_FakeEncoder(table))


def test_dense_scorer_embeds_each_text_once():
    table = {"q": [1, 0], "a": [1, 0], "b": [0, 1], "c": [0.6, 0.8]}
    scorer = _scorer(table)
    prior = [(0, "a"), (1, "b")]
    assert select_history("dense", "q", prior, 1, dense=scorer) == [(0, "a")]
    select_history("dense", "q", prior + [(2, "c")], 2, dense=scorer)
    assert scorer.encoder.texts == 4 and scorer.texts_encoded == 4


def test_dense_topk_returned_chronologically():
    scorer = _scorer({"q": [1, 0], "a": [0, 1], "b": [0.9, 0.1], "c": [1, 0]})
    assert select_history("dense", "q", [(0, "a"), (1, "b"), (2, "c")], 2, dense=scorer) == \
        [(1, "b"), (2, "c")]


def test_hybrid_is_rrf_over_relevance_ranks():
    prior = [(0, "red apple"), (1, "green pear"), (2, "blue sky")]
    # Lexical favors 0; dense favors 2; step 1 is second in both, so RRF picks it first.
    scorer = _scorer({"red pear sky": [1, 0, 0], "red apple": [0, 0, 1],
                      "green pear": [0.8, 0.6, 0], "blue sky": [1, 0, 0]})
    lexical = bm25_scores("red pear sky", prior)
    dense = scorer.scores("red pear sky", prior)
    fused = []
    for j in range(3):
        rl = sorted(range(3), key=lambda i: (-lexical[i], -i)).index(j) + 1
        rd = sorted(range(3), key=lambda i: (-dense[i], -i)).index(j) + 1
        fused.append(1 / (RRF_K + rl) + 1 / (RRF_K + rd))
    best = max(range(3), key=lambda j: (fused[j], j))
    assert hybrid_search("red pear sky", prior, 1, scorer) == [prior[best]]
    assert [i for i, _ in hybrid_search("red pear sky", prior, 2, scorer)] == sorted(
        [i for i, _ in hybrid_search("red pear sky", prior, 2, scorer)])


def test_dense_and_hybrid_need_a_scorer():
    with pytest.raises(ValueError, match="dense-model"):
        select_history("hybrid", "q", [(0, "a")], 1)


def test_runner_counts_shorter_history_and_retrieval_time():
    recs = [Sequence("s", "validation", "QA", "benign", ("a", "b", "c", "d"), None)]
    judge = PaperJudge(MockPaperGenerator(), "{subtasks}", "T", require_full_history=False)
    _, usage = run_sequences(recs, judge, "recent", k=1, skip_first_turn=True)
    # turns 2,3,4 see 1 of 1, 1 of 2, 1 of 3 earlier steps
    assert usage["decisions_history_shorter_than_full"] == 2
    assert usage["retrieval_seconds"] >= 0 and usage["history_items"] == 3


def test_paper_prompt_selected_history_only_when_allowed():
    view = MonitorView("s", 3, "d", ((0, "a"), (2, "c")))
    with pytest.raises(ValueError, match="full chronological"):
        paper_prompt(view, "{subtasks}")
    assert paper_prompt(view, "{subtasks}", require_full_history=False) == "a\nc\nd"
    with pytest.raises(ValueError, match="chronological"):
        paper_prompt(MonitorView("s", 3, "d", ((2, "c"), (0, "a"))), "{subtasks}", False)
    with pytest.raises(ValueError, match="chronological"):
        paper_prompt(MonitorView("s", 1, "d", ((1, "x"),)), "{subtasks}", False)


def test_condition_names():
    assert condition_name("full", None) == "full" and condition_name("bm25", 3) == "bm25_k3"
    with pytest.raises(ValueError):
        condition_name("oracle", 3)


def _recs():
    return [Sequence("h", "x", "QA", "harmful", ("ok", "synthetic-risk-a", "synthetic-risk-b"), 2),
            Sequence("b", "x", "QA", "benign", ("ok", "fine", "calm"), None)]


def test_stages_fit_then_reuse_threshold():
    judge = PaperJudge(MockPaperGenerator(), "{subtasks}", "T", require_full_history=False)
    val = run_stage(_recs(), judge, "full", None, None, "validation", 0.05)
    a = val["readouts"]["A_continuous_fpr_calibrated"]
    assert a["threshold_source"] == "fitted_on_validation" and a["summary"]["sequence_fpr"] == 0
    assert val["readouts"]["B_parsed_cutoff_1"]["threshold"] == 1.0
    assert val["protocol_version"] == "phase3-a1" and val["readout_agreement"]["agreement_rate"] is None
    with pytest.raises(ValueError, match="frozen"):
        run_stage(_recs(), judge, "full", None, None, "test", 0.05)
    with pytest.raises(ValueError, match="do not pass"):
        run_stage(_recs(), judge, "full", None, None, "validation", 0.05, frozen_threshold=0.5)
    test = run_stage(_recs(), judge, "full", None, None, "test", 0.05, frozen_threshold=a["threshold"])
    ta = test["readouts"]["A_continuous_fpr_calibrated"]
    assert ta["threshold"] == a["threshold"] and ta["threshold_source"] == "frozen_file"
    h = next(r for r in test["per_sequence"] if r["id"] == "h")
    assert h["harm_step"] == 3 and len(h["scores_continuous"]) == 2 and h["first_fire_step_A"] in (2, 3)
    assert all(not math.isnan(v) for r in test["per_sequence"] for v in r["scores_continuous"])


def test_stage_refuses_judge_without_continuous_score():
    class _TextOnly(MockPaperGenerator):
        def generate(self, system, user):
            return super().generate(system, user).__class__(text="1")
    judge = PaperJudge(_TextOnly(), "{subtasks}", "T", require_full_history=False)
    with pytest.raises(ValueError, match="no continuous score"):
        run_stage(_recs(), judge, "full", None, None, "validation", 0.05)


def test_misparsed_scores_never_become_thresholds():
    from stateful_safety.metrics import calibrate_threshold
    recs = [Sequence("b1", "x", "QA", "benign", ("a", "b"), None),
            Sequence("b2", "x", "QA", "benign", ("a", "b"), None)]
    scores = {"b1": [math.nan, 1.0], "b2": [math.nan, 3.0]}
    # budget 0.5: at 1.0 both benign fire (FPR 1.0); at 3.0 only one does (FPR 0.5)
    assert calibrate_threshold(recs, scores, 0.5) == 3.0  # without a range, a misparse can win
    assert calibrate_threshold(recs, scores, 0.5, valid_range=(0.0, 1.0)) == math.inf


# ---- freeze script -------------------------------------------------------------------
spec = importlib.util.spec_from_file_location("frz", ROOT / "scripts/freeze_phase3_thresholds.py")
frz = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frz)
PLAN = json.loads((ROOT / "configs/phase3_plan.json").read_text())


def _val_result(tmp_path, cond, tag="ok", **override):
    mode, _, k = cond.partition("_k")
    data = {"status": "PHASE3_RUN", "stage": "validation", "condition": cond, "mode": mode,
            "protocol_version": "phase3-a1",
            "paper_prompt": PLAN["judge"]["paper_prompt"], "fpr_budget": 0.05,
            "model": {"model_revision": PLAN["judge"]["model_revision"], "dtype": "torch.bfloat16"},
            "dense_encoder": {"revision": PLAN["dense_encoder"]["revision"]} if mode in ("dense", "hybrid") else None,
            "readouts": {"A_continuous_fpr_calibrated": {
                "threshold": 0.5, "summary": {"sequence_fpr": 0.04, "timely_recall": 0.8}}}}
    data.update(override)
    path = tmp_path / f"{cond}_{tag}.json"
    path.write_text(json.dumps(data))
    return path


def test_freeze_happy_path_and_guards(tmp_path):
    plan = ROOT / "configs/phase3_plan.json"
    paths = [_val_result(tmp_path, c) for c in PLAN["pilot_conditions"]]
    frozen = frz.build(plan, paths, margin=0.05, primary_k=3)
    assert set(frozen["conditions"]) == set(PLAN["pilot_conditions"])
    assert frozen["equivalence_margin_timely_recall"] == 0.05 and len(frozen["plan_sha256"]) == 64
    with pytest.raises(ValueError, match="missing"):
        frz.build(plan, paths[:-1], 0.05, 3)
    with pytest.raises(ValueError, match="more than one"):
        frz.build(plan, paths + [paths[0]], 0.05, 3)
    bad = _val_result(tmp_path, "full", tag="badjudge", model={"model_revision": "0" * 40, "dtype": "torch.bfloat16"})
    with pytest.raises(ValueError, match="judge does not match"):
        frz.build(plan, [bad] + paths[2:], 0.05, 3)
    mock = _val_result(tmp_path, "current", tag="mock", status="SMOKE_TEST_ONLY")
    with pytest.raises(ValueError, match="mock"):
        frz.build(plan, [mock] + paths[1:], 0.05, 3)
    test_stage = _val_result(tmp_path, "full", tag="teststage", stage="test")
    with pytest.raises(ValueError, match="not a validation"):
        frz.build(plan, [paths[0], test_stage] + paths[2:], 0.05, 3)
    with pytest.raises(ValueError, match="margin"):
        frz.build(plan, paths, 0.0, 3)
    old = _val_result(tmp_path, "full", tag="superseded", protocol_version=None)
    with pytest.raises(ValueError, match="superseded"):
        frz.build(plan, [paths[0], old] + paths[2:], 0.05, 3)
    assert frozen["protocol_version"] == "phase3-a1" and len(frozen["amendment_sha256"]) == 64


# ---- CLI -----------------------------------------------------------------------------
def _cli(*args):
    return subprocess.run([sys.executable, "-m", "stateful_safety", *args], capture_output=True,
                          text=True, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "src")})


BASE = ("--schema", "decomposedharm", "--index-base", "1", "--backend", "mock")


@needs_upstream
def test_cli_validation_then_test(tmp_path):
    val_out = tmp_path / "val.json"
    proc = _cli("history-eval", "--stage", "validation", "--mode", "recent", "--k", "1",
                "--data", str(VAL), *BASE, "--out", str(val_out))
    assert proc.returncode == 0, proc.stderr
    val = json.loads(val_out.read_text())
    assert val["condition"] == "recent_k1" and val["status"] == "SMOKE_TEST_ONLY"
    assert val["usage"]["decisions_history_shorter_than_full"] > 0
    thr = val["readouts"]["A_continuous_fpr_calibrated"]["threshold"]
    frozen = {"protocol_version": "phase3-a1", "fpr_budget": 0.05, "equivalence_margin_timely_recall": 0.05,
              "conditions": {"recent_k1": {"threshold": thr}}}
    (tmp_path / "thr.json").write_text(json.dumps(frozen))
    test_out = tmp_path / "test.json"
    proc = _cli("history-eval", "--stage", "test", "--mode", "recent", "--k", "1", "--data", str(TEST),
                *BASE, "--thresholds", str(tmp_path / "thr.json"), "--out", str(test_out))
    assert proc.returncode == 0, proc.stderr
    test = json.loads(test_out.read_text())
    assert test["readouts"]["A_continuous_fpr_calibrated"]["threshold"] == thr
    for needle in ("synthetic-risk", "Fictional"):
        assert needle not in test_out.read_text()


@needs_upstream
def test_cli_refuses_unsafe_combinations(tmp_path):
    common = ("--data", str(TEST), *BASE)
    assert "requires --thresholds" in _cli("history-eval", "--stage", "test", "--mode", "full", *common).stderr
    (tmp_path / "t.json").write_text(json.dumps({"protocol_version": "phase3-a1", "fpr_budget": 0.05,
                                                "equivalence_margin_timely_recall": None,
                                                "conditions": {"full": {"threshold": 0.5}}}))
    (tmp_path / "old.json").write_text(json.dumps({"fpr_budget": 0.05, "equivalence_margin_timely_recall": 0.05,
                                                  "conditions": {"full": {"threshold": 0.5}}}))
    assert "not protocol phase3-a1" in _cli("history-eval", "--stage", "test", "--mode", "full", *common,
                                            "--thresholds", str(tmp_path / "old.json")).stderr
    assert "no equivalence margin" in _cli("history-eval", "--stage", "test", "--mode", "full", *common,
                                           "--thresholds", str(tmp_path / "t.json")).stderr
    assert "do not pass --thresholds" in _cli("history-eval", "--stage", "validation", "--mode", "full",
                                              "--data", str(VAL), *BASE, "--thresholds", str(tmp_path / "t.json")).stderr
    assert "--k (>= 1) required" in _cli("history-eval", "--stage", "validation", "--mode", "bm25",
                                         "--data", str(VAL), *BASE).stderr
    assert "meaningless" in _cli("history-eval", "--stage", "validation", "--mode", "full", "--k", "3",
                                 "--data", str(VAL), *BASE).stderr


def test_two_token_readout_math():
    from stateful_safety.paper import two_token_readout
    ids = {"0": [0, 3], "1": [1]}
    row = [math.log(0.2), math.log(0.5), math.log(0.2), math.log(0.1)]  # softmax = itself
    p_one, mass = two_token_readout(row, ids)
    assert p_one == pytest.approx(0.5 / 0.8) and mass == pytest.approx(0.8)


def test_first_token_ids_rules():
    from stateful_safety.paper import first_token_ids

    class _Tok:
        def __init__(self, table):
            self.table = table

        def encode(self, text, add_special_tokens):
            return self.table[text]
    ok = first_token_ids(_Tok({"0": [15], " 0": [220, 15], "1": [16], " 1": [220, 16]}))
    assert ok == {"0": [15], "1": [16]}
    with pytest.raises(ValueError, match="single-token"):
        first_token_ids(_Tok({"0": [9, 9], " 0": [8, 8], "1": [16], " 1": [16]}))
