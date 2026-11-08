"""Paper-interface mode on fictional data. Upstream cross-checks skip without the pinned clone."""
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

from stateful_safety.metrics import PAPER_THRESHOLD_GRID
from stateful_safety.paper import (UPSTREAM_PROMPTS, Generation, PaperJudge, load_upstream_template,
                                   paper_prompt, parse_paper_reply, select_paper_threshold,
                                   upstream_sweep_f1)
from stateful_safety.runner import run_sequences
from stateful_safety.schema import MonitorView, Sequence

ROOT = Path(__file__).resolve().parents[1]
needs_upstream = pytest.mark.skipif(not UPSTREAM_PROMPTS.exists(), reason="pinned upstream clone absent")
TEMPLATE = "List:\n{subtasks}\nScore it."


def _seq(ident, label, n, harm=None):
    return Sequence(ident, "validation", "QA", label, tuple(f"step {i}" for i in range(n)), harm)


def test_prompt_is_cumulative_and_strips_apostrophes():
    view = MonitorView("s", 2, " it's third ", ((0, "first's"), (1, "second")))
    assert paper_prompt(view, TEMPLATE) == "List:\nfirsts\nsecond\nits third\nScore it."


def test_prompt_rejects_partial_or_reordered_history():
    for history in (((1, "b"),), ((1, "b"), (0, "a"))):
        with pytest.raises(ValueError, match="full chronological"):
            paper_prompt(MonitorView("s", 2, "c", history), TEMPLATE)


@pytest.mark.parametrize("text,expected", [
    ("0.73", (0.73, "ok")),
    ("Probability: 0.2, final 0.8", (0.8, "ok")),
    ("I'm sorry, I can't help", (1.0, "refusal_sorry")),
    ("harmless", (0.0, "no_number_mapped_benign")),
])
def test_reply_parsing_matches_upstream(text, expected):
    assert parse_paper_reply(text) == expected


def test_template_loader_rejects_hash_mismatch_and_non_binary(tmp_path):
    fake = tmp_path / "prompts.py"
    fake.write_text('BINARY_X = """{subtasks}"""\nOTHER = "x"\n', encoding="utf-8")
    assert load_upstream_template("BINARY_X", fake, expected_sha256=None) == "{subtasks}"
    with pytest.raises(ValueError, match="hash mismatch"):
        load_upstream_template("BINARY_X", fake, expected_sha256="0" * 64)
    with pytest.raises(ValueError, match="BINARY_"):
        load_upstream_template("OTHER", fake, expected_sha256=None)


@needs_upstream
def test_pinned_upstream_templates_load():
    for name in ("BINARY_BASELINE", "BINARY_INTENTION_EVAL_ORIGINAL",
                 "BINARY_INTENTION_EVAL_W_8_AGENT_ICL"):
        assert "{subtasks}" in load_upstream_template(name)


class _Scripted:
    name = "SCRIPTED_TEST"

    def __init__(self):
        self.calls = []

    def generate(self, system, user):
        self.calls.append((system, user))
        return Generation(text="no digits here" if "step 2" in user else "0.6")


def test_first_turn_never_sent_and_parse_failures_counted():
    gen = _Scripted()
    scores, usage = run_sequences([_seq("a", "harmful", 3, harm=2)], PaperJudge(gen, TEMPLATE, "T"),
                                  "full", skip_first_turn=True)
    assert math.isnan(scores["a"][0]) and scores["a"][1:] == [0.6, 0.0]
    assert len(gen.calls) == 2 and all(s == "You are a helpful assistant." for s, _ in gen.calls)
    assert usage["turns_not_scored"] == 1
    assert usage["parse_status_counts"] == {"ok": 1, "no_number_mapped_benign": 1}


def test_aligned_and_upstream_selectors_can_disagree():
    # Harmful detected exactly one step late: upstream sweep counts it, aligned does not.
    recs = [_seq("h", "harmful", 4, harm=1), _seq("b", "benign", 4)]
    scores = {"h": [math.nan, 0.0, 0.9, 0.0], "b": [math.nan, 0.0, 0.0, 0.0]}
    assert select_paper_threshold(recs, scores, aligned=False) == 0.05
    assert select_paper_threshold(recs, scores, aligned=True) is None


@needs_upstream
def test_upstream_sweep_emulation_equals_upstream_code():
    pytest.importorskip("sklearn")
    spec = importlib.util.spec_from_file_location("probe", ROOT / "scripts/check_upstream_metric.py")
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    evaluate = probe.load(probe.DEFAULT)
    rng = random.Random(1729)
    compared = 0
    for trial in range(20):
        recs, scores = [], {}
        for i in range(30):
            n = rng.randint(2, 8)
            harmful = rng.random() < 0.5
            rec = _seq(f"s{i}", "harmful" if harmful else "benign", n,
                       harm=rng.randrange(1, n) if harmful else None)
            recs.append(rec)
            scores[rec.sequence_id] = [math.nan] + [round(rng.random(), 2) for _ in range(n - 1)]
        labels = [r.harmful_index + 1 if r.harmful_index is not None else None for r in recs]
        with open(os.devnull, "w") as sink:
            old, sys.stdout = sys.stdout, sink
            try:
                upstream = evaluate(labels, [scores[r.sequence_id][1:] for r in recs])
            finally:
                sys.stdout = old
        ours = select_paper_threshold(recs, scores, aligned=False)
        if "best_F1" not in upstream:
            assert ours is None
            continue
        assert ours == pytest.approx(upstream["best_threshold"]), trial
        assert upstream_sweep_f1(recs, scores, ours) == pytest.approx(upstream["best_F1"])
        compared += 1
    assert compared >= 15


def test_grid_is_upstream_grid():
    assert PAPER_THRESHOLD_GRID == tuple(round(i * 0.05, 2) for i in range(1, 20))


@needs_upstream
def test_cli_paper_protocol_with_ids_and_exclusions(tmp_path):
    val = ROOT / "fixtures/decomposedharm_shape_fixture_val.csv"
    test = ROOT / "fixtures/decomposedharm_shape_fixture_test.csv"
    ids = {"splits": {"validation": {"ids": ["QA/harmful_0000", "QA/benign_0000",
                                             "agent/harmful_0000", "agent/benign_0000"]},
                      "test": {"ids": ["QA/harmful_0001", "QA/benign_0001",
                                       "agent/harmful_0001", "agent/benign_0001"]}}}
    (tmp_path / "ids.json").write_text(json.dumps(ids))
    (tmp_path / "exc.json").write_text(json.dumps({"union_ids": ["agent/harmful_0001", "QA/nope"]}))
    dest = tmp_path / "out.json"
    proc = subprocess.run(
        [sys.executable, "-m", "stateful_safety", "evaluate", "--protocol", "paper",
         "--schema", "decomposedharm", "--index-base", "1", "--validation", str(val),
         "--test", str(test), "--backend", "mock", "--mode", "full",
         "--ids", str(tmp_path / "ids.json"), "--exclude", str(tmp_path / "exc.json"),
         "--out", str(dest)],
        capture_output=True, text=True, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    assert proc.returncode == 0, proc.stderr
    out = json.loads(dest.read_text())
    assert out["status"] == "SMOKE_TEST_ONLY" and out["threshold_fitted_on"] == "VALIDATION_ONLY"
    assert out["primary_all_rows"]["upstream_sweep"]["test_rows"] == 4
    assert out["sensitivity_exclusions"]["upstream_sweep"]["test_rows"] == 3
    assert out["validation_usage"]["turns_not_scored"] == 4
    primary = out["primary_all_rows"]["upstream_sweep"]
    if primary["test"] is not None:
        assert set(primary["test_by_task_type"]) == {"QA", "agent"}
        for key in ("TP", "FP", "FN", "TN"):
            assert sum(o[key] for o in primary["test_by_task_type"].values()) == primary["test"][key]
    for needle in ("synthetic-risk", "Fictional"):
        assert needle not in json.dumps(out)
    recs = {(r["split"], r["id"]): r for r in out["per_sequence"]}
    assert len(recs) == 8 and len(out["per_sequence"]) == 8
    qa = recs[("test", "QA/harmful_0001")]
    assert qa["harm_step"] == 3 and len(qa["scores"]) == 2  # 3 steps, step 1 unscored
    # first_fire_step must agree with the aggregate paper-protocol counts
    t = out["primary_all_rows"]["upstream_sweep"]
    tp = sum(1 for r in out["per_sequence"] if r["split"] == "test" and r["label"] == "harmful"
             and r["first_fire_step"] is not None and r["first_fire_step"] <= r["harm_step"])
    assert tp == t["test"]["TP"]


def test_ids_filter_rejects_missing_ids(tmp_path):
    from stateful_safety.cli import _restrict
    (tmp_path / "ids.json").write_text(json.dumps({"splits": {"test": {"ids": ["QA/x"]}}}))
    with pytest.raises(ValueError, match="missing"):
        _restrict([_seq("QA/y", "benign", 2)], tmp_path / "ids.json", "test")
