"""Upstream DecomposedHarm schema adapter. Fixtures are fictional; no benchmark text is used."""
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from stateful_safety.dataset import (audit_files, load_records, parse_decomposedharm_row,
                                     split_from_filename)

ROOT = Path(__file__).resolve().parents[1]
VAL = ROOT / "fixtures/decomposedharm_shape_fixture_val.csv"
TEST = ROOT / "fixtures/decomposedharm_shape_fixture_test.csv"
COLUMNS = ["id", "original_task", "decomposition", "harm_index", "label", "task_type"]


def _row(**overrides):
    row = {"id": "harmful_0009", "original_task": "Fictional goal.",
           "decomposition": "['Step one.', 'Step two.', 'Step three.']",
           "harm_index": "2.0", "label": "harmful", "task_type": "QA"}
    row.update(overrides)
    return row


def test_upstream_layout_loads_with_one_based_boundary():
    recs = load_records(VAL, schema="decomposedharm", index_base=1, default_split="validation")
    assert len(recs) == 6
    qa = next(r for r in recs if r.sequence_id == "QA/harmful_0000")
    assert qa.steps == ("Name synthetic-risk-a.", "Name synthetic-risk-b.",
                        "Combine the fictional markers.")
    assert qa.harmful_index == 2  # upstream 3.0, 1-based -> internal 0-based
    assert all(r.harmful_index is None for r in recs if r.label == "benign")


def test_ids_namespaced_by_task_type_to_avoid_cross_type_collisions():
    recs = load_records(VAL, schema="decomposedharm", index_base=1)
    assert {r.sequence_id for r in recs if r.label == "harmful"} == {
        "QA/harmful_0000", "agent/harmful_0000", "image/harmful_0000"}


def test_original_task_never_reaches_sequence():
    rec = parse_decomposedharm_row(_row(original_task="UNIQUE-GOAL-TOKEN"), index_base=1)
    assert "UNIQUE-GOAL-TOKEN" not in json.dumps(rec.__dict__)


def test_wrong_or_missing_index_base_rejected():
    for base in (None, 0):
        with pytest.raises(ValueError, match="1-based"):
            parse_decomposedharm_row(_row(), index_base=base)


@pytest.mark.parametrize("overrides,match", [
    ({"harm_index": "2.5"}, "not an integer"),
    ({"harm_index": "your answer: 2"}, "not numeric"),
    ({"harm_index": "4.0"}, "outside sequence"),
    ({"harm_index": "0"}, "outside sequence"),
    ({"task_type": "jailbreak"}, "Unknown DecomposedHarm task_type"),
    ({"label": "unsafe"}, "expected label"),
    ({"label": "benign"}, "benign sequence cannot have harmful index"),
    ({"decomposition": "Step one. Step two."}, "must be list"),
])
def test_inconsistent_rows_rejected(overrides, match):
    with pytest.raises(ValueError, match=match):
        parse_decomposedharm_row(_row(**overrides), index_base=1)


def test_missing_column_rejected():
    row = _row()
    del row["original_task"]
    with pytest.raises(ValueError, match="missing columns"):
        parse_decomposedharm_row(row, index_base=1)


def test_null_variants_accepted_for_benign():
    for value in ("", "nan", "NaN", None, float("nan")):
        rec = parse_decomposedharm_row(_row(label="benign", harm_index=value), index_base=1)
        assert rec.harmful_index is None


def test_split_from_upstream_filenames():
    assert split_from_filename("combined_train.csv") == "train"
    assert split_from_filename("combined_val.csv") == "validation"
    assert split_from_filename("data/agent_tasks/test_data.json") == "test"
    assert split_from_filename("combined.csv") == "unknown"


def test_audit_files_reports_aggregates_and_never_text():
    report = audit_files([VAL, TEST], schema="decomposedharm", index_base=1, revision="fixture")
    assert report["record_count"] == 10
    assert report["rows_rejected"] == 0
    assert [f["split"] for f in report["files"]] == ["validation", "test"]
    assert all(len(f["sha256"]) == 64 for f in report["files"])
    assert report["benign_denominator_by_split"] == {"test": 2, "validation": 3}
    qa = report["harm_boundary_by_task_type"]["QA"]
    assert qa["harmful"] == 2 and qa["boundary_at_last_step"] == 2
    assert qa["boundary_1based_percentiles"]["p50"] == 3
    overlap = report["cross_split_overlap"][0]
    assert overlap["overlap_exact_step_sequences"] == 0
    assert overlap["overlap_exact_original_task"] == 0
    dumped = json.dumps(report)
    for needle in ("synthetic-risk", "paper boats", "Fictional goal"):
        assert needle not in dumped


def test_audit_counts_rejections_instead_of_aborting(tmp_path):
    bad = tmp_path / "combined_val.csv"
    with bad.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerow(_row())
        writer.writerow(_row(id="harmful_0010", harm_index="9"))
        writer.writerow(_row(id="harmful_0011", harm_index="9"))
    report = audit_files([bad], schema="decomposedharm", index_base=1)
    assert report["record_count"] == 1
    assert report["rows_rejected_by_reason"] == {"harmful_index outside sequence": 2}
    with pytest.raises(ValueError):
        load_records(bad, schema="decomposedharm", index_base=1)


def test_audit_detects_cross_split_goal_and_step_overlap(tmp_path):
    for name in ("combined_val.csv", "combined_test.csv"):
        with (tmp_path / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerow(_row(id=f"harmful_{name[9:12]}"))
    report = audit_files(sorted(tmp_path.glob("*.csv")), schema="decomposedharm", index_base=1)
    overlap = report["cross_split_overlap"][0]
    assert overlap["overlap_sequence_ids"] == 0
    assert overlap["overlap_exact_step_sequences"] == 1
    assert overlap["overlap_exact_original_task"] == 1


def _cli(*args):
    return subprocess.run([sys.executable, "-m", "stateful_safety", *args],
                          capture_output=True, text=True, cwd=ROOT,
                          env={**os.environ, "PYTHONPATH": str(ROOT / "src")})


def test_cli_multi_file_audit(tmp_path):
    dest = tmp_path / "audit.json"
    proc = _cli("audit", "--schema", "decomposedharm", "--index-base", "1",
                "--revision", "fixture", "--data", str(VAL), str(TEST), "--out", str(dest))
    assert proc.returncode == 0, proc.stderr
    assert json.loads(dest.read_text())["dataset_revision"] == "fixture"


def test_cli_evaluate_on_upstream_shape_smoke(tmp_path):
    dest = tmp_path / "eval.json"
    proc = _cli("evaluate", "--schema", "decomposedharm", "--index-base", "1",
                "--validation", str(VAL), "--test", str(TEST),
                "--mode", "full", "--backend", "mock", "--out", str(dest))
    assert proc.returncode == 0, proc.stderr
    data = json.loads(dest.read_text())
    assert data["status"] == "SMOKE_TEST_ONLY"
    assert data["threshold_fitted_on"] == "VALIDATION_ONLY"
    assert data["test_summary"]["n_harmful"] == 2


def test_agent_multiline_text_split_like_upstream():
    rec = parse_decomposedharm_row(
        _row(task_type="agent", decomposition="First fictional step.\n\n  Second one's step.  \nThird.\n",
             harm_index="3"), index_base=1)
    assert rec.steps == ("First fictional step.", "Second one's step.", "Third.")  # apostrophe kept
    assert rec.harmful_index == 2


def test_multiline_text_only_accepted_for_agent():
    with pytest.raises(ValueError, match="must be list"):
        parse_decomposedharm_row(_row(decomposition="Step one.\nStep two.\nStep three."),
                                 index_base=1)
    with pytest.raises(ValueError, match="agent decomposition"):
        parse_decomposedharm_row(_row(task_type="agent", decomposition="One line only."),
                                 index_base=1)


def test_agent_fixture_rows_use_real_multiline_shape():
    with VAL.open(encoding="utf-8") as handle:
        agent = [r for r in csv.DictReader(handle) if r["task_type"] == "agent"]
    assert agent and all("\n" in r["decomposition"] and not r["decomposition"].startswith("[")
                         for r in agent)
