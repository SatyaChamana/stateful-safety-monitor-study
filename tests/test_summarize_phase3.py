"""Phase 3 summary script on fictional result files (protocol phase3-a1)."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sp3", ROOT / "scripts/summarize_phase3.py")
sp3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sp3)
A, B = "A_continuous_fpr_calibrated", "B_parsed_cutoff_1"


def _run(cond, fires, stage="validation", version="phase3-a1"):
    benign_fire = 2 if cond == "current" else None
    recs = [{"id": f"h{i}", "label": "harmful", "harm_step": 3, "first_fire_step_A": f,
             "first_fire_step_B": f} for i, f in enumerate(fires)] + \
           [{"id": "b0", "label": "benign", "harm_step": None, "first_fire_step_A": None, "first_fire_step_B": None},
            {"id": "b1", "label": "benign", "harm_step": None, "first_fire_step_A": benign_fire,
             "first_fire_step_B": benign_fire}]
    timely = sum(f is not None and f <= 3 for f in fires)
    summary = {"sequence_fpr": 0.5 if cond == "current" else 0.0, "n_benign": 2,
               "timely_recall": timely / len(fires), "timely_recall_denominator": len(fires),
               "sequence_detection_recall": sum(f is not None for f in fires) / len(fires)}
    return {"condition": cond, "stage": stage, "protocol_version": version,
            "readouts": {A: {"threshold": 0.42, "summary": summary}, B: {"threshold": 1.0, "summary": summary}},
            "readout_agreement": {"agreement_rate": 0.98, "decisions_parsed_0_or_1": 50},
            "per_sequence": recs,
            "usage": {"decisions": 10, "decisions_history_shorter_than_full": 4, "input_tokens_total": 1000,
                      "seconds_per_decision_end_to_end": 0.3,
                      "readout_mass": {"median": 0.99, "below_0_5": 1}}}


def test_paired_counts():
    full = sp3._outcomes(_run("full", [2, 3, None, 2]), A)
    cur = sp3._outcomes(_run("current", [2, 4, None, None]), A)
    r = sp3.paired_diff(full, cur, n_boot=200)
    assert r["timely_recall_diff"]["point"] == pytest.approx(-0.5)
    assert r["fpr_diff"]["point"] == pytest.approx(0.5)
    assert (r["timely_only_in_reference"], r["timely_only_in_candidate"]) == (2, 0)


def test_summary_renders_both_readouts_and_rejects_mixing(tmp_path):
    for cond, fires in (("full", [2, 3, None, 2]), ("current", [2, 4, None, None])):
        (tmp_path / f"{cond}.json").write_text(json.dumps(_run(cond, fires)))
    md = sp3.summarize(sorted(tmp_path.glob("*.json")))
    assert f"#### Readout `{A}`" in md and f"#### Readout `{B}`" in md
    assert md.count("| `current` | -0.500") == 2 and "0.990; 1" in md
    (tmp_path / "x.json").write_text(json.dumps(_run("bm25_k3", [2, 2, 2, 2], stage="test")))
    with pytest.raises(ValueError, match="mixed"):
        sp3.summarize(sorted(tmp_path.glob("*.json")))
    (tmp_path / "x.json").write_text(json.dumps(_run("bm25_k3", [2, 2, 2, 2], version=None)))
    with pytest.raises(ValueError, match="mixed"):
        sp3.summarize(sorted(tmp_path.glob("*.json")))
