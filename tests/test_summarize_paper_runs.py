"""Summary script on fictional result JSONs."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("spr", ROOT / "scripts/summarize_paper_runs.py")
spr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(spr)


def _rec(i, label, harm, fire):
    return {"id": i, "split": "test", "task_type": "QA", "label": label, "harm_step": harm,
            "scores": [], "first_fire_step": fire}


def test_outcome_rules():
    assert spr._outcome(_rec("a", "harmful", 3, 3)) == "TP"
    assert spr._outcome(_rec("a", "harmful", 3, 4)) == "FN"
    assert spr._outcome(_rec("a", "harmful", 3, None)) == "FN"
    assert spr._outcome(_rec("a", "benign", None, 2)) == "FP"


def test_paired_counts_and_identical_runs():
    a = {"h1": "TP", "h2": "FN", "h3": "FN", "b1": "TN", "b2": "FP"}
    b = {"h1": "FN", "h2": "TP", "h3": "TP", "b1": "FP", "b2": "FP"}
    r = spr.paired(a, b, n_boot=200)
    assert (r["harmful_caught_only_by_a"], r["harmful_caught_only_by_b"]) == (1, 2)
    assert (r["benign_flagged_only_by_a"], r["benign_flagged_only_by_b"]) == (0, 1)
    same = spr.paired(a, a, n_boot=50)
    assert same["F1_diff"]["point"] == 0 and same["F1_diff"]["ci95"] == [0, 0]


def test_summarize_renders_tables(tmp_path):
    def run(prompt, fires):
        recs = [_rec(f"h{i}", "harmful", 3, f) for i, f in enumerate(fires)] + \
               [_rec(f"b{i}", "benign", None, None) for i in range(3)]
        counts = {"TP": sum(f is not None and f <= 3 for f in fires), "FP": 0, "TN": 3}
        counts["FN"] = len(fires) - counts["TP"]
        test = {**counts, "F1": 0.5, "DSR_recall": 0.5, "too_late": 0, "never_triggered": counts["FN"]}
        return {"paper_prompt": prompt, "per_sequence": recs,
                "test_usage": {"decisions": 10, "parse_status_counts": {"ok": 10}},
                "primary_all_rows": {"upstream_sweep": {"threshold_from_validation": 0.5, "test": test,
                                                        "test_by_task_type": {"QA": test}}}}
    for name, fires in (("A", [2, None, 3]), ("B", [2, 2, None])):
        (tmp_path / f"{name}.json").write_text(json.dumps(run(name, fires)))
    md = spr.summarize(sorted(tmp_path.glob("*.json")))
    assert "### Test results" in md and "### By task type" in md and "a=`A`, b=`B`" in md
