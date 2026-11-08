"""Figure script runs end to end on fictional result files and writes all six PNGs."""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("mkfig", ROOT / "scripts/make_figures.py")
mkfig = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mkfig)
A = "A_continuous_fpr_calibrated"
CONDS = ["current", "full"] + [f"{m}_k{k}" for m in ("recent", "bm25", "dense", "hybrid") for k in (3, 5)]


def _phase3(cond, i):
    recs = []
    for t in ("QA", "agent", "image"):
        recs.append({"id": f"{t}/h", "label": "harmful", "task_type": t, "harm_step": 3,
                     "scores_continuous": [0.1 * i, 0.9], "first_fire_step_A": 3, "first_fire_step_B": 3})
        recs.append({"id": f"{t}/b", "label": "benign", "task_type": t, "harm_step": None,
                     "scores_continuous": [0.2, 0.05 * i], "first_fire_step_A": None, "first_fire_step_B": None})
    summ = {"sequence_fpr": 0.01 * i, "timely_recall": 0.7 + 0.01 * i, "n_benign": 3}
    return {"condition": cond, "stage": "test", "protocol_version": "phase3-a1", "per_sequence": recs,
            "usage": {"input_tokens_total": 1500 * 10 + i, "decisions": 10},
            "readouts": {A: {"summary": summ,
                             "summary_by_task_type": {t: {"timely_recall": 0.5, "sequence_fpr": 0.05}
                                                      for t in ("QA", "agent", "image")}}}}


def test_writes_six_figures(tmp_path, monkeypatch):
    res, out = tmp_path / "results", tmp_path / "figs"
    res.mkdir()
    for prompt, _ in mkfig.PAPER_PROMPTS:
        test = {"TP": 80, "FP": 5, "FN": 20, "TN": 95, "F1": 0.85}
        (res / f"paper_eval_{prompt}_x.json").write_text(
            json.dumps({"primary_all_rows": {"upstream_sweep": {"test": test}}}))
    for i, cond in enumerate(CONDS):
        (res / f"phase3a1_test_{cond}_x.json").write_text(json.dumps(_phase3(cond, i)))
    monkeypatch.setattr(sys, "argv", ["make_figures.py", "--results", str(res), "--out", str(out)])
    monkeypatch.syspath_prepend(str(ROOT / "src"))
    mkfig.main()
    assert len(list(out.glob("fig*.png"))) == 6
