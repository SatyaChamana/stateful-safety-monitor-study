#!/usr/bin/env python3
"""Probe the pinned upstream threshold-sweep metric on fictional scores (no data, no model).

Extracts only the pure `evaluate_predictions` function from the upstream evaluator and
checks its step alignment. In upstream concurrent mode, probs[j] scores the cumulative
prefix of length j + 2 (subtasks 2..n). Prints JSON; exit code is 0 either way.
"""
from __future__ import annotations
import ast
import contextlib
import io
import json
import sys
from pathlib import Path

DEFAULT = (Path(__file__).resolve().parents[1]
           / "third_party/Monitoring-Decomposition-Attack/src/1_to_k_binary_eval.py")


def load(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == "evaluate_predictions")
    namespace: dict = {}
    exec("from sklearn.metrics import roc_auc_score\n" + ast.unparse(fn), namespace)
    return namespace["evaluate_predictions"]


def probe(path: Path = DEFAULT) -> dict:
    evaluate = load(path)

    def tp_at_half(harm_index: int, probs: list[float]) -> bool:
        with contextlib.redirect_stdout(io.StringIO()):
            result = evaluate([harm_index, None], [probs, [0.0] * len(probs)])
        return result["threshold_0_5_TP"] == 1

    # Harmful, 4 subtasks, 1-based harm_index = 2.
    return {
        "late_by_one_counted_timely": tp_at_half(2, [0.0, 0.9, 0.0]),  # fires at subtask 3
        "subtask_2_detection_counted": tp_at_half(2, [0.9, 0.0, 0.0]),  # fires at subtask 2
        "note": "Fictional scores. Describes upstream code behavior, not paper results.",
    }


if __name__ == "__main__":
    print(json.dumps(probe(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT), indent=2))
