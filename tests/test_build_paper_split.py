"""Paper-split pinning on fictional upstream-shaped files."""
import csv
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("bps", ROOT / "scripts/build_paper_split.py")
bps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bps)
COLUMNS = ["id", "original_task", "decomposition", "harm_index", "label", "task_type"]


def _write(tmp_path, csv_rows, json_rows, split="test"):
    with (tmp_path / f"combined_{split}.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.writer(h)
        w.writerow(COLUMNS)
        w.writerows(csv_rows)
    (tmp_path / f"{split}_data.json").write_text(json.dumps(json_rows), encoding="utf-8")
    return tmp_path / f"combined_{split}.csv", tmp_path / f"{split}_data.json"


def test_colliding_ids_resolved_by_content(tmp_path):
    csv_rows = [["harmful_0000", "g", "['a', 'b']", "2.0", "harmful", "QA"],
                ["harmful_0000", "g", "x\ny\n", "2", "harmful", "agent"]]
    json_rows = [{"id": "harmful_0000", "original_task": "g", "decomposition": "x\ny\n",
                  "harm_index": 2, "label": "harmful"}]
    assert bps.match_split(*_write(tmp_path, csv_rows, json_rows)) == ["agent/harmful_0000"]


def test_unmatched_row_aborts(tmp_path):
    csv_rows = [["benign_0000", "g", "['a', 'b']", "", "benign", "QA"]]
    json_rows = [{"id": "benign_0000", "original_task": "g", "decomposition": "['a', 'c']",
                  "harm_index": None, "label": "benign"}]
    with pytest.raises(ValueError, match="matched 0"):
        bps.match_split(*_write(tmp_path, csv_rows, json_rows))
