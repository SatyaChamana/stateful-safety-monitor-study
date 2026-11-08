"""Regression check of the documented upstream threshold-sweep step alignment."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "third_party/Monitoring-Decomposition-Attack/src/1_to_k_binary_eval.py"


@pytest.mark.skipif(not UPSTREAM.exists(), reason="pinned upstream clone not present")
def test_upstream_concurrent_sweep_is_shifted_by_one_step():
    pytest.importorskip("sklearn")
    spec = importlib.util.spec_from_file_location("probe", ROOT / "scripts/check_upstream_metric.py")
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    result = probe.probe(UPSTREAM)
    # Documented in docs/reference_implementation_notes.md section 3.
    assert result["late_by_one_counted_timely"] is True
    assert result["subtask_2_detection_counted"] is False
