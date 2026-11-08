"""Count-based sequence bootstrap for paper-protocol F1/DSR (fictional counts)."""
import pytest

from stateful_safety.statistics import bootstrap_outcome_counts


def test_point_estimates_and_interval_contain_point():
    out = bootstrap_outcome_counts(tp=60, fp=20, fn=40, tn=80, n_boot=500)
    assert out["F1"]["point"] == pytest.approx(2 * 0.75 * 0.6 / 1.35)
    assert out["DSR_recall"]["point"] == pytest.approx(0.6)
    for name in ("F1", "DSR_recall"):
        lo, hi = out[name]["ci95"]
        assert lo <= out[name]["point"] <= hi and 0 < out[name]["se"] < 0.1
    assert out["unit"] == "SEQUENCE_BOOTSTRAP"


def test_deterministic_with_seed():
    a = bootstrap_outcome_counts(5, 2, 3, 6, n_boot=200, seed=7)
    assert a == bootstrap_outcome_counts(5, 2, 3, 6, n_boot=200, seed=7)


def test_empty_rejected():
    with pytest.raises(ValueError):
        bootstrap_outcome_counts(0, 0, 0, 0)
