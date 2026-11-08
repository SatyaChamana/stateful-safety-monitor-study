"""Model revision pinning and the validation-only paper smoke command (fictional data only)."""
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from stateful_safety.cli import _smoke_sample
from stateful_safety.paper import UPSTREAM_PROMPTS, validate_model_revision
from stateful_safety.schema import Sequence

ROOT = Path(__file__).resolve().parents[1]
VAL = ROOT / "fixtures/decomposedharm_shape_fixture_val.csv"
TEST = ROOT / "fixtures/decomposedharm_shape_fixture_test.csv"
SHA = "0123456789abcdef0123456789abcdef01234567"
needs_upstream = pytest.mark.skipif(not UPSTREAM_PROMPTS.exists(), reason="pinned upstream clone absent")


def _cli(*args):
    return subprocess.run([sys.executable, "-m", "stateful_safety", *args], capture_output=True,
                          text=True, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "src")})


@pytest.mark.parametrize("bad", [None, "", "main", SHA[:7], SHA.upper(), SHA + "0"])
def test_revision_must_be_full_commit(bad):
    with pytest.raises(ValueError, match="40-character"):
        validate_model_revision(bad)


def test_revision_accepts_full_commit():
    assert validate_model_revision(SHA) == SHA


@pytest.mark.parametrize("command", [
    ["pilot", "--data", str(VAL), "--schema", "decomposedharm", "--index-base", "1"],
    ["evaluate", "--validation", str(VAL), "--test", str(TEST), "--schema", "decomposedharm",
     "--index-base", "1"],
    ["evaluate", "--protocol", "paper", "--validation", str(VAL), "--test", str(TEST),
     "--schema", "decomposedharm", "--index-base", "1"],
    ["paper-smoke", "--validation", str(VAL), "--schema", "decomposedharm", "--index-base", "1"],
])
def test_hf_backend_refuses_unpinned_model_before_loading(command):
    if "paper" in command or "paper-smoke" in command:
        if not UPSTREAM_PROMPTS.exists():
            pytest.skip("pinned upstream clone absent")
    base = [*command, "--backend", "hf", "--model-id", "meta-llama/Llama-3.1-8B-Instruct"]
    proc = _cli(*base, "--dtype", "float16", "--model-revision", "main")
    assert proc.returncode != 0
    assert "40-character commit hash" in proc.stderr
    proc = _cli(*base, "--model-revision", SHA)  # pinned, but dtype left implicit
    assert proc.returncode != 0
    assert "--dtype must be one of" in proc.stderr


def test_generator_passes_revision_to_both_loaders(monkeypatch):
    seen = []

    class _Loader:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            seen.append((model_id, kwargs.get("revision"), kwargs.get("dtype")))
            return types.SimpleNamespace(
                dtype="bfloat16", device="cpu",
                encode=lambda text, add_special_tokens: [ord(text)] if len(text) == 1 else [32, ord(text[-1])])

    fake_torch = types.ModuleType("torch")
    fake_torch.float16 = "float16-object"
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    fake = types.ModuleType("transformers")
    fake.AutoTokenizer = fake.AutoModelForCausalLM = _Loader
    monkeypatch.setitem(sys.modules, "transformers", fake)
    from stateful_safety.paper import TransformersGenerator
    gen = TransformersGenerator("org/model", SHA, "float16")
    assert seen == [("org/model", SHA, None), ("org/model", SHA, "float16-object")]
    with pytest.raises(ValueError, match="--dtype"):
        TransformersGenerator("org/model", SHA, "auto")
    assert gen.metadata()["model_revision"] == SHA
    assert gen.metadata()["gpu"] is None  # fake torch has no CUDA


def _seq(ident, label, task_type):
    return Sequence(ident, "validation", task_type, label, ("a", "b"),
                    1 if label == "harmful" else None)


def test_smoke_sample_is_deterministic_and_balanced():
    recs = [_seq(f"{t}/{label}_{i}", label, t) for t in ("QA", "agent", "image")
            for label in ("harmful", "benign") for i in range(5)]
    first = _smoke_sample(recs, 6)
    assert [r.sequence_id for r in first] == [r.sequence_id for r in _smoke_sample(recs[::-1], 6)]
    assert sorted((r.label, r.task_type) for r in first) == sorted(
        (label, t) for label in ("harmful", "benign") for t in ("QA", "agent", "image"))


def test_smoke_has_no_test_argument():
    proc = _cli("paper-smoke", "--validation", str(VAL), "--test", str(TEST))
    assert proc.returncode != 0 and "unrecognized arguments" in proc.stderr


@needs_upstream
def test_paper_smoke_mock_reports_reply_health_only(tmp_path):
    dest = tmp_path / "smoke.json"
    proc = _cli("paper-smoke", "--validation", str(VAL), "--schema", "decomposedharm",
                "--index-base", "1", "--n", "4", "--backend", "mock", "--out", str(dest))
    assert proc.returncode == 0, proc.stderr
    out = json.loads(dest.read_text())
    assert out["status"] == "SMOKE_TEST_ONLY" and out["split"] == "validation"
    assert out["sequences_scored"] == 4
    assert out["model_calls"] == sum(out["parse_status_counts"].values()) > 0
    assert out["parse_ok_rate"] == 1.0
    assert sum(out["parsed_score_buckets"].values()) == out["model_calls"]
    assert set(out["parsed_score_buckets"]) <= {"between_0_and_1", "exactly_0", "exactly_1", "above_1"}
    assert out["replies_over_5_tokens"] == 0  # mock replies carry no token counts
    assert set(out["sample_by_type_label"].values()) == {1}
    keys = " ".join(out).lower()
    for metric in ("threshold", "f1", "dsr", "recall", "precision", "tp", "fp"):
        assert metric not in keys.split() and f"_{metric}" not in keys
    dumped = json.dumps(out)
    for text in ("synthetic-risk", "Fictional", "paper boats"):
        assert text not in dumped


def test_score_buckets():
    from stateful_safety.cli import _score_bucket
    assert [_score_bucket(v) for v in (0.0, 1.0, 0.45, 3.0, 2024.0)] == [
        "exactly_0", "exactly_1", "between_0_and_1", "above_1", "above_1"]
