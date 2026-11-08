"""HF generate path with fakes shaped like newer transformers (BatchEncoding from
apply_chat_template). Regression for clipper job 798351 (AttributeError on .shape)."""
import contextlib
import math
import sys
import types

import pytest

SHA = "0123456789abcdef0123456789abcdef01234567"


class _Ids(list):
    """Minimal tensor stand-in: shape, .to(), 2-D indexing, slicing."""

    @property
    def shape(self):
        return (len(self), len(self[0]))

    def to(self, device):
        return self


class _BatchEncoding(dict):
    """Like transformers' BatchEncoding: dict access works, attribute .shape does not."""

    def __getattr__(self, item):
        raise AttributeError(item)


class _Tokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens):
        return [ord(text)] if len(text) == 1 else [32, ord(text[-1])]  # "0"->48, "1"->49

    def apply_chat_template(self, messages, add_generation_prompt, return_tensors, return_dict=True):
        ids = _Ids([[1, 2, 3, 4]])
        if return_dict:
            return _BatchEncoding(input_ids=ids, attention_mask=_Ids([[1, 1, 1, 1]]))
        return ids

    def decode(self, ids, skip_special_tokens):
        return "0.9" if list(ids) == [7, 8] else "unexpected"


class _Model:
    device = "cpu"
    dtype = "bfloat16"

    def __init__(self):
        self.kwargs = None

    def generate(self, **kwargs):
        self.kwargs = kwargs
        row = [0.0] * 60
        row[48], row[49] = math.log(0.25), math.log(0.75)  # P("0")=.25, P("1")=.75 (+ tiny rest)
        sequences = [kwargs["input_ids"][0] + [7, 8]]
        if not kwargs.get("return_dict_in_generate"):
            return sequences  # older call style (diagnostic judge)
        return types.SimpleNamespace(sequences=sequences, logits=([row],))  # (steps) x (batch, vocab)


@pytest.fixture
def fake_hf(monkeypatch):
    torch = types.ModuleType("torch")
    torch.bfloat16 = "bf16"
    torch.inference_mode = contextlib.nullcontext
    model = _Model()
    loader = types.SimpleNamespace(from_pretrained=lambda *a, **k: _Tokenizer())
    mloader = types.SimpleNamespace(from_pretrained=lambda *a, **k: model)
    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer, transformers.AutoModelForCausalLM = loader, mloader
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    return model


def test_paper_generator_handles_batch_encoding(fake_hf):
    from stateful_safety.paper import TransformersGenerator
    out = TransformersGenerator("org/m", SHA, "bfloat16").generate("sys", "user")
    assert (out.text, out.input_tokens, out.output_tokens) == ("0.9", 4, 2)
    assert fake_hf.kwargs["attention_mask"] == [[1, 1, 1, 1]]
    assert fake_hf.kwargs["do_sample"] is False
    assert fake_hf.kwargs["return_dict_in_generate"] is True and fake_hf.kwargs["output_logits"] is True
    assert out.p_one == pytest.approx(0.75) and 0 < out.p01_mass < 1


def test_diagnostic_judge_handles_batch_encoding(fake_hf, monkeypatch):
    from stateful_safety.judge import TransformersJudge
    from stateful_safety.schema import MonitorView
    judge = TransformersJudge("org/m", revision=SHA, dtype="bfloat16")
    monkeypatch.setattr(judge.tokenizer, "decode", lambda ids, skip_special_tokens: "RISK_SCORE: 0.25")
    decision = judge.score(MonitorView("s", 0, "x", ()))
    assert decision.score == 0.25 and decision.input_tokens == 4
    assert "attention_mask" in fake_hf.kwargs
