import json

import pytest

from guardjudge.data import load_jsonl
from guardjudge.schema import build_messages, format_target, parse_verdict


def test_format_target_roundtrip():
    out = format_target("unsafe", "jailbreak")
    assert json.loads(out) == {"verdict": "unsafe", "category": "jailbreak"}
    v = parse_verdict(out)
    assert v.unsafe and v.category == "jailbreak" and v.parsed_ok


def test_safe_forces_category_none():
    assert json.loads(format_target("safe", "jailbreak"))["category"] == "none"


def test_parse_noisy_json():
    v = parse_verdict('Sure.\n{"verdict": "unsafe", "category": "tool_abuse"} done')
    assert v.unsafe and v.category == "tool_abuse"


def test_parse_native_qwen3guard():
    assert parse_verdict("Safety: Unsafe\nCategories: Violent").unsafe
    assert parse_verdict("Safety: Controversial").unsafe
    assert not parse_verdict("Safety: Controversial", controversial_is_unsafe=False).unsafe
    assert not parse_verdict("Safety: Safe\nCategories: None").unsafe


def test_parse_garbage_flags_failure():
    v = parse_verdict("I cannot say")
    assert not v.parsed_ok and not v.unsafe


def test_bad_stage_rejected():
    with pytest.raises(ValueError):
        build_messages("nonsense", "x")


def test_demo_data_is_valid():
    for name in ("train", "val", "test"):
        recs = load_jsonl(f"data/demo_{name}.jsonl")
        assert recs
        assert {r["stage"] for r in recs} <= {"input", "tool_call", "rag", "output"}


def test_parse_llama_guard_native():
    v = parse_verdict("\n\nunsafe\nS2")
    assert v.unsafe and v.parsed_ok
    assert not parse_verdict("safe").unsafe


# --- SFT tokenization (stub tokenizer: no model download needed) ---

from guardjudge.data import IGNORE_INDEX, pad_batch, tokenize_example  # noqa: E402


class StubTokenizer:
    """One token per character; the chat template is a simple tag format."""
    eos_token_id = 0
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kw):
        assert tokenize is False
        s = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        if kw.get("enable_thinking") is False:
            s += "<nothink>"
        return s + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


REC = {"stage": "input", "text": "hi", "label": "unsafe", "category": "jailbreak"}


def test_loss_only_on_verdict():
    ex = tokenize_example(StubTokenizer(), REC, max_length=10_000)
    target = [l for l in ex["labels"] if l != IGNORE_INDEX]
    assert "".join(map(chr, target[:-1])) == format_target("unsafe", "jailbreak")
    assert target[-1] == StubTokenizer.eos_token_id
    assert len(ex["input_ids"]) == len(ex["labels"]) == len(ex["attention_mask"])
    # labels are a suffix of input_ids
    assert ex["input_ids"][-len(target):] == target


def test_chat_template_kwargs_are_passed():
    a = tokenize_example(StubTokenizer(), REC, 10_000)
    b = tokenize_example(StubTokenizer(), REC, 10_000, {"enable_thinking": False})
    assert len(b["input_ids"]) == len(a["input_ids"]) + len("<nothink>")


def test_too_long_is_skipped():
    assert tokenize_example(StubTokenizer(), REC, max_length=20) is None


def test_pad_batch():
    b = pad_batch([{"input_ids": [5, 6, 7], "attention_mask": [1, 1, 1], "labels": [-100, 6, 7]},
                   {"input_ids": [5], "attention_mask": [1], "labels": [5]}], pad_token_id=0)
    assert b["input_ids"][1] == [5, 0, 0]
    assert b["attention_mask"][1] == [1, 0, 0]
    assert b["labels"][1] == [5, IGNORE_INDEX, IGNORE_INDEX]


def test_chatml_override():
    from guardjudge.data import CHATML_NO_THINK, apply_chat_template_override

    class Tok:
        chat_template = "original"

    t = Tok()
    apply_chat_template_override(t, None)
    assert t.chat_template == "original"
    apply_chat_template_override(t, "chatml")
    assert t.chat_template == CHATML_NO_THINK
    with pytest.raises(ValueError):
        apply_chat_template_override(t, "nope")


def test_chatml_renders_like_qwen3():
    jinja2 = pytest.importorskip("jinja2")
    from guardjudge.data import CHATML_NO_THINK

    out = jinja2.Template(CHATML_NO_THINK).render(
        messages=[{"role": "system", "content": "S"}, {"role": "user", "content": "U"}],
        add_generation_prompt=True)
    assert out == ("<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nU<|im_end|>\n"
                   "<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_client_fails_closed_on_garbage(monkeypatch):
    pytest.importorskip("openai")
    import asyncio
    from types import SimpleNamespace

    from guardjudge.client import JudgeClient

    def fake_client(answer):
        async def create(**kw):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=answer))])
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    c = JudgeClient(base_url="http://x/v1", fail_open=False)
    c._client = fake_client("I cannot say")
    v = asyncio.run(c.judge("input", "hi"))
    assert v.unsafe and not v.parsed_ok
    c.fail_open = True
    assert not asyncio.run(c.judge("input", "hi")).unsafe
    c._client = fake_client('{"verdict": "safe", "category": "none"}')
    c.fail_open = False
    assert not asyncio.run(c.judge("input", "hi")).unsafe


def test_compat_training_args():
    import dataclasses
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("train_script", "scripts/train.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["train_script"] = mod
    spec.loader.exec_module(mod)

    @dataclasses.dataclass
    class V4:
        output_dir: str = ""
        group_by_length: bool = False

    @dataclasses.dataclass
    class V5:
        output_dir: str = ""
        train_sampling_strategy: str = "random"

    assert mod.compat_training_args(V4, {"output_dir": "o"}, True) == {"output_dir": "o", "group_by_length": True}
    assert mod.compat_training_args(V5, {"output_dir": "o", "bogus": 1}, True) == {
        "output_dir": "o", "train_sampling_strategy": "group_by_length"}


def test_verdict_only_loss_matches_stock_loss():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("train_script", "scripts/train.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["train_script"] = mod
    spec.loader.exec_module(mod)

    torch.manual_seed(0)
    cfg = transformers.Qwen3Config(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, head_dim=8)
    model = transformers.AutoModelForCausalLM.from_config(cfg)
    ids = torch.randint(0, 97, (3, 12))
    att = torch.ones_like(ids)
    att[2, 9:] = 0  # padded row
    labels = torch.full_like(ids, IGNORE_INDEX)
    labels[0, 8:] = ids[0, 8:]
    labels[1, 5:] = ids[1, 5:]
    labels[2, 6:9] = ids[2, 6:9]
    stock = model(input_ids=ids, attention_mask=att, labels=labels).loss
    s, n = mod.verdict_only_loss(model, ids, att, labels)
    assert int(n) == int((labels[:, 1:] != IGNORE_INDEX).sum())
    assert torch.allclose(stock, s / n, atol=1e-6)
