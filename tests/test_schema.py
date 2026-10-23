import json

import pytest

from guardjudge.data import load_jsonl, to_prompt_completion
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
        pc = to_prompt_completion(recs[0])
        assert pc["prompt"][-1]["role"] == "user" and pc["completion"][0]["role"] == "assistant"
