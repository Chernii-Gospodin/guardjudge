"""JSONL dataset helpers and SFT tokenization.

Record format (one JSON per line):
  {"stage": "input|tool_call|rag|output", "text": "...", "context": "... (optional)",
   "label": "safe|unsafe", "category": "none|jailbreak|..."}

Tokenization is done by hand (no TRL): the prompt is rendered with the model's own chat
template, the JSON verdict + EOS is appended, and the loss is computed only on the verdict
tokens (prompt labels are set to -100). The same `render_prompt` is used at inference time,
so training and serving see exactly the same text.
"""
from __future__ import annotations

import json
from pathlib import Path

from .schema import CATEGORIES, STAGES, build_messages, format_target

IGNORE_INDEX = -100

# Plain ChatML with an empty <think> block: renders exactly what Qwen3-0.6B's own template renders
# with enable_thinking=False. Used for models whose built-in template rewrites the conversation,
# e.g. Qwen3Guard-Gen wraps every chat into its own fixed classification instruction. With this
# override Qwen3-0.6B and Qwen3Guard-Gen-0.6B see byte-identical prompts during fine-tuning.
CHATML_NO_THINK = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n<think>\n\n</think>\n\n{% endif %}"
)
CHAT_TEMPLATES = {"chatml": CHATML_NO_THINK}


def apply_chat_template_override(tokenizer, name: str | None) -> None:
    """Replace the tokenizer's chat template by a named one ("chatml"); None keeps the original.

    The override is saved together with the fine-tuned tokenizer, so vLLM / serve.py use it too.
    """
    if not name:
        return
    if name not in CHAT_TEMPLATES:
        raise ValueError(f"unknown chat_template {name!r}; expected one of {sorted(CHAT_TEMPLATES)}")
    tokenizer.chat_template = CHAT_TEMPLATES[name]


def load_jsonl(path: str | Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("stage") not in STAGES:
                raise ValueError(f"{path}:{i}: bad stage {rec.get('stage')!r}")
            if rec.get("label") not in ("safe", "unsafe"):
                raise ValueError(f"{path}:{i}: bad label {rec.get('label')!r}")
            if rec.get("category", "none") not in CATEGORIES:
                raise ValueError(f"{path}:{i}: bad category {rec.get('category')!r}")
            if not isinstance(rec.get("text"), str) or not rec["text"].strip():
                raise ValueError(f"{path}:{i}: empty text")
            records.append(rec)
    return records


def render_prompt(tokenizer, messages: list[dict], chat_template_kwargs: dict | None = None) -> str:
    """Chat-template the prompt and open the assistant turn (what the model continues from)."""
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **(chat_template_kwargs or {})
    )


def tokenize_example(tokenizer, rec: dict, max_length: int,
                     chat_template_kwargs: dict | None = None) -> dict | None:
    """Build input_ids / labels for one record. Returns None if it does not fit in max_length."""
    prompt = render_prompt(tokenizer, build_messages(rec["stage"], rec["text"], rec.get("context")),
                           chat_template_kwargs)
    target = format_target(rec["label"], rec.get("category", "none"))
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    if len(prompt_ids) + len(target_ids) > max_length:
        return None
    return {
        "input_ids": prompt_ids + target_ids,
        "attention_mask": [1] * (len(prompt_ids) + len(target_ids)),
        "labels": [IGNORE_INDEX] * len(prompt_ids) + target_ids,
    }


def tokenize_records(tokenizer, records: list[dict], max_length: int,
                     chat_template_kwargs: dict | None = None) -> tuple[list[dict], int]:
    """Tokenize all records; returns (examples, number_skipped_as_too_long)."""
    out, skipped = [], 0
    for rec in records:
        ex = tokenize_example(tokenizer, rec, max_length, chat_template_kwargs)
        if ex is None:
            skipped += 1
        else:
            out.append(ex)
    return out, skipped


def pad_batch(features: list[dict], pad_token_id: int) -> dict[str, list[list[int]]]:
    """Right-pad a batch (labels with -100). Returns plain lists; the collator wraps them in tensors."""
    width = max(len(f["input_ids"]) for f in features)
    batch = {"input_ids": [], "attention_mask": [], "labels": []}
    for f in features:
        n = width - len(f["input_ids"])
        batch["input_ids"].append(f["input_ids"] + [pad_token_id] * n)
        batch["attention_mask"].append(f["attention_mask"] + [0] * n)
        batch["labels"].append(f["labels"] + [IGNORE_INDEX] * n)
    return batch
