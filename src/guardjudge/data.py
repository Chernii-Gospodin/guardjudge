"""JSONL dataset helpers.

Record format (one JSON per line):
  {"stage": "input|tool_call|rag|output", "text": "...", "context": "... (optional)",
   "label": "safe|unsafe", "category": "none|jailbreak|..."}
"""
from __future__ import annotations

import json
from pathlib import Path

from .schema import STAGES, build_messages, format_target


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
            records.append(rec)
    return records


def to_prompt_completion(rec: dict) -> dict:
    """TRL prompt-completion format: loss is computed on the completion only."""
    return {
        "prompt": build_messages(rec["stage"], rec["text"], rec.get("context")),
        "completion": [
            {"role": "assistant", "content": format_target(rec["label"], rec.get("category", "none"))}
        ],
    }
