"""Convert the team dataset (guardrails_dataset) into the GuardJudge JSONL format.

Source records ({train,validation,test}.jsonl in the dataset folder):
  {"id", "text": str | {"trusted_instruction", "user_task", "untrusted_content"},
   "is_unsafe": 0|1, "category": "harmful_content|prompt_injection", "subcategory",
   "language", "source", "group_id", "provenance"}

Output records (data/guardrails/{train,val,test}.jsonl):
  {"stage": "input", "text", "context"?, "label": "safe|unsafe", "category",
   "id", "source", "language", "subcategory"}

Mapping:
  * every record is a user input -> stage "input" (the dataset has no tool_call/rag/output rows);
  * structured text: untrusted_content -> text, trusted_instruction + user_task -> context;
  * is_unsafe=1 -> label "unsafe", category = source category (harmful_content | prompt_injection);
    is_unsafe=0 -> label "safe", category "none" (the source category only names the domain there).
The official splits are kept as is (they are already group-disjoint, see group_id).

  python scripts/prepare_dataset.py --src ../guardrails_dataset --out data/guardrails
"""
import argparse
import json
from collections import Counter
from pathlib import Path

SPLITS = {"train": "train", "validation": "val", "test": "test"}
KEEP = ("id", "source", "language", "subcategory")


def convert(rec: dict) -> dict:
    text, context = rec["text"], None
    if isinstance(text, dict):
        context = (f"Trusted instruction: {text.get('trusted_instruction', '')}\n"
                   f"User task: {text.get('user_task', '')}")
        text = text.get("untrusted_content", "")
    unsafe = int(rec["is_unsafe"]) == 1
    category = rec.get("category", "harmful_content") if unsafe else "none"
    if category not in ("harmful_content", "prompt_injection", "none"):
        category = "harmful_content"
    out = {"stage": "input", "text": text, "label": "unsafe" if unsafe else "safe", "category": category}
    if context:
        out["context"] = context
    out.update({k: rec.get(k) for k in KEEP if rec.get(k) is not None})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="../guardrails_dataset", help="folder with train/validation/test.jsonl")
    ap.add_argument("--out", default="data/guardrails")
    a = ap.parse_args()

    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for src_name, dst_name in SPLITS.items():
        stats, n, skipped = Counter(), 0, 0
        with open(src / f"{src_name}.jsonl", encoding="utf-8") as fin, \
                open(out / f"{dst_name}.jsonl", "w", encoding="utf-8") as fout:
            for line in fin:
                if not line.strip():
                    continue
                rec = convert(json.loads(line))
                if not isinstance(rec["text"], str) or not rec["text"].strip():
                    skipped += 1
                    continue
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                stats[(rec["label"], rec["category"])] += 1
                n += 1
        print(f"{dst_name:<5} {n:>6} rows (skipped empty: {skipped})  "
              + ", ".join(f"{l}/{c}={k}" for (l, c), k in sorted(stats.items())))


if __name__ == "__main__":
    main()
