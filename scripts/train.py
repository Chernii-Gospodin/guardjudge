"""Full fine-tuning (all weights, no LoRA/PEFT) of a small LLM into a guardrail judge.

The model learns to answer the judge prompt with a strict JSON verdict. Loss is computed
only on the verdict tokens. Master weights are kept in fp32; `precision` selects mixed
precision (fp16 on T4, bf16 on Ampere+), so a ~0.6B model fits on a 16 GB T4.

Usage:
  python scripts/train.py --config configs/train_full.yaml
  python scripts/train.py --config configs/train_full.yaml --dry-run   # show one tokenized example
  python scripts/train.py --config configs/train_guardrails.yaml       --set model_name=Qwen/Qwen3Guard-Gen-0.6B --set train.epochs=1   # override config values
  python scripts/train.py --config ... --resume                       # continue from last checkpoint
"""
import argparse
import json
import math
import random
import shutil
from pathlib import Path

import yaml

from guardjudge.data import (IGNORE_INDEX, apply_chat_template_override, load_jsonl, pad_batch,
                             tokenize_records)


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply `dotted.key=value` overrides; values are parsed as YAML (numbers, bools, null)."""
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"--set expects key=value, got {item!r}")
        node, parts = cfg, key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(value)
    return cfg


def subsample(records: list[dict], n: int | None, seed: int) -> list[dict]:
    if n and n < len(records):
        return random.Random(seed).sample(records, n)
    return records


def resolve_precision(precision: str, cuda: bool, bf16_ok: bool) -> tuple[bool, bool]:
    """Return (bf16, fp16) flags for the Trainer, downgrading bf16 to fp16 when unsupported."""
    precision = (precision or "fp16").lower()
    if precision not in ("fp16", "bf16", "fp32"):
        raise ValueError(f"unknown precision {precision!r}; expected fp16 | bf16 | fp32")
    if not cuda or precision == "fp32":
        return False, False
    if precision == "bf16" and not bf16_ok:
        print("[warn] bf16 is not supported on this GPU; falling back to fp16.")
        return False, True
    return precision == "bf16", precision == "fp16"


def show_example(tok, ex: dict) -> None:
    ids, labels = ex["input_ids"], ex["labels"]
    start = next(i for i, l in enumerate(labels) if l != IGNORE_INDEX)
    print("=" * 72)
    print("PROMPT (no loss):\n" + tok.decode(ids[:start]))
    print("-" * 72)
    print("TARGET (loss):\n" + tok.decode(ids[start:]))
    print("=" * 72)
    print(f"tokens: prompt={start} target={len(ids) - start}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_full.yaml")
    ap.add_argument("--dry-run", action="store_true", help="tokenize data, print one example, exit")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config value, e.g. --set train.epochs=1 (repeatable)")
    ap.add_argument("--resume", action="store_true", help="resume from the last checkpoint in checkpoint_dir")
    args = ap.parse_args()
    cfg = apply_overrides(yaml.safe_load(open(args.config, encoding="utf-8")), args.set)
    t = cfg["train"]
    tmpl_kwargs = cfg.get("chat_template_kwargs") or {}

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    apply_chat_template_override(tok, cfg.get("chat_template"))

    train_recs = subsample(load_jsonl(cfg["train_file"]), cfg.get("max_train_samples"), t["seed"])
    eval_recs = subsample(load_jsonl(cfg["eval_file"]), cfg.get("max_eval_samples"), t["seed"])
    train_ex, skip_tr = tokenize_records(tok, train_recs, t["max_length"], tmpl_kwargs)
    eval_ex, skip_ev = tokenize_records(tok, eval_recs, t["max_length"], tmpl_kwargs)
    print(f"[info] train={len(train_ex)} eval={len(eval_ex)} "
          f"skipped as longer than max_length={t['max_length']}: train={skip_tr} eval={skip_ev}")
    if not train_ex:
        raise SystemExit("no training examples left; increase max_length or check the data")
    show_example(tok, train_ex[0])
    if args.dry_run:
        return

    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, set_seed

    set_seed(t["seed"])
    cuda = torch.cuda.is_available()
    use_bf16, use_fp16 = resolve_precision(cfg.get("precision", "fp16"), cuda,
                                           cuda and torch.cuda.is_bf16_supported())
    print(f"[info] model={cfg['model_name']} cuda={cuda} bf16={use_bf16} fp16={use_fp16}")
    if not cuda:
        print("[warn] no CUDA GPU: training on CPU is only practical for tiny smoke-test models.")

    # fp32 master weights; autocast to fp16/bf16 is done by the Trainer (mixed precision).
    model = AutoModelForCausalLM.from_pretrained(cfg["model_name"], dtype=torch.float32)
    model.config.use_cache = False
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[info] trainable parameters: {n_params / 1e6:.1f}M (full fine-tuning)")

    def collate(features):
        return {k: torch.tensor(v) for k, v in pad_batch(features, tok.pad_token_id).items()}

    steps_per_epoch = math.ceil(len(train_ex) / (t["per_device_batch_size"] * t["grad_accum_steps"]))
    out = Path(cfg["output_dir"])
    # fp32 weights + optimizer state: ~16 bytes/param (~10 GB for 0.6B). In Colab keep it on the
    # local disk (/content/...), not on Google Drive.
    ckpt_dir = Path(cfg.get("checkpoint_dir") or out / "checkpoints")
    # "epoch" or a number of optimizer steps (more frequent saves survive Colab disconnects)
    every = t.get("eval_save_every", "epoch")
    strategy = "epoch" if every == "epoch" else "steps"
    targs = TrainingArguments(
        output_dir=str(ckpt_dir),
        num_train_epochs=t["epochs"],
        per_device_train_batch_size=t["per_device_batch_size"],
        per_device_eval_batch_size=t["per_device_batch_size"],
        gradient_accumulation_steps=t["grad_accum_steps"],
        learning_rate=float(t["learning_rate"]),
        weight_decay=t.get("weight_decay", 0.0),
        lr_scheduler_type=t["lr_scheduler"],
        warmup_steps=max(1, int(t["warmup_ratio"] * steps_per_epoch * t["epochs"])),
        max_grad_norm=t.get("max_grad_norm", 1.0),
        logging_steps=t["logging_steps"],
        eval_strategy=strategy if eval_ex else "no",
        save_strategy=strategy,
        eval_steps=None if strategy == "epoch" else int(every),
        save_steps=None if strategy == "epoch" else int(every),
        save_total_limit=1,
        load_best_model_at_end=bool(eval_ex),
        metric_for_best_model="eval_loss" if eval_ex else None,
        fp16=use_fp16,
        bf16=use_bf16,
        optim=t.get("optim", "adamw_torch"),
        gradient_checkpointing=t["gradient_checkpointing"],
        group_by_length=t.get("group_by_length", False),
        remove_unused_columns=False,
        seed=t["seed"],
        report_to="none",
    )
    trainer = Trainer(
        model=model, args=targs, data_collator=collate,
        train_dataset=Dataset.from_list(train_ex),
        eval_dataset=Dataset.from_list(eval_ex) if eval_ex else None,
    )
    resume = args.resume and ckpt_dir.is_dir() and any(ckpt_dir.glob("checkpoint-*"))
    if args.resume and not resume:
        print(f"[warn] --resume: no checkpoints in {ckpt_dir}, starting from scratch")
    trainer.train(resume_from_checkpoint=True if resume else None)

    # Save a regular HF checkpoint: it is served as is by vLLM / scripts/serve.py, no merge step.
    model.config.use_cache = True
    save_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[
        cfg.get("save_dtype", "bf16")]
    model.to(save_dtype)  # halves the checkpoint size vs fp32
    trainer.save_model(str(out))
    tok.save_pretrained(str(out))
    with open(out / "guardjudge.json", "w", encoding="utf-8") as f:
        json.dump({"base_model": cfg["model_name"], "chat_template": cfg.get("chat_template"),
                   "chat_template_kwargs": tmpl_kwargs, "max_length": t["max_length"],
                   "train_examples": len(train_ex), "epochs": t["epochs"],
                   "log_history": trainer.state.log_history}, f, indent=2)
    if not cfg.get("keep_checkpoints", False):
        shutil.rmtree(ckpt_dir, ignore_errors=True)
    print(f"[info] saved fine-tuned judge to {out}")


if __name__ == "__main__":
    main()
