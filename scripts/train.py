"""LoRA SFT of the judge model.

Plain LoRA + fp16 by default (T4-compatible). QLoRA (4-bit base) is an optional
fallback via `load_in_4bit: true` in the config.

Usage: python scripts/train.py --config configs/train_lora.yaml
"""
import argparse

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from guardjudge.data import load_jsonl, to_prompt_completion


def resolve_precision(precision: str) -> tuple[bool, bool, torch.dtype]:
    """Map a precision name to (use_bf16, use_fp16, dtype), downgrading bf16 when unsupported."""
    precision = (precision or "fp16").lower()
    if precision == "fp32":
        return False, False, torch.float32
    if precision == "bf16":
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            print("[warn] bf16 is not supported on this device; falling back to fp16.")
            return False, True, torch.float16
        return True, False, torch.bfloat16
    if precision != "fp16":
        raise ValueError(f"unknown precision {precision!r}; expected fp16 | bf16 | fp32")
    return False, True, torch.float16


def resolve_device_map(name: str):
    """Pin to a single GPU by default; accelerate sharding (`auto`) breaks DDP on multi-GPU hosts."""
    if name == "single":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "device_map: single requires a CUDA GPU (training is GPU-only). "
                "Set device_map: auto to opt into CPU placement."
            )
        return {"": torch.cuda.current_device()}
    return "auto"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_lora.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    t, l = cfg["train"], cfg["lora"]

    use_bf16, use_fp16, dtype = resolve_precision(cfg.get("precision", "fp16"))
    load_in_4bit = bool(cfg.get("load_in_4bit", False))
    device_map = resolve_device_map(cfg.get("device_map", "single"))
    print(f"[info] precision={cfg.get('precision', 'fp16')} bf16={use_bf16} fp16={use_fp16} "
          f"load_in_4bit={load_in_4bit} device_map={device_map}")

    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if load_in_4bit:
        from peft import prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg["model_name"], quantization_config=bnb, device_map=device_map
        )
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=t["gradient_checkpointing"]
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["model_name"], torch_dtype=dtype, device_map=device_map
        )
        # PEFT + gradient checkpointing needs the frozen base to produce input grads.
        if t["gradient_checkpointing"]:
            model.enable_input_require_grads()

    train_ds = Dataset.from_list([to_prompt_completion(r) for r in load_jsonl(cfg["train_file"])])
    eval_ds = Dataset.from_list([to_prompt_completion(r) for r in load_jsonl(cfg["eval_file"])])

    peft_cfg = LoraConfig(
        r=l["r"], lora_alpha=l["alpha"], lora_dropout=l["dropout"],
        target_modules=l["target_modules"], bias="none", task_type="CAUSAL_LM",
    )
    sft_cfg = SFTConfig(
        output_dir=cfg["output_dir"],
        num_train_epochs=t["epochs"],
        per_device_train_batch_size=t["per_device_batch_size"],
        per_device_eval_batch_size=t["per_device_batch_size"],
        gradient_accumulation_steps=t["grad_accum_steps"],
        learning_rate=float(t["learning_rate"]),
        lr_scheduler_type=t["lr_scheduler"],
        warmup_ratio=t["warmup_ratio"],
        logging_steps=t["logging_steps"],
        eval_strategy="epoch",
        save_strategy="epoch",
        fp16=use_fp16,
        bf16=use_bf16,
        optim=t.get("optim", "adamw_torch"),
        gradient_checkpointing=t["gradient_checkpointing"],
        max_length=t["max_length"],
        seed=t["seed"],
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model, args=sft_cfg, train_dataset=train_ds, eval_dataset=eval_ds,
        processing_class=tok, peft_config=peft_cfg,
    )
    trainer.train()
    trainer.save_model(cfg["output_dir"])  # saves the LoRA adapter
    tok.save_pretrained(cfg["output_dir"])


if __name__ == "__main__":
    main()
