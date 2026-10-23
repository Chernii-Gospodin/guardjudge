"""Merge the trained LoRA adapter into the base model for vLLM/Ollama serving.

Memory-safe defaults: `auto` picks bf16 on Ampere+ / fp16 elsewhere and a CUDA device
when available, avoiding the ~2x RAM cost of an fp32 merge on low-memory hosts.
Usage: python scripts/merge_lora.py --base Qwen/Qwen3Guard-Gen-4B --adapter outputs/<run> --out merged/guardjudge-4b
"""
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def resolve_dtype(name: str) -> torch.dtype:
    if name == "auto":
        return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    return _DTYPES[name]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", choices=["auto", "fp16", "bf16", "fp32"], default="auto",
                    help="auto: bf16 on Ampere+ / fp16 elsewhere (low memory); fp32 needs ~2x model size in RAM")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                    help="auto: cuda when available, else cpu")
    a = ap.parse_args()

    dtype = resolve_dtype(a.dtype)
    device = a.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_map = "cpu" if device == "cpu" else "auto"
    print(f"[info] dtype={a.dtype} ({dtype}) device={device}")

    model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=dtype, device_map=device_map)
    model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
    model.save_pretrained(a.out)
    AutoTokenizer.from_pretrained(a.base).save_pretrained(a.out)


if __name__ == "__main__":
    main()
