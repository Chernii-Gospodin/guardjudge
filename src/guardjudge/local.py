"""In-process judge on top of HF Transformers (no server needed).

Used by scripts/evaluate.py (--backend local) and scripts/serve.py. torch/transformers are
imported lazily so the rest of the package works without them.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from .data import apply_chat_template_override, render_prompt
from .schema import Verdict, build_messages, parse_verdict

# Qwen3 is a hybrid "thinking" model: without this the template invites a <think> block.
# Tokenizers whose template does not use the variable simply ignore it.
DEFAULT_CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}


class LocalJudge:
    def __init__(self, model_path: str, device: str | None = None, dtype: str = "auto",
                 chat_template_kwargs: dict | None = None, chat_template: str | None = None,
                 max_new_tokens: int = 32):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if dtype == "auto":
            if self.device == "cpu":
                torch_dtype = torch.float32
            elif torch.cuda.is_bf16_supported():
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float16
        else:
            torch_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype]

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # A fine-tuned checkpoint already carries its (possibly overridden) template.
        apply_chat_template_override(self.tokenizer, chat_template)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch_dtype).to(self.device)
        self.model.eval()
        if chat_template_kwargs is None:
            # A checkpoint from scripts/train.py records the kwargs it was trained with.
            meta = Path(model_path) / "guardjudge.json"
            chat_template_kwargs = (json.loads(meta.read_text(encoding="utf-8"))
                                    .get("chat_template_kwargs", {}) if meta.is_file()
                                    else DEFAULT_CHAT_TEMPLATE_KWARGS)
        self.chat_template_kwargs = chat_template_kwargs
        self.max_new_tokens = max_new_tokens
        self._lock = threading.Lock()  # generate() is not safe to call concurrently
        self._torch = torch

    def _gen_kwargs(self, max_new_tokens: int | None) -> dict:
        return dict(
            max_new_tokens=max_new_tokens or self.max_new_tokens,
            do_sample=False,
            temperature=None, top_p=None, top_k=None,  # silence sampling-config warnings
            pad_token_id=self.tokenizer.pad_token_id,
        )

    def generate(self, messages: list[dict], max_new_tokens: int | None = None,
                 chat_template_kwargs: dict | None = None) -> tuple[str, int, int]:
        """Greedy-decode one answer. Returns (text, prompt_tokens, completion_tokens)."""
        kwargs = {**self.chat_template_kwargs, **(chat_template_kwargs or {})}
        prompt = render_prompt(self.tokenizer, messages, kwargs)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        n_prompt = inputs["input_ids"].shape[1]
        with self._lock, self._torch.inference_mode():
            out = self.model.generate(**inputs, **self._gen_kwargs(max_new_tokens))
        new = out[0][n_prompt:]
        return self.tokenizer.decode(new, skip_special_tokens=True), n_prompt, int(new.shape[0])

    def generate_batch(self, batch: list[list[dict]], max_new_tokens: int | None = None) -> list[str]:
        """Greedy-decode several conversations at once (left padding). Same answers as generate()
        up to numerical noise; used for fast offline evaluation."""
        prompts = [render_prompt(self.tokenizer, m, self.chat_template_kwargs) for m in batch]
        side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True,
                                    add_special_tokens=False).to(self.device)
        finally:
            self.tokenizer.padding_side = side
        with self._lock, self._torch.inference_mode():
            out = self.model.generate(**inputs, **self._gen_kwargs(max_new_tokens))
        new = out[:, inputs["input_ids"].shape[1]:]
        return self.tokenizer.batch_decode(new, skip_special_tokens=True)

    def judge(self, stage: str, text: str, context: str | None = None) -> Verdict:
        raw, _, _ = self.generate(build_messages(stage, text, context))
        return parse_verdict(raw)
