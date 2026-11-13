"""Evaluate a judge (fine-tuned or baseline) on a labelled JSONL set.

Reports precision / recall / F1 / false-positive rate / parse failures / latency, overall and
by stage, source, language and attack category.

Backends:
  --backend local : load the model in-process with Transformers (no server; e.g. right after
                    training in Colab). --model is a local path or a HF repo id. Quality is
                    computed with batched generation (--batch-size); latency is measured
                    separately on --latency-samples examples one at a time (batch 1, like a
                    real guardrail call).
  --backend http  : any OpenAI-compatible endpoint (scripts/serve.py, vLLM, Ollama); one
                    request per example, latency = request round-trip.

Prompt styles:
  --prompt-style judge : our system prompt + JSON answer (fine-tuned models, zero-shot base model)
  --prompt-style raw   : plain conversation for guard models with their own template
                         (Qwen3Guard-Gen, Llama Guard). For stage=output the context goes into the
                         user turn and the fragment into the assistant turn.

Examples (before / after fine-tuning):
  python scripts/evaluate.py --data data/guardrails/test.jsonl --model Qwen/Qwen3-0.6B \
      --name "Qwen3-0.6B (zero-shot)" --out results/qwen3_before.json
  python scripts/evaluate.py --data data/guardrails/test.jsonl --model Qwen/Qwen3Guard-Gen-0.6B \
      --prompt-style raw --name "Qwen3Guard-Gen-0.6B (native)" --out results/qwen3guard_before.json
  python scripts/evaluate.py --data data/guardrails/test.jsonl --model outputs/guardjudge-qwen3-0.6b \
      --name "Qwen3-0.6B + full FT" --out results/qwen3_after.json
"""
import argparse
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

from guardjudge.data import load_jsonl, render_prompt
from guardjudge.metrics import binary_metrics, category_metrics, group_metrics, recall_by_category
from guardjudge.schema import CATEGORIES, build_messages, parse_verdict

# less fragmentation when batch shapes vary a lot (must be set before CUDA is initialised)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def raw_messages(rec: dict) -> list[dict]:
    if rec["stage"] == "output":
        return [{"role": "user", "content": rec.get("context") or ""},
                {"role": "assistant", "content": rec["text"]}]
    text = rec["text"]
    if rec.get("context"):  # structured input: show the guard the whole thing
        text = f"{rec['context']}\n\n{text}"
    return [{"role": "user", "content": text}]


class LocalBackend:
    def __init__(self, a):
        from guardjudge.local import LocalJudge

        tmpl = json.loads(a.chat_template_kwargs) if a.chat_template_kwargs else None
        self.judge = LocalJudge(a.model, device=a.device, dtype=a.dtype, chat_template_kwargs=tmpl,
                                chat_template=a.chat_template, max_new_tokens=a.max_new_tokens)
        self.batch_size = a.batch_size
        self.max_batch_tokens = a.max_batch_tokens
        self.oom_failures = 0

    def one(self, msgs):
        return self.judge.generate(msgs)[0]

    def _prompt_len(self, msgs) -> int:
        prompt = render_prompt(self.judge.tokenizer, msgs, self.judge.chat_template_kwargs)
        return len(self.judge.tokenizer(prompt, add_special_tokens=False)["input_ids"])

    def _make_batches(self, lengths: list[int]) -> list[list[int]]:
        """Sort by length and pack: a batch holds at most batch_size rows and its padded size
        (rows x longest prompt) stays under max_batch_tokens, so a few very long inputs
        (some test rows have ~5k tokens) run in small batches instead of blowing up memory."""
        order = sorted(range(len(lengths)), key=lengths.__getitem__)
        batches, cur = [], []
        for i in order:
            width = lengths[i]  # sorted ascending -> the new row is the longest in the batch
            if cur and (len(cur) >= self.batch_size or (len(cur) + 1) * width > self.max_batch_tokens):
                batches.append(cur)
                cur = []
            cur.append(i)
        if cur:
            batches.append(cur)
        return batches

    def _run(self, batch_msgs: list[list[dict]]) -> list[str]:
        """generate_batch with an OOM fallback: halve the batch until it fits."""
        import torch

        try:
            return self.judge.generate_batch(batch_msgs)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(batch_msgs) == 1:
                self.oom_failures += 1
                print("  [warn] OOM on a single example; counted as unparsed answer", flush=True)
                return [""]
            mid = len(batch_msgs) // 2
            return self._run(batch_msgs[:mid]) + self._run(batch_msgs[mid:])

    def many(self, all_msgs: list[list[dict]], progress: bool = True) -> list[str]:
        lengths = [self._prompt_len(m) for m in all_msgs]
        batches = self._make_batches(lengths)
        print(f"  prompt tokens: median {sorted(lengths)[len(lengths) // 2]}, max {max(lengths)}; "
              f"{len(batches)} batches", flush=True)
        answers: list[str | None] = [None] * len(all_msgs)
        t0, done = time.perf_counter(), 0
        for n, idx in enumerate(batches):
            for i, ans in zip(idx, self._run([all_msgs[i] for i in idx])):
                answers[i] = ans
            done += len(idx)
            if progress and (n % 20 == 0 or done == len(all_msgs)):
                print(f"  {done}/{len(all_msgs)}  {time.perf_counter() - t0:.0f}s", flush=True)
        return answers


class HttpBackend:
    def __init__(self, a):
        from openai import OpenAI

        self.client = OpenAI(base_url=a.base_url, api_key="EMPTY")
        self.model = a.model
        self.max_new_tokens = a.max_new_tokens
        self.extra = {"chat_template_kwargs": json.loads(a.chat_template_kwargs or '{"enable_thinking": false}')}

    def one(self, msgs):
        resp = self.client.chat.completions.create(model=self.model, messages=msgs, temperature=0.0,
                                                   max_tokens=self.max_new_tokens, extra_body=self.extra)
        return resp.choices[0].message.content or ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--backend", choices=["local", "http"], default="local")
    ap.add_argument("--model", default="outputs/guardjudge-qwen3-0.6b",
                    help="local: path or HF id; http: served model name")
    ap.add_argument("--name", default=None, help="label of this run in reports (default: --model)")
    ap.add_argument("--base-url", default="http://localhost:8001/v1")
    ap.add_argument("--device", default=None, help="local backend: cuda | cpu (default: auto)")
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    ap.add_argument("--chat-template", default=None, choices=["chatml"],
                    help="override the model's chat template (local backend)")
    ap.add_argument("--chat-template-kwargs", default=None,
                    help='JSON, default {"enable_thinking": false}')
    ap.add_argument("--prompt-style", choices=["judge", "raw"], default="judge")
    ap.add_argument("--controversial", choices=["unsafe", "safe"], default="unsafe",
                    help="how to count Qwen3Guard 'Controversial' answers")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=16, help="local backend: max rows per batch")
    ap.add_argument("--max-batch-tokens", type=int, default=8192,
                    help="local backend: max rows x longest prompt per batch (long inputs -> small batches)")
    ap.add_argument("--latency-samples", type=int, default=100,
                    help="local backend: examples timed one by one (0 = skip)")
    ap.add_argument("--warmup", type=int, default=3, help="untimed warm-up calls")
    ap.add_argument("--limit", type=int, default=None, help="evaluate a random subset of N rows")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    ap.add_argument("--save-predictions", action="store_true", help="also dump per-example answers")
    a = ap.parse_args()

    records = load_jsonl(a.data)
    if a.limit and a.limit < len(records):
        records = random.Random(a.seed).sample(records, a.limit)

    def to_msgs(rec):
        return (build_messages(rec["stage"], rec["text"], rec.get("context"))
                if a.prompt_style == "judge" else raw_messages(rec))

    all_msgs = [to_msgs(r) for r in records]
    backend = LocalBackend(a) if a.backend == "local" else HttpBackend(a)
    for msgs in all_msgs[:a.warmup]:
        backend.one(msgs)

    latencies: list[float | None] = [None] * len(records)
    if a.backend == "local":
        answers = backend.many(all_msgs) if a.batch_size > 1 else [backend.one(m) for m in all_msgs]
        timed = random.Random(a.seed + 1).sample(range(len(records)), min(a.latency_samples, len(records)))
        for i in timed:
            t0 = time.perf_counter()
            backend.one(all_msgs[i])
            latencies[i] = time.perf_counter() - t0
    else:
        answers = []
        for i, msgs in enumerate(all_msgs):
            t0 = time.perf_counter()
            answers.append(backend.one(msgs))
            latencies[i] = time.perf_counter() - t0

    rows, by_stage = [], defaultdict(list)
    for rec, raw, lat in zip(records, answers, latencies):
        v = parse_verdict(raw, controversial_is_unsafe=a.controversial == "unsafe")
        row = {"id": rec.get("id"), "source": rec.get("source"), "language": rec.get("language"),
               "stage": rec["stage"], "category": rec.get("category", "none"),
               "gold": rec["label"] == "unsafe", "pred": v.unsafe, "pred_category": v.category,
               "parsed_ok": v.parsed_ok, "latency": lat, "raw": raw, "text": rec["text"]}
        rows.append(row)
        by_stage[rec["stage"]].append(row)

    report = {"name": a.name or a.model, "model": a.model, "backend": a.backend,
              "prompt_style": a.prompt_style, "chat_template": a.chat_template, "data": a.data,
              "overall": binary_metrics(rows),
              "oom_skipped": getattr(backend, "oom_failures", 0),
              "by_stage": {s: binary_metrics(r) for s, r in by_stage.items()},
              # optional metadata from data/guardrails (prepare_dataset.py)
              "by_source": group_metrics(rows, "source"),
              "by_language": group_metrics(rows, "language"),
              # binary detection rate per attack family (only unsafe examples)
              "recall_by_category": recall_by_category(rows),
              # did the judge also name the right category? Only for our JSON format: native guard
              # formats (Qwen3Guard, Llama Guard) use their own taxonomy.
              "category_metrics": (category_metrics(rows, known=set(CATEGORIES) - {"none"})
                                   if a.prompt_style == "judge" else None)}

    def fmt(x):
        return f"{x:>9.3f}" if x is not None else f"{'-':>9}"

    print(f"== {report['name']}")
    print(f"{'stage':<10}{'n':>6}{'P':>8}{'R':>8}{'F1':>8}{'FPR':>8}{'fail':>6}{'p50,s':>9}{'p95,s':>9}")
    for name, m in [("overall", report["overall"]), *report["by_stage"].items()]:
        print(f"{name:<10}{m['n']:>6}{m['precision']:>8.3f}{m['recall']:>8.3f}{m['f1']:>8.3f}"
              f"{m['fpr']:>8.3f}{m['parse_fail']:>6}{fmt(m['latency_p50_s'])}{fmt(m['latency_p95_s'])}")
    for key in ("by_source", "by_language"):
        if report[key]:
            print(key + ":", {g: f"F1={m['f1']:.3f} FPR={m['fpr']:.3f} n={m['n']}"
                              for g, m in report[key].items()})
    print("recall by attack category:", report["recall_by_category"])
    cm = report["category_metrics"]
    if cm:
        print(f"category: macro-F1={cm['macro_f1']:.3f} accuracy={cm['accuracy']:.3f} "
              f"correct category on detected attacks={cm['category_accuracy_on_detected']} "
              f"(n={cm['detected_n']})")
        print("confusion (gold -> predicted):", cm["confusion"])

    if a.out:
        if a.save_predictions:
            report["predictions"] = rows
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
