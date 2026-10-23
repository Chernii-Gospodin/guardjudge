"""Evaluate a judge (fine-tuned or baseline) behind an OpenAI-compatible endpoint.

Reports per-stage precision / recall / F1 / false-positive rate / latency (p50, p95).

  python scripts/evaluate.py --data data/demo_test.jsonl --base-url http://localhost:8001/v1 \
      --model guardjudge --prompt-style judge --out results/guardjudge.json

--prompt-style judge : our system prompt + JSON answer (fine-tuned model)
--prompt-style raw   : plain text, for baselines with their own chat template (e.g. Qwen3Guard-Gen
                       without fine-tuning). For stage=output the context is sent as the user turn
                       and the fragment as the assistant turn.
"""
import argparse
import json
import statistics
import time
from collections import defaultdict

from openai import OpenAI

from guardjudge.data import load_jsonl
from guardjudge.schema import build_messages, parse_verdict


def raw_messages(rec: dict) -> list[dict]:
    if rec["stage"] == "output":
        return [{"role": "user", "content": rec.get("context") or ""},
                {"role": "assistant", "content": rec["text"]}]
    return [{"role": "user", "content": rec["text"]}]


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    return values[min(len(values) - 1, int(round(q * (len(values) - 1))))]


def metrics(rows: list[dict]) -> dict:
    tp = sum(r["pred"] and r["gold"] for r in rows)
    fp = sum(r["pred"] and not r["gold"] for r in rows)
    fn = sum((not r["pred"]) and r["gold"] for r in rows)
    tn = sum((not r["pred"]) and (not r["gold"]) for r in rows)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    lat = [r["latency"] for r in rows]
    return {
        "n": len(rows), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
        "fpr": round(fp / (fp + tn), 4) if fp + tn else 0.0,
        "parse_fail": sum(not r["parsed_ok"] for r in rows),
        "latency_p50_s": round(statistics.median(lat), 4) if lat else 0.0,
        "latency_p95_s": round(pct(lat, 0.95), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--base-url", default="http://localhost:8001/v1")
    ap.add_argument("--model", default="guardjudge")
    ap.add_argument("--prompt-style", choices=["judge", "raw"], default="judge")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    client = OpenAI(base_url=a.base_url, api_key="EMPTY")
    rows, by_stage = [], defaultdict(list)
    for rec in load_jsonl(a.data):
        msgs = (build_messages(rec["stage"], rec["text"], rec.get("context"))
                if a.prompt_style == "judge" else raw_messages(rec))
        t0 = time.perf_counter()
        resp = client.chat.completions.create(model=a.model, messages=msgs,
                                              temperature=0.0, max_tokens=64)
        latency = time.perf_counter() - t0
        v = parse_verdict(resp.choices[0].message.content or "")
        row = {"stage": rec["stage"], "gold": rec["label"] == "unsafe", "pred": v.unsafe,
               "parsed_ok": v.parsed_ok, "latency": latency}
        rows.append(row)
        by_stage[rec["stage"]].append(row)

    report = {"overall": metrics(rows), **{s: metrics(r) for s, r in by_stage.items()}}
    print(f"{'stage':<10}{'n':>5}{'P':>8}{'R':>8}{'F1':>8}{'FPR':>8}{'p50,s':>9}{'p95,s':>9}")
    for name, m in report.items():
        print(f"{name:<10}{m['n']:>5}{m['precision']:>8.3f}{m['recall']:>8.3f}{m['f1']:>8.3f}"
              f"{m['fpr']:>8.3f}{m['latency_p50_s']:>9.3f}{m['latency_p95_s']:>9.3f}")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
