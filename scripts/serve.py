"""Minimal OpenAI-compatible server for the judge (FastAPI + Transformers).

Lightweight alternative to vLLM: works on CPU and on Windows, enough for NeMo Guardrails,
JudgeClient and `scripts/evaluate.py --backend http`. Requests are processed one at a time.

  python scripts/serve.py --model outputs/guardjudge-qwen3-0.6b --port 8001
  curl http://localhost:8001/health

Endpoints: GET /health, GET /v1/models, POST /v1/chat/completions,
POST /judge  {"stage": "input", "text": "...", "context": null} -> {"verdict", "category", ...}
"""
import argparse
import time
import uuid

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from guardjudge.local import LocalJudge
from guardjudge.schema import STAGES, build_messages, parse_verdict


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[dict]
    max_tokens: int | None = 32
    temperature: float | None = 0.0  # ignored: decoding is always greedy
    chat_template_kwargs: dict | None = None


class JudgeRequest(BaseModel):
    stage: str
    text: str
    context: str | None = None


def build_app(judge: LocalJudge, served_name: str) -> FastAPI:
    app = FastAPI(title="GuardJudge")

    @app.get("/health")
    def health():
        return {"status": "ok", "model": served_name, "device": judge.device}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": served_name, "object": "model", "owned_by": "guardjudge"}]}

    @app.post("/v1/chat/completions")
    def chat(req: ChatRequest):  # sync def -> FastAPI runs it in a worker thread
        text, n_in, n_out = judge.generate(req.messages, min(req.max_tokens or 32, 256),
                                           req.chat_template_kwargs)
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion",
            "created": int(time.time()), "model": served_name,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": n_in, "completion_tokens": n_out, "total_tokens": n_in + n_out},
        }

    @app.post("/judge")
    def judge_endpoint(req: JudgeRequest):
        if req.stage not in STAGES:
            return {"error": f"stage must be one of {STAGES}"}
        t0 = time.perf_counter()
        raw, _, _ = judge.generate(build_messages(req.stage, req.text, req.context))
        v = parse_verdict(raw)
        return {"verdict": "unsafe" if v.unsafe else "safe", "category": v.category,
                "parsed_ok": v.parsed_ok, "latency_s": round(time.perf_counter() - t0, 4)}

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="outputs/guardjudge-qwen3-0.6b")
    ap.add_argument("--served-name", default="guardjudge")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    a = ap.parse_args()
    judge = LocalJudge(a.model, device=a.device, dtype=a.dtype)
    uvicorn.run(build_app(judge, a.served_name), host=a.host, port=a.port)


if __name__ == "__main__":
    main()
