"""Async client for the judge served behind an OpenAI-compatible API (scripts/serve.py / vLLM)."""
from __future__ import annotations

import os

from openai import AsyncOpenAI

from .schema import Verdict, build_messages, parse_verdict


class JudgeClient:
    def __init__(self, base_url: str | None = None, model: str | None = None,
                 api_key: str = "EMPTY", timeout: float = 30.0, fail_open: bool | None = None):
        self.base_url = base_url or os.getenv("GUARDJUDGE_BASE_URL", "http://localhost:8001/v1")
        self.model = model or os.getenv("GUARDJUDGE_MODEL", "guardjudge")
        if fail_open is None:
            fail_open = os.getenv("GUARDJUDGE_FAIL_OPEN", "0") == "1"
        self.fail_open = fail_open
        self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, timeout=timeout)

    async def judge(self, stage: str, text: str, context: str | None = None) -> Verdict:
        try:
            resp = await self._client.chat.completions.create(
                model=self.model,
                messages=build_messages(stage, text, context),
                temperature=0.0,
                max_tokens=32,
                # Qwen3: no <think> block. Understood by vLLM and scripts/serve.py.
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            verdict = parse_verdict(resp.choices[0].message.content or "")
        except Exception:
            # Fail closed by default: if the judge is unavailable, block.
            return Verdict(unsafe=not self.fail_open, category="none", parsed_ok=False)
        if not verdict.parsed_ok:
            # An answer we cannot read is not a "safe" verdict: same policy as an unavailable judge.
            return Verdict(unsafe=not self.fail_open, category="none", parsed_ok=False)
        return verdict
