"""Custom NeMo Guardrails actions that call the GuardJudge model.

Environment variables:
  GUARDJUDGE_BASE_URL   default http://localhost:8001/v1
  GUARDJUDGE_MODEL      default guardjudge
  GUARDJUDGE_FAIL_OPEN  1 = allow when the judge is unreachable (default 0 = block)
"""
from typing import Optional

from nemoguardrails.actions import action
from nemoguardrails.actions.actions import ActionResult

from guardjudge.client import JudgeClient

_client = JudgeClient()


@action(is_system_action=True)
async def guardjudge_check(stage: str, text: Optional[str] = None, context: Optional[str] = None) -> bool:
    """True if the judge considers the fragment unsafe."""
    verdict = await _client.judge(stage, text or "", context)
    return verdict.unsafe


@action(is_system_action=True)
async def guardjudge_filter_chunks(context: Optional[dict] = None):
    """Drop retrieved RAG chunks that the judge marks as unsafe (e.g. indirect prompt injection)."""
    chunks = (context or {}).get("relevant_chunks") or []
    if isinstance(chunks, str):
        chunks = [chunks]
    kept = []
    for chunk in chunks:
        verdict = await _client.judge("rag", str(chunk))
        if not verdict.unsafe:
            kept.append(chunk)
    return ActionResult(return_value=len(chunks) - len(kept), context_updates={"relevant_chunks": kept})
